#!/usr/bin/env python3
"""
Noise diagnostics: is an unexpected flakeval result chance, a scoring artifact, a calibration problem, or a real effect?

Use it when a same-system control (the "A/A check") comes out flagged as a real difference, or whenever you want evidence
that a study's numbers can be trusted. It reads the same settings file as the study.

  analyze      Offline, no model calls, a few seconds. Re-scores the saved raw answers (strict and lenient), then:
                 1. shows the accuracy of every individual run,
                 2. tests whether runs of the same configuration are really interchangeable (what flakeval assumes), with
                    an exact test that shuffles each question's results across runs,
                 3. measures flakeval's false-alarm rate on data shaped like yours,
                 4. splits a strict-scoring gap into "formatting" and "correctness",
                 5. shows which questions produced the gap.
  determinism  About 5 minutes of GPU. Re-asks saved questions with the SAME seeds and checks the answers come back
               identical. If they do not, the server itself adds noise that flakeval cannot see by design.
  replicate    Hours of GPU (it is the strongest evidence). Runs extra same-configuration repeats with new seeds, so
               that analyze has enough runs to test the false-alarm rate properly.

  python examples/template_case_studies/noise_diagnostics.py analyze --dir results_template_gsm8k
  python examples/template_case_studies/noise_diagnostics.py determinism --dir results_template_gsm8k
  python examples/template_case_studies/noise_diagnostics.py replicate --dir results_template_gsm8k --extra 2

Add --settings path/to/other_settings.py for a different study (default: gsm8k_eval.py next to this file).
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from flakeval import Scores, compare, study
from flakeval.models import call_seed, make_caller
from flakeval.progress import Progress, hms
from flakeval.scoring import ANSWER_LINE, fold_text, score_output

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ loading
def config_for(settings_path: str | Path, out_dir: str | None = None) -> dict:
    spec = importlib.util.spec_from_file_location("study_settings", settings_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = study.config_from_settings(vars(module))
    if out_dir:
        cfg["out_dir"] = out_dir
    return cfg


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def available_systems(out_dir: Path) -> list[str]:
    suffix = ".outputs.jsonl"
    return sorted(p.name[: -len(suffix)] for p in Path(out_dir).glob(f"*{suffix}"))


def scores_from_records(records: list[dict], items: dict, scorer: str, custom=None) -> Scores:
    """Re-mark the saved raw answers with `scorer` (the saved scores are ignored)."""
    by_item: dict[str, dict[int, float]] = defaultdict(dict)
    for r in records:
        item = items[r["item_id"]]
        by_item[r["item_id"]][int(r["run"])] = score_output(scorer, r["answer"], item["expected"], item["row"], custom)
    ids = sorted(by_item)
    k = min(len(v) for v in by_item.values())
    return Scores(ids, [[by_item[i][run] for run in sorted(by_item[i])[:k]] for i in ids])


def same_config_names(baseline: str, available: list[str]) -> list[str]:
    """The baseline plus its A/A repeats: <baseline>_repeat, <baseline>_repeat2, ..."""
    pattern = re.compile(rf"^{re.escape(baseline)}_repeat(\d*)$")
    repeats = sorted((n for n in available if pattern.match(n)), key=lambda n: int(pattern.match(n).group(1) or 1))
    return ([baseline] if baseline in available else []) + repeats


def sampling_seed_for(name: str, baseline: str) -> int:
    """The base seed the study used for this system (baseline 0, <baseline>_repeat 1, <baseline>_repeatN N)."""
    m = re.fullmatch(rf"{re.escape(baseline)}_repeat(\d*)", name)
    return int(m.group(1) or 1) if m else 0


# ------------------------------------------------------------------ statistics
def system_effect_permutation(groups: list[np.ndarray], n_perm: int = 20000, seed: int = 0) -> float:
    """Do systems differ in mean accuracy more than chance? Shuffles individual runs between systems.
    `groups` holds one array of per-run accuracies per system. Under the null every run is interchangeable."""
    sizes = {len(g) for g in groups}
    if len(groups) < 2 or len(sizes) != 1:
        return float("nan")
    k, s = sizes.pop(), len(groups)
    pooled = np.concatenate(groups)
    observed = np.var([g.mean() for g in groups], ddof=1)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        means = rng.permutation(pooled).reshape(s, k).mean(axis=1)
        hits += np.var(means, ddof=1) >= observed - 1e-12
    return float((hits + 1) / (n_perm + 1))


def system_spread(groups: list[np.ndarray], n_perm: int = 2000, seed: int = 0) -> tuple[float, float]:
    """(observed SD of the systems' mean accuracies, the SD expected if the same runs were randomly assigned to systems)."""
    k, s = len(groups[0]), len(groups)
    pooled = np.concatenate(groups)
    rng = np.random.default_rng(seed)
    null = [np.std(rng.permutation(pooled).reshape(s, k).mean(axis=1), ddof=1) for _ in range(n_perm)]
    return float(np.std([g.mean() for g in groups], ddof=1)), float(np.mean(null))


def run_dispersion_test(X: np.ndarray, n_shuffles: int = 5000, seed: int = 0) -> tuple[float, float, float, float]:
    """Do individual runs differ in accuracy by more than chance? X has one column per run (rows = questions).

    Null: within every question the runs are interchangeable (what flakeval assumes). It is simulated by shuffling each
    question's results across the runs, which keeps every question's difficulty and removes any run-level structure.
    Returns (observed SD of run accuracies, null mean SD, null 95th percentile SD, p-value)."""
    rng = np.random.default_rng(seed)
    observed = float(X.mean(axis=0).std(ddof=1))
    null = np.array([rng.permuted(X, axis=1).mean(axis=0).std(ddof=1) for _ in range(n_shuffles)])
    p = (1 + int((null >= observed - 1e-12).sum())) / (n_shuffles + 1)
    return observed, float(null.mean()), float(np.quantile(null, 0.95)), float(p)


def agreement_test(X: np.ndarray, k: int, n_shuffles: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Do runs of the SAME system agree with each other more than with runs of OTHER systems?
    (Related seeds, a shared server state, or a shared prompt cache could make them agree.) Columns of X are runs, k per
    system. Statistic: mean agreement of same-system run pairs minus mean agreement of different-system pairs.
    Null: each question's results shuffled across all runs. Returns (observed, null mean, one-sided p)."""
    runs_total = X.shape[1]
    system_of = np.repeat(np.arange(runs_total // k), k)
    upper = np.triu(np.ones((runs_total, runs_total), dtype=bool), 1)
    same = upper & (system_of[:, None] == system_of[None, :])
    different = upper & (system_of[:, None] != system_of[None, :])
    if k < 2 or not different.any():
        return float("nan"), float("nan"), float("nan")

    def statistic(M: np.ndarray) -> float:
        agree = (M[:, :, None] == M[:, None, :]).mean(axis=0)
        return float(agree[same].mean() - agree[different].mean())

    observed = statistic(X)
    rng = np.random.default_rng(seed)
    null = np.array([statistic(rng.permuted(X, axis=1)) for _ in range(n_shuffles)])
    return observed, float(null.mean()), float((1 + int((null >= observed - 1e-12).sum())) / (n_shuffles + 1))


def pair_gap_permutation(Xa: np.ndarray, Xb: np.ndarray, n_shuffles: int = 5000, seed: int = 0) -> tuple[float, float]:
    """An independent check of flakeval's p-value for one comparison of two groups of runs (same null as above)."""
    k = Xa.shape[1]
    pooled = np.hstack([Xa, Xb])
    observed = float(Xb.mean() - Xa.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_shuffles):
        shuffled = rng.permuted(pooled, axis=1)
        hits += abs(shuffled[:, k:].mean() - shuffled[:, :k].mean()) >= abs(observed) - 1e-12
    return observed, float((hits + 1) / (n_shuffles + 1))


def false_alarm_calibration(X: np.ndarray, ids: list[str], k: int, n_datasets: int = 30, draws: int = 60,
                            B: int = 500, seed: int = 0) -> list[float]:
    """The share of random same-configuration groupings that flakeval flags as REAL, in datasets where the null is true
    by construction (each question's runs shuffled). Its average is flakeval's false-alarm rate for THIS data."""
    rng = np.random.default_rng(seed)
    shares = []
    for d in range(n_datasets):
        res = regroup(rng.permuted(X, axis=1), ids, k, draws, B, seed=seed + d * 10007)
        shares.append(sum(r[4].startswith("REAL") for r in res) / len(res))
    return shares


def regroup(X: np.ndarray, ids: list[str], k: int, max_draws: int, B: int, seed: int = 0) -> list[tuple]:
    """Compare random disjoint groups of k runs drawn from the pooled runs of ONE configuration.
    X has one column per run. Returns (group_a, group_b, delta, p, verdict) for each draw. If few distinct
    pairings exist, all of them are used."""
    runs_total = X.shape[1]
    pairs = []
    for ga in itertools.combinations(range(runs_total), k):
        rest = [r for r in range(runs_total) if r not in ga]
        pairs += [(ga, gb) for gb in itertools.combinations(rest, k) if ga < gb]
    rng = np.random.default_rng(seed)
    if len(pairs) > max_draws:
        pairs = []
        for _ in range(max_draws):
            perm = rng.permutation(runs_total)
            pairs.append((tuple(sorted(perm[:k])), tuple(sorted(perm[k : 2 * k]))))
    out = []
    for i, (ga, gb) in enumerate(pairs):
        c = compare(Scores(ids, X[:, list(ga)]), Scores(ids, X[:, list(gb)]), B=B, seed=seed + i)
        out.append((ga, gb, c.delta, c.p_value, c.verdict))
    return out


# ------------------------------------------------------------------ analyze
def build_report(cfg: dict, out_dir: Path, scorers: list[str], n_draws: int = 400, B: int = 2000,
                 n_perm: int = 20000) -> str:
    items = study.prepare_items(cfg)
    baseline = cfg["systems"][0]["name"]
    names = available_systems(out_dir)
    if not names:
        raise SystemExit(f"No *.outputs.jsonl files in {out_dir}. Run the study first (with SAVE_OUTPUTS = True), or "
                         f"pass --dir with the right results folder.")
    records = {n: read_records(out_dir / f"{n}.outputs.jsonl") for n in names}
    custom = cfg.get("custom_score")
    mats = {sc: {n: scores_from_records(records[n], items, sc, custom) for n in names} for sc in scorers}
    same = same_config_names(baseline, names)
    out = [f"Noise diagnostics for {out_dir}", f"baseline system: {baseline}",
           f"same-configuration systems found: {', '.join(same) or 'none'}", ""]

    # 1. every run
    out.append("1. ACCURACY OF EVERY INDIVIDUAL RUN")
    for sc in scorers:
        out.append(f"   scorer = {sc}")
        for n in names:
            x = mats[sc][n].x
            runs = "  ".join(f"{a:.3f}" for a in x.mean(axis=0))
            out.append(f"     {n:16s} runs: {runs}   mean {x.mean():.3f}   SD of runs {x.mean(axis=0).std(ddof=1) * 100:.1f} pts")
    out.append("")

    # 2. are runs exchangeable?
    out.append("2. ARE RUNS OF THE SAME CONFIGURATION INTERCHANGEABLE (WHAT FLAKEVAL ASSUMES)?")
    if len(same) < 2:
        out.append("   needs the baseline and at least one repeat")
    for sc in scorers if len(same) >= 2 else []:
        k = mats[sc][same[0]].x.shape[1]
        X = np.hstack([mats[sc][n].x for n in same])
        obs_sd, null_sd, null_q95, p_sd = run_dispersion_test(X, n_shuffles=n_perm // 4)
        out.append(f"   scorer = {sc}")
        out.append(f"     SD of the {X.shape[1]} run accuracies: {obs_sd * 100:.2f} pts observed; {null_sd * 100:.2f} pts expected "
                   f"if runs are interchangeable (95th percentile {null_q95 * 100:.2f}); p = {p_sd:.3f}")
        obs_ag, null_ag, p_ag = agreement_test(X, k, n_shuffles=max(n_perm // 10, 200))
        out.append(f"     runs of the same system agree with each other {obs_ag * 100:+.2f} points more often than with runs of "
                   f"other systems (expected {null_ag * 100:+.2f} if interchangeable); one-sided p = {p_ag:.3f}")
        groups = [mats[sc][n].x.mean(axis=0) for n in same]
        spread, expected = system_spread(groups, n_perm=n_perm // 4)
        out.append(f"     spread between the {len(groups)} systems' mean accuracies: SD {spread * 100:.2f} pts observed; "
                   f"{expected * 100:.2f} pts expected if the runs were randomly assigned to systems")
        if len(groups) >= 3:
            out.append(f"     systems as groups (shuffling whole runs between systems): p = "
                       f"{system_effect_permutation(groups, n_perm=n_perm):.3f}")
        for n in same[1:]:
            gap, p_perm = pair_gap_permutation(mats[sc][same[0]].x, mats[sc][n].x, n_shuffles=n_perm // 4)
            c = compare(mats[sc][same[0]], mats[sc][n], B=B)
            out.append(f"     {n} vs {same[0]}: gap {gap * 100:+.1f} pts; flakeval p = {c.p_value:.3f} ({c.verdict.lower()}); "
                       f"independent permutation p = {p_perm:.3f}")
    out.append("   Reading: a small p for the SD means some runs are better or worse than chance allows. A small p for the "
               "agreement means runs of one system resemble each other (related seeds or shared server state?). Either way the "
               "runs are NOT interchangeable. If both are unremarkable and flakeval's p agrees with the permutation p, the "
               "flagged control is consistent with chance.")
    out.append("")

    # 3. empirical false-alarm rate
    out.append("3. HOW OFTEN DOES FLAKEVAL RAISE A FALSE ALARM ON THIS KIND OF DATA?")
    if len(same) < 2:
        out.append("   needs the baseline and at least one repeat")
    for sc in scorers if len(same) >= 2 else []:
        k = mats[sc][same[0]].x.shape[1]
        ids = mats[sc][same[0]].ids
        X = np.hstack([mats[sc][n].x for n in same])
        res = regroup(X, ids, k, n_draws, B)
        flagged = sum(1 for r in res if r[4].startswith("REAL"))
        shares = false_alarm_calibration(X, ids, k, B=B // 4)
        out.append(f"   scorer = {sc}: in data where nothing can be going on (each question's runs shuffled), flakeval flagged "
                   f"{np.mean(shares):.1%} of random same-configuration comparisons on average (range {min(shares):.0%} to "
                   f"{max(shares):.0%} across {len(shares)} such datasets). A calibrated test is near 5%.")
        out.append(f"     in YOUR data, {flagged} of {len(res)} regroupings of the same runs were flagged ({flagged / len(res):.1%}); "
                   f"these regroupings share the same runs, so one dataset's share is noisy.")
        if len(same) >= 3:
            pairs = [(a, b, compare(mats[sc][a], mats[sc][b], B=B)) for a, b in itertools.combinations(same, 2)]
            real = [f"{a} vs {b} ({c.delta * 100:+.1f} pts, p = {c.p_value:.3f})" for a, b, c in pairs if c.verdict.startswith("REAL")]
            out.append(f"     all {len(pairs)} pairs of the {len(same)} same-configuration systems: flakeval flagged {len(real)} as a real "
                       f"difference" + (": " + "; ".join(real) if real else "") + " (these pairs share systems, so they are not independent)")
    out.append("   Reading: an average near or below 5% means flakeval is not over-confident on this data when runs are "
               "interchangeable. Section 2 says whether they are.")
    out.append("")

    # 4. formatting vs correctness
    out.append("4. HOW MUCH OF A STRICT-SCORING GAP IS FORMATTING?")
    if len(scorers) < 2 or len(same) < 2:
        out.append("   needs two scorers (strict then lenient) and at least one repeat")
    else:
        strict, lenient = scorers[0], scorers[1]
        for n in same[1:]:
            gs = mats[strict][n].x.mean() - mats[strict][same[0]].x.mean()
            gl = mats[lenient][n].x.mean() - mats[lenient][same[0]].x.mean()
            share = f"{(gs - gl) / gs:.0%} of the strict gap" if abs(gs) >= 0.01 else "gap under 1 point: too small to apportion"
            out.append(f"   {n} vs {same[0]}: {strict} gap {gs * 100:+.1f} pts, {lenient} gap {gl * 100:+.1f} pts, "
                       f"formatting part {(gs - gl) * 100:+.1f} pts ({share})")
    out.append("   Replies without an 'Answer:' line (strict scoring marks them wrong), per system:")
    out.append(f"     {'system':16s} {'replies':>8s} {'no Answer: line':>16s} {'rescued by lenient':>19s}")
    if len(scorers) >= 2:
        lenient = scorers[1]
        for n in names:
            recs = records[n]
            no_line = [r for r in recs if ANSWER_LINE.search(fold_text(r["answer"])) is None]
            rescued = sum(score_output(lenient, r["answer"], items[r["item_id"]]["expected"],
                                       items[r["item_id"]]["row"], custom) for r in no_line)
            out.append(f"     {n:16s} {len(recs):8d} {len(no_line):8d} ({len(no_line) / len(recs):5.1%}) {int(rescued):10d}")
    out.append("")

    # 5. where the gap comes from
    out.append("5. WHICH QUESTIONS PRODUCED THE BASELINE-VS-FIRST-REPEAT GAP?")
    if len(same) < 2:
        out.append("   needs the baseline and at least one repeat")
    else:
        sc = scorers[0]
        a, b = mats[sc][same[0]], mats[sc][same[1]]
        d = b.x.mean(axis=1) - a.x.mean(axis=1)
        order = np.argsort(-np.abs(d))
        top10 = np.abs(d[order[:10]]).sum() / max(np.abs(d).sum(), 1e-12)
        out.append(f"   scorer = {sc}: {int((np.abs(d) > 1e-12).sum())} of {len(d)} questions differ at all; the 10 largest "
                   f"differences make up {top10:.0%} of the total absolute difference; mean difference {d.mean() * 100:+.1f} pts")
        for i in order[:5]:
            out.append(f"     {a.ids[i]}: {same[0]} {a.x[i].mean():.2f}, {same[1]} {b.x[i].mean():.2f}")
    # 6. timeline
    out.append("")
    out.append("6. DOES THE GAP DEPEND ON WHEN THE QUESTIONS WERE ASKED?")
    if len(same) < 2:
        out.append("   needs the baseline and at least one repeat")
    else:
        sc = scorers[0]
        order = [i for i in items if i in set(mats[sc][same[0]].ids)]          # the order the study asked them in
        blocks = np.array_split(np.arange(len(order)), 4)
        out.append(f"   scorer = {sc}; questions in the order they were asked, in four blocks (each question's runs were asked together)")
        out.append(f"     {'block':8s} " + " ".join(f"{n:>14s}" for n in same))
        for bi, idx in enumerate(blocks, 1):
            cells = []
            for n in same:
                pos = {q: j for j, q in enumerate(mats[sc][n].ids)}
                cells.append(f"{np.mean([mats[sc][n].x[pos[order[i]]].mean() for i in idx]):14.3f}")
            out.append(f"     {bi:<8d} " + " ".join(cells))
        out.append("   Reading: a gap that is present in every block points to a uniform difference between the runs; a gap in only "
                   "some blocks points to something that changed during the run.")
    out += ["",
            "HOW TO READ THIS",
            "  - Section 2 not significant and flakeval's p close to the permutation p, and section 3 near or below 5%:",
            "    the flagged control is consistent with chance. Report it, and keep the same-system control in every study.",
            "  - Section 2 significant (runs differ more than chance allows): flakeval's p-values are too optimistic for this",
            "    task. Treat borderline p-values (0.01 to 0.05) as unreliable here, say so, and run `determinism`.",
            "  - Section 3 clearly above 5%: flakeval itself is over-confident on this kind of data; do not publish p-values",
            "    without that caveat.",
            "  - Section 4 large: the strict-scoring gap is mostly formatting, not correctness. Prefer the lenient result."]
    return "\n".join(out)


# ------------------------------------------------------------------ determinism
def determinism_check(cfg: dict, out_dir: Path, system_name: str, n_cells: int, seed: int = 0, show=print,
                      mode: str = "random") -> dict:
    """Re-ask saved answers with the SAME seeds and compare.

    mode "random":   individual (question, run) cells, asked in random order, each after an unrelated question.
    mode "sequence": whole questions, all runs asked back to back in the original order (as the study did), so any
                     server-side reuse of the previous request (a prompt cache) happens the same way as before.
    If "sequence" reproduces the answers but "random" does not, the server's answers depend on the request history."""
    items = study.prepare_items(cfg)
    baseline = cfg["systems"][0]
    base_name = baseline["name"]
    system = next((s for s in cfg["systems"] if s["name"] == system_name), baseline if system_name.startswith(base_name) else None)
    if system is None:
        raise SystemExit(f"Unknown system {system_name!r}.")
    path = out_dir / f"{system_name}.outputs.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run the study first, or pass --dir.")
    saved = {(r["item_id"], int(r["run"])): r["answer"] for r in read_records(path)}
    rng = random.Random(seed)
    if mode == "sequence":
        ids = sorted({i for i, _ in saved})
        chosen = rng.sample(ids, min(max(1, n_cells // max(1, len({r for _, r in saved}))), len(ids)))
        cells = [(i, r) for i in chosen for r in sorted(r for j, r in saved if j == i)]      # per question, runs in order
    else:
        cells = rng.sample(sorted(saved), min(n_cells, len(saved)))
    sampling_seed = sampling_seed_for(system_name, base_name)
    caller = make_caller(system, cfg["max_tokens"])
    template = system.get("prompt_template", "{input}")
    progress = Progress(len(cells))
    progress.start_system(f"{system_name} ({mode})", 1, 1, len(cells))
    by_run: dict[int, list[int]] = defaultdict(lambda: [0, 0])          # run -> [checked, identical]
    different, score_changed = [], 0
    for item_id, run in cells:
        item = items[item_id]
        answer = caller(template.replace("{input}", item["input"]), system.get("system_prompt") or "",
                        system.get("temperature"), call_seed(sampling_seed, item_id, run))
        by_run[run][0] += 1
        if answer == saved[(item_id, run)]:
            by_run[run][1] += 1
        else:
            old = saved[(item_id, run)]
            common = next((i for i in range(min(len(old), len(answer))) if old[i] != answer[i]), min(len(old), len(answer)))
            different.append((item_id, run, common))
            a = score_output(cfg["scorer"], old, item["expected"], item["row"], cfg.get("custom_score"))
            b = score_output(cfg["scorer"], answer, item["expected"], item["row"], cfg.get("custom_score"))
            score_changed += int(a != b)
        progress.advance()
    progress.close()
    identical = sum(v[1] for v in by_run.values())
    summary = {"mode": mode, "checked": len(cells), "identical": identical, "different": len(different),
               "score_changed": score_changed, "by_run": {r: tuple(v) for r, v in sorted(by_run.items())}}
    show(f"\n[{mode}] Re-asked {len(cells)} saved answers of {system_name!r} with the same seeds (base {sampling_seed}): "
         f"{identical} came back identical, {len(different)} differed, {score_changed} of those changed score.")
    show("  by run: " + ", ".join(f"run {r}: {v[1]}/{v[0]} identical" for r, v in sorted(by_run.items())))
    for item_id, run, common in different[:3]:
        show(f"  differs: {item_id} run {run} (identical for the first {common} characters)")
    return summary


DETERMINISM_READING = (
    "Reading: all identical in both modes = the server is deterministic for a given seed. Identical in 'sequence' but not in "
    "'random' = the answer depends on the request history (for example a reused prompt cache), not on random noise. "
    "Different in both = genuine server-side noise (non-deterministic GPU arithmetic). In every case flakeval's comparisons "
    "stay valid as long as both systems were run the same way, but exact reproduction of a study is not possible.")


# ------------------------------------------------------------------ replicate
def replicate(cfg: dict, out_dir: Path, extra: int, first_seed: int = 2, yes: bool = False) -> list[str]:
    cfg = {**cfg, "out_dir": str(out_dir)}
    items = study.prepare_items(cfg)
    baseline = cfg["systems"][0]
    plan = [(f"{baseline['name']}_repeat{s}", s) for s in range(first_seed, first_seed + extra)]
    todo = [(n, s) for n, s in plan if not (out_dir / f"{n}.jsonl").exists()]
    calls = len(items) * cfg["runs"] * len(todo)
    for n, _ in plan:
        if (out_dir / f"{n}.jsonl").exists():
            print(f"  skipping {n}: already exists")
    print(f"{len(items)} questions x {cfg['runs']} runs x {len(todo)} extra repeat(s) = {calls} model calls")
    if todo and not yes:
        if input("Proceed? [y/N] ").strip().lower() != "y":
            raise SystemExit("Cancelled.")
    progress = Progress(calls) if todo else None
    try:
        for k, (name, seed) in enumerate(todo, 1):
            progress.start_system(name, k, len(todo), len(items) * cfg["runs"])
            t0 = time.time()
            scores, _ = study.evaluate_system(baseline, name, items, cfg, seed, cfg["runs"], save=True, progress=progress)
            progress.message(f"  {name}: mean score {scores.mean:.3f} ({hms(time.time() - t0)})")
    finally:
        if progress:
            progress.close()
    return [n for n, _ in plan]


# ------------------------------------------------------------------ command line
def analyze_command(out_dir) -> str:
    """The command to run next. Forward slashes on purpose: a Windows path with backslashes breaks when pasted into
    Git Bash, which treats the backslash as an escape character."""
    return f"python examples/template_case_studies/noise_diagnostics.py analyze --dir {Path(out_dir).as_posix()}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["analyze", "determinism", "replicate"])
    p.add_argument("--settings", default=str(HERE / "gsm8k_eval.py"), help="the study's settings file")
    p.add_argument("--dir", help="results folder (default: the settings file's OUT_DIR)")
    p.add_argument("--scorers", nargs="+", default=["answer_line", "final_number"], help="strict then lenient scorer")
    p.add_argument("--draws", type=int, default=400, help="analyze: random regroupings per scorer")
    p.add_argument("--system", help="determinism: which system's answers to re-ask (default: the baseline)")
    p.add_argument("--cells", type=int, default=60, help="determinism: how many saved answers to re-ask (per mode)")
    p.add_argument("--mode", choices=["random", "sequence", "both"], default="both",
                   help="determinism: random cells, whole questions in the original back-to-back order, or both")
    p.add_argument("--extra", type=int, default=2, help="replicate: how many extra repeats")
    p.add_argument("--first-seed", type=int, default=2, help="replicate: base seed of the first extra repeat")
    p.add_argument("--yes", action="store_true")
    a = p.parse_args(argv)
    cfg = config_for(a.settings, a.dir)
    out_dir = Path(cfg["out_dir"])
    if a.command == "analyze":
        report = build_report(cfg, out_dir, a.scorers, n_draws=a.draws)
        (out_dir / "noise_diagnostics.txt").write_text(report + "\n", encoding="utf-8")
        print(report)
        print(f"\nsaved to {out_dir / 'noise_diagnostics.txt'}")
    elif a.command == "determinism":
        for mode in (["random", "sequence"] if a.mode == "both" else [a.mode]):
            determinism_check(cfg, out_dir, a.system or cfg["systems"][0]["name"], a.cells, mode=mode)
        print("\n" + DETERMINISM_READING)
    else:
        replicate(cfg, out_dir, a.extra, a.first_seed, a.yes)
        print("\nNext: " + analyze_command(out_dir))


if __name__ == "__main__":
    main()
