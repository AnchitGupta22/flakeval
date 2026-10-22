"""The engine behind the template: load your data, ask each system your questions several times, score the
answers, and report whether the differences are real. Nothing here needs editing; users write a small settings file
(see `flakeval init`) or build a settings dictionary and call run_study().

    from flakeval.study import default_config, run_study
    cfg = default_config()
    cfg.update(data_path="my.csv", input_column="question", expected_column="answer", systems=[...])
    run_study(cfg, yes=True)
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import random
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from .cli import format_comparison, format_summary
from .io import load_jsonl, repeat, save_jsonl
from .models import call_seed, make_caller
from .progress import Progress, hms
from .scoring import SCORER_HELP, SCORERS, question_hash, score_output
from .stats import Scores, compare, summarize


# ---------------------------------------------------------------- configuration
# The names a user writes in their settings file (left) and the keys the engine uses internally (right).
SETTING_NAMES = {
    "DATA_PATH": "data_path", "INPUT_COLUMN": "input_column", "EXPECTED_COLUMN": "expected_column",
    "ID_COLUMN": "id_column", "MAX_ITEMS": "max_items", "ITEM_SEED": "item_seed", "SCORER": "scorer",
    "SYSTEMS": "systems", "RUNS": "runs", "MAX_TOKENS": "max_tokens", "WORKERS": "workers",
    "RUN_A_A_TEST": "run_a_a_test", "OUT_DIR": "out_dir", "SAVE_OUTPUTS": "save_outputs",
}
DEFAULTS = dict(
    data_path=None, input_column=None, expected_column=None, id_column=None, max_items=None, item_seed=0,
    scorer="contains", custom_score=None, systems=None, runs=5, max_tokens=300, workers=1, run_a_a_test=True,
    out_dir="my_results", save_outputs=True,
)
REQUIRED = {"data_path": "DATA_PATH", "input_column": "INPUT_COLUMN", "systems": "SYSTEMS"}


def default_config() -> dict:
    """A fresh settings dictionary holding every default. Fill in the required keys and pass it to run_study()."""
    return dict(DEFAULTS)


def complete_config(cfg: dict, need_systems: bool = True) -> dict:
    """Fill in defaults and check the settings, with plain-English errors.
    (Loading data alone does not need SYSTEMS, so prepare_items() passes need_systems=False.)"""
    unknown = sorted(set(cfg) - set(DEFAULTS))
    if unknown:
        raise SystemExit(f"Unknown setting(s): {unknown}. Known settings: {sorted(DEFAULTS)}")
    merged = {**DEFAULTS, **cfg}
    if merged["scorer"] not in SCORERS:  # fail now, not after hours of model calls
        raise SystemExit(f"Unknown SCORER {merged['scorer']!r}. Choose one of:\n{SCORER_HELP}")
    if merged["scorer"] == "custom" and not callable(merged["custom_score"]):
        raise SystemExit('SCORER = "custom" needs a function called custom_score(output, expected, row) in your file.')
    for key, label in REQUIRED.items():
        if key == "systems" and not need_systems:
            continue
        if merged[key] in (None, "", []):
            raise SystemExit(f"Missing required setting {label}.")
    return merged


def config_from_settings(settings: Mapping) -> dict:
    """Read the ALL-CAPS settings (plus a function called custom_score) from a settings file's namespace.

    Typically called as run_cli(globals()). A name that looks like a misspelled setting (for example RUN instead
    of RUNS) stops the run with a hint, because silently using the default would waste a long run."""
    known = list(SETTING_NAMES)
    for name in settings:
        if name.isupper() and not name.startswith("_") and name not in SETTING_NAMES:
            close = difflib.get_close_matches(name, known, n=1, cutoff=0.8)
            if close:
                raise SystemExit(f"Unknown setting {name}. Did you mean {close[0]}? (If {name} is your own variable, "
                                 f"rename it so it does not look like a setting, for example _{name}.)")
    cfg = {key: settings[name] for name, key in SETTING_NAMES.items() if name in settings}
    if callable(settings.get("custom_score")):
        cfg["custom_score"] = settings["custom_score"]
    return complete_config(cfg)


# ---------------------------------------------------------------- data
def load_rows(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Data file not found: {path}  (check DATA_PATH; it is relative to where you run the script)")
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".csv":
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    else:
        raise SystemExit(f"Data file must end in .csv or .jsonl, got: {path}")
    if not rows:
        raise SystemExit(f"Data file is empty: {path}")
    return rows


def prepare_items(cfg: dict) -> dict[str, dict]:
    cfg = complete_config(cfg, need_systems=False)
    rows = load_rows(cfg["data_path"])
    available = list(rows[0].keys())
    for label, col in (("INPUT_COLUMN", cfg["input_column"]), ("EXPECTED_COLUMN", cfg["expected_column"]),
                       ("ID_COLUMN", cfg["id_column"])):
        if col is not None and col not in available:
            raise SystemExit(f"{label} = {col!r} is not a column in your data. Columns found: {available}")
    items: dict[str, dict] = {}
    for i, row in enumerate(rows):
        item_id = str(row[cfg["id_column"]]) if cfg["id_column"] else f"row{i}"
        if item_id in items:
            raise SystemExit(f"Duplicate id {item_id!r}. ID_COLUMN must be unique per row (or set it to None).")
        expected = row[cfg["expected_column"]] if cfg["expected_column"] else None
        items[item_id] = {"id": item_id, "input": str(row[cfg["input_column"]]), "expected": expected, "row": row}
    if cfg["max_items"] and cfg["max_items"] < len(items):
        keep = random.Random(cfg["item_seed"]).sample(sorted(items), cfg["max_items"])
        items = {k: items[k] for k in sorted(keep)}
    return items


# ---------------------------------------------------------------- running
def evaluate_system(system: dict, name: str, items: dict, cfg: dict, sampling_seed: int, runs: int, save: bool,
                    progress: "Progress | None" = None):
    caller = make_caller(system, cfg["max_tokens"])
    template = system.get("prompt_template", "{input}")
    records, lock = [], threading.Lock()
    out_dir = Path(cfg["out_dir"])

    def score(item: dict, run_idx: int) -> float:
        prompt = template.replace("{input}", item["input"])
        answer = caller(prompt, system.get("system_prompt") or "", system.get("temperature"),
                        call_seed(sampling_seed, item["id"], run_idx))
        value = score_output(cfg["scorer"], answer, item["expected"], item["row"], cfg.get("custom_score"))
        with lock:
            records.append({"item_id": item["id"], "run": run_idx, "score": value, "answer": answer,
                            "question_hash": question_hash(item["input"])})
        if progress:
            progress.advance()
        return value

    path = out_dir / f"{name}.jsonl" if save else None
    scores = repeat(score, items, runs=runs, workers=cfg["workers"], save=path, with_run=True)
    if save and cfg["save_outputs"]:
        with open(out_dir / f"{name}.outputs.jsonl", "w", encoding="utf-8") as f:
            for r in sorted(records, key=lambda r: (r["item_id"], r["run"])):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return scores, records


def shorten(text: str, head: int = 150, tail: int = 250) -> str:
    """Show the start AND the end of a long answer: with step-by-step answers the final result is at the end."""
    if len(text) <= head + tail:
        return repr(text)
    return f"{text[:head]!r} ... [{len(text) - head - tail} characters not shown] ... {text[-tail:]!r}"


def dry_run(cfg: dict, items: dict) -> None:
    print("DRY RUN: 1 run on the first 2 questions per system. Nothing is saved.\n")
    sample = dict(list(items.items())[:2])
    for system in cfg["systems"]:
        print(f"=== system {system['name']!r} ===")
        _, records = evaluate_system(system, system["name"], sample, {**cfg, "workers": 1}, 0, 1, save=False)
        for r in sorted(records, key=lambda r: r["item_id"]):
            item = items[r["item_id"]]
            print(f"question: {item['input'][:200]}\nexpected: {item['expected']}\nanswer:   {shorten(r['answer'])}\n"
                  f"score:    {r['score']}  (scorer = {cfg['scorer']})\n")
    print("If the answers and scores look right, run again without --dry-run.")


def run_study(cfg: dict, dry: bool = False, yes: bool = False) -> str:
    """Run every system on every question RUNS times, then print and save the report. Returns the report text."""
    cfg = complete_config(cfg)
    items = prepare_items(cfg)
    systems = cfg["systems"]
    if len(systems) < 2:
        raise SystemExit("SYSTEMS needs at least two entries: a baseline and something to compare it with.")
    names = [s["name"] for s in systems]
    if len(set(names)) != len(names):
        raise SystemExit("Each system needs a different 'name'.")
    if dry:
        dry_run(cfg, items)
        return ""

    out_dir = Path(cfg["out_dir"])
    jobs = [(s, s["name"], 0) for s in systems]
    if cfg["run_a_a_test"]:
        jobs.append((systems[0], f"{systems[0]['name']}_repeat", 1))  # same system, independent random seeds
    todo = [j for j in jobs if not (out_dir / f"{j[1]}.jsonl").exists()]
    calls = len(items) * cfg["runs"] * len(todo)
    print(f"{len(items)} questions x {cfg['runs']} runs x {len(todo)} system runs to do = {calls} model calls")
    for _, name, _ in jobs:
        if (out_dir / f"{name}.jsonl").exists():
            print(f"  skipping {name}: {out_dir / (name + '.jsonl')} already exists (delete it to redo)")
    if todo and not yes:
        if not sys.stdin.isatty():
            raise SystemExit("Not running interactively: pass --yes to confirm the number of calls above.")
        if input("Proceed? [y/N] ").strip().lower() != "y":
            raise SystemExit("Cancelled.")

    progress = Progress(calls) if todo else None
    try:
        for k, (system, name, seed) in enumerate(todo, 1):
            progress.start_system(name, k, len(todo), len(items) * cfg["runs"])
            t0 = time.time()
            scores, _ = evaluate_system(system, name, items, cfg, seed, cfg["runs"], save=True, progress=progress)
            progress.message(f"  {name}: mean score {scores.mean:.3f} ({hms(time.time() - t0)})")
    finally:
        if progress:
            progress.close()

    return write_report(cfg, names)


def write_report(cfg: dict, names: list[str]) -> str:
    out_dir = Path(cfg["out_dir"])
    baseline = load_jsonl(out_dir / f"{names[0]}.jsonl")
    lines = ["=" * 72, f"flakeval report: baseline = {names[0]}", "=" * 72, "",
             format_summary(names[0], summarize(baseline)), ""]
    comparisons = [(f"{names[0]}_repeat", "A/A test: the baseline against itself (should be WITHIN NOISE)")] \
        if cfg["run_a_a_test"] else []
    comparisons += [(n, f"{names[0]} vs {n}") for n in names[1:]]
    for name, title in comparisons:
        block = format_comparison(compare(baseline, load_jsonl(out_dir / f"{name}.jsonl")))
        if name.endswith("_repeat") and "IDENTICAL RESULTS" in block:
            block += ("\nNote: for this A/A test, identical results are the best possible outcome. The baseline gave the "
                      "same mark on every question in both runs, so on these questions chance did not change any score. "
                      "(It also means this data cannot show run-to-run noise; harder questions or more of them might.)")
        lines += [f"--- {title} ---", block, ""]
    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"Saved to {out_dir}/ (report.txt, one .jsonl per system" +
          (", and .outputs.jsonl with raw answers)" if cfg["save_outputs"] else ")"))
    return report


def rescore_study(cfg: dict) -> str:
    """Re-mark the SAVED raw answers with the current scorer and rebuild the report. Calls no model.
    Fixing a scorer or a wrong expected answer is fine; editing the question text is refused (the answers no longer fit)."""
    cfg = complete_config(cfg)
    items = prepare_items(cfg)
    out_dir = Path(cfg["out_dir"])
    names = [s["name"] for s in cfg["systems"]]
    jobs = names + ([f"{names[0]}_repeat"] if cfg["run_a_a_test"] else [])
    # extra same-configuration repeats (for example added by noise_diagnostics.py replicate) are re-marked too
    suffix = ".outputs.jsonl"
    jobs += [n for n in sorted(p.name[: -len(suffix)] for p in out_dir.glob(f"{names[0]}_repeat*{suffix}")) if n not in jobs]
    for name in jobs:
        path = out_dir / f"{name}.outputs.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} not found. --rescore needs the saved raw answers (SAVE_OUTPUTS = True) from an "
                             f"earlier run with the same OUT_DIR and system names.")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        before = sum(r["score"] for r in records) / len(records)
        by_item: dict[str, dict[int, float]] = {}
        for r in records:
            if r["item_id"] not in items:
                raise SystemExit(f"Item {r['item_id']!r} in {path.name} is not in your current data. Is DATA_PATH / "
                                 f"ID_COLUMN / MAX_ITEMS the same as in the original run?")
            item = items[r["item_id"]]
            if r.get("question_hash") and r["question_hash"] != question_hash(item["input"]):
                raise SystemExit(f"Question {r['item_id']!r} has been edited since this run, so its saved answers belong "
                                 f"to the old wording. Rescoring would mix them up; delete {out_dir}/ and run again.")
            r["score"] = score_output(cfg["scorer"], r["answer"], item["expected"], item["row"], cfg.get("custom_score"))
            by_item.setdefault(r["item_id"], {})[int(r["run"])] = r["score"]
        ids = sorted(by_item)
        k = min(len(by_item[i]) for i in ids)
        scores = Scores(ids, [[by_item[i][run] for run in sorted(by_item[i])[:k]] for i in ids])
        save_jsonl(scores, out_dir / f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name}: mean score {before:.3f} -> {scores.mean:.3f}")
    print()
    return write_report(cfg, names)


# ---------------------------------------------------------------- command line for a settings file
def run_cli(settings: Mapping, argv: list[str] | None = None) -> None:
    """The last line of a settings file: `run_cli(globals())`.

    Adds --dry-run, --yes and --rescore, plus quick overrides of a few settings so you can try something small
    without editing the file: --max-items, --runs, --workers, --out-dir and --scorer."""
    p = argparse.ArgumentParser(description="Compare your models/prompts on your data with flakeval.")
    p.add_argument("--dry-run", action="store_true", help="try 2 questions once per system and print what happens")
    p.add_argument("--yes", action="store_true", help="skip the confirmation question before the real run")
    p.add_argument("--rescore", action="store_true",
                   help="re-mark the saved answers with your current scorer and rebuild the report (no model calls)")
    p.add_argument("--max-items", type=int, help="override MAX_ITEMS: use only a random subset of this many questions")
    p.add_argument("--runs", type=int, help="override RUNS: times each question is asked per system")
    p.add_argument("--workers", type=int, help="override WORKERS: parallel requests")
    p.add_argument("--out-dir", help="override OUT_DIR: where results are written")
    p.add_argument("--scorer", choices=SCORERS, help="override SCORER (for example to re-mark saved answers)")
    a = p.parse_args(argv)
    cfg = config_from_settings(settings)
    for flag, key in (("max_items", "max_items"), ("runs", "runs"), ("workers", "workers"),
                      ("out_dir", "out_dir"), ("scorer", "scorer")):
        if getattr(a, flag) is not None:
            cfg[key] = getattr(a, flag)
    cfg = complete_config(cfg)
    if a.rescore:
        rescore_study(cfg)
    else:
        run_study(cfg, dry=a.dry_run, yes=a.yes)
