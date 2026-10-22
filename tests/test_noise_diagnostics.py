"""Tests for examples/template_case_studies/noise_diagnostics.py, using simulated results with a known planted truth."""
import csv
import importlib.util
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from flakeval import study  # noqa: E402
from test_study import FakeServer  # noqa: E402

spec = importlib.util.spec_from_file_location("noise_diagnostics", ROOT / "examples" / "template_case_studies" / "noise_diagnostics.py")
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)


# ------------------------------------------------------------ simulated studies
def item_probabilities(n, rng):
    """Difficulty shaped like a reasoning benchmark: about a third of the questions are unstable."""
    return np.where(rng.random(n) < 0.35, rng.uniform(0.25, 0.75, n), np.where(rng.random(n) < 0.7, 0.97, 0.03))


def write_study(tmp_path, spec, n=80, runs=3, seed=0, boxed_share=0.2):
    """Writes data.csv and one .outputs.jsonl per system. `spec` maps system name -> accuracy shift (a batch effect)."""
    rng = np.random.default_rng(seed)
    p = item_probabilities(n, rng)
    data = tmp_path / "data.csv"
    gold = {f"q{i}": 100 + i for i in range(n)}
    with open(data, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "answer"])
        for i in range(n):
            w.writerow([f"q{i}", f"What is {i} plus 100?", gold[f"q{i}"]])
    out = tmp_path / "res"
    out.mkdir()
    boxed_cells = {}
    for name, shift in spec.items():
        rows, boxed = [], 0
        for i in range(n):
            for r in range(runs):
                ok = rng.random() < min(max(p[i] + shift, 0), 1)
                value = gold[f"q{i}"] if ok else gold[f"q{i}"] + 1
                if ok and rng.random() < boxed_share:
                    answer, boxed = f"so the result is \\boxed{{{value}}}", boxed + 1       # correct, but no Answer: line
                else:
                    answer = f"reasoning...\nAnswer: {value}"
                rows.append({"item_id": f"q{i}", "run": r, "score": 0.0, "answer": answer})
        (out / f"{name}.outputs.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
        boxed_cells[name] = boxed
    cfg = study.default_config()
    cfg.update(data_path=str(data), input_column="question", expected_column="answer", id_column="id",
               systems=[{"name": "qwen", "model": "m", "base_url": "http://unused"}, {"name": "llama", "model": "m", "base_url": "http://unused"}],
               runs=runs, out_dir=str(out), scorer="answer_line")
    return cfg, out, boxed_cells


SCORERS = ["answer_line", "final_number"]


# ------------------------------------------------------------ naming helpers
def test_same_config_names_and_seed_inference():
    names = ["llama", "qwen", "qwen_check", "qwen_repeat", "qwen_repeat10", "qwen_repeat2", "qwen.extra"]
    assert N.same_config_names("qwen", names) == ["qwen", "qwen_repeat", "qwen_repeat2", "qwen_repeat10"]
    assert N.same_config_names("qwen", ["llama"]) == []
    assert (N.sampling_seed_for("qwen", "qwen"), N.sampling_seed_for("qwen_repeat", "qwen"),
            N.sampling_seed_for("qwen_repeat7", "qwen"), N.sampling_seed_for("llama", "qwen")) == (0, 1, 7, 0)


# ------------------------------------------------------------ the statistical tests
def test_tests_have_a_low_false_alarm_rate_when_runs_are_interchangeable():
    """Each test is a 5% test, so judge it by its rejection rate over many null datasets (a single dataset can hit the 5%)."""
    rng = np.random.default_rng(1)
    p = item_probabilities(200, rng)
    rejected = {"dispersion": 0, "agreement": 0, "system": 0}
    datasets = 40
    for i in range(datasets):
        X = np.hstack([(rng.random((200, 3)) < p[:, None]).astype(float) for _ in range(4)])
        rejected["dispersion"] += N.run_dispersion_test(X, n_shuffles=300, seed=i)[3] < 0.05
        rejected["agreement"] += N.agreement_test(X, 3, n_shuffles=200, seed=i)[2] < 0.05
        rejected["system"] += N.system_effect_permutation([X[:, 3 * j:3 * j + 3].mean(axis=0) for j in range(4)], n_perm=500, seed=i) < 0.05
    for name, count in rejected.items():
        assert count / datasets <= 0.20, (name, count)                    # nominal 5%; 20% leaves room for sampling noise


def test_dispersion_and_agreement_tests_detect_planted_structure():
    rng = np.random.default_rng(2)
    p = item_probabilities(200, rng)
    # a batch effect: one system is 20 points better in every run
    X = np.hstack([(rng.random((200, 3)) < np.clip(p + s, 0, 1)[:, None]).astype(float) for s in (0.20, 0, 0, 0)])
    assert N.run_dispersion_test(X, n_shuffles=800)[3] < 0.01
    # dependence: within a system, later runs copy the first run half of the time
    cols = []
    for _ in range(4):
        first = (rng.random(200) < p).astype(float)
        cols += [first] + [np.where(rng.random(200) < 0.5, first, (rng.random(200) < p).astype(float)) for _ in range(2)]
    assert N.agreement_test(np.stack(cols, axis=1), 3, n_shuffles=800)[2] < 0.01


def test_pair_gap_permutation_agrees_with_the_obvious_cases():
    rng = np.random.default_rng(3)
    p = item_probabilities(200, rng)
    a = (rng.random((200, 3)) < p[:, None]).astype(float)
    same = (rng.random((200, 3)) < p[:, None]).astype(float)
    better = (rng.random((200, 3)) < np.clip(p + 0.15, 0, 1)[:, None]).astype(float)
    assert N.pair_gap_permutation(a, same, n_shuffles=800)[1] > 0.05
    gap, p_value = N.pair_gap_permutation(a, better, n_shuffles=800)
    assert gap > 0.05 and p_value < 0.01


def test_false_alarm_calibration_is_near_the_nominal_rate_when_the_null_is_true():
    rng = np.random.default_rng(4)
    p = item_probabilities(200, rng)
    X = (rng.random((200, 12)) < p[:, None]).astype(float)
    shares = N.false_alarm_calibration(X, [f"q{i}" for i in range(200)], 3, n_datasets=20, draws=40, B=300)
    assert 0 <= np.mean(shares) <= 0.12                        # nominal 5%; never wildly over-confident


def test_regroup_enumerates_all_pairings_when_there_are_few_runs():
    X = (np.random.default_rng(5).random((50, 6)) < 0.5).astype(float)
    res = N.regroup(X, [f"q{i}" for i in range(50)], 3, max_draws=400, B=200)
    assert len(res) == 10 and len({(r[0], r[1]) for r in res}) == 10      # C(6,3)/2 distinct splits, each once


# ------------------------------------------------------------ the report
def test_report_contains_every_section_and_correct_counts(tmp_path):
    cfg, out, boxed = write_study(tmp_path, {"qwen": 0, "qwen_repeat": 0, "qwen_repeat2": 0, "llama": -0.05})
    report = N.build_report(cfg, out, SCORERS, n_draws=20, B=300, n_perm=800)
    for heading in ("1. ACCURACY OF EVERY INDIVIDUAL RUN", "2. ARE RUNS OF THE SAME CONFIGURATION INTERCHANGEABLE",
                    "3. HOW OFTEN DOES FLAKEVAL RAISE A FALSE ALARM", "4. HOW MUCH OF A STRICT-SCORING GAP IS FORMATTING",
                    "5. WHICH QUESTIONS PRODUCED", "HOW TO READ THIS"):
        assert heading in report, heading
    assert "same-configuration systems found: qwen, qwen_repeat, qwen_repeat2" in report
    assert "all 3 pairs of the 3 same-configuration systems" in report
    # the replies without an 'Answer:' line are exactly the boxed ones we planted
    row = [line for line in report.splitlines() if line.strip().startswith("qwen ") and "(" in line and "%" in line][0]
    assert str(boxed["qwen"]) in row.split()
    # per-run accuracies match an independent recomputation
    items = study.prepare_items(cfg)
    recs = N.read_records(out / "qwen.outputs.jsonl")
    strict = N.scores_from_records(recs, items, "answer_line")
    assert f"{strict.x.mean(axis=0)[0]:.3f}" in report


def test_report_needs_outputs_and_explains_when_they_are_missing(tmp_path):
    cfg, out, _ = write_study(tmp_path, {"qwen": 0, "qwen_repeat": 0})
    (out / "qwen.outputs.jsonl").unlink()
    (out / "qwen_repeat.outputs.jsonl").unlink()
    with pytest.raises(SystemExit, match="outputs.jsonl"):
        N.build_report(cfg, out, SCORERS)


def test_formatting_share_of_a_strict_gap_is_reported(tmp_path):
    # the repeat has EXACTLY the baseline's correctness, but boxes a third of its answers: strict sees a gap, lenient none
    cfg, out, _ = write_study(tmp_path, {"qwen": 0, "qwen_repeat": 0}, boxed_share=0.0, seed=8)
    recs = N.read_records(out / "qwen.outputs.jsonl")
    for r in recs[::3]:
        r["answer"] = r["answer"].replace("Answer: ", "\\boxed{") + "}"
    (out / "qwen_repeat.outputs.jsonl").write_text("\n".join(json.dumps(x) for x in recs) + "\n", encoding="utf-8")
    report = N.build_report(cfg, out, SCORERS, n_draws=10, B=200, n_perm=400)
    line = [l for l in report.splitlines() if "formatting part" in l][0]
    assert "final_number gap +0.0 pts" in line                 # the same answers, so no difference once formatting is ignored
    assert "answer_line gap -" in line and "(100% of the strict gap)" in line


# ------------------------------------------------------------ the GPU-facing commands (fake model server)
class NoisyServer:
    """A server that gives a different answer every time, even for the same seed (hidden nondeterminism)."""

    def __init__(self):
        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps({"choices": [{"message": {"content": f"Paris {random.random()}"}}]}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1"

    def close(self):
        self.srv.shutdown()


def study_cfg(tmp_path, url, runs=2, n=12):
    data = tmp_path / "d.csv"
    data.write_text("id,q,a\n" + "\n".join(f"{i},Capital of France {i}?,Paris" for i in range(n)) + "\n")
    cfg = study.default_config()
    cfg.update(data_path=str(data), input_column="q", expected_column="a", id_column="id", runs=runs, max_tokens=30,
               out_dir=str(tmp_path / "res"),
               systems=[{"name": "qwen", "model": "m", "base_url": url, "temperature": 0.7},
                        {"name": "llama", "model": "m", "base_url": url, "temperature": 0.7}])
    return cfg


def test_determinism_check_passes_on_a_deterministic_server_and_catches_a_noisy_one(tmp_path):
    fake = FakeServer()
    cfg = study_cfg(tmp_path, fake.url)
    study.run_study(cfg, yes=True)
    summary = N.determinism_check(cfg, Path(cfg["out_dir"]), "qwen", n_cells=10, show=lambda *a: None)
    assert summary["checked"] == 10 and summary["identical"] == 10 and summary["different"] == 0
    summary = N.determinism_check(cfg, Path(cfg["out_dir"]), "qwen_repeat", n_cells=10, show=lambda *a: None)
    assert summary["identical"] == 10                                  # uses the repeat's own base seed (1)
    fake.close()

    noisy = NoisyServer()
    noisy_cfg = {**cfg, "systems": [{**s, "base_url": noisy.url} for s in cfg["systems"]]}
    summary = N.determinism_check(noisy_cfg, Path(cfg["out_dir"]), "qwen", n_cells=10, show=lambda *a: None)
    assert summary["different"] == 10 and summary["identical"] == 0
    noisy.close()


def test_replicate_adds_repeats_with_new_seeds_and_resumes(tmp_path):
    fake = FakeServer()
    cfg = study_cfg(tmp_path, fake.url)
    study.run_study(cfg, yes=True)
    out = Path(cfg["out_dir"])
    before = fake.calls
    names = N.replicate(cfg, out, extra=2, first_seed=2, yes=True)
    assert names == ["qwen_repeat2", "qwen_repeat3"] and fake.calls == before + 12 * 2 * 2
    for n in names:
        assert (out / f"{n}.jsonl").exists() and (out / f"{n}.outputs.jsonl").exists()
    base = {(r["item_id"], r["run"]): r["answer"] for r in N.read_records(out / "qwen.outputs.jsonl")}
    rep = {(r["item_id"], r["run"]): r["answer"] for r in N.read_records(out / "qwen_repeat2.outputs.jsonl")}
    assert any(base[k] != rep[k] for k in base)                        # different seeds -> different samples
    N.replicate(cfg, out, extra=2, first_seed=2, yes=True)
    assert fake.calls == before + 12 * 2 * 2                           # second call skips what exists
    assert N.same_config_names("qwen", N.available_systems(out)) == ["qwen", "qwen_repeat", "qwen_repeat2", "qwen_repeat3"]
    fake.close()


def test_command_line_analyze_writes_the_report_file(tmp_path, capsys):
    cfg, out, _ = write_study(tmp_path, {"qwen": 0, "qwen_repeat": 0})
    settings = tmp_path / "settings.py"
    settings.write_text(
        f'DATA_PATH = {cfg["data_path"]!r}\nINPUT_COLUMN = "question"\nEXPECTED_COLUMN = "answer"\nID_COLUMN = "id"\n'
        f'SYSTEMS = [{{"name": "qwen", "model": "m", "base_url": "http://unused"}}, '
        f'{{"name": "llama", "model": "m", "base_url": "http://unused"}}]\nOUT_DIR = "ignored"\nSCORER = "answer_line"\n')
    N.main(["analyze", "--settings", str(settings), "--dir", str(out), "--draws", "10"])
    assert (out / "noise_diagnostics.txt").exists() and "HOW TO READ THIS" in capsys.readouterr().out


def test_system_spread_matches_random_assignment_when_nothing_is_going_on_and_exceeds_it_when_systems_differ():
    rng = np.random.default_rng(6)
    p = item_probabilities(200, rng)
    def system(shift): return (rng.random((200, 3)) < np.clip(p + shift, 0, 1)[:, None]).astype(float).mean(axis=0)
    obs, expected = N.system_spread([system(0) for _ in range(5)], n_perm=800)
    assert 0.3 * expected < obs < 3 * expected
    obs, expected = N.system_spread([system(s) for s in (0.0, 0.05, 0.10, 0.15, 0.20)], n_perm=800)
    assert obs > 1.5 * expected                      # the null pool already contains the differences, so this is a modest ratio


# ------------------------------------------------------------ request-history (prompt-cache-like) servers
class CachingServer:
    """Answers depend on request history: a question asked right after the SAME question gets a 'cached' answer."""

    def __init__(self):
        outer = self
        self.last = None

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = body["messages"][-1]["content"]
                text = "Paris cached" if prompt == outer.last else "Paris fresh"
                outer.last = prompt
                out = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1"

    def close(self):
        self.srv.shutdown()


def test_determinism_check_tells_history_dependence_apart_from_random_noise(tmp_path):
    server = CachingServer()
    cfg = study_cfg(tmp_path, server.url, runs=3, n=12)
    study.run_study(cfg, yes=True)                         # item-major: each question's 3 runs back to back
    out = Path(cfg["out_dir"])
    quiet = lambda *a: None

    random_mode = N.determinism_check(cfg, out, "qwen", n_cells=36, show=quiet, mode="random")
    r0, r1, r2 = (random_mode["by_run"][r] for r in (0, 1, 2))
    assert r0[1] / r0[0] >= 0.75                           # run 0 was answered "fresh" originally, and is again
    assert r1[1] / r1[0] <= 0.3 and r2[1] / r2[0] <= 0.3   # runs 1-2 were answered "cached" originally, but come back fresh

    sequence = N.determinism_check(cfg, out, "qwen", n_cells=36, show=quiet, mode="sequence")
    assert sequence["checked"] == 36 and sequence["identical"] == 36       # restoring the original order restores every answer
    assert set(sequence["by_run"]) == {0, 1, 2}
    server.close()


def test_determinism_summary_has_a_per_run_breakdown_and_modes(tmp_path):
    fake = FakeServer()
    cfg = study_cfg(tmp_path, fake.url, runs=2, n=8)
    study.run_study(cfg, yes=True)
    out = Path(cfg["out_dir"])
    s = N.determinism_check(cfg, out, "qwen", n_cells=16, show=lambda *a: None, mode="random")
    assert s["mode"] == "random" and sum(v[0] for v in s["by_run"].values()) == s["checked"] == 16
    seq = N.determinism_check(cfg, out, "qwen", n_cells=8, show=lambda *a: None, mode="sequence")
    assert seq["mode"] == "sequence" and seq["checked"] == 8 and seq["identical"] == 8     # 4 questions x 2 runs
    fake.close()


def test_report_includes_a_timeline_section_with_four_blocks(tmp_path):
    cfg, out, _ = write_study(tmp_path, {"qwen": 0, "qwen_repeat": 0}, n=80, seed=12)
    report = N.build_report(cfg, out, SCORERS, n_draws=10, B=200, n_perm=400)
    section = report[report.index("6. DOES THE GAP DEPEND ON WHEN"):]
    block_rows = [l for l in section.splitlines() if l.strip()[:2] in ("1 ", "2 ", "3 ", "4 ")]
    assert len(block_rows) == 4 and "qwen_repeat" in section


def test_next_step_command_uses_forward_slashes_even_for_windows_paths():
    from pathlib import PureWindowsPath
    cmd = N.analyze_command(PureWindowsPath("examples\\template_case_studies\\results_template_gsm8k"))
    assert "\\" not in cmd and cmd.endswith("--dir examples/template_case_studies/results_template_gsm8k")
