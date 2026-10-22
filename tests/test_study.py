import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
from flakeval import models, progress, scoring  # noqa: E402
from flakeval import study as T  # noqa: E402


class FakeServer:
    """Minimal OpenAI-compatible /chat/completions server. Answers 'Paris' or 'Rome' depending on the seed."""

    def __init__(self, fail_first=0, auth_expected=None):
        outer = self
        self.calls, self.fail_first, self.seen_auth, self.bodies = 0, fail_first, [], []

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls += 1
                outer.seen_auth.append(self.headers.get("Authorization"))
                outer.bodies.append(body)
                if outer.calls <= outer.fail_first:
                    self.send_response(503); self.send_header("Content-Length", "0"); self.end_headers(); return
                text = "Paris" if body.get("seed", 0) % 2 == 0 else "Rome"
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


def cfg_for(tmp_path, url, **over):
    data = tmp_path / "data.csv"
    data.write_text("id,q,a\n" + "\n".join(f"{i},Capital of France?,Paris" for i in range(30)) + "\n")
    base = dict(data_path=str(data), input_column="q", expected_column="a", id_column="id", max_items=None,
                item_seed=0, scorer="contains", custom_score=None, runs=3, max_tokens=50, workers=2,
                run_a_a_test=True, out_dir=str(tmp_path / "out"), save_outputs=True,
                systems=[{"name": "a", "model": "m", "base_url": url, "temperature": 0.7},
                         {"name": "b", "model": "m", "base_url": url, "temperature": 0.7}])
    base.update(over)
    return base


def test_scorers():
    s = scoring.score_output
    assert s("exact", " Paris. ", "paris", {}) == 1.0 and s("exact", "It is Paris", "Paris", {}) == 0.0
    assert s("contains", "The capital is Paris!", "paris", {}) == 1.0
    assert s("number", "so 7*8 = 56 apples", "56", {}) == 1.0 and s("number", "I think 55", "56", {}) == 0.0
    assert s("number", "1,234", "1234", {}) == 1.0 and s("number", "no digits", "5", {}) == 0.0
    assert s("custom", "abc", None, {}, custom=lambda o, e, r: len(o) / 10) == 0.3
    with pytest.raises(SystemExit):
        s("nonsense", "x", "y", {})


def test_full_run_writes_files_and_report(tmp_path, capsys):
    fake = FakeServer()
    cfg = cfg_for(tmp_path, fake.url)
    report = T.run_study(cfg, yes=True)
    out = tmp_path / "out"
    for name in ("a", "b", "a_repeat"):
        assert (out / f"{name}.jsonl").exists() and (out / f"{name}.outputs.jsonl").exists()
    assert (out / "report.txt").exists() and "VERDICT" in report and "A/A test" in report
    assert fake.calls == 30 * 3 * 3                      # 3 system runs (a, b, a_repeat)
    assert {b["seed"] for b in fake.bodies} != {0}       # seeds vary between calls
    fake.close()


def test_resume_skips_existing_results(tmp_path):
    fake = FakeServer()
    cfg = cfg_for(tmp_path, fake.url)
    T.run_study(cfg, yes=True)
    before = fake.calls
    T.run_study(cfg, yes=True)
    assert fake.calls == before                           # nothing re-run, no extra spend
    fake.close()


def test_dry_run_writes_nothing_and_prints(tmp_path, capsys):
    fake = FakeServer()
    cfg = cfg_for(tmp_path, fake.url)
    T.run_study(cfg, dry=True)
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "answer:" in out and "score:" in out
    assert not (tmp_path / "out").exists() and fake.calls == 2 * 2   # 2 questions x 2 systems, 1 run
    fake.close()


def test_retry_on_busy_server(tmp_path, monkeypatch):
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    fake = FakeServer(fail_first=2)
    caller = models.make_caller({"name": "x", "model": "m", "base_url": fake.url}, 20)
    assert caller("hi", "", 0.5, 1) in ("Paris", "Rome") and fake.calls == 3
    fake.close()


def test_api_key_is_sent_from_env_and_missing_key_is_reported(tmp_path, monkeypatch):
    fake = FakeServer()
    system = {"name": "x", "model": "m", "base_url": fake.url, "api_key_env": "MY_TEST_KEY"}
    monkeypatch.delenv("MY_TEST_KEY", raising=False)
    with pytest.raises(SystemExit, match="MY_TEST_KEY"):
        models.make_caller(system, 20)
    monkeypatch.setenv("MY_TEST_KEY", "secret123")
    models.make_caller(system, 20)("hi", "sys", None, 3)
    assert fake.seen_auth[-1] == "Bearer secret123" and "temperature" not in fake.bodies[-1]
    assert fake.bodies[-1]["messages"][0] == {"role": "system", "content": "sys"}
    fake.close()


def test_custom_call_function_and_token_param(tmp_path):
    fake = FakeServer()
    def mine(prompt, system_prompt, temperature, seed, max_tokens):
        return "Paris" if seed % 3 else "Berlin"
    cfg = cfg_for(tmp_path, fake.url, systems=[
        {"name": "a", "model": "m", "base_url": fake.url, "token_limit_param": "max_completion_tokens",
         "send_seed": False},
        {"name": "mine", "call": mine}], run_a_a_test=False)
    T.run_study(cfg, yes=True)
    assert (tmp_path / "out" / "mine.jsonl").exists()
    assert "max_completion_tokens" in fake.bodies[0] and "seed" not in fake.bodies[0]
    fake.close()


def test_helpful_errors(tmp_path):
    fake = FakeServer()
    with pytest.raises(SystemExit, match="Columns found"):
        T.run_study(cfg_for(tmp_path, fake.url, input_column="nope"), yes=True)
    with pytest.raises(SystemExit, match="not found"):
        T.run_study(cfg_for(tmp_path, fake.url, data_path=str(tmp_path / "missing.csv")), yes=True)
    with pytest.raises(SystemExit, match="at least two"):
        T.run_study(cfg_for(tmp_path, fake.url, systems=[{"name": "a", "model": "m", "base_url": fake.url}]), yes=True)
    fake.close()


def test_subset_is_reproducible_and_jsonl_supported(tmp_path):
    data = tmp_path / "d.jsonl"
    data.write_text("\n".join(json.dumps({"id": i, "q": f"q{i}", "a": "x"}) for i in range(50)))
    cfg = cfg_for(tmp_path, "http://unused", data_path=str(data), max_items=10)
    a, b = T.prepare_items(cfg), T.prepare_items(cfg)
    assert len(a) == 10 and list(a) == list(b)


def test_shipped_sample_data_loads():
    from importlib import resources
    cfg = T.default_config()
    cfg.update(data_path=str(resources.files("flakeval") / "templates" / "sample_data.csv"),
               input_column="question", expected_column="answer", id_column="id")
    assert len(T.prepare_items(cfg)) == 12


# ---------------------------------------------------------------- progress bar
import io  # noqa: E402
import os  # noqa: E402


class FakeStream(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def test_hms():
    assert progress.hms(3725) == "1:02:05" and progress.hms(0) == "0:00:00" and progress.hms(-5) == "0:00:00"


def test_progress_log_mode_prints_plain_lines_every_ten_percent():
    s = FakeStream(False)
    p = progress.Progress(20, stream=s)
    p.start_system("model_a", 1, 2, 20)
    for _ in range(20):
        p.advance()
    p.message("  model_a: mean score 0.5")
    p.close()
    out = s.getvalue()
    assert "\r" not in out and out.count("progress:") == 10 and "100%" in out and "model_a (system 1/2)" in out
    assert "  model_a: mean score 0.5" in out


def test_progress_terminal_mode_redraws_one_line_and_fits_width(monkeypatch):
    monkeypatch.setattr(progress.shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((60, 20)))
    s = FakeStream(True)
    p = progress.Progress(10, stream=s)
    p.start_system("a_rather_long_system_name", 1, 3, 10)
    for _ in range(10):
        p.advance()
    p.message("done with system")
    p.close()
    frames = [f for f in s.getvalue().replace("\n", "\r").split("\r") if f.strip()]
    assert all(len(f) <= 59 for f in frames), [len(f) for f in frames]      # never wraps, so \r redraw stays clean
    assert any("100%" in f for f in frames) and "done with system" in frames
    assert s.getvalue().endswith("\n")


def test_run_study_shows_progress_and_still_reports(tmp_path, capsys):
    fake = FakeServer()
    T.run_study(cfg_for(tmp_path, fake.url), yes=True)
    out = capsys.readouterr().out
    assert "progress:" in out and "100%" in out and "(system 3/3)" in out and "VERDICT" in out
    fake.close()


# ---------------------------------------------------------------- scorer robustness and --rescore
def test_look_alike_characters_are_scored_as_equal():
    s = scoring.score_output
    assert s("contains", "The chemical formula for water is H₂O.", "H2O", {}) == 1.0   # subscript two (real case)
    assert s("exact", "H₂O", "H2O", {}) == 1.0
    assert s("number", "the answer is ５６", "56", {}) == 1.0                              # full-width digits
    assert s("contains", "The ﬁnal answer", "final", {}) == 1.0                          # ligature


def test_rescore_fixes_scores_from_saved_answers_without_calling_a_model(tmp_path, capsys):
    fake = FakeServer()
    data = tmp_path / "water.csv"
    data.write_text("id,q,a\n" + "\n".join(f"{i},Formula of water?,H2O" for i in range(6)) + "\n")
    cfg = cfg_for(tmp_path, fake.url, data_path=str(data), runs=2)
    T.run_study(cfg, yes=True)
    calls_before = fake.calls

    out = tmp_path / "out"
    for name in ("a", "b", "a_repeat"):          # recreate the real-world situation: correct answers marked wrong
        path = out / f"{name}.outputs.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            r["answer"], r["score"] = "The chemical formula is H₂O.", 0.0
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    report = T.rescore_study(cfg)
    printed = capsys.readouterr().out
    assert fake.calls == calls_before                                   # no model was called
    assert "0.000 -> 1.000" in printed and "VERDICT" in report
    for name in ("a", "b", "a_repeat"):
        assert T.load_jsonl(out / f"{name}.jsonl").mean == 1.0
        assert all(json.loads(l)["score"] == 1.0 for l in (out / f"{name}.outputs.jsonl").read_text().splitlines())
    fake.close()


def test_rescore_explains_missing_outputs_and_mismatched_data(tmp_path):
    fake = FakeServer()
    cfg = cfg_for(tmp_path, fake.url)
    with pytest.raises(SystemExit, match="SAVE_OUTPUTS"):
        T.rescore_study(cfg)
    T.run_study(cfg, yes=True)
    smaller = tmp_path / "other.csv"
    smaller.write_text("id,q,a\n999,x,y\n")
    with pytest.raises(SystemExit, match="not in your current data"):
        T.rescore_study({**cfg, "data_path": str(smaller)})
    fake.close()


def test_identical_a_a_result_gets_an_explanatory_note(tmp_path):
    fake = FakeServer()
    cfg = cfg_for(tmp_path, fake.url, scorer="custom", custom_score=lambda o, e, r: 1.0)   # every answer marked 1
    report = T.run_study(cfg, yes=True)
    assert "IDENTICAL RESULTS" in report and "best possible outcome" in report
    fake.close()


def test_rescore_refuses_edited_questions_but_allows_fixed_expected_answers(tmp_path):
    fake = FakeServer()
    cfg = cfg_for(tmp_path, fake.url, runs=2)
    T.run_study(cfg, yes=True)
    data = Path(cfg["data_path"])
    original = data.read_text()

    data.write_text(original.replace("Capital of France?,Paris", "Capital of France?,Rome"))   # gold label fixed
    T.rescore_study(cfg)                                                                          # allowed

    data.write_text(original.replace("0,Capital of France?", "0,Capital of Spain?", 1))           # question edited
    with pytest.raises(SystemExit, match="has been edited"):
        T.rescore_study(cfg)
    fake.close()


def test_dry_run_shows_the_end_of_long_answers(tmp_path, capsys):
    fake = FakeServer()
    long_answer = "step " * 200 + "so the final result is 482"
    cfg = cfg_for(tmp_path, fake.url, systems=[
        {"name": "a", "call": lambda **kw: long_answer}, {"name": "b", "call": lambda **kw: long_answer}])
    T.run_study(cfg, dry=True)
    out = capsys.readouterr().out
    assert "so the final result is 482" in out and "characters not shown" in out
    assert T.shorten("short") == "'short'"
    fake.close()


def test_call_seed_is_reproducible_but_varies_per_item_run_and_base_seed():
    """Guards the bug where every 'repeat' was identical because the same seed was reused."""
    assert models.call_seed(0, "q1", 0) == models.call_seed(0, "q1", 0)
    assert len({models.call_seed(0, "q1", r) for r in range(5)}) == 5
    assert models.call_seed(0, "q1", 0) != models.call_seed(1, "q1", 0)
    assert models.call_seed(0, "q1", 0) != models.call_seed(0, "q2", 0)


def test_rescore_also_remarks_extra_repeats_added_later(tmp_path):
    fake = FakeServer()
    data = tmp_path / "water.csv"
    data.write_text("id,q,a\n" + "\n".join(f"{i},Formula of water?,H2O" for i in range(6)) + "\n")
    cfg = cfg_for(tmp_path, fake.url, data_path=str(data), runs=2)
    T.run_study(cfg, yes=True)
    out = tmp_path / "out"
    for ext in (".jsonl", ".outputs.jsonl"):                       # pretend `replicate` added a_repeat2 and a_repeat3
        for n in ("a_repeat2", "a_repeat3"):
            (out / f"{n}{ext}").write_text((out / f"a_repeat{ext}").read_text(encoding="utf-8"), encoding="utf-8")
    for n in ("a", "b", "a_repeat", "a_repeat2", "a_repeat3"):    # correct answers that are marked wrong
        path = out / f"{n}.outputs.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            r["answer"], r["score"] = "The formula is H\u2082O.", 0.0
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    T.rescore_study(cfg)
    for n in ("a", "a_repeat", "a_repeat2", "a_repeat3"):
        assert T.load_jsonl(out / f"{n}.jsonl").mean == 1.0, n
    fake.close()
