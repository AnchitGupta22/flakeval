import importlib.util
import subprocess
import sys

import pytest

from flakeval import study
from flakeval.cli import main as cli_main
from test_study import FakeServer


# ---------------------------------------------------------------- flakeval init
def test_init_creates_template_and_sample_data(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli_main(["init"]) == 0
    assert (tmp_path / "my_eval.py").exists() and (tmp_path / "sample_data.csv").exists()
    out = capsys.readouterr().out
    assert "my_eval.py --dry-run" in out


def test_init_never_overwrites_without_force(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cli_main(["init"])
    (tmp_path / "my_eval.py").write_text("# my careful edits\n")
    assert cli_main(["init"]) == 1
    assert (tmp_path / "my_eval.py").read_text() == "# my careful edits\n"
    assert "--force" in capsys.readouterr().out
    assert cli_main(["init", "--force"]) == 0
    assert "careful edits" not in (tmp_path / "my_eval.py").read_text()


def test_init_custom_name_in_subfolder_and_no_sample_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli_main(["init", "study/second_eval.py", "--no-sample-data"]) == 0
    assert (tmp_path / "study" / "second_eval.py").exists() and not (tmp_path / "study" / "sample_data.csv").exists()


# ---------------------------------------------------------------- settings handling
def test_settings_are_read_from_a_namespace_and_defaults_fill_the_rest():
    ns = {"DATA_PATH": "d.csv", "INPUT_COLUMN": "q", "SYSTEMS": [{"name": "a"}, {"name": "b"}], "RUNS": 7,
          "custom_score": lambda o, e, r: 1.0, "_private": 1, "SOMETHING_ELSE_ENTIRELY": 3}
    cfg = study.config_from_settings(ns)
    assert cfg["runs"] == 7 and cfg["max_tokens"] == 300 and cfg["run_a_a_test"] is True
    assert callable(cfg["custom_score"]) and cfg["out_dir"] == "my_results"


def test_misspelled_setting_stops_with_a_hint_but_unrelated_constants_are_fine():
    base = {"DATA_PATH": "d.csv", "INPUT_COLUMN": "q", "SYSTEMS": [{}, {}]}
    with pytest.raises(SystemExit, match="Did you mean RUNS"):
        study.config_from_settings({**base, "RUN": 10})
    with pytest.raises(SystemExit, match="Did you mean MAX_ITEMS"):
        study.config_from_settings({**base, "MAXITEMS": 10})
    study.config_from_settings({**base, "MY_PROMPT_PREFIX": "hello"})          # own variable: allowed


def test_missing_required_and_unknown_programmatic_settings_are_explained():
    with pytest.raises(SystemExit, match="Missing required setting SYSTEMS"):
        study.complete_config({"data_path": "d.csv", "input_column": "q"})
    with pytest.raises(SystemExit, match="Unknown setting"):
        study.complete_config({"data_path": "d.csv", "input_column": "q", "systems": [1], "run": 3})


# ---------------------------------------------------------------- the shipped template
def _load_template(path):
    spec = importlib.util.spec_from_file_location("user_settings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # __name__ is not "__main__", so nothing runs
    return module


def test_shipped_template_is_a_valid_settings_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli_main(["init"])
    cfg = study.config_from_settings(vars(_load_template(tmp_path / "my_eval.py")))
    assert len(cfg["systems"]) == 2 and cfg["data_path"] == "sample_data.csv" and cfg["scorer"] == "contains"
    assert len(study.prepare_items(cfg)) == 12


def test_the_template_file_is_only_settings_not_machinery():
    from importlib import resources
    text = (resources.files("flakeval") / "templates" / "my_eval.py").read_text(encoding="utf-8")
    assert "run_cli(globals())" in text and "def make_caller" not in text and "class Progress" not in text
    assert len(text.splitlines()) < 140


# ---------------------------------------------------------------- the real user flow, in a fresh folder
def test_user_flow_init_edit_dry_run_run_rescore(tmp_path):
    fake = FakeServer()
    run = lambda *args: subprocess.run([sys.executable, *args], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert subprocess.run([sys.executable, "-m", "flakeval.cli", "init"], cwd=tmp_path, capture_output=True).returncode == 0
    f = tmp_path / "my_eval.py"
    f.write_text(f.read_text(encoding="utf-8").replace("http://localhost:11434/v1", fake.url).replace("RUNS = 5", "RUNS = 2"),
                 encoding="utf-8")

    dry = run("my_eval.py", "--dry-run")
    assert dry.returncode == 0 and "DRY RUN" in dry.stdout and not (tmp_path / "my_results").exists()

    real = run("my_eval.py", "--yes")
    assert real.returncode == 0, real.stderr
    assert "VERDICT" in real.stdout and "A/A test" in real.stdout
    for name in ("model_a", "model_b", "model_a_repeat"):
        assert (tmp_path / "my_results" / f"{name}.jsonl").exists()

    calls = fake.calls
    again = run("my_eval.py", "--rescore")
    assert again.returncode == 0 and fake.calls == calls and "mean score" in again.stdout   # no model calls
    fake.close()


def test_a_typo_in_the_users_file_fails_fast_with_a_hint(tmp_path):
    subprocess.run([sys.executable, "-m", "flakeval.cli", "init"], cwd=tmp_path, capture_output=True)
    f = tmp_path / "my_eval.py"
    f.write_text(f.read_text(encoding="utf-8").replace("RUNS = 5", "RUN = 5"), encoding="utf-8")
    r = subprocess.run([sys.executable, "my_eval.py", "--yes"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "Did you mean RUNS" in (r.stderr + r.stdout)
    assert not (tmp_path / "my_results").exists()


def test_readme_library_example_works(tmp_path):
    """The Python-library snippet in the README: only the minimum settings, everything else defaulted."""
    fake = FakeServer()
    data = tmp_path / "my_questions.csv"
    data.write_text("question,answer\n" + "\n".join(f"Capital of France {i}?,Paris" for i in range(8)) + "\n")
    cfg = study.default_config()
    cfg.update(data_path=str(data), input_column="question", expected_column="answer", runs=2,
               out_dir=str(tmp_path / "out"),
               systems=[{"name": "a", "model": "qwen2.5:3b", "base_url": fake.url},
                        {"name": "b", "model": "llama3.2:3b", "base_url": fake.url}])
    assert "VERDICT" in study.run_study(cfg, yes=True)
    fake.close()


# ---------------------------------------------------------------- early validation and override flags
BASE = {"DATA_PATH": "d.csv", "INPUT_COLUMN": "q", "SYSTEMS": [{"name": "a"}, {"name": "b"}]}


def test_unknown_scorer_is_rejected_before_any_model_is_called():
    with pytest.raises(SystemExit, match="choice_letter") as e:       # the message lists every valid scorer
        study.config_from_settings({**BASE, "SCORER": "exakt"})
    assert "final_number" in str(e.value) and "answer_line" in str(e.value)


def test_custom_scorer_without_a_function_is_explained():
    with pytest.raises(SystemExit, match="custom_score"):
        study.config_from_settings({**BASE, "SCORER": "custom"})
    assert study.config_from_settings({**BASE, "SCORER": "custom", "custom_score": lambda o, e, r: 1.0})


def test_override_flags_replace_settings_without_editing_the_file(monkeypatch):
    seen = {}
    monkeypatch.setattr(study, "run_study", lambda cfg, dry=False, yes=False: seen.update(cfg=cfg, dry=dry, yes=yes))
    settings = {**BASE, "RUNS": 5, "MAX_ITEMS": None, "OUT_DIR": "my_results", "WORKERS": 1, "SCORER": "contains"}
    study.run_cli(settings, ["--max-items", "10", "--runs", "2", "--workers", "4", "--out-dir", "trial",
                             "--scorer", "number", "--dry-run", "--yes"])
    cfg = seen["cfg"]
    assert (cfg["max_items"], cfg["runs"], cfg["workers"], cfg["out_dir"], cfg["scorer"]) == (10, 2, 4, "trial", "number")
    assert seen["dry"] is True and seen["yes"] is True
    study.run_cli(settings, [])                                         # no flags: the file's own settings win
    assert seen["cfg"]["runs"] == 5 and seen["cfg"]["scorer"] == "contains" and seen["cfg"]["out_dir"] == "my_results"


def test_override_flag_with_an_invalid_scorer_is_refused_by_argparse(capsys):
    with pytest.raises(SystemExit):
        study.run_cli(BASE, ["--scorer", "nonsense"])
    assert "invalid choice" in capsys.readouterr().err
