"""Tests for examples/template_case_studies/: the two case studies written as ordinary settings files.

They are self-contained examples of using the template: nothing is imported from the older case-study scripts. A few
tests below compare against values captured from those older scripts while they still existed (prompt wording,
scoring of real model replies, which questions get exported). Those values are frozen here as plain data, so the
tests keep guarding against accidental changes without needing the old code.
"""
import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "examples" / "template_case_studies"
sys.path.insert(0, str(ROOT / "tests"))

from flakeval import study  # noqa: E402
from flakeval.scoring import score_output  # noqa: E402
from test_study import FakeServer  # noqa: E402


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, FOLDER / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module)
    return found


# ------------------------------------------------------------ they really are self-contained examples
def test_settings_files_import_only_the_engine():
    for name in ("arc_eval", "gsm8k_eval"):
        assert imported_modules(FOLDER / f"{name}.py") == {"flakeval.study"}, name


def test_data_export_script_stands_alone():
    mods = imported_modules(FOLDER / "prepare_data.py")
    assert not any(m and (m.startswith("run_ollama") or m.startswith("rescore") or "case_study" in m) for m in mods)


@pytest.mark.parametrize("name,scorer,names,temps,runs,tokens", [
    ("arc_eval", "choice_letter", ["qwen", "llama", "qwen_t03"], [1.0, 1.0, 0.3], 5, 16),
    ("gsm8k_eval", "answer_line", ["qwen", "llama", "qwen_check"], [0.7, 0.7, 0.7], 3, 800),
])
def test_settings_files_are_valid_and_match_the_case_study_designs(name, scorer, names, temps, runs, tokens):
    cfg = study.config_from_settings(vars(load_module(name)))
    assert cfg["scorer"] == scorer and cfg["runs"] == runs and cfg["max_tokens"] == tokens and cfg["run_a_a_test"]
    assert [s["name"] for s in cfg["systems"]] == names and [s["temperature"] for s in cfg["systems"]] == temps
    assert cfg["out_dir"] == f"examples/template_case_studies/results_template_{name.split('_')[0]}"


# Wording used by the original GSM8K case study (frozen). The "check" variant adds one sentence to the base prompt.
BASE_PROMPT = "Solve the problem. Think step by step, then finish with a final line of the form 'Answer: <number>'."
CHECK_PROMPT = BASE_PROMPT + " Double-check your arithmetic before giving the final answer."


def test_gsm8k_prompts_match_the_original_case_study_wording():
    systems = {s["name"]: s for s in study.config_from_settings(vars(load_module("gsm8k_eval")))["systems"]}
    assert systems["qwen"]["prompt_template"] == "{input}\n\n" + BASE_PROMPT
    assert systems["llama"]["prompt_template"] == "{input}\n\n" + BASE_PROMPT
    assert systems["qwen_check"]["prompt_template"] == "{input}\n\n" + CHECK_PROMPT


# ------------------------------------------------------------ built-in scorers on real model replies (frozen values)
# (reply ending, expected answer, correct under strict "answer_line", correct under lenient "final_number")
MATH = [
    ('Therefore, Tom has \\( 91 \\) trees left.', '91', 0, 1),
    ("Therefore, Elaina's usual concerts run for \\( 25 \\) minutes.", '25', 0, 1),
    ('Therefore, Dennis will spend: / \\[  / \\boxed{13} / \\]', '13', 0, 1),
    ('Thus, Abraham will earn a total of **$140** after selling all his unused land.', '140', 0, 1),
    ('Soledad must hike an additional: \\[ \\boxed{60} \\text{ miles per day} \\]', '60', 0, 1),
    ('Therefore, Janet now has a total of **98** pens.', '98', 0, 1),
    ('\\[ \\boxed{8} \\text{ centimeters shorter after burning from 1:00 PM to 5:00 PM.}', '8', 0, 1),   # boxed beats "5"
    ('### Final Answer: Vince stapled a total of 1800 reports.', '1800', 0, 1),
    ('Daisy bought a bag of carrots that weighs **7 pounds**.', '7', 0, 1),
    ('so 3+4=7\nAnswer: 7', '7', 1, 1),
    ('Answer: 5 ... wait, redo\nAnswer: -12', '-12', 1, 1),                                              # last Answer: wins
    ('no digits at all', '5', 0, 0),
    ('Final answer: 8', '8', 1, 1),
    ('Answer: $482', '482', 1, 1),
    ('Answer: 1,200,000', '1200000', 1, 1),
]

# (reply, the letter the original single-letter parser extracted from it, or None)
LETTERS = [('B', 'B'), ('The answer is C.', 'C'), ('A', 'A'), ('D', 'D'), ('I think the answer is B because ...', 'B'),
           ('(C)', 'C'), ('b', None), ('no idea', None), ('E', None)]


def test_math_scorers_on_real_replies():
    for text, gold, strict, lenient in MATH:
        assert score_output("answer_line", text, gold, {}) == strict, ("strict", text)
        assert score_output("final_number", text, gold, {}) == lenient, ("lenient", text)


def test_choice_scorer_on_typical_replies():
    for text, letter in LETTERS:
        for gold in "ABCD":
            assert score_output("choice_letter", text, gold, {}) == float(letter == gold), (text, gold)


# ------------------------------------------------------------ the data export picks the same questions as before
def _install_fake_datasets(monkeypatch):
    arc = [{"id": f"Q{i}", "question": f"Question {i}?", "choices": {"label": ["A", "B", "C", "D"],
            "text": ["w", "x", "y", "z"]}, "answerKey": "BCDA"[i % 4]} for i in range(12)]
    arc.append({"id": "numeric", "question": "Numeric labels?", "choices": {"label": ["1", "2", "3"],
                "text": ["p", "q", "r"]}, "answerKey": "2"})
    arc.append({"id": "broken", "question": "Bad key?", "choices": {"label": ["A", "B"], "text": ["a", "b"]},
                "answerKey": "Z"})
    gsm = [{"question": f"What is {i} plus 1?", "answer": f"reasoning here\n#### {i + 1:,}"} for i in range(1200, 1212)]
    gsm[0]["answer"] = "adds up\n#### 1,200,000"
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda name, *a, **kw: arc if name == "allenai/ai2_arc" else gsm
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_export_selects_the_expected_questions(monkeypatch, tmp_path):
    """Frozen from the original case-study loaders: same filtering, same seeded shuffle, same prompt and answers."""
    _install_fake_datasets(monkeypatch)
    prep = load_module("prepare_data")
    arc, gsm = prep._arc_rows(6, 0), prep._gsm8k_rows(6, 0)
    assert [r["id"] for r in arc] == ["Q1", "Q10", "Q9", "Q5", "Q11", "Q2"]          # the 'broken' row is filtered out
    assert [r["answer"] for r in arc] == ["C", "D", "C", "C", "A", "D"]
    assert arc[0]["question"] == "Question 1?\n\nA. w\nB. x\nC. y\nD. z\n\nAnswer with only the letter of the correct choice."
    assert [r["id"] for r in gsm] == ["gsm8k_test_1", "gsm8k_test_9", "gsm8k_test_8", "gsm8k_test_5", "gsm8k_test_10",
                                      "gsm8k_test_2"]
    assert [r["answer"] for r in gsm] == ["1202", "1210", "1209", "1206", "1211", "1203"]
    assert [r["id"] for r in prep._arc_rows(6, 1)] != [r["id"] for r in arc]        # another seed, another subset

    numeric = [r for r in prep._arc_rows(20, 0) if r["id"] == "numeric"][0]         # labels "1","2","3" -> letters
    assert numeric["answer"] == "B" and "A. p" in numeric["question"]
    assert "1200000" in [r["answer"] for r in prep._gsm8k_rows(12, 0)]              # never 1.2e+06 or 1200000.0

    paths = prep.prepare(str(tmp_path / "data"), n_items=6)
    cfg = study.default_config()
    cfg.update(data_path=str(paths["gsm8k"]), input_column="question", expected_column="answer", id_column="id")
    items = study.prepare_items(cfg)
    assert len(items) == 6 and all("." not in v["expected"] for v in items.values())


# ------------------------------------------------------------ end to end, the way a user would run them
def _settings_pointing_at(module, fake, data_csv, out_dir, runs=2):
    settings = dict(vars(module))
    settings.update(DATA_PATH=str(data_csv), OUT_DIR=str(out_dir), RUNS=runs,
                    SYSTEMS=[{**s, "base_url": fake.url} for s in module.SYSTEMS])
    return settings


def test_gsm8k_settings_file_runs_and_rescores_with_flags_only(monkeypatch, tmp_path):
    _install_fake_datasets(monkeypatch)
    paths = load_module("prepare_data").prepare(str(tmp_path / "data"), n_items=8)
    fake = FakeServer()
    settings = _settings_pointing_at(load_module("gsm8k_eval"), fake, paths["gsm8k"], tmp_path / "res")
    study.run_cli(settings, ["--yes"])
    out = tmp_path / "res"
    for n in ("qwen", "llama", "qwen_check", "qwen_repeat"):
        assert (out / f"{n}.jsonl").exists() and (out / f"{n}.outputs.jsonl").exists()
    calls = fake.calls
    for n in ("qwen", "llama", "qwen_check", "qwen_repeat"):             # right answers, but not as an 'Answer:' line
        path = out / f"{n}.outputs.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        gold = {r["id"]: r["answer"] for r in __import__("csv").DictReader(open(paths["gsm8k"], encoding="utf-8"))}
        for r in rows:
            r["answer"] = f"So the total is \\boxed{{{gold[r['item_id']]}}} in the end."
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    study.run_cli(settings, ["--rescore"])                                # strict, as in the file
    assert study.load_jsonl(out / "qwen.jsonl").mean == 0.0
    study.run_cli(settings, ["--scorer", "final_number", "--rescore"])    # lenient, chosen on the command line
    assert study.load_jsonl(out / "qwen.jsonl").mean == 1.0 and fake.calls == calls
    fake.close()


def test_arc_settings_file_runs_with_quick_trial_flags(monkeypatch, tmp_path, capsys):
    _install_fake_datasets(monkeypatch)
    paths = load_module("prepare_data").prepare(str(tmp_path / "data"), n_items=10)
    fake = FakeServer()
    settings = _settings_pointing_at(load_module("arc_eval"), fake, paths["arc"], tmp_path / "res", runs=5)
    study.run_cli(settings, ["--yes", "--max-items", "4", "--runs", "2", "--out-dir", str(tmp_path / "trial")])
    assert "4 questions x 2 runs x 4 system runs" in capsys.readouterr().out
    assert fake.calls == 4 * 2 * 4 and (tmp_path / "trial" / "qwen_t03.jsonl").exists() and not (tmp_path / "res").exists()
    assert {b["max_tokens"] for b in fake.bodies} == {16}
    fake.close()
