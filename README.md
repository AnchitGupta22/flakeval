<p align="center">
    <img src="https://raw.githubusercontent.com/AnchitGupta22/flakeval/main/assets/flakeval_logo.png" alt="Logo" width="240" />
    <br/>
    <br/>
    <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg" /></a>
    <a href="https://pypi.org/project/flakeval/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/flakeval.svg?maxAge=86400" /></a>
    <a href="https://pypi.org/project/flakeval/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/flakeval" /></a>
    <a href="https://github.com/AnchitGupta22/flakeval/actions"><img alt="Tests"
    src="https://github.com/AnchitGupta22/flakeval/actions/workflows/tests.yml/badge.svg"/></a>
    <a href="https://anchitgupta22.github.io/flakeval/"><img alt="Documentation" src="https://img.shields.io/badge/Documentation-Read%20the%20docs-red" /></a>
</p>

---

**Is your AI improvement real, or just luck?**

*(Say it "flake-eval": flaky + evaluation.)*

You change a prompt, a model or a setting, and your test score goes from 72% to 74%. Did it really get better, or would you get 74% just by running the same test again? `flakeval` answers that with a plain verdict.

- Works on results you already have (a simple JSONL file), or runs your evaluation for you.
- Gives a verdict, a range for the true difference, and the **smallest difference your test can reliably detect**.
- Tells you whether you need **more runs** or **more test questions**.
- Exits with an error on a real regression, so it works as a CI check.
- One dependency (numpy). Runs on a laptop.

**Full documentation, case studies and FAQ: https://anchitgupta22.github.io/flakeval/**

---

## Contents

1. [The idea in plain English](#the-idea-in-plain-english)
2. [Quick start](#quick-start)
3. [Case studies (summary)](#case-studies-summary)
4. [FAQ](#faq)
5. [How it works](#how-it-works)
6. [Limitations](#limitations)
7. [Repository layout](#repository-layout)
8. [Roadmap](#roadmap)

---

## The idea in plain English

Language models answer with some randomness, and your test only has a limited number of questions. Together this wobble is called **noise**. A 2-point change in score could be a real improvement or just noise. `flakeval` measures the noise so you know which.

| Verdict | Meaning | What to do |
|---|---|---|
| **REAL IMPROVEMENT** | B scored higher; chance is an unlikely explanation. | Trust it. |
| **REAL REGRESSION** | B scored lower; chance is an unlikely explanation. | Investigate before shipping B. |
| **WITHIN NOISE** | The test cannot tell (this does **not** mean "equal"). | Add more **runs** or more **questions**, as the advice line says. |
| **IDENTICAL RESULTS** | A and B scored the same on every item. | Check your change actually reached the model. |

Example (simulated: B is truly ~2 points better, 200 items, 5 runs each):

```
$ flakeval compare a.jsonl b.jsonl
A: 0.6800 (5 runs)   B: 0.6860 (5 runs)   items: 200
delta (B - A): +0.0060   95% CI: [-0.0310, +0.0430]   p = 0.7955
smallest reliably detectable difference: ~0.0535
VERDICT: WITHIN NOISE
```

The true difference could be anywhere from 3.1 points worse to 4.3 points better, and this test can only reliably spot gaps of about 5 points, so a 2-point improvement is invisible to it.

---

## Quick start

Requires Python 3.10+.

```bash
pip install flakeval
```

### Option A: you already have scores

Write one JSON line per item per run (use at least 3 runs per item, and the same ids in both files):

```json
{"item_id": "q17", "run": 2, "score": 1.0}
```

```bash
flakeval report a.jsonl              # how noisy is one system?
flakeval compare a.jsonl b.jsonl     # is B really different from A?
flakeval compare a.jsonl b.jsonl --json   # machine-readable
```

`flakeval compare` exits with code 1 only on a statistically real regression, so it works as a **CI gate**.

From Python:

```python
from flakeval import repeat, compare

a = repeat(eval_model_a, items, runs=5, workers=8, save="a.jsonl")
b = repeat(eval_model_b, items, runs=5, workers=8, save="b.jsonl")
print(compare(a, b))
```

If your function uses a sampling seed, make it **different for every run** (`repeat(..., with_run=True)` passes the run number). A fixed seed hides the noise this tool measures.

### Option B: let flakeval run your models

```bash
flakeval init                  # creates my_eval.py (settings) and sample_data.csv
python my_eval.py --dry-run    # 2 questions per system, prints answers and scores; saves nothing
python my_eval.py              # the real run (shows the number of calls and asks first)
```

Put your questions in a CSV or JSONL (`id, input text, expected answer`), then edit the settings at the top of `my_eval.py`. Example: does a new prompt beat the current one on the same model? The first system is the baseline.

```python
DATA_PATH = "tickets.csv"
INPUT_COLUMN = "ticket"
EXPECTED_COLUMN = "label"
ID_COLUMN = "id"
SCORER = "exact"

SYSTEMS = [
    {"name": "current_prompt", "model": "qwen2.5:3b", "base_url": "http://localhost:11434/v1",
     "api_key_env": None, "temperature": 0.7, "system_prompt": "",
     "prompt_template": "Classify this ticket as billing, bug or question.\n\nTicket: {input}"},
    {"name": "new_prompt", "model": "qwen2.5:3b", "base_url": "http://localhost:11434/v1",
     "api_key_env": None, "temperature": 0.7, "system_prompt": "",
     "prompt_template": "Reply with exactly one label: billing, bug or question.\n\nTicket: {input}"},
]

RUNS = 5          # 3 to 5 is a good range
MAX_TOKENS = 20
```

- **Models:** anything with an OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, hosted APIs). Put the *name* of the environment variable holding your key in `api_key_env`, never the key itself. For anything else, pass your own function as `"call": my_function`.
- **Built-in scorers:** `exact`, `contains`, `number`, `choice_letter`, `answer_line`, `final_number`, or `"custom"` (your own `custom_score(output, expected, row)` returning 0 to 1).
- **Overrides:** `--max-items 10`, `--runs 2`, `--workers 4`, `--out-dir DIR`, `--scorer NAME`. Try `--max-items 10 --runs 2 --out-dir trial` for a cheap rehearsal.
- **Output:** a printed report plus `my_results/` with scores, raw answers (`<system>.outputs.jsonl`) and `report.txt`. The baseline is run twice (A/A check) by default, showing how much two identical runs differ by chance.
- **Safe to rerun:** finished systems are skipped, and a server failure stops the run instead of recording fake zeros.
- **Fixed your scorer?** `python my_eval.py --rescore` re-marks saved answers without calling any model.
- **Pure Python, no settings file:** use `default_config()` and `run_study(cfg)` from `flakeval.study`.

**Read some raw answers that scored 0 before trusting the numbers.** A scorer that wrongly marks good answers down looks exactly like model noise.

**Privacy:** nothing is uploaded. Rows go only to the endpoints you list. If your data is confidential, add `my_results/` to `.gitignore`.

### Option C: clone the repo (tests, demo, case studies)

```bash
git clone https://github.com/AnchitGupta22/flakeval && cd flakeval
pip install -e ".[dev,examples]"
python -m pytest
cd examples && python make_demo.py && flakeval compare a.jsonl b.jsonl
```

If `flakeval` is not found, use `python -m flakeval.cli compare a.jsonl b.jsonl`.

---

## Case studies (summary)

Two worked examples live in `examples/template_case_studies/` (`arc_eval.py`, `gsm8k_eval.py`), run on `qwen2.5:3b` and `llama3.2:3b` with Ollama. Full tables, methodology and diagnostics are in the [documentation](https://anchitgupta22.github.io/flakeval/).

- **ARC (science multiple choice, 200 items x 5 runs):** the same model twice scored 82.6% and 82.9% (within noise). `llama3.2:3b` scored 11.7 points lower than `qwen2.5:3b` (p < 0.001), flagged as real.
- **GSM8K (math reasoning, 200 items x 3 runs):** under strict scoring, the same system twice differed by 4.8 points (p = 0.024). Five extra repeats ranged from 69.7% to 74.5%, so this was chance. A "double-check" prompt result of -4.7 points fell inside that natural spread, so no conclusion could be drawn.

Takeaways:

1. Always include the "same system twice" check; it shows how big a gap chance alone can produce.
2. Reasoning tasks are far noisier than single-letter answers (32.5% of GSM8K items changed score across 3 runs vs 0.5% on ARC).
3. Check the "smallest detectable difference" **before** you start. If it is bigger than the improvement you hope for, your test is too small.
4. Treat borderline p-values (about 0.01 to 0.05) with caution, and always report how you scored the answers.

---

## FAQ

**Isn't it all just chance?** Yes, partly, and `flakeval` cannot remove that. It **measures how much chance is in your result**, like a margin of error in a poll, and tells you whether more runs or more questions would shrink it. Halving the smallest detectable difference takes about four times the data.

**Why not just set temperature to 0?** That removes most model randomness but not the sampling of your questions, and real users do not run at temperature 0.

**How many questions and runs do I need?** Look at the "smallest detectable difference" in the report. A reasonable start is 100 or more questions and 3 to 5 runs.

---

## How it works

Scores form an items x runs matrix, and comparisons are paired by item id. The 95% interval comes from a bootstrap that resamples **items** (run noise is already inside each item's mean). The p-value comes from a paired sign-flip permutation test. A verdict of "real" requires the interval to exclude zero **and** p < 0.05. The smallest detectable difference is about 2.8 x the standard error (80% power, alpha 0.05).

In simulation (60 items, 5 runs), two identical systems were called "different" 5.3% of the time (target 5%), and a true 10-point gap was detected 71% of the time.

**Pitfall guarded against:** many servers use a fixed default seed (Ollama's is 0), so "repeated" runs are identical and show zero noise. The template sends a different seed per question and run, and `flakeval` warns when no item changes score across runs.

## Limitations

- Assumes your items are a random sample of the questions you care about.
- Compares two systems at a time; no multiple-comparison correction yet.
- Scores are treated as plain numbers; an LLM judge's own randomness is not modeled separately.
- p-values assume runs of one configuration are interchangeable.
- The case studies use one benchmark each, two small models and one machine. They illustrate the method; they are not a model ranking.

## Repository layout

```
src/flakeval/          the installable package (stats, io, cli, study, scoring, models, progress, templates)
examples/              demo data and the two case studies (not part of the installed package)
tests/                 pytest suite
```

## Roadmap

- [ ] Importers for lm-evaluation-harness, Inspect, and promptfoo outputs
- [ ] HTML report
- [ ] Adaptive mode: keep sampling until the verdict is decided
- [ ] LLM-judge variance and multi-system comparisons with correction

## License

MIT. See [LICENSE](LICENSE).

## Citation

If `flakeval` is useful in your work, please cite it (see [`CITATION.cff`](CITATION.cff)). **Author:** Anchit Gupta
