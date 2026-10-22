#!/usr/bin/env python3
"""
Case study 2 as an ordinary flakeval settings file: GSM8K math word problems, answered step by step.

This is exactly the kind of file `flakeval init` gives you (compare it with src/flakeval/templates/my_eval.py). It
imports nothing from the other case-study scripts: the data is a plain CSV, and the scoring is a built-in scorer.

  1. Once:            python examples/template_case_studies/prepare_data.py     (exports the questions to a CSV)
  2. Check the setup: python examples/template_case_studies/gsm8k_eval.py --dry-run
  3. Run it:          python examples/template_case_studies/gsm8k_eval.py
  Quick trial:        python examples/template_case_studies/gsm8k_eval.py --max-items 10 --runs 2 --out-dir results_trial
  Lenient marking:    python examples/template_case_studies/gsm8k_eval.py --scorer final_number --rescore
                      (re-marks the saved answers, no model calls; copy the results folder first to keep the strict ones)

Needs Ollama running with qwen2.5:3b and llama3.2:3b pulled. 200 questions x 3 runs x 4 system runs = 2,400 calls.
"""
from flakeval.study import run_cli

# ---- Data (exported by prepare_data.py): columns id, question, answer ----
DATA_PATH = "examples/template_case_studies/case_study_data/gsm8k_200.csv"
INPUT_COLUMN = "question"
EXPECTED_COLUMN = "answer"
ID_COLUMN = "id"
MAX_ITEMS = None
ITEM_SEED = 0

# ---- Scoring ----
# "answer_line"  strict: the reply must end with a line like "Answer: 482".
# "final_number" lenient: also accepts \boxed{482}, **482**, or "... total is 482" (the last number in the reply).
SCORER = "answer_line"

# ---- What to compare: the first system is the baseline ----
_ASK = "{input}\n\nSolve the problem. Think step by step, then finish with a final line of the form 'Answer: <number>'."
SYSTEMS = [
    {"name": "qwen", "model": "qwen2.5:3b", "base_url": "http://localhost:11434/v1", "api_key_env": None,
     "temperature": 0.7, "system_prompt": "", "prompt_template": _ASK},
    {"name": "llama", "model": "llama3.2:3b", "base_url": "http://localhost:11434/v1", "api_key_env": None,
     "temperature": 0.7, "system_prompt": "", "prompt_template": _ASK},
    # the same model with a small prompt tweak: does "double-check" really help?
    {"name": "qwen_check", "model": "qwen2.5:3b", "base_url": "http://localhost:11434/v1", "api_key_env": None,
     "temperature": 0.7, "system_prompt": "",
     "prompt_template": _ASK + " Double-check your arithmetic before giving the final answer."},
]

# ---- Runs and output ----
RUNS = 3
MAX_TOKENS = 800        # step-by-step answers need room; replies cut off at the limit are scored as they are
WORKERS = 1             # keep 1 for a single local GPU
RUN_A_A_TEST = True
OUT_DIR = "examples/template_case_studies/results_template_gsm8k"
SAVE_OUTPUTS = True

if __name__ == "__main__":
    run_cli(globals())
