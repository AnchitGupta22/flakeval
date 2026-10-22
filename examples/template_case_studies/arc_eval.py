#!/usr/bin/env python3
"""
Case study 1 as an ordinary flakeval settings file: ARC-Challenge science questions, answered with a single letter.

This is exactly the kind of file `flakeval init` gives you (compare it with src/flakeval/templates/my_eval.py). It
imports nothing from the other case-study scripts: the data is a plain CSV, and the scoring is a built-in scorer.

  1. Once:            python examples/template_case_studies/prepare_data.py     (exports the questions to a CSV)
  2. Check the setup: python examples/template_case_studies/arc_eval.py --dry-run
  3. Run it:          python examples/template_case_studies/arc_eval.py
  Quick trial:        python examples/template_case_studies/arc_eval.py --max-items 20 --runs 2 --out-dir results_trial

Needs Ollama running with qwen2.5:3b and llama3.2:3b pulled. 200 questions x 5 runs x 4 system runs = 4,000 short calls.
"""
from flakeval.study import run_cli

# ---- Data (exported by prepare_data.py): columns id, question (the full prompt with choices), answer (a letter) ----
DATA_PATH = "examples/template_case_studies/case_study_data/arc_challenge_200.csv"
INPUT_COLUMN = "question"
EXPECTED_COLUMN = "answer"
ID_COLUMN = "id"
MAX_ITEMS = None
ITEM_SEED = 0

# ---- Scoring: the first standalone capital letter A-H in the reply must be the correct letter ----
SCORER = "choice_letter"

# ---- What to compare: the first system is the baseline ----
SYSTEMS = [
    {"name": "qwen", "model": "qwen2.5:3b", "base_url": "http://localhost:11434/v1", "api_key_env": None,
     "temperature": 1.0, "system_prompt": "", "prompt_template": "{input}"},
    {"name": "llama", "model": "llama3.2:3b", "base_url": "http://localhost:11434/v1", "api_key_env": None,
     "temperature": 1.0, "system_prompt": "", "prompt_template": "{input}"},
    # the same model at a low temperature: does a subtle setting change survive the noise?
    {"name": "qwen_t03", "model": "qwen2.5:3b", "base_url": "http://localhost:11434/v1", "api_key_env": None,
     "temperature": 0.3, "system_prompt": "", "prompt_template": "{input}"},
]

# ---- Runs and output ----
RUNS = 5
MAX_TOKENS = 16         # a single letter is all we need
WORKERS = 1             # keep 1 for a single local GPU
RUN_A_A_TEST = True
OUT_DIR = "examples/template_case_studies/results_template_arc"
SAVE_OUTPUTS = True

if __name__ == "__main__":
    run_cli(globals())
