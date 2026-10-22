#!/usr/bin/env python3
"""
flakeval settings file: compare YOUR models or prompts on YOUR data, with proper noise estimates.

HOW TO USE
  1. Edit the settings below. Nothing else in this file needs changing (the engine lives in the flakeval package).
  2. Check your setup on a couple of rows first (no files written):   python my_eval.py --dry-run
  3. Run it for real:                                                python my_eval.py
  Changed your scorer, or found it marking good answers wrong?    python my_eval.py --rescore
  That re-marks the saved answers without calling any model (needs SAVE_OUTPUTS = True).
  A progress bar shows how far along it is (with an estimated time left). Result files go to OUT_DIR, and the
  last thing printed is the verdict for each comparison.
  (You got this file from `flakeval init`. Run that again in any folder to start a new evaluation.)

WHAT IT DOES
  For every system in SYSTEMS it asks each question in your data RUNS times (with a different random seed each time),
  scores every answer, and then uses flakeval to say whether the difference between the first system (the baseline)
  and each other system is real or just noise. With RUN_A_A_TEST on, it also runs the baseline twice to show how
  much two identical runs differ by chance.

PRIVACY
  flakeval uploads nothing. Your rows are sent ONLY to the model endpoints you list in SYSTEMS. With a local model
  (for example Ollama) nothing leaves your computer. Scores and raw model answers are written to OUT_DIR on your disk;
  if your data is confidential, keep OUT_DIR out of git (add it to .gitignore).
  API keys: put the key in an environment variable and name that variable in api_key_env. Never paste a key here.

COST
  Total model calls = (number of questions) x RUNS x (number of systems, plus 1 for the A/A test). The script prints
  this number and asks for confirmation before it starts. Try a small MAX_ITEMS first if you pay per call.
"""
from flakeval.study import run_cli

# =====================================================================================================================
# EDIT THE SETTINGS BELOW
# =====================================================================================================================

# ---- Your data: a .csv (with a header row) or a .jsonl file (one JSON object per line) ----
DATA_PATH = "sample_data.csv"  # relative to the folder you run this file from
INPUT_COLUMN = "question"      # column with the text sent to the model
EXPECTED_COLUMN = "answer"     # column with the correct answer (set to None if your SCORER does not need one)
ID_COLUMN = "id"               # column with a unique id per row (set to None to just use the row number)
MAX_ITEMS = None               # test on a random subset of this many rows (None = all rows). Aim for 100+ if you can.
ITEM_SEED = 0                  # fixes which subset is chosen, so every system sees the same questions

# ---- How to score an answer (1 = good, 0 = bad) ----
#   "exact"          the answer equals the expected text (ignores case, spaces and trailing punctuation)
#   "contains"       the expected text appears somewhere in the answer
#   "number"         the LAST number in the answer equals the expected number
#   "choice_letter"  multiple choice: the FIRST standalone capital letter A-H in the answer is the expected letter
#   "answer_line"    math, strict: the number after the LAST "Answer:" in the reply equals the expected number
#   "final_number"   math, lenient: like answer_line, but also accepts \boxed{...} and then the last number
#   "custom"         your own function (see custom_score below), for anything else
SCORER = "contains"


def custom_score(output: str, expected: str, row: dict) -> float:
    """Only used when SCORER = "custom". Return a number, usually 1.0 (good) or 0.0 (bad), or anything from 0 to 1.
    `output` is the model's answer, `expected` is the value from EXPECTED_COLUMN, `row` is the whole data row."""
    return 1.0 if output.strip() else 0.0


# ---- The systems to compare. The FIRST one is the baseline; every other one is compared against it. ----
# Two models, or the same model with two different prompts, whatever you want to compare.
#   name             any short label (used for file names)
#   model            the model name your server expects
#   base_url         an OpenAI-compatible endpoint. Local examples: Ollama "http://localhost:11434/v1",
#                    LM Studio "http://localhost:1234/v1". Hosted: see your provider's docs for its base URL.
#   api_key_env      NAME of the environment variable holding the API key (None for local servers)
#   temperature      randomness of the answers (None = do not send the setting, for models that reject it)
#   system_prompt    optional instructions that go before every question
#   prompt_template  how each question is sent; {input} is replaced by the text of INPUT_COLUMN
# Optional per system: "send_seed": False (if a provider rejects the seed setting),
#                      "token_limit_param": "max_completion_tokens" (needed by some newer hosted models).
SYSTEMS = [
    {
        "name": "model_a",
        "model": "qwen2.5:3b",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,
        "temperature": 0.7,
        "system_prompt": "",
        "prompt_template": "{input}",
    },
    {
        "name": "model_b",
        "model": "llama3.2:3b",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,
        "temperature": 0.7,
        "system_prompt": "",
        "prompt_template": "{input}",
    },
]

# ---- Using something that is not an OpenAI-compatible endpoint (another SDK, a local Python model, ...)? ----
# Write a function like the one below and give it to a system with  "call": my_model  (instead of model/base_url).
# It receives the prompt and must return the model's answer as plain text. Example for the Anthropic Python SDK:
#
#   import anthropic
#   client = anthropic.Anthropic()                      # reads the ANTHROPIC_API_KEY environment variable
#   def my_model(prompt, system_prompt, temperature, seed, max_tokens):
#       msg = client.messages.create(model="your-model-name", max_tokens=max_tokens, temperature=temperature,
#                                    system=system_prompt or anthropic.NOT_GIVEN,
#                                    messages=[{"role": "user", "content": prompt}])
#       return msg.content[0].text
#   SYSTEMS[1] = {"name": "my_api_model", "call": my_model, "temperature": 0.7,
#                 "system_prompt": "", "prompt_template": "{input}"}
#
# (That API has no seed setting. That is fine: its answers vary on their own. Just ignore `seed`.)

# ---- How many runs and how long answers can be ----
RUNS = 5                  # times each question is asked per system (3 to 5 is a good range)
MAX_TOKENS = 300          # longest answer allowed; answers cut off here are scored as they are
WORKERS = 1               # parallel requests. Raise for hosted APIs (e.g. 8); keep 1 for a single local GPU
RUN_A_A_TEST = True       # also run the baseline a second time, as a sanity check on how much chance alone moves scores

# ---- Output ----
OUT_DIR = "my_results"    # score files, raw answers and report.txt go here
SAVE_OUTPUTS = True       # keep the raw model answers (useful for checking your scorer; may contain your data)

# =====================================================================================================================
# Nothing below needs editing.
# =====================================================================================================================
if __name__ == "__main__":
    run_cli(globals())  # reads the settings above and handles --dry-run, --yes and --rescore
