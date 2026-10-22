"""Turning a model's answer into a score: the built-in scorers and a few text helpers."""
from __future__ import annotations

import hashlib
import re
import unicodedata

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
ANSWER_LINE = re.compile(r"answer:[\s*$]*(-?[\d,]*\.?\d+)", re.IGNORECASE)      # "Answer: 18", "Answer: **$1,200**"
BOXED = re.compile(r"\\boxed\{\s*\\?\$?\s*(-?[\d,]*\.?\d+)")                       # \boxed{13}
CHOICE = re.compile(r"\b([A-H])\b")                                                # a standalone capital letter

SCORERS = ("exact", "contains", "number", "choice_letter", "answer_line", "final_number", "custom")
SCORER_HELP = """\
  "exact"         the answer equals the expected text (ignores case, spaces and trailing punctuation)
  "contains"      the expected text appears somewhere in the answer
  "number"        the LAST number in the answer equals the expected number
  "choice_letter" multiple choice: the FIRST standalone capital letter A-H in the answer equals the expected letter
  "answer_line"   strict math: the number after the LAST "Answer:" in the reply equals the expected number
  "final_number"  lenient math: like answer_line, but if there is no "Answer:" line it also accepts \\boxed{...}
                  and finally the last number in the reply
  "custom"        your own function called custom_score"""


def fold_text(text) -> str:
    """Treat look-alike characters as the same (for example the subscript in H₂O becomes H2O)."""
    return unicodedata.normalize("NFKC", str(text))


def question_hash(text: str) -> str:
    """Short fingerprint of a question, saved with each answer so --rescore can tell if the question was edited."""
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:10]


def normalize_text(text) -> str:
    return re.sub(r"\s+", " ", fold_text(text).strip().lower()).strip(" .!?\"'")


def to_number(text: str):
    try:
        return float(fold_text(text).replace(",", ""))
    except ValueError:
        return None


def _last_number(pattern: re.Pattern, text: str):
    found = pattern.findall(text)
    return to_number(found[-1]) if found else None


def _matches_number(got, expected) -> float:
    want = to_number(str(expected).replace("$", ""))
    return float(got is not None and want is not None and abs(got - want) < 1e-6)


def score_output(scorer: str, output: str, expected, row: dict, custom=None) -> float:
    if scorer == "exact":
        return float(normalize_text(output) == normalize_text(expected))
    if scorer == "contains":
        return float(normalize_text(expected) in normalize_text(output))
    if scorer == "number":
        found = NUMBER.findall(fold_text(output))
        got, want = (to_number(found[-1]) if found else None), to_number(expected)
        return float(got is not None and want is not None and abs(got - want) < 1e-6)
    if scorer == "choice_letter":
        match = CHOICE.search(fold_text(output).strip())
        return float(bool(match) and match.group(1) == str(expected).strip().upper())
    if scorer in ("answer_line", "final_number"):
        text = fold_text(output)
        got = _last_number(ANSWER_LINE, text)
        if got is None and scorer == "final_number":
            got = _last_number(BOXED, text)
            if got is None:
                got = _last_number(NUMBER, text)
        return _matches_number(got, expected)
    if scorer == "custom":
        return float(custom(output, expected, row))
    raise SystemExit(f"Unknown SCORER {scorer!r}. Choose one of:\n{SCORER_HELP}")
