"""Export the questions used in the two case studies as plain CSV files, for use with a flakeval settings file.

    python examples/template_case_studies/prepare_data.py                  # 200 questions each
    python examples/template_case_studies/prepare_data.py --n-items 500    # a bigger sample

Writes (default folder: case_study_data/):
    arc_challenge_200.csv   columns: id, question (the full prompt with the choices), answer (a letter)
    gsm8k_200.csv           columns: id, question, answer (a number)

This script stands alone, and it is also a small model of how to prepare your own data: whatever you start from, the
goal is just a CSV (or JSONL) with one row per question, a unique id, the text to send, and the expected answer.
The questions are picked with the same shuffle and seed as the dedicated case-study scripts, so the first N match.
Needs: pip install -e ".[examples]"   (the `datasets` package) and an internet connection the first time.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

LETTERS = "ABCDEFGH"


def _arc_rows(n_items: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rows = []
    for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"):
        labels = list(r["choices"]["label"])
        if r["answerKey"] not in labels:
            continue
        options = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(r["choices"]["text"]))
        prompt = f"{r['question']}\n\n{options}\n\nAnswer with only the letter of the correct choice."
        rows.append({"id": r["id"], "question": prompt, "answer": LETTERS[labels.index(r["answerKey"])]})
    random.Random(seed).shuffle(rows)
    return rows[:n_items]


def _gsm8k_rows(n_items: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rows = []
    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test")):
        answer = float(r["answer"].split("####")[-1].replace(",", "").replace("$", "").strip().rstrip("."))
        rows.append({"id": f"gsm8k_test_{i}", "question": r["question"],
                     "answer": str(int(answer)) if answer.is_integer() else str(answer)})
    random.Random(seed).shuffle(rows)
    return rows[:n_items]


def _write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer"])
        writer.writeheader()
        writer.writerows(rows)


def prepare(out_dir: str = "examples/template_case_studies/case_study_data", n_items: int = 200, seed: int = 0) -> dict[str, Path]:
    out = Path(out_dir)
    paths = {"arc": out / f"arc_challenge_{n_items}.csv", "gsm8k": out / f"gsm8k_{n_items}.csv"}
    _write(_arc_rows(n_items, seed), paths["arc"])
    _write(_gsm8k_rows(n_items, seed), paths["gsm8k"])
    return paths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="examples/template_case_studies/case_study_data")
    p.add_argument("--n-items", type=int, default=200)
    p.add_argument("--item-seed", type=int, default=0)
    a = p.parse_args()
    for path in prepare(a.out_dir, a.n_items, a.item_seed).values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
