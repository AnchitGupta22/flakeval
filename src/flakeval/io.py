"""Reading/writing results and the `repeat` helper that runs an eval many times.

Results format (JSONL, one line per item per run):
    {"item_id": "q17", "run": 2, "score": 1.0}
`correct` (bool) is accepted in place of `score`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .stats import Scores


def load_jsonl(path: str | Path) -> Scores:
    by_item: dict[str, dict[int, float]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                item = str(row["item_id"])
                run = int(row.get("run", len(by_item[item])))
                score = float(row["score"] if "score" in row else row["correct"])
            except KeyError as e:
                raise ValueError(f"{path}:{line_no}: missing field {e}") from None
            by_item[item][run] = score
    if not by_item:
        raise ValueError(f"{path}: no results found")
    k = min(len(r) for r in by_item.values())
    ids = sorted(by_item)
    x = np.array([[by_item[i][r] for r in sorted(by_item[i])[:k]] for i in ids])
    return Scores(ids, x)


def save_jsonl(scores: Scores, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, item in enumerate(scores.ids):
            for r in range(scores.n_runs):
                f.write(json.dumps({"item_id": item, "run": r, "score": float(scores.x[i, r])}) + "\n")


def repeat(
    fn: Callable[[Any], float],
    items: Iterable[Any] | dict[str, Any],
    runs: int = 5,
    workers: int = 1,
    save: str | Path | None = None,
    with_run: bool = False,
) -> Scores:
    """Run `fn(item) -> score` `runs` times for every item.

    `items` may be a dict {item_id: input} or a plain iterable (ids become "0", "1", ...).
    Use workers > 1 for API-bound evals (threads).
    With with_run=True, fn is called as fn(item, run_index), e.g. to derive a different
    sampling seed for every run.
    """
    pairs = list(items.items()) if isinstance(items, dict) else [(str(i), x) for i, x in enumerate(items)]
    jobs = [(i, r) for i in range(len(pairs)) for r in range(runs)]
    out = np.zeros((len(pairs), runs))

    def work(job: tuple[int, int]) -> tuple[int, int, float]:
        i, r = job
        return i, r, float(fn(pairs[i][1], r) if with_run else fn(pairs[i][1]))

    if workers <= 1:
        results = map(work, jobs)
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        results = pool.map(work, jobs)
    for i, r, v in results:
        out[i, r] = v
    scores = Scores([p[0] for p in pairs], out)
    if save:
        save_jsonl(scores, save)
    return scores
