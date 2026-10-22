"""Generate two simulated eval result files so you can try the CLI.

Model A truly scores ~0.70, model B ~0.72 (a small real gain), 200 items x 5 runs.
Both are evaluated on the same items.
"""
import numpy as np

from flakeval import Scores, save_jsonl

rng = np.random.default_rng(42)
n, k = 200, 5
difficulty = np.clip(rng.normal(0.70, 0.18, n), 0.02, 0.98)
for name, bump in [("a", 0.0), ("b", 0.02)]:
    p = np.clip(difficulty + bump, 0, 1)
    x = (rng.random((n, k)) < p[:, None]).astype(float)
    save_jsonl(Scores([f"q{i}" for i in range(n)], x), f"{name}.jsonl")
print("wrote a.jsonl and b.jsonl")
