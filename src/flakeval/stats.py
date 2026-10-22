"""Core statistics for flakeval.

Data model: a `Scores` object holds an (n_items x n_runs) matrix. Each cell is
the score of one eval item on one repeated run of the same system.

Two sources of noise affect an eval score:
  * run noise      - the same item gives different scores across runs
                     (sampling temperature, batching, API drift)
  * item sampling  - your eval set is a finite sample of all possible questions

Confidence intervals come from a bootstrap over ITEMS only, using each item's mean score across runs.
Run noise is already inside those item means, so also resampling runs would count it twice (we measured
~25% too-wide intervals that way). The run-noise share of the uncertainty is estimated separately, from the
within-item variance of the runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_CHUNK = 500  # bootstrap replicates per vectorised batch (keeps memory small)


@dataclass
class Scores:
    ids: list[str]
    x: np.ndarray  # shape (n_items, n_runs)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        if self.x.ndim != 2:
            raise ValueError("scores must be a 2-D (items x runs) matrix")
        if len(self.ids) != self.x.shape[0]:
            raise ValueError("ids length must match number of rows")

    @property
    def n_items(self) -> int:
        return self.x.shape[0]

    @property
    def n_runs(self) -> int:
        return self.x.shape[1]

    @property
    def mean(self) -> float:
        return float(self.x.mean())

    def subset(self, ids: list[str]) -> "Scores":
        pos = {i: j for j, i in enumerate(self.ids)}
        return Scores(list(ids), self.x[[pos[i] for i in ids]])


# --------------------------------------------------------------------------- #
# single system
# --------------------------------------------------------------------------- #
@dataclass
class Summary:
    n_items: int
    n_runs: int
    mean: float
    ci: tuple[float, float]
    se_total: float
    se_run: float
    run_noise_share: float  # fraction of variance of the mean due to run noise
    flaky_fraction: float  # items whose score is not constant across runs
    flaky_items: list[tuple[str, float]] = field(default_factory=list)  # (id, std)
    warnings: list[str] = field(default_factory=list)


def _boot_item_means(v: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap the mean of per-item values v (shape (n,)) by resampling items."""
    n = len(v)
    out = np.empty(B)
    done = 0
    while done < B:
        b = min(_CHUNK * 4, B - done)
        out[done : done + b] = v[rng.integers(0, n, size=(b, n))].mean(axis=1)
        done += b
    return out


def _run_noise_var_of_mean(x: np.ndarray) -> float:
    """Variance of the overall mean that is due to run-to-run noise: sum_i s_i^2 / k / n^2."""
    n, k = x.shape
    return float(x.var(axis=1, ddof=1).sum() / k / n**2)


def _constant_hint(v: float) -> str:
    if v == 1.0:
        return " If 1 is the best possible score, this test is too easy to tell systems apart (a ceiling effect)."
    if v == 0.0:
        return " If 0 is the worst possible score, this test is too hard to tell systems apart (a floor effect)."
    return ""


def _constant_message(v: float) -> str:
    """Explain the case where every single score is the same value."""
    return f"Every score was {v:g} in every run.{_constant_hint(v)} Use harder, more varied, or more questions."


def summarize(s: Scores, B: int = 4000, seed: int = 0, top_flaky: int = 10) -> Summary:
    rng = np.random.default_rng(seed)
    warnings: list[str] = []
    item_means = s.x.mean(axis=1)
    total = _boot_item_means(item_means, B, rng)
    se_total = float(total.std(ddof=1))
    if s.n_runs > 1:
        se_run = float(np.sqrt(_run_noise_var_of_mean(s.x)))
    else:
        se_run = float("nan")
        warnings.append("Only 1 run per item: run-to-run noise cannot be measured. Use >= 3 runs.")
    share = min(1.0, se_run**2 / se_total**2) if se_total > 0 and not np.isnan(se_run) else float("nan")

    std = s.x.std(axis=1, ddof=1) if s.n_runs > 1 else np.zeros(s.n_items)
    flaky_mask = std > 1e-12
    if s.n_runs > 1 and not flaky_mask.any():
        if np.all(s.x == s.x.flat[0]):
            warnings.append(_constant_message(float(s.x.flat[0])))
        else:
            warnings.append(
                "No item changed score across runs. The runs look identical (fixed sampling seed, temperature 0, "
                "caching, or a very confident model), so run-to-run noise is NOT being measured here."
            )
    order = np.argsort(-std)[:top_flaky]
    flaky = [(s.ids[i], float(std[i])) for i in order if flaky_mask[i]]
    lo, hi = np.percentile(total, [2.5, 97.5])
    return Summary(
        n_items=s.n_items,
        n_runs=s.n_runs,
        mean=s.mean,
        ci=(float(lo), float(hi)),
        se_total=se_total,
        se_run=se_run,
        run_noise_share=float(share),
        flaky_fraction=float(flaky_mask.mean()),
        flaky_items=flaky,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# paired comparison A -> B
# --------------------------------------------------------------------------- #
@dataclass
class Comparison:
    n_items: int
    runs_a: int
    runs_b: int
    mean_a: float
    mean_b: float
    delta: float  # B - A
    ci: tuple[float, float]
    p_value: float
    se_total: float
    se_run: float
    run_noise_share: float
    mde: float  # min. detectable difference at 80% power, alpha=0.05
    verdict: str
    advice: str
    warnings: list[str] = field(default_factory=list)


def compare(a: Scores, b: Scores, B: int = 10000, seed: int = 0) -> Comparison:
    rng = np.random.default_rng(seed)
    ids = sorted(set(a.ids) & set(b.ids))
    warnings: list[str] = []
    if len(ids) < 2:
        raise ValueError("need at least 2 shared item ids to compare systems")
    dropped = (a.n_items - len(ids)) + (b.n_items - len(ids))
    if dropped:
        warnings.append(f"{dropped} item(s) not present in both files were ignored (paired comparison).")
    xa, xb = a.subset(ids).x, b.subset(ids).x
    n = len(ids)
    ka, kb = xa.shape[1], xb.shape[1]

    d_items = xb.mean(axis=1) - xa.mean(axis=1)  # per-item paired difference of mean scores
    d_total = _boot_item_means(d_items, B, rng)

    delta = float(xb.mean() - xa.mean())
    lo, hi = np.percentile(d_total, [2.5, 97.5])
    identical = bool(np.array_equal(xa, xb))
    same_constant = identical and bool(np.all(xa == xa.flat[0]))
    for name, x in (("A", xa), ("B", xb)):
        if x.shape[1] > 1 and np.all(x == x[:, [0]]) and not same_constant:
            if np.all(x == x.flat[0]):
                warnings.append(f"System {name}: " + _constant_message(float(x.flat[0])))
            else:
                warnings.append(
                    f"System {name}: no item changed score across runs. Its runs look identical (fixed seed, "
                    "temperature 0, caching?), so run-to-run noise is NOT being measured for it."
                )
    # p-value: paired sign-flip permutation test. Null hypothesis: A and B are the same system, so within each
    # item the labels are exchangeable and the sign of every item difference is arbitrary.
    hits, done = 0, 0
    while done < B:
        m = min(_CHUNK * 4, B - done)
        signs = rng.choice((-1.0, 1.0), size=(m, n))
        hits += int((np.abs((signs * d_items).mean(axis=1)) >= abs(delta) - 1e-12).sum())
        done += m
    p = float((hits + 1) / (B + 1))
    se_total = float(d_total.std(ddof=1))
    if ka > 1 and kb > 1:
        var_run = (xa.var(axis=1, ddof=1) / ka + xb.var(axis=1, ddof=1) / kb).sum() / n**2
        se_run = float(np.sqrt(var_run))
        share = min(1.0, se_run**2 / se_total**2) if se_total > 0 else float("nan")
    else:
        se_run, share = float("nan"), float("nan")
        warnings.append("Only 1 run for at least one system: run-to-run noise cannot be measured. Use >= 3 runs each.")
    mde = 2.8 * se_total  # (z_.975 + z_.80) * SE

    eps = 1e-9
    if identical:
        verdict = "IDENTICAL RESULTS"
        if same_constant:
            v = float(xa.flat[0])
            advice = (f"Both systems scored {v:g} on every question in every run.{_constant_hint(v)} "
                      "Use harder, more varied, or more questions.")
        else:
            advice = (
                "Every item scored exactly the same in A and B. If you expected a difference, check that the setting "
                "you changed actually reached the model (seed, temperature, model name, caching)."
            )
    elif (lo > eps or hi < -eps) and p < 0.05:
        verdict = f"REAL {'IMPROVEMENT' if delta > 0 else 'REGRESSION'}"
        advice = "The 95% interval excludes zero after accounting for run noise and question sampling."
    else:
        verdict = "WITHIN NOISE"
        if se_total == 0:
            advice = "No measurable variation in the difference; the systems behave almost identically on these items."
        elif np.isnan(share):
            advice = f"Cannot separate noise sources. This eval can only detect differences larger than ~{mde:.3f}."
        elif share > 0.6:
            advice = (
                f"Most uncertainty is run-to-run noise ({share:.0%}). Running each item more times "
                f"will tighten the interval. Currently detectable difference: ~{mde:.3f}."
            )
        elif share >= 0.4:
            advice = (
                f"Uncertainty is split between run-to-run noise ({share:.0%}) and eval-set size ({1 - share:.0%}). "
                f"Both more runs and more items would help. Currently detectable difference: ~{mde:.3f}."
            )
        else:
            advice = (
                f"Most uncertainty comes from the limited number of eval items ({1 - share:.0%}). "
                f"More runs will not help much; add more items. Currently detectable difference: ~{mde:.3f}."
            )
    return Comparison(
        n_items=n, runs_a=ka, runs_b=kb,
        mean_a=float(xa.mean()), mean_b=float(xb.mean()), delta=delta,
        ci=(float(lo), float(hi)), p_value=p,
        se_total=se_total, se_run=se_run, run_noise_share=float(share),
        mde=float(mde), verdict=verdict, advice=advice, warnings=warnings,
    )
