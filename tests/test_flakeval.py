import numpy as np

from flakeval import Scores, compare, load_jsonl, repeat, save_jsonl, summarize


def make(p, n=300, k=5, seed=0, item_sd=0.15, ids=None, item_seed=None):
    """Binary outcomes: each item has its own difficulty, then run-level coin flips."""
    d_rng = np.random.default_rng(seed if item_seed is None else item_seed)
    difficulty = np.clip(d_rng.normal(p, item_sd, n), 0.02, 0.98)
    rng = np.random.default_rng(seed)
    x = (rng.random((n, k)) < difficulty[:, None]).astype(float)
    return Scores(ids or [f"q{i}" for i in range(n)], x)


def test_summary_basic():
    s = summarize(make(0.7))
    assert 0.65 < s.mean < 0.75
    assert s.ci[0] < s.mean < s.ci[1]
    assert s.flaky_fraction > 0.3


def test_null_difference_is_within_noise():
    # same underlying items, same true quality, fresh run noise: should rarely be called "real"
    false_alarms = 0
    for seed in range(20):
        a = make(0.7, seed=100 + seed, item_seed=seed)
        b = make(0.7, seed=200 + seed, item_seed=seed)
        if compare(a, b, B=1000, seed=seed).verdict != "WITHIN NOISE":
            false_alarms += 1
    assert false_alarms <= 4  # nominal 5% => expect ~1 of 20


def test_large_effect_is_detected():
    a = make(0.60, seed=1, item_seed=7)
    b = make(0.70, seed=2, item_seed=7)
    c = compare(a, b, B=2000)
    assert c.verdict == "REAL IMPROVEMENT" and c.delta > 0 and c.p_value < 0.05


def test_single_run_warns():
    c = compare(make(0.6, k=1), make(0.62, k=1), B=500)
    assert any("1 run" in w for w in c.warnings)


def test_roundtrip_and_truncation(tmp_path):
    s = make(0.5, n=10, k=3)
    path = tmp_path / "r.jsonl"
    save_jsonl(s, path)
    with open(path, "a") as f:
        f.write('{"item_id": "q0", "run": 3, "score": 1}\n')  # extra run for one item only
    loaded = load_jsonl(path)
    assert loaded.n_runs == 3 and loaded.n_items == 10


def test_repeat_runner():
    calls = []

    def fn(x):
        calls.append(x)
        return float(x > 1)

    s = repeat(fn, {"a": 1, "b": 2}, runs=4, workers=2)
    assert s.x.shape == (2, 4) and len(calls) == 8 and s.x[1].mean() == 1.0


def test_unpaired_items_warn():
    a = make(0.6, n=50)
    b = Scores(a.ids[:40], a.x[:40])
    assert compare(a, b, B=300).warnings


def test_identical_results_are_flagged():
    a = make(0.7, seed=3)
    c = compare(a, Scores(a.ids, a.x.copy()), B=300)
    assert c.verdict == "IDENTICAL RESULTS"


def test_zero_run_noise_is_warned():
    # every run of every item is the same => run noise is not being measured
    x = np.tile((np.random.default_rng(0).random((100, 1)) < 0.7).astype(float), (1, 5))
    s = Scores([f"q{i}" for i in range(100)], x)
    assert any("No item changed" in w for w in summarize(s).warnings)
    assert any("no item changed" in w for w in compare(s, make(0.7, n=100, seed=5), B=300).warnings)


def test_sparse_one_sided_differences_are_not_called_real():
    # only 4 items differ (each by +0.2): bootstrap CI alone would exclude 0, but this is not strong evidence
    base = np.tile((np.random.default_rng(1).random((200, 1)) < 0.6).astype(float), (1, 5))
    b = base.copy()
    flip = np.where(base[:, 0] == 0)[0][:4]
    b[flip, 0] = 1.0
    ids = [f"q{i}" for i in range(200)]
    assert compare(Scores(ids, base), Scores(ids, b), B=3000).verdict == "WITHIN NOISE"


def test_save_creates_missing_folder(tmp_path):
    path = tmp_path / "new" / "nested" / "r.jsonl"
    save_jsonl(make(0.5, n=5, k=2), path)
    assert path.exists()


def test_interval_width_is_calibrated():
    """Regression test: resampling both items and runs double-counted run noise (~25% too-wide intervals)."""
    rng = np.random.default_rng(0)
    n, k, sims = 60, 5, 100
    ids = [f"q{i}" for i in range(n)]
    widths, deltas = [], []
    for s in range(sims):
        diff = np.clip(rng.normal(0.75, 0.2, n), 0.02, 0.98)
        a = Scores(ids, (rng.random((n, k)) < diff[:, None]).astype(float))
        b = Scores(ids, (rng.random((n, k)) < diff[:, None]).astype(float))  # identical system
        c = compare(a, b, B=400, seed=s)
        widths.append(c.ci[1] - c.ci[0])
        deltas.append(c.delta)
    ratio = np.mean(widths) / (2 * 1.96 * np.std(deltas))
    assert 0.75 < ratio < 1.12, ratio


def test_all_perfect_scores_are_explained_as_a_ceiling_effect():
    import numpy as np
    from flakeval import summarize
    from flakeval.cli import format_summary
    ids = [f"q{i}" for i in range(12)]
    a = Scores(ids, np.ones((12, 5)))
    b = Scores(ids, np.ones((12, 5)))
    text = format_summary("a", summarize(a))
    assert "nan" not in text.lower() and "n/a" in text and "too easy" in text
    c = compare(a, b, B=300)
    assert c.verdict == "IDENTICAL RESULTS" and "too easy" in c.advice
    assert not any("fixed seed" in w for w in c.warnings)       # no misleading seed/temperature hints here


def test_one_constant_system_still_gets_a_specific_warning():
    import numpy as np
    ids = [f"q{i}" for i in range(40)]
    rng = np.random.default_rng(0)
    perfect = Scores(ids, np.ones((40, 4)))
    varied = Scores(ids, (rng.random((40, 4)) < 0.7).astype(float))
    c = compare(perfect, varied, B=300)
    assert any(w.startswith("System A: Every score was 1") for w in c.warnings)


def test_flaky_line_shows_a_count_so_one_unstable_item_is_not_reported_as_zero_percent():
    import numpy as np
    from flakeval import summarize
    from flakeval.cli import format_summary
    x = np.ones((200, 5))
    x[17, 2] = 0.0                                   # exactly one item changes score in one run
    ids = [f"q{i}" for i in range(200)]
    text = format_summary("a", summarize(Scores(ids, x)))
    assert "1 of 200 (0.5%)" in text and "q17" in text
    assert "0 of 200 (0.0%)" in format_summary("a", summarize(Scores(ids, np.ones((200, 5)))))
