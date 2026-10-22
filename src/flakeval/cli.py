"""Command line interface: `flakeval report`, `flakeval compare`, and `flakeval init` (copies the settings template)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from importlib import resources
from pathlib import Path

from .io import load_jsonl
from .stats import Comparison, Summary, compare, summarize


def format_summary(name: str, s: Summary) -> str:
    lines = [
        f"== {name} ==",
        f"items: {s.n_items}   runs per item: {s.n_runs}",
        f"mean score: {s.mean:.4f}   95% CI: [{s.ci[0]:.4f}, {s.ci[1]:.4f}]",
    ]
    if s.n_runs > 1:
        if s.run_noise_share != s.run_noise_share:  # NaN: the scores did not vary at all
            lines.append("uncertainty from run-to-run noise: n/a (the scores did not vary at all)")
        else:
            lines.append(f"uncertainty from run-to-run noise: {s.run_noise_share:.0%} (rest: eval-set sampling)")
        n_flaky = round(s.flaky_fraction * s.n_items)  # show the count: "0%" next to a listed unstable item looks like a bug
        lines.append(f"flaky items (score changes between runs): {n_flaky} of {s.n_items} ({s.flaky_fraction:.1%})")
        if s.flaky_items:
            lines.append("most unstable items: " + ", ".join(f"{i} (sd {sd:.2f})" for i, sd in s.flaky_items[:5]))
    lines += [f"WARNING: {w}" for w in s.warnings]
    return "\n".join(lines)


def format_comparison(c: Comparison) -> str:
    lines = [
        f"A: {c.mean_a:.4f} ({c.runs_a} runs)   B: {c.mean_b:.4f} ({c.runs_b} runs)   items: {c.n_items}",
        f"delta (B - A): {c.delta:+.4f}   95% CI: [{c.ci[0]:+.4f}, {c.ci[1]:+.4f}]   p = {c.p_value:.4f}",
        f"smallest reliably detectable difference: ~{c.mde:.4f}",
        f"VERDICT: {c.verdict}",
        c.advice,
    ]
    lines += [f"WARNING: {w}" for w in c.warnings]
    return "\n".join(lines)


def init_command(filename: str, force: bool = False, with_sample_data: bool = True) -> int:
    """Copy the settings template (and a tiny sample dataset) out of the installed package."""
    templates = resources.files("flakeval") / "templates"
    dest = Path(filename)
    if dest.exists() and not force:
        print(f"{dest} already exists, so nothing was changed. Use --force to overwrite it "
              f"(you would lose your edits), or choose another name: flakeval init other_name.py")
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text((templates / "my_eval.py").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"created {dest}")
    sample = dest.parent / "sample_data.csv"
    if with_sample_data and not sample.exists():
        sample.write_text((templates / "sample_data.csv").read_text(encoding="utf-8"), encoding="utf-8")
        print(f"created {sample}  (12 easy questions, just to try things out)")
    print(f"\nNext:\n  1. open {dest} and edit the settings at the top (data path, models, how to score)\n"
          f"  2. python {dest} --dry-run     # tries 2 questions, saves nothing\n"
          f"  3. python {dest}               # the real run")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="flakeval", description="Is your eval improvement real, or just noise?")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="noise report for one system")
    r.add_argument("results")
    c = sub.add_parser("compare", help="paired comparison of system A vs B")
    c.add_argument("a")
    c.add_argument("b")
    i = sub.add_parser("init", help="create a settings file (my_eval.py) to test your own models and data")
    i.add_argument("filename", nargs="?", default="my_eval.py")
    i.add_argument("--force", action="store_true", help="overwrite the file if it exists")
    i.add_argument("--no-sample-data", action="store_true", help="do not also create sample_data.csv")
    for sp in (r, c):
        sp.add_argument("--json", action="store_true", help="machine-readable output")
        sp.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.cmd == "init":
        return init_command(args.filename, args.force, not args.no_sample_data)
    if args.cmd == "report":
        s = summarize(load_jsonl(args.results), seed=args.seed)
        print(json.dumps(dataclasses.asdict(s), indent=2) if args.json else format_summary(args.results, s))
    else:
        res = compare(load_jsonl(args.a), load_jsonl(args.b), seed=args.seed)
        print(json.dumps(dataclasses.asdict(res), indent=2) if args.json else format_comparison(res))
        # non-zero exit on regression => usable as a CI gate
        if res.verdict == "REAL REGRESSION":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
