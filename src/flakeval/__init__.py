from importlib.metadata import version, PackageNotFoundError

from .io import load_jsonl, repeat, save_jsonl
from .stats import Comparison, Scores, Summary, compare, summarize

__all__ = [
    "Scores",
    "Summary",
    "Comparison",
    "summarize",
    "compare",
    "repeat",
    "load_jsonl",
    "save_jsonl",
]

try:
    __version__ = version("flakeval")
except PackageNotFoundError:
    __version__ = "0.0.0"