"""Explicit comparison entrypoint; --help needs only Python's standard library."""
from pathlib import Path
import sys

# Works in this workspace and in the Windows package's sibling ragkit directory.
if not (Path(__file__).resolve().parent / 'ragkit').is_dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit.retrieval_evaluation import main

if __name__ == '__main__':
    raise SystemExit(main())
