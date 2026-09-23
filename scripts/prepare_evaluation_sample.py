"""CLI for persistent evaluation sample selection (no OCR)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_common import sample_main

if __name__ == "__main__":
    sample_main()
