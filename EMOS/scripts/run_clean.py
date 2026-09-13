#!/usr/bin/env python3
"""Record selected clean episodes: python scripts/run_clean.py --episode 20."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.fault_injection.cli import main

if __name__ == "__main__":
    raise SystemExit(main(clean=True))
