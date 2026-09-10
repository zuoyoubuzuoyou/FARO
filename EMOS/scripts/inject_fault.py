#!/usr/bin/env python3
"""Run from the EMOS checkout: python scripts/inject_fault.py --help."""

import sys
from pathlib import Path

# Allow --help and offline --dry-run without installing the project packages.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.fault_injection.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
