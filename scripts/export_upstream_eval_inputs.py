#!/usr/bin/env python3
"""Compatibility wrapper for the packaged upstream input exporter."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gems_rag.upstream_exports import main


if __name__ == "__main__":
    raise SystemExit(main())
