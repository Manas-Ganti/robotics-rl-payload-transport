"""Pytest configuration: make the repo root importable.

Lets ``pytest`` run from the repo root with no install step and no PYTHONPATH
fiddling -- ``from env.reward import ...`` resolves the same way it does for the
entrypoint scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
