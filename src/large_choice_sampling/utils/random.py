from __future__ import annotations

from random import Random


def make_rng(seed: int | None = None) -> Random:
    return Random(seed)
