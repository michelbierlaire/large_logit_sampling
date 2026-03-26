"""Common type aliases used across the project.

This module defines lightweight semantic aliases for core scalar and utility
types used throughout the large-choice-set sampling framework.

:param RandomGenerator: Alias for the random-number generator type used by the
    sampling and search components.
:param Probability: Alias for scalar values interpreted as probabilities.
:param LogWeight: Alias for scalar values interpreted as log-weights or other
    utility-like scores.
:return: This module does not define functions and therefore does not return a
    value.

Michel Bierlaire
Wed Mar 25 2026, 17:21:20
"""
from __future__ import annotations

from random import Random
from typing import TypeAlias

# Alias for the pseudo-random number generator object passed through the
# sampling and search interfaces.
RandomGenerator: TypeAlias = Random
# Alias for scalar values representing probabilities or probability-like
# quantities.
Probability: TypeAlias = float
# Alias for scalar values representing log-weights, utilities, or related
# scores.
LogWeight: TypeAlias = float
