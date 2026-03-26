"""Abstract interface for good-set generation.

This module defines the common interface implemented by search procedures that
construct the elite set ``G`` and the frontier-based representation of the
remainder of the choice set. Concrete subclasses are responsible for exploring
an implicit choice-space representation and returning the corresponding
``GoodSetResult``.

Michel Bierlaire
Wed Mar 25 2026, 17:22:09
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from large_choice_sampling.core.choice_space import ChoiceSpace
from large_choice_sampling.core.results import GoodSetResult
from large_choice_sampling.core.types import RandomGenerator


class GoodSetGenerator(ABC):
    """Abstract base class for good-set generators.

    A good-set generator explores an implicit choice-space representation and
    returns a decomposition of the full choice set into:

    - a finite set of explicitly discovered complete alternatives, and
    - a frontier-based implicit representation of the unexplored remainder.

    Concrete implementations may rely on best-first search, beam search, or
    other structured search strategies.
    """

    @abstractmethod
    def generate(
        self, choice_space: ChoiceSpace, rng: RandomGenerator
    ) -> GoodSetResult:
        """Generate the good set and the frontier-based remainder representation.

        :param choice_space: Implicit representation of the choice set,
            providing the operations needed by the search strategy.
        :param rng: Random number generator used by implementations that rely
            on randomized exploration or tie-breaking.
        :return: A ``GoodSetResult`` containing the discovered good set,
            utility information, frontier regions, and search diagnostics.
        """
