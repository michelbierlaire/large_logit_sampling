"""Stopping criteria for good-set search procedures.

This module defines the abstract interface used to decide when a search over an
implicit choice-space representation should terminate. Concrete stopping rules
can be based on the number of expanded states, the number of complete
alternatives found, or any other diagnostic maintained by the search
algorithm.

Michel Bierlaire
Wed Mar 25 2026, 14:52:33
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class StoppingCriterion(ABC):
    """Abstract base class for search stopping rules.

    A stopping criterion inspects summary statistics produced during the search
    process and decides whether the search should terminate.
    """

    @abstractmethod
    def should_stop(self, *, states_expanded: int, leaves_found: int) -> bool:
        """Decide whether the search should stop.

        :param states_expanded: Number of search states that have been removed
            from the frontier and processed.
        :param leaves_found: Number of complete alternatives discovered so far.
        :return: ``True`` if the search should terminate, ``False`` otherwise.
        """


@dataclass(frozen=True)
class MaxStatesStopping(StoppingCriterion):
    """Stop the search after a maximum number of expanded states.

    :param max_states: Maximum allowed number of expanded states.
    """

    max_states: int

    def should_stop(self, *, states_expanded: int, leaves_found: int) -> bool:
        """Check whether the state-expansion limit has been reached.

        :param states_expanded: Number of search states that have been removed
            from the frontier and processed.
        :param leaves_found: Number of complete alternatives discovered so far.
            This argument is accepted for interface compatibility and is not
            used by this stopping rule.
        :return: ``True`` if ``states_expanded`` is greater than or equal to
            ``max_states``, ``False`` otherwise.
        """
        return states_expanded >= self.max_states


@dataclass(frozen=True)
class MaxAlternativesStopping(StoppingCriterion):
    """Stop the search after a maximum number of complete alternatives.

    :param max_alternatives: Maximum allowed number of complete alternatives
        discovered during the search.
    """

    max_alternatives: int

    def should_stop(self, *, states_expanded: int, leaves_found: int) -> bool:
        """Check whether the alternative-discovery limit has been reached.

        :param states_expanded: Number of search states that have been removed
            from the frontier and processed. This argument is accepted for
            interface compatibility and is not used by this stopping rule.
        :param leaves_found: Number of complete alternatives discovered so far.
        :return: ``True`` if ``leaves_found`` is greater than or equal to
            ``max_alternatives``, ``False`` otherwise.
        """
        return leaves_found >= self.max_alternatives
