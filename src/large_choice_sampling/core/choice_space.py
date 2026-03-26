"""Abstract interface for implicit large choice spaces.

This module defines the core abstraction used throughout the sampling
framework to represent combinatorial choice sets
without explicit enumeration.

A ``ChoiceSpace`` encapsulates the structure of the set of alternatives
through a state-space representation (tree or graph), where:

- ``SearchState`` represents a partial or complete alternative,
- complete states correspond to actual alternatives,
- non-complete states represent subsets of alternatives,
- expansion defines how the space is explored.

The interface provides the minimal set of operations required by the
sampling algorithms:

- navigation in the state space (root, expand),
- detection and extraction of complete alternatives,
- evaluation of utilities,
- computation of upper bounds used for search and accept-reject.

Concrete implementations are application-specific (e.g., route choice,
activity-based models) and must implement all abstract methods.

Michel Bierlaire
Wed Mar 25 2026, 17:15:33
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from .alternatives import Alternative
from .states import SearchState


class ChoiceSpace(ABC):
    """Abstract interface for an implicit choice space.

    A choice space is represented implicitly through a collection of
    states forming a tree or graph. Each state corresponds to a subset
    of alternatives, and complete states correspond to individual
    alternatives.

    This abstraction allows algorithms to explore very large choice
    sets without enumerating them explicitly.
    """

    @abstractmethod
    def root_state(self) -> SearchState:
        """Return the root state representing the full choice space.

        The root state corresponds to the entire set of feasible
        alternatives.

        :return: The root ``SearchState`` covering the entire choice set.
        """

    @abstractmethod
    def expand(self, state: SearchState) -> Iterable[SearchState]:
        """Generate the child states of a given state.

        The returned states must form a partition of the alternatives
        represented by the input state.

        :param state: The current ``SearchState`` to expand.
        :return: An iterable of child ``SearchState`` objects.
        """

    @abstractmethod
    def is_complete(self, state: SearchState) -> bool:
        """Check whether a state corresponds to a complete alternative.

        :param state: The ``SearchState`` to evaluate.
        :return: ``True`` if the state represents a single complete
            alternative, ``False`` otherwise.
        """

    @abstractmethod
    def to_alternative(self, state: SearchState) -> Alternative:
        """Convert a complete state into an ``Alternative`` instance.

        This method must only be called on states for which
        ``is_complete(state)`` is ``True``.

        :param state: A complete ``SearchState``.
        :return: The corresponding ``Alternative`` object.
        """

    @abstractmethod
    def utility(self, alternative: Alternative) -> float:
        """Compute the utility of a complete alternative.

        :param alternative: A complete ``Alternative``.
        :return: The utility value associated with the alternative.
        """

    @abstractmethod
    def utility_from_state(self, state: SearchState) -> float:
        """Compute the utility associated with a state.

        For complete states, this returns the utility of the
        corresponding alternative. For partial states, this typically
        represents the accumulated utility so far.

        :param state: A ``SearchState`` (complete or partial).
        :return: The utility value associated with the state.
        """

    @abstractmethod
    def upper_bound(self, state: SearchState) -> float:
        """Compute an upper bound on the utility reachable from a state.

        This bound must satisfy:
        U(state) >= max_{i in C(state)} V_i,
        and is used to guide the search and construct accept-reject
        bounds.

        :param state: A ``SearchState``.
        :return: An upper bound on the utility of any completion of the state.
        """
