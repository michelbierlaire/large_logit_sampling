"""State abstractions for implicit large choice sets.

This module defines the generic state objects used to represent subsets of a
very large choice set without explicit enumeration. A search state corresponds
to a region of the choice space, while a tail region is a compact wrapper used
in the accept-reject construction to describe an unexplored portion of the
remainder together with its optimistic utility bound.

Michel Bierlaire
Wed Mar 25 2026, 14:52:33
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchState:
    """Base class for a search state representing a subset of alternatives.

    A search state is the generic building block used to represent an implicit
    region of the choice space. Concrete applications may extend this class to
    store richer structural information, such as partial paths, schedules, or
    partially specified combinatorial objects.

    :param label: Human-readable identifier for the state. This is primarily
        intended for debugging, logging, and diagnostics.
    :param metadata: Optional dictionary containing application-specific
        annotations associated with the state.
    """

    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TailRegion:
    """Compact representation of an unexplored remainder region.

    A tail region wraps a search state that has not been fully expanded and the
    corresponding optimistic bound on the best utility reachable from that
    state. In the route-choice prototype, a tail region is simply an
    unexplored partial path together with its utility upper bound.

    :param state: Search state representing the unexplored region of the choice
        set.
    :param upper_bound: Upper bound on the utility of any complete alternative
        contained in the region represented by ``state``.
    """

    state: SearchState
    upper_bound: float
