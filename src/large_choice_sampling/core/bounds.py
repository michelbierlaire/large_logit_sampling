r"""Bound-related data structures for accept-reject sampling.

This module defines immutable containers that summarize the quantities
needed to build and evaluate the accept-reject bound in the
elite/tail decomposition framework described in the paper.

- ``TailBoundInfo`` stores per-state (tail region) information used to
  compute the tail contribution to the global bound.
- ``BoundInfo`` aggregates the elite and tail contributions into a
  single global bound together with the mixing parameter.

All classes are frozen dataclasses to ensure immutability and safe
sharing across components of the algorithm.

Michel Bierlaire
Wed Mar 25 2026, 17:15:31
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TailBoundInfo:
    r"""Information required to compute a tail-region bound.

    :param upper_bound_utility: Upper bound on the utility over the
        region, i.e., a value U(s) such that V(i) \le U(s) for all
        i in the region. The numerator bound is then exp(U(s)).
    :param tail_multiplier: Multiplicative factor accounting for the
        inverse lower bound on the proposal probability over the
        region (e.g., product of branching factors along a path or
        1 / (\pi(s) * \underline{q}_s(i))).
    :param tail_bound: Precomputed bound for the region equal to
        exp(upper_bound_utility) * tail_multiplier.

    :return: An immutable container with all quantities needed to
        evaluate the tail contribution to the global bound.
    """
    upper_bound_utility: float
    tail_multiplier: float
    tail_bound: float


@dataclass(frozen=True)
class BoundInfo:
    r"""Aggregated information for the global accept-reject bound.

    :param alpha: Mixing parameter between the elite and tail
        proposals, with alpha in (0, 1).
    :param elite_mass: Sum of g(i) over the elite set G, i.e., Z_G.
    :param elite_bound: Bound for the elite component, equal to
        Z_G / alpha.
    :param tail_bound: Bound for the tail component, equal to
        \widetilde{B}_R / (1 - alpha).
    :param global_bound: Overall bound c_u = max(elite_bound,
        tail_bound).

    :return: An immutable container summarizing all components of the
        accept-reject bound used during sampling.
    """
    alpha: float
    elite_mass: float
    elite_bound: float
    tail_bound: float
    global_bound: float
