"""Computation of accept-reject bounds for the mixture proposal.

This module implements the formulas described in the sections
"Bound construction" and "Global bound" of the document. It provides
utilities to compute the elite bound, the tail bound, and the resulting
global bound used in the accept-reject algorithm.

Michel Bierlaire
Wed Mar 25 2026, 14:52:33
"""

from __future__ import annotations

from large_choice_sampling.core.bounds import BoundInfo
from large_choice_sampling.tail.base import TailProposal


def compute_elite_bound(elite_mass: float, alpha: float) -> float:
    """
    Compute the elite contribution to the accept-reject bound.

    The elite bound corresponds to the ratio g(i)/q(i) for alternatives
    in the elite set G, which simplifies to Z_G / alpha.

    :param elite_mass: Total mass of the elite set, i.e., Z_G = sum_{i in G} g(i).
    :type elite_mass: float
    :param alpha: Mixing parameter of the proposal distribution.
    :type alpha: float
    :return: The elite bound Z_G / alpha.
    :rtype: float
    :raises ValueError: If elite_mass is non-positive or alpha is not in (0,1).
    """
    if elite_mass <= 0.0:
        raise ValueError("elite_mass must be positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1).")
    return elite_mass / alpha


def compute_tail_bound(tail_proposal: TailProposal, alpha: float) -> float:
    """
    Compute the tail contribution to the accept-reject bound.

    The tail bound corresponds to the supremum of g(i)/q(i) over the tail
    region R. It is obtained from the tail proposal bound term divided by
    (1 - alpha).

    :param tail_proposal: Tail proposal object providing the bound term.
    :type tail_proposal: TailProposal
    :param alpha: Mixing parameter of the proposal distribution.
    :type alpha: float
    :return: The tail bound term divided by (1 - alpha).
    :rtype: float
    :raises ValueError: If alpha is not in (0,1).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1).")
    return tail_proposal.bound_term() / (1.0 - alpha)


def compute_global_bound(
    elite_mass: float, tail_proposal: TailProposal, alpha: float
) -> BoundInfo:
    """
    Compute the global accept-reject bound for the mixture proposal.

    This function combines the elite and tail bounds and returns the
    maximum of the two, which is a valid upper bound for g(i)/q(i) over
    all alternatives.

    :param elite_mass: Total mass of the elite set, i.e., Z_G.
    :type elite_mass: float
    :param tail_proposal: Tail proposal object providing the bound term.
    :type tail_proposal: TailProposal
    :param alpha: Mixing parameter of the proposal distribution.
    :type alpha: float
    :return: A BoundInfo object containing the elite bound, tail bound,
        and global bound.
    :rtype: BoundInfo
    """
    elite_bound = compute_elite_bound(elite_mass, alpha)
    tail_bound = compute_tail_bound(tail_proposal, alpha)
    return BoundInfo(
        alpha=alpha,
        elite_mass=elite_mass,
        elite_bound=elite_bound,
        tail_bound=tail_bound,
        global_bound=max(elite_bound, tail_bound),
    )
