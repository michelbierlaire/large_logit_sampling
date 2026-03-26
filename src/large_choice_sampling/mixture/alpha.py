"""Utilities for computing and validating the mixture weight alpha.

This module implements helper functions corresponding to the section
"Optimal mixing" in the document. In particular, it provides:

- a function to compute the optimal mixing parameter that balances the
  elite and tail contributions in the accept-reject bound, and
- a function to ensure that the mixing parameter remains in a valid
  open interval.

The parameter alpha controls the trade-off between sampling from the
elite set and the tail distribution in the mixture proposal.

Michel Bierlaire
Wed Mar 25 2026, 17:23:28
"""

from __future__ import annotations


def compute_balanced_alpha(elite_mass: float, tail_term: float) -> float:
    """Compute the optimal mixing weight between elite and tail components.

    This function returns the value of \\(\\alpha\\in(0,1)\\) that balances the two
    terms in the accept-reject bound:

        elite_mass / alpha = tail_term / (1 - alpha)

    Solving this equation yields:

        alpha = elite_mass / (elite_mass + tail_term)

    :param elite_mass: Positive scalar corresponding to the total weight of the
        elite set (typically \\(Z_G\\)).
    :param tail_term: Positive scalar corresponding to the tail bound term
        (typically \\(\\widetilde{B}_R\\)).
    :return: The balanced mixing weight \\(\\alpha\\in(0,1)\\).
    :raises ValueError: If either argument is not strictly positive.
    """
    if elite_mass <= 0.0:
        raise ValueError("elite_mass must be strictly positive.")
    if tail_term <= 0.0:
        raise ValueError("tail_term must be strictly positive.")
    return elite_mass / (elite_mass + tail_term)


def clip_alpha(
    alpha: float, min_alpha: float = 1e-6, max_alpha: float = 1.0 - 1e-6
) -> float:
    """Clip a mixing weight to a valid open interval.

    This function ensures that the mixing parameter \\(\\alpha\\) remains strictly
    within the open interval \\((0,1)\\), which is required for a valid mixture
    between elite and tail proposals.

    :param alpha: The mixing weight to be clipped.
    :param min_alpha: Lower bound of the admissible interval (default: 1e-6).
    :param max_alpha: Upper bound of the admissible interval (default: 1 - 1e-6).
    :return: The clipped value of \\(\\alpha\\).
    :raises ValueError: If the interval is invalid.
    """
    if min_alpha <= 0.0 or max_alpha >= 1.0 or min_alpha >= max_alpha:
        raise ValueError("Invalid alpha clipping interval.")
    return max(min_alpha, min(alpha, max_alpha))
