"""Base interface for tail proposal distributions.

This module defines the abstract interface used to represent the tail proposal
component q_R in the accept-reject sampling framework described in the
methodology. A tail proposal is responsible for generating alternatives from
the implicit remainder of the choice set and evaluating their proposal
probabilities.

The interface is designed so that implementations can rely on generative
procedures rather than explicit enumeration of the choice set.

Michel Bierlaire
Wed Mar 25 2026, 14:52:33
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from large_choice_sampling.core.alternatives import Alternative
from large_choice_sampling.core.types import RandomGenerator


class TailProposal(ABC):
    @abstractmethod
    def sample(self, rng: RandomGenerator) -> tuple[Alternative, float]:
        """Sample an alternative from the tail proposal.

        The method implements the generative mechanism defining q_R. It returns
        both the sampled alternative and its proposal probability under q_R.

        :param rng: Random number generator used to sample from the proposal.
        :type rng: RandomGenerator
        :return: A tuple (alternative, probability), where `alternative` is the
            sampled element of the choice set and `probability` is its proposal
            probability q_R(i).
        :rtype: tuple[Alternative, float]
        """

    @abstractmethod
    def prob(self, alternative: Alternative) -> float:
        """Evaluate the tail proposal probability of a given alternative.

        This method returns the probability q_R(i) assigned to a specific
        alternative under the tail proposal distribution.

        :param alternative: The alternative whose proposal probability is
            requested.
        :type alternative: Alternative
        :return: The value of q_R(i).
        :rtype: float
        """

    @abstractmethod
    def bound_term(self) -> float:
        """Return an upper bound used in the accept-reject procedure.

        This method returns a value that upper-bounds the ratio g(j) / q_R(j)
        over all alternatives in the support of the tail proposal. It is used
        to construct the global accept-reject constant.

        :return: A scalar upper bound on g(j) / q_R(j) over the tail.
        :rtype: float
        """
