"""Proposal distributions for sampling alternatives.

This module defines the abstraction of a proposal distribution used in the
accept-reject framework described in the documentation. A proposal is a
distribution over alternatives from which candidates can be drawn, together
with the ability to evaluate the probability of any given alternative.

The proposal may be composed of several components (e.g., elite and tail),
and the sampled object explicitly records which component generated the
alternative. This information can be useful for diagnostics or adaptive
algorithms.

Michel Bierlaire
Wed Mar 25 2026, 14:52:33
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .alternatives import Alternative
from .types import RandomGenerator


@dataclass(frozen=True)
class ProposalSample:
    """Container for a sample drawn from a proposal distribution.

    This class stores both the sampled alternative and the information
    required to evaluate acceptance probabilities in the accept-reject
    algorithm.

    :param alternative: The sampled alternative.
    :param proposal_probability: The probability assigned by the proposal
        distribution to this alternative.
    :param component: Identifier of the proposal component that generated
        the sample (e.g., "elite" or "tail").
    """

    alternative: Alternative
    proposal_probability: float
    component: str


class Proposal(ABC):
    """Abstract base class for proposal distributions.

    A proposal defines a probability distribution over alternatives and
    provides two fundamental operations:

    - sampling: generate an alternative together with its proposal
      probability,
    - evaluation: compute the probability of any given alternative.

    Concrete implementations may correspond to mixture proposals combining
    several components (e.g., elite and tail distributions).
    """

    @abstractmethod
    def sample(self, rng: RandomGenerator) -> ProposalSample:
        """Draw a sample from the proposal distribution.

        :param rng: Random number generator used to perform the sampling.
        :return: A ``ProposalSample`` containing the sampled alternative,
            its proposal probability, and the component that generated it.
        """

    @abstractmethod
    def prob(self, alternative: Alternative) -> float:
        """Evaluate the proposal probability of an alternative.

        :param alternative: The alternative for which the probability is
            to be computed.
        :return: The probability assigned by the proposal distribution to
            the given alternative.
        """
