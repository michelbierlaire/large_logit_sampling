"""Mixture proposal combining an elite component and a tail component.

This module implements the proposal distribution described in the paper section
"Proposal construction", especially the subsection "Mixture proposal". The
resulting proposal is a convex combination of:

- an elite proposal defined exactly on the finite set ``G`` of discovered
  alternatives, and
- a tail proposal defined on the implicit remainder ``R``.

The class also provides utilities to sample from the mixture and to evaluate
proposal probabilities for accepted-reject sampling.

Michel Bierlaire
Wed Mar 25 2026, 17:23:38
"""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
from math import exp

from large_choice_sampling.core.alternatives import Alternative
from large_choice_sampling.core.proposals import Proposal, ProposalSample
from large_choice_sampling.core.types import RandomGenerator
from large_choice_sampling.tail.base import TailProposal
from typing import Any


@dataclass
class MixtureProposal(Proposal):
    """Proposal distribution mixing elite and tail components.

    The elite component is defined exactly on a finite list of alternatives,
    with weights proportional to ``exp(utility)``. The tail component is
    delegated to a ``TailProposal`` instance. The overall proposal assigns
    probability ``alpha`` to the elite component and ``1 - alpha`` to the tail
    component.

    :param elite_alternatives: Ordered list of alternatives contained in the
        elite set.
    :param elite_utilities: Mapping from each elite alternative to its utility
        value.
    :param tail_proposal: Proposal defined on the implicit tail of the choice
        set.
    :param alpha: Mixture weight of the elite proposal. It must lie strictly
        between 0 and 1.
    """

    elite_alternatives: list[Alternative]
    elite_utilities: dict[Alternative, float]
    tail_proposal: TailProposal
    alpha: float

    def __post_init__(self) -> None:
        """Validate inputs and precompute elite masses for fast sampling.

        The elite weights are proportional to ``exp(utility)`` and are stored
        together with their cumulative sums so that elite alternatives can be
        sampled efficiently using inverse transform sampling.

        :return: None.
        :raises ValueError: If the elite component is empty or if ``alpha`` is
            not strictly between 0 and 1.
        """
        if not self.elite_alternatives:
            raise ValueError(
                "The elite component must contain at least one alternative."
            )
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be strictly between 0 and 1.")
        self._elite_weights: list[float] = [
            exp(self.elite_utilities[a]) for a in self.elite_alternatives
        ]
        self._elite_mass: float = sum(self._elite_weights)
        cumulative: list[float] = []
        total: float = 0.0
        for weight in self._elite_weights:
            total += weight
            cumulative.append(total)
        self._elite_cumulative: list[float] = cumulative

    @property
    def elite_mass(self) -> float:
        """Return the total unnormalized mass of the elite component.

        :return: Sum of ``exp(utility)`` over all elite alternatives.
        """
        return self._elite_mass

    def _sample_elite(self, rng: RandomGenerator) -> ProposalSample:
        """Sample one alternative from the elite component.

        The elite alternatives are sampled according to probabilities
        proportional to ``exp(utility)`` within the elite set, and then scaled
        by the mixture weight ``alpha``.

        :param rng: Random number generator used to draw the elite sample.
        :return: A ``ProposalSample`` describing the sampled alternative, its
            overall proposal probability, and the generating component.
        """
        draw: float = rng.random() * self._elite_mass
        index: int = bisect_left(self._elite_cumulative, draw)
        alternative: Alternative = self.elite_alternatives[index]
        probability: float = (
            self.alpha * exp(self.elite_utilities[alternative]) / self._elite_mass
        )
        return ProposalSample(
            alternative=alternative,
            proposal_probability=probability,
            component="elite",
        )

    def _sample_tail(self, rng: RandomGenerator) -> ProposalSample:
        """Sample one alternative from the tail component.

        :param rng: Random number generator used to draw the tail sample.
        :return: A ``ProposalSample`` describing the sampled alternative, its
            overall proposal probability, and the generating component.
        """
        alternative, tail_probability = self.tail_proposal.sample(rng)
        alternative: Alternative = alternative
        tail_probability: float = tail_probability
        probability: float = (1.0 - self.alpha) * tail_probability
        return ProposalSample(
            alternative=alternative,
            proposal_probability=probability,
            component="tail",
        )

    def sample(self, rng: RandomGenerator) -> ProposalSample:
        """Sample one alternative from the mixture proposal.

        With probability ``alpha``, the sample is drawn from the elite
        component. Otherwise, it is drawn from the tail component.

        :param rng: Random number generator used to draw from the mixture.
        :return: A ``ProposalSample`` containing the sampled alternative, its
            overall proposal probability, and the component that generated it.
        """
        if rng.random() < self.alpha:
            return self._sample_elite(rng)
        return self._sample_tail(rng)

    def prob(self, alternative: Alternative) -> float:
        """Evaluate the proposal probability of an alternative.

        The returned value is the sum of the elite and tail contributions. If
        the alternative is not supported by one of the components, that
        component contributes zero.

        :param alternative: Alternative whose proposal probability is to be
            evaluated.
        :return: Overall probability assigned to ``alternative`` by the mixture
            proposal.
        """
        elite_prob: float = 0.0
        if alternative in self.elite_utilities:
            elite_prob = (
                self.alpha * exp(self.elite_utilities[alternative]) / self._elite_mass
            )

        tail_prob: float = 0.0
        try:
            tail_prob = (1.0 - self.alpha) * self.tail_proposal.prob(alternative)
        except KeyError:
            tail_prob = 0.0
        return elite_prob + tail_prob
