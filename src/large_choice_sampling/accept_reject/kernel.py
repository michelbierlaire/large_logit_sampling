"""Accept-reject sampling kernel.

This module implements the final sampling phase of the methodology described in
the paper. It assumes that a proposal distribution and a valid global bound on
``g(i) / q(i)`` have already been constructed. The sampler then repeatedly
proposes candidates and accepts them with the standard accept-reject
probability, producing independent draws from the target distribution.

Michel Bierlaire
Wed Mar 25 2026, 17:13:17
"""

from __future__ import annotations


from dataclasses import dataclass, field
from math import exp

from typing import Any

from large_choice_sampling.core.bounds import BoundInfo
from large_choice_sampling.core.choice_space import ChoiceSpace
from large_choice_sampling.core.results import AcceptRejectResult
from large_choice_sampling.core.types import RandomGenerator
from large_choice_sampling.mixture.proposal import MixtureProposal
from .diagnostics import AcceptanceStats


@dataclass
class AcceptRejectSampler:
    """Sample independently from the target distribution using accept-reject.

    The class combines three ingredients:

    - a ``ChoiceSpace`` able to evaluate the utility of a proposed alternative,
    - a ``MixtureProposal`` able to generate candidates together with their
      proposal probabilities,
    - a ``BoundInfo`` object containing a valid global bound for the
      accept-reject correction.

    :param choice_space: Choice-space object used to evaluate the utility of a
        proposed alternative.
    :param proposal: Proposal distribution used to generate candidate
        alternatives.
    :param bound_info: Precomputed bounds used by the accept-reject algorithm.
    :param stats: Mutable statistics object updated during sampling.
    """

    choice_space: ChoiceSpace
    proposal: MixtureProposal
    bound_info: BoundInfo
    stats: AcceptanceStats = field(default_factory=AcceptanceStats)

    def acceptance_probability(
        self, utility: float, proposal_probability: float
    ) -> float:
        """Compute the accept-reject probability for a proposed alternative.

        The probability is
        ``exp(utility) / (global_bound * proposal_probability)``. A safety
        clamp to ``1.0`` is applied to protect against numerical round-off,
        although the precomputed bound should theoretically keep the ratio in
        the unit interval.

        :param utility: Utility of the proposed alternative.
        :param proposal_probability: Probability assigned to the proposed
            alternative by the proposal distribution.
        :return: Acceptance probability of the proposed alternative.
        :raises ValueError: If ``proposal_probability`` is not strictly
            positive.
        """
        if proposal_probability <= 0.0:
            raise ValueError("Proposal probability must be strictly positive.")
        probability: float = exp(utility) / (
            self.bound_info.global_bound * proposal_probability
        )
        return min(1.0, probability)

    def sample(
        self, rng: RandomGenerator, max_attempts: int = 10000
    ) -> AcceptRejectResult:
        """Generate one accepted draw from the target distribution.

        The method repeatedly samples a candidate from the proposal,
        evaluates its utility, computes the accept-reject probability, and
        accepts or rejects the candidate using the provided random number
        generator. If a candidate is accepted, the method returns an
        ``AcceptRejectResult`` describing the accepted draw.

        :param rng: Random number generator used for proposal sampling and the
            accept-reject decision.
        :param max_attempts: Maximum number of proposal attempts before raising
            an error.
        :return: Description of the accepted draw, including the accepted
            alternative, the number of proposal draws, the proposal
            probability, and the proposal component that generated it.
        :raises RuntimeError: If no candidate is accepted within
            ``max_attempts`` iterations.
        """
        for attempt in range(1, max_attempts + 1):
            self.stats.attempts += 1
            proposal_sample: Any = self.proposal.sample(rng)
            utility: float = self.choice_space.utility(proposal_sample.alternative)
            accept_prob: float = self.acceptance_probability(
                utility=utility,
                proposal_probability=proposal_sample.proposal_probability,
            )
            if rng.random() <= accept_prob:
                self.stats.accepts += 1
                return AcceptRejectResult(
                    alternative=proposal_sample.alternative,
                    proposal_draws=attempt,
                    accepted_probability=accept_prob,
                    proposal_probability=proposal_sample.proposal_probability,
                    component=proposal_sample.component,
                )
        raise RuntimeError("Maximum number of attempts reached before acceptance.")
