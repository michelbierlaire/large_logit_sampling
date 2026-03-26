"""Uniform tail proposal over frontier regions.

This module implements the tail proposal described in the paper sections
"Proposal construction" and "Bound construction". The proposal first selects
one unexplored frontier region uniformly, then delegates to the choice-space
completion mechanism to generate a complete alternative within that region.

The resulting proposal is exact only for the sampled alternative, because the
prototype caches the realized completion probability rather than reconstructing
it for arbitrary alternatives after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from large_choice_sampling.core.alternatives import Alternative
from large_choice_sampling.core.choice_space import ChoiceSpace
from large_choice_sampling.core.results import GoodSetResult
from large_choice_sampling.core.states import TailRegion
from large_choice_sampling.core.types import RandomGenerator
from .base import TailProposal


@dataclass
class UniformRegionTailProposal(TailProposal):
    """Tail proposal obtained by sampling uniformly over frontier regions.

    The proposal proceeds in two stages. First, it selects one tail region
    uniformly among the frontier regions returned by the good-set search.
    Second, it invokes the choice-space-specific completion procedure to sample
    a complete alternative inside the selected region.

    The implementation assumes that the regions are disjoint, so that each
    alternative belongs to a unique region. This assumption holds for the
    route-choice prototype, where frontier states correspond to partial simple
    paths with disjoint completion sets.

    :param choice_space: Choice-space object providing the completion sampler
        and a lower bound on completion probabilities.
    :param regions: Tail regions defining the unexplored part of the choice
        set.
    :param region_probability: Probability assigned to each region under the
        uniform state-selection rule.
    :param utility_upper_bound: Maximum upper bound over all regions. It is
        used in the computation of the tail bound term.
    :param completion_probability_cache: Cache mapping sampled alternatives to
        their realized tail proposal probabilities.
    """

    choice_space: ChoiceSpace
    regions: list[TailRegion]
    region_probability: float
    utility_upper_bound: float
    completion_probability_cache: dict[Alternative, float] | None = None

    def __post_init__(self) -> None:
        """Initialize the completion-probability cache when needed.

        :return: None.
        """
        if self.completion_probability_cache is None:
            self.completion_probability_cache = {}

    @classmethod
    def from_goodset_result(
        cls,
        choice_space: ChoiceSpace,
        result: GoodSetResult,
    ) -> "UniformRegionTailProposal":
        """Build a uniform-region tail proposal from a good-set result.

        The frontier regions stored in the good-set result define the tail of
        the choice set. This constructor assigns equal probability to each
        region and records the largest regional utility upper bound for use in
        the tail-bound computation.

        :param choice_space: Choice-space object used later for completion
            sampling and lower-bound evaluation.
        :param result: Search result containing the frontier regions that make
            up the tail.
        :return: A `UniformRegionTailProposal` built from the frontier of the
            supplied good-set result.
        :raises ValueError: If the good-set result does not contain any
            frontier region.
        """
        regions = list(result.frontier_regions)
        if not regions:
            raise ValueError("Cannot build a tail proposal from an empty frontier.")
        region_probability = 1.0 / len(regions)
        utility_upper_bound = max(region.upper_bound for region in regions)
        return cls(
            choice_space=choice_space,
            regions=regions,
            region_probability=region_probability,
            utility_upper_bound=utility_upper_bound,
        )

    def sample(self, rng: RandomGenerator) -> tuple[Alternative, float]:
        """Sample one alternative from the tail proposal.

        A region is selected uniformly, then the choice-space completion
        procedure is used to generate a complete alternative within that
        region. The realized proposal probability is cached for later access by
        :meth:`prob`.

        :param rng: Random number generator used for region selection and
            completion sampling.
        :return: A pair `(alternative, probability)` where `alternative` is the
            sampled complete alternative and `probability` is its realized tail
            proposal probability.
        """
        index = rng.randrange(len(self.regions))
        region = self.regions[index]
        alternative, completion_probability = self.choice_space.sample_completion(
            region.state, rng
        )
        probability = self.region_probability * completion_probability
        self.completion_probability_cache[alternative] = probability
        return alternative, probability

    def prob(self, alternative: Alternative) -> float:
        """Return the cached tail proposal probability of an alternative.

        In this prototype, probabilities are guaranteed only for alternatives
        that have already been sampled, because the completion probability is
        cached at sampling time.

        :param alternative: Alternative whose tail proposal probability is
            requested.
        :return: Cached tail proposal probability of the alternative.
        :raises KeyError: If the alternative has not been sampled before and no
            cached probability is available.
        """
        cached = self.completion_probability_cache.get(alternative)
        if cached is not None:
            return cached
        raise KeyError(
            "Tail probability for the requested alternative is not cached. "
            "In this prototype, probabilities are guaranteed only for sampled alternatives."
        )

    def bound_term(self) -> float:
        """Compute the tail contribution to the global accept-reject bound.

        The bound uses the largest utility upper bound among the regions and
        the smallest lower bound on completion probability across all frontier
        regions.

        :return: Tail bound term used in the global accept-reject constant.
        :raises ValueError: If a completion lower bound is non-positive.
        """
        min_region_completion = min(
            self.choice_space.completion_lower_bound(region.state)
            for region in self.regions
        )
        if min_region_completion <= 0.0:
            raise ValueError("Completion lower bound must be strictly positive.")
        return exp(self.utility_upper_bound) / (
            self.region_probability * min_region_completion
        )
