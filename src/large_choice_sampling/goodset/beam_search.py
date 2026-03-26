"""Beam-search-based generation of the good set.

This module implements a bounded search procedure over the implicit choice-space
representation. The search keeps only the best states according to the
state-specific upper bound provided by the choice-space object.

Complete states are converted into alternatives and collected in the good set.
The remaining frontier states define tail regions that represent the unexplored
part of the choice set.

Michel Bierlaire
Wed Mar 25 2026, 13:49:43
"""

from __future__ import annotations


from dataclasses import dataclass
from .base import GoodSetGenerator

from large_choice_sampling.core.choice_space import ChoiceSpace
from large_choice_sampling.core.results import GoodSetResult
from large_choice_sampling.core.states import TailRegion
from large_choice_sampling.core.types import RandomGenerator


@dataclass
class BeamSearchGoodSetGenerator(GoodSetGenerator):
    """Generate a good set using beam search.

    At each depth, the algorithm expands the current beam, collects complete
    alternatives into the good set, and retains only the states with the
    largest upper bounds for the next iteration.

    :param beam_width: Maximum number of frontier states retained after each
        expansion step.
    :param max_depth_iterations: Maximum number of beam expansion iterations.
    """

    beam_width: int = 10
    max_depth_iterations: int = 20

    def generate(
        self, choice_space: ChoiceSpace, rng: RandomGenerator
    ) -> GoodSetResult:
        """Run beam search and build the corresponding good-set decomposition.

        The method starts from the root state of the choice space. Complete
        states are converted into alternatives and stored in the good set,
        while incomplete states are expanded. After each iteration, only the
        states with the highest upper bounds are kept in the beam. The final
        beam defines the tail regions of the unexplored search space.

        :param choice_space: Implicit representation of the choice set together
            with expansion, completion, utility, and upper-bound operations.
        :param rng: Random number generator. It is part of the common generator
            interface, although this implementation does not use randomness.
        :return: A `GoodSetResult` containing the discovered good set,
            utilities of the discovered alternatives, frontier regions, and
            diagnostic information about the search.
        """
        beam: list = [choice_space.root_state()]
        good_set: list = []
        utilities: dict = {}
        states_expanded = 0
        leaves_found = 0

        for _ in range(self.max_depth_iterations):
            candidates: list = []
            for state in beam:
                states_expanded += 1
                if choice_space.is_complete(state):
                    alt = choice_space.to_alternative(state)
                    util = choice_space.utility(alt)
                    good_set.append(alt)
                    utilities[alt] = util
                    leaves_found += 1
                else:
                    candidates.extend(choice_space.expand(state))
            if not candidates:
                break
            beam = sorted(candidates, key=choice_space.upper_bound, reverse=True)[
                : self.beam_width
            ]

        frontier_regions: list[TailRegion] = [
            TailRegion(state=s, upper_bound=choice_space.upper_bound(s)) for s in beam
        ]
        best_unseen = max(
            (r.upper_bound for r in frontier_regions), default=float("-inf")
        )
        return GoodSetResult(
            good_set=good_set,
            utilities=utilities,
            frontier_regions=frontier_regions,
            best_unseen_upper_bound=best_unseen,
            states_expanded=states_expanded,
            leaves_found=leaves_found,
            diagnostics={"beam_width": float(self.beam_width)},
        )
