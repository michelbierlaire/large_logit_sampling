"""Best-first generation of the elite set and tail frontier.

This module implements the search procedure described in the paper section
"Structured representation of the choice set", especially the subsection
"Search and decomposition", together with the construction phase of the
algorithm. The search explores the state tree using the state upper bound as a
priority, collects complete alternatives into the elite set ``G``, and leaves
unexpanded frontier states as the implicit representation of the tail ``R``.

Michel Bierlaire
Wed Mar 25 2026, 16:45:41
"""

from __future__ import annotations


import heapq
from dataclasses import dataclass
from itertools import count

from typing import Any

from large_choice_sampling.core.choice_space import ChoiceSpace
from large_choice_sampling.core.results import GoodSetResult
from large_choice_sampling.core.states import TailRegion
from large_choice_sampling.core.types import RandomGenerator
from .base import GoodSetGenerator
from .stopping import MaxAlternativesStopping, MaxStatesStopping, StoppingCriterion


@dataclass
class BestFirstGoodSetGenerator(GoodSetGenerator):
    """Generate the good set using best-first search.

    States are explored in decreasing order of their upper bound. Complete
    states are converted into alternatives and stored in the good set. The
    remaining frontier states define tail regions that represent the unexplored
    part of the choice set.

    :param stopping: Stopping rule controlling when the search terminates. If
        ``None``, a default rule based on the number of alternatives found is
        used.
    """

    stopping: StoppingCriterion | None = None

    def __post_init__(self) -> None:
        """Initialize the default stopping rule when none is provided.

        :return: None.
        """
        if self.stopping is None:
            self.stopping = MaxAlternativesStopping(max_alternatives=10)

    def generate(
        self, choice_space: ChoiceSpace, rng: RandomGenerator
    ) -> GoodSetResult:
        """Explore the choice-space tree and build the elite/tail decomposition.

        The method starts from the root state, repeatedly extracts the frontier
        state with the largest upper bound, and either stores the corresponding
        alternative in the good set or expands the state into children. The
        states that remain in the frontier when the search stops are converted
        into tail regions.

        :param choice_space: Implicit representation of the choice set. It must
            provide operations to obtain the root state, test completeness,
            convert complete states to alternatives, expand non-complete states,
            evaluate utilities, and compute upper bounds.
        :param rng: Random number generator. It is part of the common interface
            of good-set generators, although this implementation is entirely
            deterministic and does not use randomness.
        :return: A ``GoodSetResult`` containing the discovered elite set,
            utilities of the discovered alternatives, frontier regions,
            summary diagnostics, and an upper bound on the best unseen
            alternative.
        """
        good_set: list[Any] = []
        utilities: dict[Any, float] = {}
        leaves_found = 0
        states_expanded = 0

        frontier: list[tuple[float, int, Any]] = []
        serial = count()
        root: Any = choice_space.root_state()
        heapq.heappush(frontier, (-choice_space.upper_bound(root), next(serial), root))

        while frontier and not self.stopping.should_stop(
            states_expanded=states_expanded,
            leaves_found=leaves_found,
        ):
            _, _, state = heapq.heappop(frontier)
            state: Any = state
            states_expanded += 1

            if choice_space.is_complete(state):
                alternative: Any = choice_space.to_alternative(state)
                utility: float = choice_space.utility(alternative)
                good_set.append(alternative)
                utilities[alternative] = utility
                leaves_found += 1
                continue

            for child in choice_space.expand(state):
                child: Any = child
                ub: float = choice_space.upper_bound(child)
                heapq.heappush(frontier, (-ub, next(serial), child))

        frontier_regions: list[TailRegion] = [
            TailRegion(state=item[2], upper_bound=-item[0]) for item in frontier
        ]
        best_unseen: float = max(
            (region.upper_bound for region in frontier_regions), default=float("-inf")
        )

        diagnostics: dict[str, float] = {
            "frontier_size": float(len(frontier_regions)),
            "states_expanded": float(states_expanded),
            "leaves_found": float(leaves_found),
        }
        return GoodSetResult(
            good_set=good_set,
            utilities=utilities,
            frontier_regions=frontier_regions,
            best_unseen_upper_bound=best_unseen,
            states_expanded=states_expanded,
            leaves_found=leaves_found,
            diagnostics=diagnostics,
        )
