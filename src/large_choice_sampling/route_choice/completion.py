"""Sequential completion proposals for route-choice tail sampling.

This module implements the stochastic completion mechanism described in the
paper section on proposal construction and the specialization to path sampling.
Starting from a partial route state, the completion procedure extends the path
step by step until the destination is reached. It also provides a conservative
lower bound on completion probabilities used in the tail bound construction.

Michel Bierlaire
Wed Mar 25 2026, 17:30:03
"""

from __future__ import annotations

from dataclasses import dataclass

from large_choice_sampling.core.types import RandomGenerator
from .alternative import RouteAlternative
from .network import Link, RouteNetwork
from .state import RouteState


@dataclass
class UniformRouteCompletion:
    """Uniform sequential completion of a partial route.

    The completion mechanism repeatedly selects one feasible outgoing link
    uniformly at random until the destination is reached. The resulting path
    probability is the product of the local uniform transition probabilities.

    :param network: Directed route network defining the feasible outgoing links
        from each node.
    :param destination: Identifier of the destination node.
    """

    network: RouteNetwork
    destination: int

    def feasible_links(self, state: RouteState) -> list[Link]:
        """Return feasible outgoing links for a partial route state.

        A link is feasible if it leaves the current node, does not revisit an
        already visited node, and the state still allows at least one
        additional step.

        :param state: Partial route state from which the next link is to be
            selected.
        :return: List of feasible outgoing links.
        """
        visited: set[int] = set(state.path_nodes)
        candidates: list[Link] = []
        for link in self.network.outgoing(state.current_node):
            if link.head in visited:
                continue
            if state.remaining_steps <= 0:
                continue
            candidates.append(link)
        return candidates

    def sample(
        self, state: RouteState, rng: RandomGenerator
    ) -> tuple[RouteAlternative, float]:
        """Generate a complete route by sequential uniform completion.

        Starting from the supplied partial route state, the method repeatedly
        selects one feasible outgoing link uniformly at random among the links
        that still allow the destination to be reached within the remaining
        step budget. It returns both the completed route and the probability of
        generating that route under the completion mechanism.

        :param state: Partial route state to be completed.
        :param rng: Random number generator used to sample outgoing links.
        :return: Pair consisting of the completed route and its generation
            probability.
        :raises RuntimeError: If no feasible completion can be generated from
            the supplied state.
        """
        current_node: int = state.current_node
        path_nodes: list[int] = list(state.path_nodes)
        total_travel_time: float = state.total_travel_time
        remaining_steps: int = state.remaining_steps
        probability: float = 1.0

        while current_node != self.destination:
            current_state: RouteState = RouteState(
                current_node=current_node,
                path_nodes=tuple(path_nodes),
                total_travel_time=total_travel_time,
                remaining_steps=remaining_steps,
            )
            links: list[Link] = self.feasible_links(current_state)
            links = [
                link
                for link in links
                if self._can_reach_destination(
                    path_nodes, link.head, remaining_steps - 1
                )
            ]
            if not links:
                raise RuntimeError(
                    "No feasible completion found from the supplied state."
                )
            index: int = rng.randrange(len(links))
            link = links[index]
            probability *= 1.0 / len(links)
            path_nodes.append(link.head)
            total_travel_time += link.travel_time
            current_node = link.head
            remaining_steps -= 1

        return (
            RouteAlternative(
                nodes=tuple(path_nodes), total_travel_time=total_travel_time
            ),
            probability,
        )

    def completion_lower_bound(self, state: RouteState) -> float:
        """Return a conservative lower bound on completion probability.

        The bound is based on the worst-case branching factor of the network
        and on the maximum number of remaining extension steps allowed by the
        state. It is valid for the uniform completion mechanism under the
        simple-path and maximum-length assumptions used in this prototype.

        :param state: Partial route state from which a completion may be
            generated.
        :return: Lower bound on the probability of generating any feasible
            completion from the given state.
        :raises ValueError: If the network contains no outgoing choices.
        """
        max_out_degree = max(
            (len(links) for links in self.network.adjacency.values()), default=1
        )
        if max_out_degree <= 0:
            raise ValueError("Network must contain at least one outgoing choice.")
        steps = max(0, state.remaining_steps)
        return (1.0 / max_out_degree) ** max(1, steps)

    def _can_reach_destination(
        self, path_nodes: list[int], start: int, remaining_steps: int
    ) -> bool:
        """Check whether the destination remains reachable.

        The method performs a depth-limited graph search from the candidate
        next node while respecting the simple-path restriction induced by the
        nodes already visited in the current partial route.

        :param path_nodes: Nodes already visited in the current partial route.
        :param start: Candidate next node from which reachability is tested.
        :param remaining_steps: Maximum number of additional steps allowed.
        :return: ``True`` if the destination can still be reached within the
            remaining number of steps, ``False`` otherwise.
        """
        if start == self.destination:
            return True
        if remaining_steps <= 0:
            return False
        visited: set[int] = set(path_nodes)
        stack: list[tuple[int, int]] = [(start, remaining_steps)]
        seen: set[tuple[int, int]] = set()
        while stack:
            node, steps_left = stack.pop()
            if (node, steps_left) in seen:
                continue
            seen.add((node, steps_left))
            if node == self.destination:
                return True
            if steps_left <= 0:
                continue
            for link in self.network.outgoing(node):
                if link.head in visited:
                    continue
                stack.append((link.head, steps_left - 1))
        return False
