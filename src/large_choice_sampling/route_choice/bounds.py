"""Utility and proposal-related bounds for the route-choice example.

This module provides the bound calculator used by the route-choice prototype.
It precomputes shortest remaining travel times to the destination and exposes
helpers used to build optimistic utility bounds and simple probability bounds
for sequential path generation.

Michel Bierlaire
Wed Mar 25 2026, 14:52:33
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import inf

from .network import Link, RouteNetwork


@dataclass
class RouteBoundCalculator:
    """Precompute and expose route-choice bounds.

    The calculator derives optimistic utility bounds from shortest remaining
    travel times and computes a simple lower bound on local proposal
    probabilities based on network branching.

    :param network: Directed route network defining the feasible outgoing links
        from each node.
    :param destination: Identifier of the destination node.
    :param beta_time: Marginal disutility of travel time used to convert travel
        time into utility. The implementation assumes additive utility of the
        form ``-beta_time * travel_time``.
    """

    network: RouteNetwork
    destination: int
    beta_time: float = 1.0

    def __post_init__(self) -> None:
        """Build reverse adjacency and precompute bound ingredients.

        The method constructs a reverse graph for shortest-path calculations,
        computes the shortest remaining travel time from every node to the
        destination, and stores a global lower bound on local proposal
        probabilities.

        :return: None.
        """
        self._reverse_adjacency: dict[int, list[tuple[int, float]]] = {}
        for tail, links in self.network.adjacency.items():
            self._reverse_adjacency.setdefault(tail, [])
            for link in links:
                self._reverse_adjacency.setdefault(link.head, []).append(
                    (tail, link.travel_time)
                )
        self._remaining_time = self._compute_shortest_remaining_times()
        self._min_outgoing_probability = self._compute_min_outgoing_probability()

    def _compute_shortest_remaining_times(self) -> dict[int, float]:
        """Compute shortest remaining travel times to the destination.

        The computation is performed on the reverse graph using Dijkstra's
        algorithm. The resulting dictionary maps each node to the shortest
        travel time from that node to the destination.

        :return: Dictionary mapping each node identifier to its shortest
            remaining travel time. Nodes that cannot reach the destination are
            assigned ``inf``.
        """
        dist: dict[int, float] = {node: inf for node in self.network.nodes}
        dist[self.destination] = 0.0
        heap: list[tuple[float, int]] = [(0.0, self.destination)]
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist[node]:
                continue
            for predecessor, travel_time in self._reverse_adjacency.get(node, []):
                nd = d + travel_time
                if nd < dist[predecessor]:
                    dist[predecessor] = nd
                    heapq.heappush(heap, (nd, predecessor))
        return dist

    def _compute_min_outgoing_probability(self) -> float:
        """Compute a global lower bound on local branching probabilities.

        Under a uniform local choice rule over outgoing links, the smallest
        possible transition probability is the reciprocal of the largest
        out-degree in the network.

        :return: Lower bound on the probability of selecting an outgoing link
            at any node under a uniform local proposal.
        """
        max_out = max(
            (len(links) for links in self.network.adjacency.values()), default=1
        )
        return 1.0 / max_out

    def upper_bound_utility(
        self, current_travel_time: float, current_node: int
    ) -> float:
        """Return an optimistic utility bound from the current search state.

        The bound combines the travel time already accumulated along the
        partial path with the shortest remaining travel time to the
        destination. If the destination is unreachable from the current node,
        the method returns negative infinity.

        :param current_travel_time: Travel time accumulated so far along the
            partial path.
        :param current_node: Identifier of the current node at the end of the
            partial path.
        :return: Upper bound on the utility of any feasible completion of the
            partial path.
        """
        remaining = self._remaining_time.get(current_node, inf)
        if remaining == inf:
            return float("-inf")
        return -self.beta_time * (current_travel_time + remaining)

    @property
    def min_outgoing_probability(self) -> float:
        """Return the precomputed lower bound on local proposal probabilities.

        :return: Lower bound on the probability of choosing any feasible
            outgoing link under a uniform local proposal rule.
        """
        return self._min_outgoing_probability
