"""State representation for route choice problems.

This module defines the data structure used to represent partial alternatives
(states) in the structured choice set. In the context of route choice, a state
corresponds to a partial path from the origin to a current node in the network.

Each state contains enough information to:
- identify the current position in the network,
- reconstruct the partial path,
- evaluate accumulated attributes such as travel time,
- control the depth or remaining expansion of the state.

These states are used by search procedures (e.g., best-first search) to explore
large combinatorial choice sets without explicit enumeration.

Michel Bierlaire
Wed Mar 25 2026, 14:52:33
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteState:
    """Representation of a partial path in a route choice problem.

    A ``RouteState`` encodes a state in the implicit enumeration of routes.
    It corresponds to a partial alternative that can be extended into a
    complete path. This structure is used by search algorithms to explore
    the choice set and by sampling procedures to generate candidates.

    :param current_node: Identifier of the current node in the network where
        the partial path ends.
    :param path_nodes: Tuple of node identifiers representing the sequence of
        visited nodes from the origin to the current node.
    :param total_travel_time: Accumulated travel time along the partial path.
    :param remaining_steps: Maximum number of expansion steps allowed to
        complete the path (used to control search depth or feasibility).
    :return: An immutable ``RouteState`` instance representing a partial path.
    """

    current_node: int
    path_nodes: tuple[int, ...]
    total_travel_time: float
    remaining_steps: int
