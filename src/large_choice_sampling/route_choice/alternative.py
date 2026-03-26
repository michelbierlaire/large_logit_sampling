"""Route-alternative representation for the route-choice example.

This module defines the concrete alternative object used in the route-choice
prototype. A route alternative corresponds to a complete path in the network,
represented here by the ordered tuple of visited nodes together with its total
travel time.

Michel Bierlaire
Wed Mar 25 2026, 17:24:26

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteAlternative:
    """Complete route alternative represented as a path in the network.

    Instances of this class are immutable so they can safely be used as keys in
    dictionaries storing utilities, probabilities, or diagnostics.

    :param nodes: Ordered tuple of node identifiers describing the complete
        path from origin to destination.
    :param total_travel_time: Total travel time associated with the route.
    :return: A frozen dataclass instance representing one complete route.
    """

    nodes: tuple[int, ...]
    total_travel_time: float
