"""Network data structures for route choice models.

This module defines a minimal representation of a directed transportation
network. It provides:

- A ``Link`` class representing directed edges with attributes such as
  travel time.
- A ``RouteNetwork`` class representing the network as an adjacency list.

The network is used as the underlying structure for generating paths
(alternatives) in route choice problems, where each path corresponds to
a sequence of links.

Michel Bierlaire
Wed Mar 25 2026, 17:37:49
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Link:
    """Directed link in a transportation network.

    A link connects a tail node to a head node and is associated with a
    travel time.

    :param tail: Identifier of the origin node of the link.
    :param head: Identifier of the destination node of the link.
    :param travel_time: Travel time associated with traversing the link.
    """

    tail: int
    head: int
    travel_time: float


@dataclass
class RouteNetwork:
    """Directed network represented as an adjacency list.

    The network is stored as a mapping from node identifiers to lists of
    outgoing links. Nodes without outgoing links are still represented in
    the adjacency structure.

    :param adjacency: Dictionary mapping each node to the list of outgoing
        links from that node.
    """

    adjacency: dict[int, list[Link]] = field(default_factory=dict)

    def add_link(self, tail: int, head: int, travel_time: float) -> None:
        """Add a directed link to the network.

        If the tail or head nodes are not already present in the adjacency
        structure, they are created automatically. The head node is ensured
        to exist even if it has no outgoing links.

        :param tail: Identifier of the origin node of the link.
        :param head: Identifier of the destination node of the link.
        :param travel_time: Travel time associated with the link.
        :return: None.
        """
        self.adjacency.setdefault(tail, []).append(
            Link(tail=tail, head=head, travel_time=travel_time)
        )
        self.adjacency.setdefault(head, [])

    def outgoing(self, node: int) -> list[Link]:
        """Return the outgoing links from a given node.

        :param node: Identifier of the node.
        :return: List of outgoing links. Returns an empty list if the node
            has no outgoing links or is not present in the network.
        """
        return list(self.adjacency.get(node, []))

    @property
    def nodes(self) -> list[int]:
        """Return the list of nodes in the network.

        :return: List of node identifiers present in the adjacency structure.
        """
        return list(self.adjacency.keys())
