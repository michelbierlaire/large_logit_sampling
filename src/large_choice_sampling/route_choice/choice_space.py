"""Route-choice implementation of the generic choice-space abstraction.

This module provides the concrete `ChoiceSpace` implementation used by the
route-choice example. It represents alternatives as paths in a directed
network, states as partial paths, and exposes the operations required by the
search, proposal, and bound-construction components of the sampling framework.

Michel Bierlaire
Wed Mar 25 2026, 17:28:20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from large_choice_sampling.core.choice_space import ChoiceSpace
from .alternative import RouteAlternative
from .bounds import RouteBoundCalculator
from .completion import UniformRouteCompletion
from .network import RouteNetwork
from .state import RouteState
from .utility import RouteUtilityModel


@dataclass
class RouteChoiceSpace(ChoiceSpace):
    """Implicit route-choice space based on partial-path states.

    The class exposes the generic choice-space operations required by the
    sampling framework: root-state construction, state expansion, completion
    tests, conversion of complete states into alternatives, utility evaluation,
    upper bounds, and stochastic completion of partial states.

    :param network: Directed network defining the feasible links.
    :param origin: Identifier of the origin node.
    :param destination: Identifier of the destination node.
    :param utility_model: Utility specification used to evaluate complete
        routes and partial states.
    :param max_steps: Maximum number of link-extension steps allowed when
        expanding partial paths.
    """

    network: RouteNetwork
    origin: int
    destination: int
    utility_model: RouteUtilityModel
    max_steps: int = 6
    _bound_calculator: RouteBoundCalculator = field(init=False, repr=False)
    _completion_sampler: UniformRouteCompletion = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize helper objects used by the route-choice space.

        The method creates the optimistic bound calculator and the stochastic
        completion sampler associated with the current network, destination,
        and utility model.

        :return: None.
        """
        self._bound_calculator = RouteBoundCalculator(
            network=self.network,
            destination=self.destination,
            beta_time=self.utility_model.beta_time,
        )
        self._completion_sampler = UniformRouteCompletion(
            network=self.network,
            destination=self.destination,
        )

    def root_state(self) -> RouteState:
        """Return the root state of the route-choice tree.

        The root state corresponds to the origin node, with an empty travel
        history except for the origin itself and zero accumulated travel time.

        :return: Initial partial-path state rooted at the origin.
        """
        return RouteState(
            current_node=self.origin,
            path_nodes=(self.origin,),
            total_travel_time=0.0,
            remaining_steps=self.max_steps,
        )

    def expand(self, state: RouteState) -> Iterable[RouteState]:
        """Expand a partial path into feasible child states.

        The method creates one child state for each feasible outgoing link from
        the current node, excluding links that would revisit an already visited
        node. Children whose optimistic utility bound is negative infinity are
        discarded.

        :param state: Partial-path state to expand.
        :return: Iterable of feasible child states.
        """
        if state.remaining_steps <= 0:
            return []
        visited: set[int] = set(state.path_nodes)
        children: list[RouteState] = []
        for link in self.network.outgoing(state.current_node):
            if link.head in visited:
                continue
            child: RouteState = RouteState(
                current_node=link.head,
                path_nodes=state.path_nodes + (link.head,),
                total_travel_time=state.total_travel_time + link.travel_time,
                remaining_steps=state.remaining_steps - 1,
            )
            if self.upper_bound(child) != float("-inf"):
                children.append(child)
        return children

    def is_complete(self, state: RouteState) -> bool:
        """Check whether a state corresponds to a complete route.

        :param state: State to test.
        :return: ``True`` if the current node is the destination, ``False``
            otherwise.
        """
        return state.current_node == self.destination

    def to_alternative(self, state: RouteState) -> RouteAlternative:
        """Convert a complete state into a route alternative.

        :param state: Complete state representing a full route.
        :return: Route alternative encoded by the state.
        :raises ValueError: If the provided state is not complete.
        """
        if not self.is_complete(state):
            raise ValueError("Cannot convert an incomplete state into an alternative.")
        return RouteAlternative(
            nodes=state.path_nodes, total_travel_time=state.total_travel_time
        )

    def utility(self, alternative: RouteAlternative) -> float:
        """Evaluate the utility of a complete route alternative.

        :param alternative: Complete route alternative.
        :return: Utility value of the alternative under the current utility
            model.
        """
        return self.utility_model.utility(alternative)

    def utility_from_state(self, state: RouteState) -> float:
        """Evaluate utility information associated with a state.

        For complete states, this method returns the utility of the associated
        route. For incomplete states, it returns the utility contribution of
        the partial path accumulated so far.

        :param state: Partial or complete route state.
        :return: Utility of the complete route if the state is complete,
            otherwise utility accumulated along the partial path.
        """
        if self.is_complete(state):
            return self.utility(self.to_alternative(state))
        return -self.utility_model.beta_time * state.total_travel_time

    def upper_bound(self, state: RouteState) -> float:
        """Return an optimistic utility bound for a state.

        :param state: Partial-path state.
        :return: Upper bound on the utility of any feasible completion of the
            state.
        """
        return self._bound_calculator.upper_bound_utility(
            current_travel_time=state.total_travel_time,
            current_node=state.current_node,
        )

    def sample_completion(
        self, state: RouteState, rng: Any
    ) -> tuple[RouteAlternative, float]:
        """Sample a complete route extending the given state.

        The returned probability is the probability of the sampled completion
        under the stochastic completion mechanism.

        :param state: Partial-path state to complete.
        :param rng: Random number generator used by the completion sampler.
        :return: Pair consisting of the sampled route alternative and its
            proposal probability.
        """
        return self._completion_sampler.sample(state, rng)

    def completion_lower_bound(self, state: RouteState) -> float:
        """Return a lower bound on completion probabilities from a state.

        :param state: Partial-path state.
        :return: Lower bound on the probability of generating any feasible
            completion of the state under the completion sampler.
        """
        return self._completion_sampler.completion_lower_bound(state)
