from large_choice_sampling.route_choice.choice_space import RouteChoiceSpace
from large_choice_sampling.route_choice.network import RouteNetwork
from large_choice_sampling.route_choice.utility import RouteUtilityModel


def test_route_choice_upper_bound_is_finite_on_reachable_states():
    network = RouteNetwork()
    network.add_link(1, 2, 1.0)
    network.add_link(2, 3, 1.0)
    space = RouteChoiceSpace(network, origin=1, destination=3, utility_model=RouteUtilityModel(), max_steps=3)
    root = space.root_state()
    assert space.upper_bound(root) > float("-inf")
