from random import Random

from large_choice_sampling.goodset.best_first import BestFirstGoodSetGenerator
from large_choice_sampling.goodset.stopping import MaxAlternativesStopping
from large_choice_sampling.route_choice.choice_space import RouteChoiceSpace
from large_choice_sampling.route_choice.network import RouteNetwork
from large_choice_sampling.route_choice.utility import RouteUtilityModel


def build_space() -> RouteChoiceSpace:
    network = RouteNetwork()
    network.add_link(1, 2, 1.0)
    network.add_link(1, 3, 2.0)
    network.add_link(2, 4, 1.0)
    network.add_link(3, 4, 1.0)
    return RouteChoiceSpace(network, origin=1, destination=4, utility_model=RouteUtilityModel(), max_steps=3)


def test_best_first_finds_complete_routes():
    space = build_space()
    generator = BestFirstGoodSetGenerator(stopping=MaxAlternativesStopping(max_alternatives=2))
    result = generator.generate(space, Random(123))
    assert len(result.good_set) == 2
    assert all(path.nodes[0] == 1 and path.nodes[-1] == 4 for path in result.good_set)
