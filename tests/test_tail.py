from random import Random

from large_choice_sampling.goodset.best_first import BestFirstGoodSetGenerator
from large_choice_sampling.goodset.stopping import MaxAlternativesStopping
from large_choice_sampling.route_choice.choice_space import RouteChoiceSpace
from large_choice_sampling.route_choice.network import RouteNetwork
from large_choice_sampling.route_choice.utility import RouteUtilityModel
from large_choice_sampling.tail.uniform_regions import UniformRegionTailProposal


def build_space() -> RouteChoiceSpace:
    network = RouteNetwork()
    network.add_link(1, 2, 1.0)
    network.add_link(1, 3, 1.5)
    network.add_link(2, 4, 1.0)
    network.add_link(3, 4, 1.0)
    network.add_link(2, 3, 0.5)
    return RouteChoiceSpace(network, origin=1, destination=4, utility_model=RouteUtilityModel(), max_steps=4)


def test_uniform_region_tail_can_sample():
    space = build_space()
    generator = BestFirstGoodSetGenerator(stopping=MaxAlternativesStopping(max_alternatives=1))
    result = generator.generate(space, Random(0))
    tail = UniformRegionTailProposal.from_goodset_result(space, result)
    alternative, probability = tail.sample(Random(0))
    assert alternative.nodes[0] == 1
    assert alternative.nodes[-1] == 4
    assert probability > 0.0
