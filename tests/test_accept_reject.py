from random import Random

from large_choice_sampling.accept_reject.kernel import AcceptRejectSampler
from large_choice_sampling.goodset.best_first import BestFirstGoodSetGenerator
from large_choice_sampling.goodset.stopping import MaxAlternativesStopping
from large_choice_sampling.mixture.alpha import clip_alpha, compute_balanced_alpha
from large_choice_sampling.mixture.bounds import compute_global_bound
from large_choice_sampling.mixture.proposal import MixtureProposal
from large_choice_sampling.route_choice.choice_space import RouteChoiceSpace
from large_choice_sampling.route_choice.network import RouteNetwork
from large_choice_sampling.route_choice.utility import RouteUtilityModel
from large_choice_sampling.tail.uniform_regions import UniformRegionTailProposal


def build_space() -> RouteChoiceSpace:
    network = RouteNetwork()
    network.add_link(1, 2, 1.0)
    network.add_link(1, 3, 2.0)
    network.add_link(2, 4, 1.0)
    network.add_link(3, 4, 1.0)
    network.add_link(2, 3, 0.5)
    return RouteChoiceSpace(network, origin=1, destination=4, utility_model=RouteUtilityModel(), max_steps=4)


def test_accept_reject_sampler_runs():
    rng = Random(123)
    space = build_space()
    result = BestFirstGoodSetGenerator(stopping=MaxAlternativesStopping(max_alternatives=1)).generate(space, rng)
    tail = UniformRegionTailProposal.from_goodset_result(space, result)
    elite_mass = sum(pow(2.718281828459045, u) for u in result.utilities.values())
    alpha = clip_alpha(compute_balanced_alpha(elite_mass, tail.bound_term()))
    proposal = MixtureProposal(result.good_set, result.utilities, tail, alpha)
    bound = compute_global_bound(proposal.elite_mass, tail, alpha)
    sampler = AcceptRejectSampler(choice_space=space, proposal=proposal, bound_info=bound)
    draw = sampler.sample(rng, max_attempts=1000)
    assert draw.alternative.nodes[0] == 1
    assert draw.alternative.nodes[-1] == 4
