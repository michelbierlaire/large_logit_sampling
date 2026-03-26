from random import Random

from large_choice_sampling.mixture.proposal import MixtureProposal
from large_choice_sampling.route_choice.alternative import RouteAlternative
from large_choice_sampling.tail.base import TailProposal


class DummyTail(TailProposal):
    def __init__(self, alt: RouteAlternative):
        self.alt = alt

    def sample(self, rng: Random):
        return self.alt, 1.0

    def prob(self, alternative):
        return 1.0 if alternative == self.alt else 0.0

    def bound_term(self):
        return 2.0


def test_mixture_prob():
    elite_alt = RouteAlternative(nodes=(1, 2), total_travel_time=1.0)
    tail_alt = RouteAlternative(nodes=(1, 3), total_travel_time=2.0)
    proposal = MixtureProposal(
        elite_alternatives=[elite_alt],
        elite_utilities={elite_alt: -1.0},
        tail_proposal=DummyTail(tail_alt),
        alpha=0.5,
    )
    assert proposal.prob(elite_alt) > 0.0
    assert proposal.prob(tail_alt) > 0.0
