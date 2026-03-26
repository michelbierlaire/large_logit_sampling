"""Demonstration script for route-choice sampling with the accept-reject framework.

This script builds a small directed network, constructs the corresponding
implicit route-choice space, generates a finite elite set using best-first
search, defines a tail proposal on the unexplored regions, and finally draws an
independent sample using accept-reject sampling.

Michel Bierlaire
Wed Mar 25 2026, 17:52:11
"""

from collections import Counter
from math import exp
from random import Random
from typing import Any

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


def enumerate_all_paths(space: RouteChoiceSpace) -> list[Any]:
    """Enumerate all complete paths in the small demonstration space.

    This helper is intended only for the tiny demo network, where exhaustive
    enumeration is feasible. It is used to compute the exact target
    distribution for comparison with the empirical frequencies obtained from
    accept-reject sampling.

    :param space: Route-choice space to enumerate.
    :return: List of all complete alternatives reachable within the search
        constraints of the demo space.
    """
    all_paths: list[Any] = []
    stack: list[Any] = [space.root_state()]

    while stack:
        state: Any = stack.pop()
        if space.is_complete(state):
            all_paths.append(space.to_alternative(state))
        else:
            stack.extend(space.expand(state))

    return all_paths


def theoretical_distribution(space: RouteChoiceSpace) -> dict[tuple[int, ...], float]:
    """Compute the exact target distribution on the demo network.

    The target is proportional to ``exp(utility)`` over the fully enumerated
    set of feasible paths. This computation is only possible because the demo
    network is intentionally very small.

    :param space: Route-choice space for which the exact target distribution is
        computed.
    :return: Mapping from a path, represented by its node tuple, to its exact
        target probability.
    """
    paths: list[Any] = enumerate_all_paths(space)
    weights: dict[tuple[int, ...], float] = {
        tuple(path.nodes): exp(space.utility(path)) for path in paths
    }
    total_weight: float = sum(weights.values())
    return {nodes: weight / total_weight for nodes, weight in weights.items()}


def build_demo_space() -> RouteChoiceSpace:
    """Create a small route-choice space used in the demonstration.

    The network is intentionally tiny so that the full sequence of operations
    remains easy to inspect: good-set generation, tail construction, mixture
    proposal definition, and accept-reject sampling.

    :return: A ``RouteChoiceSpace`` defined on a small directed network with a
        fixed origin, destination, utility model, and maximum number of steps.
    """
    network = RouteNetwork()
    network.add_link(1, 2, 1.0)
    network.add_link(1, 3, 2.0)
    network.add_link(2, 4, 1.0)
    network.add_link(3, 4, 1.0)
    network.add_link(2, 3, 0.5)
    network.add_link(3, 2, 0.5)
    network.add_link(2, 5, 1.0)
    network.add_link(5, 4, 1.0)
    utility_model = RouteUtilityModel(beta_time=1.0)
    return RouteChoiceSpace(
        network=network,
        origin=1,
        destination=4,
        utility_model=utility_model,
        max_steps=5,
    )


def main() -> None:
    """Run the end-to-end demonstration of the sampling framework.

    The function constructs the demo route-choice space, generates the elite
    set, builds the tail proposal, computes the mixture parameters and global
    bound, and finally produces one accepted draw from the target distribution.
    It then prints the main intermediate quantities and the accepted sample.

    :return: None.
    """
    rng: Random = Random(12345)
    space: RouteChoiceSpace = build_demo_space()

    generator: BestFirstGoodSetGenerator = BestFirstGoodSetGenerator(
        stopping=MaxAlternativesStopping(max_alternatives=3)
    )
    goodset: Any = generator.generate(space, rng)

    tail: UniformRegionTailProposal = UniformRegionTailProposal.from_goodset_result(
        space, goodset
    )
    elite_mass: float = sum(exp(u) for u in goodset.utilities.values())
    alpha: float = clip_alpha(compute_balanced_alpha(elite_mass, tail.bound_term()))
    proposal: MixtureProposal = MixtureProposal(
        elite_alternatives=goodset.good_set,
        elite_utilities=goodset.utilities,
        tail_proposal=tail,
        alpha=alpha,
    )
    bound: Any = compute_global_bound(proposal.elite_mass, tail, alpha)
    sampler: AcceptRejectSampler = AcceptRejectSampler(space, proposal, bound)

    n_samples: int = 1000
    draws: list[Any] = [sampler.sample(rng) for _ in range(n_samples)]
    total_proposal_draws: int = sum(draw.proposal_draws for draw in draws)
    total_rejected_draws: int = total_proposal_draws - n_samples

    empirical_counts: Counter[tuple[int, ...]] = Counter(
        tuple(draw.alternative.nodes) for draw in draws
    )
    empirical_distribution: dict[tuple[int, ...], float] = {
        path: count / n_samples for path, count in empirical_counts.items()
    }
    target_distribution: dict[tuple[int, ...], float] = theoretical_distribution(space)

    print("Good set:")
    for alternative in goodset.good_set:
        alternative: Any = alternative
        print(
            f"  {alternative.nodes}  time={alternative.total_travel_time:.2f}  utility={goodset.utilities[alternative]:.2f}"
        )
    print()
    print(f"alpha = {alpha:.4f}")
    print(f"elite bound = {bound.elite_bound:.4f}")
    print(f"tail bound  = {bound.tail_bound:.4f}")
    print(f"global bound= {bound.global_bound:.4f}")
    print()
    print(f"Accepted samples     = {n_samples}")
    print(f"Proposal draws total = {total_proposal_draws}")
    print(f"Rejected draws total = {total_rejected_draws}")
    print(f"Average draws/sample = {total_proposal_draws / n_samples:.2f}")
    print()
    print("Empirical vs theoretical distribution:")
    for path in sorted(target_distribution):
        empirical_prob: float = empirical_distribution.get(path, 0.0)
        theoretical_prob: float = target_distribution[path]
        print(
            f"  path={path}  empirical={empirical_prob:.4f}  theoretical={theoretical_prob:.4f}"
        )


if __name__ == "__main__":
    main()
