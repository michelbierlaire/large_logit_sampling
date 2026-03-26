from large_choice_sampling.mixture.alpha import compute_balanced_alpha


def test_compute_balanced_alpha():
    alpha = compute_balanced_alpha(2.0, 3.0)
    assert abs(alpha - 0.4) < 1e-12
