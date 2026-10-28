"""Cutflow containers, drawing and uncertainty semantics with synthetic inputs."""

import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
import numpy as np
import pytest

import analyb as ab

def test_poisson_errors_are_explicit_and_require_raw_counts():
    assert ab.CutflowData(["a", "b"], [16.0, 9.0]).errors is None
    np.testing.assert_array_equal(
        ab.CutflowData(["a", "b"], [16.0, 9.0], errors="poisson").errors, [4.0, 3.0],
    )
    with pytest.raises(ValueError, match="nonnegative raw counts"):
        ab.CutflowData(["a", "b"], [16.0, -1.0], errors="poisson")
    with pytest.raises(ValueError, match="'poisson'"):
        ab.CutflowData(["a", "b"], [16.0, 9.0], errors="sqrt")


@pytest.mark.parametrize("errors, drawn", [(None, False), ("poisson", True)])
def test_cutflow_chart_draws_only_known_errors(tmp_path, errors, drawn):
    cutflow = ab.CutflowData(["a", "b"], [16.0, 9.0], errors=errors)
    fig, ax = ab.cutflow_chart(cutflow, verbose=False, save_path=str(tmp_path))

    assert any(isinstance(c, ErrorbarContainer) for c in ax.containers) is drawn
    plt.close(fig)


# --------------------------------------------------------------------------
# nested_efficiency: first-order variance of nested weighted ratios
# --------------------------------------------------------------------------

def test_nested_efficiency_unit_weights_reduce_to_binomial_variance():
    total = 50.0
    passed = np.arange(51.0)
    eps, var, valid = ab.nested_efficiency(passed, np.full(51, total), passed, np.full(51, total))

    assert valid.all()
    np.testing.assert_allclose(eps, passed / total, rtol=0, atol=0)
    np.testing.assert_allclose(var, eps * (1 - eps) / total, rtol=1e-12, atol=1e-18)
    assert var[0] == 0.0 and var[-1] == 0.0  # all-fail and all-pass stages


def test_nested_efficiency_scalar_inputs_return_scalars():
    eps, var, valid = ab.nested_efficiency(3.0, 4.0, 3.0, 4.0)
    assert np.ndim(eps) == np.ndim(var) == np.ndim(valid) == 0
    assert eps == 0.75 and valid
    assert var == pytest.approx(0.75 * 0.25 / 4, rel=1e-14)


def test_nested_efficiency_is_invariant_under_common_weight_rescaling():
    rng = np.random.default_rng(11)
    weights = rng.normal(1.0, 0.7, size=(6, 400))
    passes = rng.random((6, 400)) < 0.4
    args = ((weights * passes).sum(1), weights.sum(1),
            (weights ** 2 * passes).sum(1), (weights ** 2).sum(1))
    eps, var, _ = ab.nested_efficiency(*args)
    for scale in (0.37, 1e150, 1e-150):
        scaled = ab.nested_efficiency(args[0] * scale, args[1] * scale,
                                      args[2] * scale ** 2, args[3] * scale ** 2)
        np.testing.assert_allclose(scaled[0], eps, rtol=1e-12)
        np.testing.assert_allclose(scaled[1], var, rtol=1e-12)


@pytest.mark.parametrize("cross", [None, 3e306])
def test_nested_efficiency_avoids_squaring_large_denominators(cross):
    # Every input and the result are representable although B^2 overflows.
    _, var, _ = ab.nested_efficiency(3e156, 4e156, 3e306, 4e306, cross_sumw=cross)
    assert var == pytest.approx(4.6875e-8, rel=1e-12)


def test_nested_efficiency_keeps_signed_ratios_outside_unit_interval():
    # A = -3, B = -4 (negative totals); A = 6, B = 4 (ratio above one).
    eps, var, valid = ab.nested_efficiency([-3.0, 6.0], [-4.0, 4.0], [5.0, 10.0], [9.0, 12.0])
    assert valid.all()
    np.testing.assert_array_equal(eps, [0.75, 1.5])
    # ((1 - eps)^2 Q_A + eps^2 (Q_B - Q_A)) / B^2 by hand.
    np.testing.assert_allclose(var, [(0.0625 * 5 + 0.5625 * 4) / 16, (0.25 * 10 + 2.25 * 2) / 16],
                               rtol=1e-14)


def test_nested_efficiency_marks_zero_and_nonfinite_denominators_invalid():
    passed = np.array([1.0, 1.0, 1.0, np.nan, 1.0])
    total = np.array([0.0, np.nan, np.inf, 2.0, 2.0])
    sumw2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    eps, var, valid = ab.nested_efficiency(passed, total, sumw2, sumw2 * 2)
    np.testing.assert_array_equal(valid, [False, False, False, False, True])
    assert np.isnan(eps[:4]).all() and np.isnan(var[:4]).all()
    assert eps[4] == 0.5 and np.isfinite(var[4])


def test_nested_efficiency_unknown_variances_keep_central_ratio():
    eps, var, valid = ab.nested_efficiency([1.0, 2.0], [4.0, 4.0], None, [4.0, 4.0])
    np.testing.assert_array_equal(eps, [0.25, 0.5])
    assert valid.all() and np.isnan(var).all()

    eps, var, valid = ab.nested_efficiency([1.0, 2.0], [4.0, 4.0], [np.nan, 2.0], [4.0, 4.0])
    np.testing.assert_array_equal(eps, [0.25, 0.5])
    assert np.isnan(var[0]) and var[1] == pytest.approx(0.25 / 4)

    _, var, _ = ab.nested_efficiency([1.0, 2.0], [4.0, 4.0], [1.0, 2.0], [4.0, 4.0],
                                     cross_sumw=[np.nan, 2.0])
    assert np.isnan(var[0]) and var[1] == pytest.approx(0.25 / 4)


def _poisson_bootstrap_variance(pass_weights, total_weights, replicas, rng):
    """Spread of A/B when every event multiplicity is an independent Poisson(1)."""
    counts = rng.poisson(1.0, size=(replicas, total_weights.size))
    return np.var((counts @ pass_weights) / (counts @ total_weights), ddof=1)


@pytest.mark.parametrize("stage_dependent", [False, True])
def test_nested_efficiency_matches_event_level_covariance(stage_dependent):
    # Signed denominator weights; passing events carry w_A = s w_B. The oracle
    # resamples events (compound Poisson), where Var(A) = Q_A, Var(B) = Q_B and
    # Cov(A, B) = sum_passed w_A w_B hold exactly.
    rng = np.random.default_rng(20261002)
    w_b = rng.normal(1.0, 0.8, size=1000)
    passes = rng.random(w_b.size) < 0.35
    scale = rng.uniform(0.4, 0.8, size=w_b.size) if stage_dependent else np.ones(w_b.size)
    w_a = np.where(passes, scale * w_b, 0.0)
    args = (w_a.sum(), w_b.sum(), np.sum(w_a ** 2), np.sum(w_b ** 2))
    cross = np.sum(w_a * w_b)

    eps, var, valid = ab.nested_efficiency(*args, cross_sumw=cross)
    assert valid and eps == pytest.approx(args[0] / args[1])
    # Algebraic oracle: sum of squared per-event influences (a_i - eps b_i) / B.
    influence = (w_a - eps * w_b) / args[1]
    assert var == pytest.approx(np.sum(influence ** 2), rel=1e-10)
    sampled = _poisson_bootstrap_variance(w_a, w_b, 10000, rng)
    assert var == pytest.approx(sampled, rel=0.06)
    default = ab.nested_efficiency(*args)[1]
    if stage_dependent:  # C = Q_A is wrong here, by far more than MC precision
        assert default / sampled > 1.25
    else:
        assert default == pytest.approx(var, rel=1e-12)


def test_nested_efficiency_stage_multiplier_requires_explicit_cross_term():
    # Unit denominator weights, acceptance 0.3, passing weights multiplied by 0.9.
    w_b = np.ones(1000)
    w_a = np.where(np.arange(1000) < 300, 0.9, 0.0)
    args = (w_a.sum(), w_b.sum(), np.sum(w_a ** 2), np.sum(w_b ** 2))
    _, explicit, _ = ab.nested_efficiency(*args, cross_sumw=np.sum(w_a * w_b))
    _, default, _ = ab.nested_efficiency(*args)  # wrong C = Q_A for changed weights

    assert default / explicit - 1 == pytest.approx(0.0857142857, abs=1e-9)
    assert np.sqrt(default / explicit) - 1 == pytest.approx(0.04198, abs=1e-5)


def test_nested_efficiency_rejects_inconsistent_second_moments():
    with pytest.raises(ValueError, match="exceeds total_sumw2"):
        ab.nested_efficiency(3.0, 4.0, 5.0, 4.0)
    with pytest.raises(ValueError, match="exceeds sqrt"):
        ab.nested_efficiency(3.0, 4.0, 1.0, 4.0, cross_sumw=2.5)
    with pytest.raises(ValueError, match="exceeds sqrt"):
        ab.nested_efficiency(3.0, 4.0, 1.0, 4.0, cross_sumw=-2.5)
    for q_a, q_b in ((-1.0, 4.0), (1.0, np.inf)):
        with pytest.raises(ValueError, match="nonnegative"):
            ab.nested_efficiency(3.0, 4.0, q_a, q_b)
    with pytest.raises(ValueError, match="cross_sumw must be finite"):
        ab.nested_efficiency(3.0, 4.0, 1.0, 4.0, cross_sumw=np.inf)
    with pytest.raises(ValueError, match="common shape"):
        ab.nested_efficiency([3.0, 1.0], [4.0, 2.0], [1.0], [4.0, 2.0])

    # Each supplied moment is validated even when another one is unknown.
    for q_a in (None, np.nan):
        with pytest.raises(ValueError, match="total_sumw2 must be finite and nonnegative"):
            ab.nested_efficiency(1.0, 2.0, q_a, -1.0)
    with pytest.raises(ValueError, match="passed_sumw2"):
        ab.nested_efficiency(1.0, 2.0, -1.0, 4.0, cross_sumw=np.nan)
    with pytest.raises(ValueError, match="common shape"):
        ab.nested_efficiency([1.0, 1.0], [2.0, 2.0], None, [4.0, 4.0], cross_sumw=[1.0])


def test_nested_efficiency_covariance_bound_is_scale_safe():
    # sqrt(Q_A * Q_B) would overflow to inf and accept an impossible C ...
    with pytest.raises(ValueError, match="exceeds sqrt"):
        ab.nested_efficiency(1.0, 1.0, 1e200, 1e200, cross_sumw=2e200)
    # ... or underflow to 0 and reject a valid one.
    eps, var, _ = ab.nested_efficiency(1e-100, 2e-100, 1e-200, 1e-200, cross_sumw=1e-200)
    assert eps == 0.5 and var == pytest.approx(0.0625, rel=1e-12)
    with pytest.raises(ValueError, match="exceeds total_sumw2"):
        ab.nested_efficiency(1.0, 1.0, 2e200, 1e200)
    _, var, _ = ab.nested_efficiency(1e-100, 2e-100, 1e-200, 1e-200)
    assert var == pytest.approx(0.0625, rel=1e-12)

    # Accumulation roundoff at an all-pass stage is clipped, not rejected.
    eps, var, _ = ab.nested_efficiency(4.0, 4.0, 4.0 * (1 + 1e-14), 4.0)
    assert eps == 1.0 and var == pytest.approx(0.0, abs=1e-15)
    _, var, _ = ab.nested_efficiency(4.0, 4.0, 4.0, 4.0, cross_sumw=4.0 * (1 + 1e-14))
    assert var == pytest.approx(0.0, abs=1e-15)
