"""The one grid rule and the one bin-by-bin quotient, through the public interface.

Expected values are worked by hand from sigma(n/d) = sqrt((sn/|d|)^2 + (|n| sd/d^2)^2),
not recomputed with the formula under test.
"""
import numpy as np
import pytest

import analyb as ab


def test_divide_worked_example_with_combined_errors():
    q = ab.divide([2.0, 0.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0],
                  [4.0, 2.0, 0.0, -2.0], [2.0, 1.0, 1.0, 1.0])
    # bin 0: 2/4 = 0.5, sqrt(0.25^2 + (2*2/16)^2) = sqrt(0.125)
    # bin 1: 0/2 = 0, the zero numerator keeps 1/2
    # bin 2: zero denominator, undefined
    # bin 3: 4/-2 = -2, sqrt(0.5^2 + (4*1/4)^2) = sqrt(1.25)
    np.testing.assert_array_equal(q.valid, [True, True, False, True])
    np.testing.assert_allclose(q.value, [0.5, 0.0, np.nan, -2.0])
    np.testing.assert_allclose(q.error, [0.125 ** 0.5, 0.5, np.nan, 1.25 ** 0.5])


def test_divide_numerator_only_and_central_only():
    q = ab.divide([2.0, 4.0], [1.0, 1.0], [4.0, -2.0])
    np.testing.assert_allclose(q.error, [0.25, 0.5])  # sigma_n / |d|, sign-free
    q = ab.divide([2.0, 4.0], None, [4.0, -2.0])
    assert q.error is None
    np.testing.assert_allclose(q.value, [0.5, -2.0])


def test_divide_never_fills_undefined_bins():
    q = ab.divide([1.0, np.nan, 2.0, np.inf, 3.0], [0.1, 0.1, 0.1, 0.1, 0.1],
                  [1.0, 1.0, np.nan, 1.0, 0.0], [0.1, 0.1, 0.1, 0.1, 0.1])
    np.testing.assert_array_equal(q.valid, [True, False, False, False, False])
    assert np.isnan(q.value[1:]).all() and np.isnan(q.error[1:]).all()


def test_errors_never_decide_whether_a_value_exists():
    # sqrt of a negative unweighted count is NaN; the ratio 4/2 is still defined
    q = ab.divide([4.0, 3.0], [np.nan, 1.0], [2.0, 2.0], [0.5, np.inf])
    np.testing.assert_array_equal(q.valid, [True, True])
    np.testing.assert_allclose(q.value, [2.0, 1.5])
    assert np.isnan(q.error[0]) and np.isinf(q.error[1])


def test_unweighted_negative_bin_keeps_its_ratio_value():
    """Default errors sqrt(counts) are NaN for a negative bin; main returned -0.5 here."""
    with np.errstate(invalid="ignore"):
        num = ab.Hist1D([5.0, -1.0, 4.0], [0.0, 1.0, 2.0, 3.0])
    den = ab.Hist1D([4.0, 2.0, 2.0], [0.0, 1.0, 2.0, 3.0])
    for include in (True, False):
        value, _ = ab.ratio(num, den, include_denominator_error=include)
        np.testing.assert_allclose(value, [1.25, -0.5, 2.0])


def test_divide_scalars_and_shape_mismatch():
    q = ab.divide(3.0, 0.3, 6.0, 0.0)
    assert float(q.value) == 0.5 and float(q.error) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        ab.divide([1.0, 2.0], [0.1, 0.1], [1.0])


def _hist(edges, density=False):
    counts = np.ones(len(edges) - 1)
    hist = ab.Hist1D(counts, edges, errors=0.1 * counts)
    return ab.divide_by_bin_width(hist) if density else hist


def test_edge_rules_differ_only_by_roundoff():
    edges = np.array([0.0, 0.1, 0.3, 1.0])
    ulp = edges.copy()
    ulp[1] = np.nextafter(0.1, 1.0)
    shifted = edges.copy()
    shifted[1] += 1e-6 * 0.1  # a material shift: one millionth of the bin
    ab.require_comparable(_hist(edges), _hist(ulp), edges="roundoff")
    with pytest.raises(ValueError):
        ab.require_comparable(_hist(edges), _hist(ulp), edges="exact")
    for rule in ab.histogram.EDGE_RULES:
        with pytest.raises(ValueError):
            ab.require_comparable(_hist(edges), _hist(shifted), edges=rule)


def test_require_comparable_state_type_and_finiteness():
    edges = [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="density"):
        ab.require_comparable(_hist(edges), _hist(edges, density=True), edges="exact")
    h2 = ab.Hist2D(np.ones((2, 2)), [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], errors=np.ones((2, 2)))
    with pytest.raises(TypeError):
        ab.require_comparable(_hist(edges), h2, edges="exact")
    bad = ab.Hist1D([1.0, np.nan], edges, errors=[0.1, 0.1])
    ab.require_comparable(_hist(edges), bad, edges="exact")
    with pytest.raises(ValueError, match="finite"):
        ab.require_comparable(_hist(edges), bad, edges="exact", finite=True)
    with pytest.raises(ValueError, match="edges must be one of"):
        ab.require_comparable(_hist(edges), _hist(edges), edges="close")


def test_ratio_and_pairwise_comparison_agree_bin_by_bin():
    """Both public ratios go through one quotient, including negative and zero bins."""
    edges = [0.0, 1.0, 2.0, 3.0, 4.0]
    num = ab.Hist1D([3.0, 0.0, 5.0, -1.0], edges, errors=[1.0, 1.0, 2.0, 0.5], is_weighted=True)
    ref = ab.Hist1D([2.0, 4.0, 0.0, -2.0], edges, errors=[1.0, 2.0, 1.0, 0.5], is_weighted=True)
    value, error = ab.ratio(num, ref)
    comparison = ab.compare_histograms(num, ref, shape=False)
    np.testing.assert_array_equal(value, comparison.ratio)
    np.testing.assert_array_equal(error, comparison.ratio_error)
    np.testing.assert_array_equal(comparison.ratio_valid, [True, True, False, True])


def test_counts_and_errors_cannot_be_given_or_assigned_off_their_grid():
    """The histogram owns its shape: no comparison can meet a mismatched one."""
    with pytest.raises(ValueError, match="shape"):
        ab.Hist2D(np.ones((3, 2)), [0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="len\(counts\) \+ 1"):
        ab.Hist1D(np.ones(2), [0.0, 1.0, 2.0, 3.0])
    h = _hist([0.0, 1.0, 2.0, 3.0])
    for attribute in ("counts", "errors"):
        with pytest.raises(ValueError, match="shape"):
            setattr(h, attribute, np.ones(2))
    with pytest.raises(AttributeError):
        h.edges = np.array([0.0, 1.0])
    g = ab.Hist2D(np.ones((2, 3)), [0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="shape"):
        g.errors = np.ones((3, 2))
