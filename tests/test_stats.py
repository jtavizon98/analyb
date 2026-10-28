"""Clopper-Pearson binomial intervals for genuine unweighted counts."""

import numpy as np
import pytest
from scipy import stats

import analyb as ab

CL = 0.6827


def _coverage(total, probability):
    """Exact coverage for fixed ``total``: sampling law k ~ Binomial(total, p)."""
    k = np.arange(total + 1)
    lower, upper = ab.clopper_pearson_interval(k, np.full(k.size, total))
    covered = (lower <= probability) & (probability <= upper)
    return float(np.sum(stats.binom.pmf(k, total, probability) * covered))


def test_clopper_pearson_exact_coverage_reaches_nominal_level():
    # Value recomputed independently in the phase D decision record.
    assert _coverage(30, 0.9) == pytest.approx(0.884419, abs=5e-7)
    for total in (1, 5, 30, 100):
        for probability in np.linspace(0.001, 0.999, 199):
            assert _coverage(total, probability) >= CL


def test_clopper_pearson_boundary_counts_reach_the_unit_interval_ends():
    total = 30
    alpha = 1 - CL
    lower, upper = ab.clopper_pearson_interval([0, total], [total, total])
    # Closed forms: upper(k=0) = 1 - (alpha/2)^(1/N), lower(k=N) = (alpha/2)^(1/N).
    np.testing.assert_allclose([lower[0], upper[1]], [0.0, 1.0], rtol=0, atol=0)
    np.testing.assert_allclose(upper[0], 1 - (alpha / 2) ** (1 / total), rtol=1e-12)
    np.testing.assert_allclose(lower[1], (alpha / 2) ** (1 / total), rtol=1e-12)

    lower, upper = ab.clopper_pearson_interval(3, 10)
    assert np.ndim(lower) == 0 and lower < 0.3 < upper

    lower, upper = ab.clopper_pearson_interval([0, 1], [0, 2])
    assert np.isnan(lower[0]) and np.isnan(upper[0])
    assert np.isfinite(lower[1]) and np.isfinite(upper[1])


@pytest.mark.parametrize("passed, total", [
    (2.5, 10),        # weighted / non-integer
    (-1, 10),         # signed
    (3, -10),
    (11, 10),         # passed exceeds total
    (np.nan, 10),
    (3, np.inf),
])
def test_clopper_pearson_rejects_weighted_signed_or_invalid_counts(passed, total):
    with pytest.raises(ValueError):
        ab.clopper_pearson_interval(passed, total)


def test_clopper_pearson_validates_counts_before_float_conversion():
    # 2**53 + 1 and 2**53 are equal after float conversion.
    with pytest.raises(ValueError):
        ab.clopper_pearson_interval(2 ** 53 + 1, 2 ** 53)
    with pytest.raises(ValueError, match="cannot exceed"):
        ab.clopper_pearson_interval(np.int64(2 ** 53), np.int64(2 ** 53 - 1))
    with pytest.raises(ValueError, match="numeric"):
        ab.clopper_pearson_interval(2 ** 70, 2 ** 71)


def test_clopper_pearson_keeps_upper_tail_for_levels_near_one():
    lower, upper = ab.clopper_pearson_interval(5, 10, cl=np.nextafter(1.0, 0.0))
    assert upper == pytest.approx(0.9998143636044187, rel=1e-12)
    assert lower == pytest.approx(1 - upper, rel=1e-9)  # symmetric at k = N/2


def test_clopper_pearson_rejects_mismatched_shapes_and_levels():
    with pytest.raises(ValueError, match="common shape"):
        ab.clopper_pearson_interval([1, 2], [3])
    with pytest.raises(ValueError, match="cl"):
        ab.clopper_pearson_interval(1, 3, cl=1.0)
