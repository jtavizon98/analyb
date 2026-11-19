"""Conservation laws of histogram operations, checked over generated inputs.

Each test states an invariant that must hold for every valid histogram:
merging bins or summing histograms moves yield between bins but never
creates or loses counts, stored variances or flow content. Expected bins are
computed independently, group by group, from the input arrays.

Variances add as stored per-bin variances, which assumes uncorrelated bins.
For densities the conserved quantities are the raw yields counts × width and
variances (error × width)²; flows are raw yields with no stored variance.
Each output value may differ from its expected sum by float roundoff
proportional to the absolute terms of that same sum, never by a scale
borrowed from another bin or quantity. Inputs are normal floats (no
subnormals), so a merged density height cannot underflow to zero.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st

import analyb as ab

values = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False, allow_subnormal=False)
sigmas = st.floats(0.0, 1e3, allow_nan=False, allow_infinity=False, allow_subnormal=False)
widths = st.floats(1e-2, 10.0, allow_nan=False, allow_infinity=False)


def assert_sums(actual, expected, scale):
    """Each actual value equals its expected sum within roundoff of `scale`,
    the sum of absolute terms of that same value (broadcast per element)."""
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    tol = 1e-12 * np.asarray(scale, float)
    assert actual.shape == expected.shape
    bad = ~(np.abs(actual - expected) <= tol)
    assert not bad.any(), f"{actual[bad]} != {expected[bad]} (tolerance {np.broadcast_to(tol, bad.shape)[bad]})"


@st.composite
def axis(draw, n):
    start = draw(st.floats(-100.0, 100.0, allow_nan=False))
    return start + np.concatenate([[0.0], np.cumsum(draw(st.lists(widths, min_size=n, max_size=n)))])


@st.composite
def hist1d(draw, nbins=None, edges=None):
    n = draw(st.integers(1, 24)) if nbins is None else nbins
    return ab.Hist1D(
        draw(st.lists(values, min_size=n, max_size=n)),
        draw(axis(n)) if edges is None else edges,
        errors=draw(st.lists(sigmas, min_size=n, max_size=n)),
        underflow=draw(values),
        overflow=draw(values),
        is_weighted=draw(st.booleans()),
        label="x",
        unit="GeV",
    )


def raw_yields(hist):
    """Raw per-bin yields and variances: counts × width for densities."""
    w = hist.widths if hist.is_density else 1.0
    return hist.counts * w, (hist.errors * w) ** 2


def assert_merged(merged, hist, cuts):
    """`merged` must be `hist` merged at bin indices `cuts` (first 0, last nbins)."""
    groups = list(zip(cuts[:-1], cuts[1:]))
    yields, variances = raw_yields(hist)
    np.testing.assert_array_equal(merged.edges, hist.edges[cuts])
    assert (merged.is_density, merged.is_weighted) == (hist.is_density, hist.is_weighted)
    assert (merged.label, merged.unit) == (hist.label, hist.unit)
    m_yields, m_variances = raw_yields(merged)
    assert_sums(m_yields, [sum(yields[a:b]) for a, b in groups],
                [sum(abs(yields[a:b])) for a, b in groups])
    assert_sums(m_variances, [sum(variances[a:b]) for a, b in groups],
                [sum(variances[a:b]) for a, b in groups])
    assert_sums(np.sum(m_yields), np.sum(yields), np.sum(np.abs(yields)))
    assert_sums(np.sum(m_variances), np.sum(variances), np.sum(variances))
    assert (merged.underflow, merged.overflow) == (hist.underflow, hist.overflow)


@st.composite
def rebinnable(draw, density=None):
    factor = draw(st.integers(1, 6))
    hist = draw(hist1d(nbins=factor * draw(st.integers(1, 6))))
    if draw(st.booleans()) if density is None else density:
        hist = ab.divide_by_bin_width(hist)
    return hist, factor


@given(rebinnable())
def test_rebin_merges_each_group_and_conserves_totals(case):
    hist, factor = case
    assert_merged(ab.rebin(hist, factor), hist, list(range(0, hist.nbins + 1, factor)))


@given(rebinnable(density=False))
def test_rebin_commutes_with_division_by_bin_width(case):
    hist, factor = case
    direct = ab.rebin(ab.divide_by_bin_width(hist), factor)
    via_raw = ab.divide_by_bin_width(ab.rebin(hist, factor))
    w = via_raw.widths
    scale = np.abs(hist.counts).reshape(-1, factor).sum(axis=1)
    err2 = (hist.errors ** 2).reshape(-1, factor).sum(axis=1)
    assert_sums(direct.counts * w, via_raw.counts * w, scale)
    assert_sums((direct.errors * w) ** 2, (via_raw.errors * w) ** 2, err2)


@st.composite
def non_divisor(draw):
    n = draw(st.integers(1, 24))
    factor = draw(st.sampled_from([f for f in range(2, 31) if n % f]))
    return draw(hist1d(nbins=n)), factor


@given(non_divisor(), st.booleans())
def test_rebin_rejects_a_factor_that_would_drop_bins(case, density):
    hist, factor = case
    hist = ab.divide_by_bin_width(hist) if density else hist
    with pytest.raises(ValueError, match="exactly divide"):
        ab.rebin(hist, factor)


@given(hist1d(), st.booleans(), st.data())
def test_rebin_to_edges_merges_each_group_and_conserves_totals(hist, density, data):
    hist = ab.divide_by_bin_width(hist) if density else hist
    interior = list(range(1, hist.nbins))
    kept = data.draw(st.lists(st.sampled_from(interior), unique=True) if interior else st.just([]))
    cuts = [0, *sorted(kept), hist.nbins]
    assert_merged(ab.rebin_to_edges(hist, hist.edges[cuts]), hist, cuts)


@st.composite
def same_binning(draw):
    first = draw(hist1d())
    hists = [first] + [draw(hist1d(nbins=first.nbins, edges=first.edges))
                       for _ in range(draw(st.integers(0, 4)))]
    if draw(st.booleans()):
        hists = [ab.divide_by_bin_width(h) for h in hists]
    return hists


@given(same_binning(), st.randoms(use_true_random=False))
def test_sum_histograms_adds_everything_in_any_order(hists, rng):
    shuffled = list(hists)
    rng.shuffle(shuffled)
    counts = np.array([h.counts for h in hists])
    err2 = np.array([h.errors ** 2 for h in hists])
    flows = np.array([[h.underflow, h.overflow] for h in hists])
    added = hists[0]
    for h in hists[1:]:
        added = added + h
    for total in (ab.sum_histograms(hists), ab.sum_histograms(shuffled), added):
        assert total.is_density == hists[0].is_density
        np.testing.assert_array_equal(total.edges, hists[0].edges)
        assert_sums(total.counts, counts.sum(axis=0), np.abs(counts).sum(axis=0))
        assert_sums(total.errors ** 2, err2.sum(axis=0), err2.sum(axis=0))
        assert_sums([total.underflow, total.overflow], flows.sum(axis=0), np.abs(flows).sum(axis=0))


@given(hist1d(), st.floats(1e-6, 1.0))
def test_sum_histograms_rejects_materially_different_bins(hist, shift):
    edges = hist.edges.copy()
    edges[-1] += shift * hist.widths[-1]
    other = ab.Hist1D(hist.counts, edges, errors=hist.errors)
    with pytest.raises(ValueError, match="identical edges"):
        ab.sum_histograms([hist, other])


@st.composite
def hist2d(draw):
    fx, fy = draw(st.integers(1, 4)), draw(st.integers(1, 4))
    shape = (fx * draw(st.integers(1, 4)), fy * draw(st.integers(1, 4)))
    size = shape[0] * shape[1]
    hist = ab.Hist2D(
        np.reshape(draw(st.lists(values, min_size=size, max_size=size)), shape),
        draw(axis(shape[0])),
        draw(axis(shape[1])),
        errors=np.reshape(draw(st.lists(sigmas, min_size=size, max_size=size)), shape),
    )
    return hist, fx, fy


@given(hist2d())
def test_hist2d_rebin_merges_each_block_and_conserves_totals(case):
    hist, fx, fy = case
    merged = ab.rebin(hist, (fx, fy))
    np.testing.assert_array_equal(merged.x_edges, hist.x_edges[::fx])
    np.testing.assert_array_equal(merged.y_edges, hist.y_edges[::fy])
    err2 = hist.errors ** 2
    nx, ny = hist.shape[0] // fx, hist.shape[1] // fy
    def blocks(a):
        return [[sum(a[i * fx + p, j * fy + q] for p in range(fx) for q in range(fy))
                 for j in range(ny)] for i in range(nx)]
    assert_sums(merged.counts, blocks(hist.counts), blocks(np.abs(hist.counts)))
    assert_sums(merged.errors ** 2, blocks(err2), blocks(err2))
    assert_sums(merged.integral(), hist.integral(), np.sum(np.abs(hist.counts)))


@st.composite
def hist2d_non_divisor(draw):
    """A 2D histogram and a factor with a nonempty trailing group on one axis."""
    n = draw(st.integers(3, 12))
    bad = draw(st.sampled_from([f for f in range(2, n) if n % f]))
    m = draw(st.integers(1, 6))
    good = draw(st.sampled_from([f for f in range(1, m + 1) if m % f == 0]))
    bad_x = draw(st.booleans())
    shape, factor = ((n, m), (bad, good)) if bad_x else ((m, n), (good, bad))
    hist = ab.Hist2D(np.ones(shape), draw(axis(shape[0])), draw(axis(shape[1])))
    return hist, factor


@given(hist2d_non_divisor())
def test_hist2d_rebin_rejects_one_non_dividing_axis(case):
    hist, factor = case
    with pytest.raises(ValueError, match="exactly divide"):
        ab.rebin(hist, factor)


def test_histograms_reject_errors_of_the_wrong_shape():
    with pytest.raises(ValueError, match="same shape"):
        ab.Hist1D([1.0, 2.0, 3.0, 4.0], [0, 1, 2, 3, 4], errors=[1.0, 1.0])
    with pytest.raises(ValueError, match="same shape"):
        ab.Hist2D(np.ones((2, 2)), [0, 1, 2], [0, 1, 2], errors=np.ones(2))


def test_sums_of_empty_histograms_stay_empty():
    empty = ab.Hist1D([], [0.0], is_density=True)
    assert ab.sum_histograms([empty, empty]).nbins == 0
    empty2d = ab.Hist2D(np.zeros((0, 1)), [0.0], [0.0, 1.0])
    assert ab.sum_histograms2d([empty2d, empty2d]).shape == (0, 1)


@pytest.mark.parametrize("factor", [0, -2, 2.0, True, np.bool_(True)])
def test_rebin_rejects_factors_that_are_not_positive_integers(factor):
    hist = ab.Hist1D([1.0, 2.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="positive integer"):
        ab.rebin(hist, factor)
    with pytest.raises(ValueError, match="positive integer"):
        ab.rebin(ab.Hist2D(np.ones((2, 2)), [0, 1, 2], [0, 1, 2]), (1, factor))


@given(hist2d(), st.floats(1e-6, 1.0), st.booleans())
def test_sum_histograms2d_rejects_materially_different_bins(case, shift, on_x):
    hist, _, _ = case
    x, y = hist.x_edges.copy(), hist.y_edges.copy()
    edges = x if on_x else y
    edges[-1] += shift * (edges[-1] - edges[-2])
    other = ab.Hist2D(hist.counts, x, y, errors=hist.errors)
    with pytest.raises(ValueError, match="identical"):
        ab.sum_histograms2d([hist, other])
