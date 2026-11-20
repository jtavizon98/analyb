"""Test histogram operations with dummy data — no ROOT data needed."""

import numpy as np
import analyb as ab


def test_hist1d_fill():
    """Hist1D fills correctly with dummy data."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    h = ab.hist1d(data, bins=20, range=(-5, 5))
    assert h is not None
    assert h.integral() == 1000


def test_hist1d_weights():
    """Hist1D handles event weights."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    weights = np.ones(1000) * 2.0
    h = ab.hist1d(data, bins=20, range=(-5, 5), weights=weights)
    assert abs(h.integral() - 2000.0) < 1e-6


def test_hist1d_all_nonfinite_raises_without_explicit_range():
    """Automatic binning rejects all-nonfinite inputs with a clear error."""
    data = np.array([np.nan, np.inf, -np.inf])
    try:
        ab.hist1d(data, bins="scott")
    except ValueError as exc:
        assert "empty finite data" in str(exc)
    else:
        raise AssertionError("hist1d should reject all-nonfinite automatic ranges")


def test_hist1d_empty_with_explicit_range():
    """Explicit ranges produce empty histograms instead of crashing."""
    h = ab.hist1d([], bins="scott", range=(0, 1))
    assert h.integral() == 0.0
    assert np.allclose(h.edges, [0.0, 1.0])


def test_variable_width_histogram_metadata_and_raw_plot_label():
    edges = np.array([0.0, 1.0, 3.0, 6.0])
    hist = ab.hist1d([0.2, 1.2, 2.2, 4.2], bins=edges, unit=r"\GeV")

    assert np.array_equal(hist.widths, [1.0, 2.0, 3.0])
    assert not hist.is_uniform
    assert hist.bin_width is None

    fig, ax = ab.hist1d_plot(hist, variable="met")
    assert ax.get_ylabel() == "Counts"
    fig.clear()


def test_uniform_histogram_bin_width_and_plot_label_are_unchanged():
    hist = ab.hist1d([0.2, 1.2, 2.2, 3.2], bins=[0.0, 2.0, 4.0], unit=r"\GeV")
    assert hist.is_uniform
    assert hist.bin_width == 2.0

    fig, ax = ab.hist1d_plot(hist, variable="met")
    assert ax.get_ylabel() == r"Counts / $2.0\;\mathrm{GeV}$"
    fig.clear()


def test_divide_by_bin_width_is_nonmutating_and_differential():
    hist = ab.Hist1D(
        counts=[4.0, 12.0, 24.0], edges=[0.0, 1.0, 3.0, 6.0],
        errors=[2.0, 6.0, 12.0], unit=r"\GeV", overflow=7.0, underflow=5.0,
        overflow_error=3.0, underflow_error=2.0, is_weighted=True,
    )
    density = ab.divide_by_bin_width(hist)

    np.testing.assert_allclose(density.counts, [4.0, 6.0, 8.0])
    np.testing.assert_allclose(density.errors, [2.0, 3.0, 4.0])
    assert density.is_density
    assert density.bin_width is None
    assert density.overflow == 7.0 and density.underflow == 5.0
    assert (density.overflow_error, density.underflow_error) == (3.0, 2.0)
    assert density.integral() == 40.0
    assert density.integral_error() == np.sqrt(184.0)
    np.testing.assert_allclose(hist.counts, [4.0, 12.0, 24.0])
    np.testing.assert_allclose(hist.errors, [2.0, 6.0, 12.0])

    fig, ax = ab.hist1d_plot(density, variable="met")
    assert ax.get_ylabel() == r"$\sum w$ / $\mathrm{GeV}$"   # weighted: sum of weights, not counts
    fig.clear()
    raw = ab.Hist1D([2.0, 4.0], [0.0, 1.0, 2.0], errors=[1.0, 2.0])
    fig, ax = ab.hist1d_plot(raw)
    assert ax.get_ylabel() == "Counts / 1.0"
    fig.clear()


def test_density_operations_are_safe():
    raw = ab.Hist1D([2.0, 4.0], [0.0, 1.0, 3.0], errors=[1.0, 2.0])
    density = ab.divide_by_bin_width(raw)
    with np.testing.assert_raises_regex(ValueError, "raw-count and per-bin-width"):
        ab.sum_histograms([raw, density])
    with np.testing.assert_raises_regex(ValueError, "raw-count and per-bin-width"):
        ab.ratio(raw, density)
    with np.testing.assert_raises_regex(ValueError, "raw-count and per-bin-width density"):
        ab.hist1d_stacked({"raw": raw, "density": density})

    rebinned = ab.rebin(density, factor=2)
    assert rebinned.is_density
    np.testing.assert_allclose(rebinned.counts, [2.0])
    np.testing.assert_allclose(rebinned.errors, [np.sqrt(5.0) / 3.0])
    assert rebinned.integral() == density.integral()
    assert rebinned.integral_error() == density.integral_error()


def test_ratio_defaults_to_combined_uncertainty_and_can_omit_denominator():
    num = ab.Hist1D([10.0], [0.0, 1.0], errors=[3.0])
    den = ab.Hist1D([5.0], [0.0, 1.0], errors=[4.0])

    value, combined = ab.ratio(num, den)
    _, numerator_only = ab.ratio(num, den, include_denominator_error=False)

    np.testing.assert_allclose(value, [2.0])
    np.testing.assert_allclose(combined, [np.hypot(3.0 / 5.0, 10.0 * 4.0 / 25.0)])
    np.testing.assert_allclose(numerator_only, [3.0 / 5.0])


def test_ratio_zero_numerator_has_finite_numerator_uncertainty():
    num = ab.Hist1D([0.0], [0.0, 1.0], errors=[2.0])
    den = ab.Hist1D([4.0], [0.0, 1.0], errors=[1.0])

    value, error = ab.ratio(num, den)

    np.testing.assert_allclose(value, [0.0])
    np.testing.assert_allclose(error, [0.5])


def test_ratio_leaves_zero_denominator_bins_undefined():
    num = ab.Hist1D([3.0, 2.0], [0.0, 1.0, 2.0], errors=[1.0, 1.0])
    den = ab.Hist1D([0.0, 4.0], [0.0, 1.0, 2.0], errors=[0.0, 2.0])

    value, error = ab.ratio(num, den)
    assert np.isnan(value[0]) and np.isnan(error[0])
    np.testing.assert_allclose(value[1], 0.5)

    filled, filled_error = ab.ratio(num, den, nan_to=0.0)
    assert filled[0] == 0.0 and filled_error[0] == 0.0


def test_ratio_requires_matching_edges():
    num = ab.Hist1D([1.0, 1.0], [0.0, 1.0, 2.0])
    roundoff = ab.Hist1D([1.0, 1.0], [0.0, 1.0 + 1e-15, 2.0])
    np.testing.assert_allclose(ab.ratio(num, roundoff)[0], [1.0, 1.0])

    for edges in ([0.0, 1.1, 2.0], [0.0, 1.0, 2.0, 3.0]):
        den = ab.Hist1D(np.ones(len(edges) - 1), edges)
        with np.testing.assert_raises_regex(ValueError, "identical edges"):
            ab.ratio(num, den)


def test_density_rebin_rejects_a_trailing_bin():
    density = ab.divide_by_bin_width(ab.Hist1D(
        [2.0, 4.0, 6.0], [0.0, 1.0, 2.0, 3.0], errors=[1.0, 2.0, 3.0],
    ))
    with np.testing.assert_raises_regex(ValueError, "exactly divide"):
        ab.rebin(density, factor=2)


def test_rebin_to_edges_merges_aligned_bins_exactly():
    hist = ab.Hist1D(
        [1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 4.0, 8.0], errors=[0.5, 1.0, 1.5, 2.0],
        overflow=5.0, underflow=6.0, overflow_error=1.0, underflow_error=2.0, is_weighted=True,
    )
    target = [0.0, 2.0 + 1e-15, 8.0]
    merged = ab.rebin(hist, edges=target)

    assert np.array_equal(merged.edges, target)
    np.testing.assert_allclose(merged.counts, [3.0, 7.0])
    np.testing.assert_allclose(merged.errors ** 2, [0.25 + 1.0, 2.25 + 4.0])
    assert merged.overflow == 5.0 and merged.underflow == 6.0
    assert (merged.overflow_error, merged.underflow_error) == (1.0, 2.0)
    assert merged.is_weighted
    np.testing.assert_allclose(hist.counts, [1.0, 2.0, 3.0, 4.0])

    density = ab.rebin(ab.divide_by_bin_width(hist), edges=target)
    assert density.is_density
    np.testing.assert_allclose(density.counts, [3.0 / 2.0, 7.0 / 6.0])
    np.testing.assert_allclose(density.integral(), hist.integral())
    np.testing.assert_allclose(density.integral_error(), hist.integral_error())


def test_rebin_to_a_narrower_range_moves_bins_into_the_flows():
    hist = ab.Hist1D(
        [1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0], errors=[0.5, 1.0, 1.5, 2.0],
        overflow=5.0, underflow=6.0, overflow_error=1.0, underflow_error=2.0, is_weighted=True,
    )
    narrowed = ab.rebin(hist, edges=[1.0, 3.0])
    np.testing.assert_array_equal(narrowed.values(flow=True), [7.0, 5.0, 9.0])
    np.testing.assert_allclose(narrowed.variances(flow=True),
                               [4.0 + 0.25, 1.0 + 2.25, 4.0 + 1.0])
    assert np.sum(narrowed.values(flow=True)) == np.sum(hist.values(flow=True))


def test_rebin_to_edges_matches_edges_in_linear_memory():
    # A full source x target distance matrix for 200k edges would need ~320 GB.
    edges = np.linspace(0.0, 1.0, 200_001)
    hist = ab.Hist1D(np.ones(edges.size - 1), edges)
    identity = ab.rebin(hist, edges=edges)
    np.testing.assert_array_equal(identity.counts, hist.counts)
    merged = ab.rebin(hist, edges=edges[::1000])
    np.testing.assert_allclose(merged.counts, 1000.0)


def test_rebin_to_edges_rejects_split_bins_and_wider_ranges():
    hist = ab.Hist1D([1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])
    for target in (
        [0.0, 1.5, 3.0],
        [0.0, 1.0, 1.0 + 1e-6, 3.0],
        [0.0, 2.0, 4.0],
        [-1.0, 1.0, 3.0],
    ):
        with np.testing.assert_raises_regex(ValueError, "align with source edges"):
            ab.rebin(hist, edges=target)
    with np.testing.assert_raises_regex(ValueError, "either factor or edges"):
        ab.rebin(hist, 1, edges=[0.0, 3.0])


def test_data_mc_plots_reject_misaligned_bins():
    mc = ab.Hist1D([4.0, 4.0], [0.0, 1.0, 2.0], errors=[1.0, 1.0])
    other = ab.Hist1D([4.0, 4.0], [0.0, 1.5, 2.0], errors=[1.0, 1.0])
    with np.testing.assert_raises_regex(ValueError, "identical edges"):
        ab.hist1d_with_ratio(other, mc)
    with np.testing.assert_raises_regex(ValueError, "identical edges"):
        ab.hist1d_stacked({"MC": mc}, data_hist=other)
    with np.testing.assert_raises_regex(ValueError, "identical edges"):
        ab.hist1d_stacked({"a": mc, "b": other})


def test_efficiency_error_lengths_are_nonnegative_at_boundary_counts():
    passed = np.array([0.0, 4.0, 10.0, 0.0])
    total = np.array([10.0, 10.0, 10.0, 0.0])
    _, eff, err_low, err_high = ab.efficiency(passed, total, bins=[0., 1., 2., 3., 4.])

    np.testing.assert_array_equal(eff[:3], [0.0, 0.4, 1.0])
    assert err_low[0] == 0.0 and err_high[0] > 0
    assert err_high[2] == 0.0 and err_low[2] > 0
    assert np.all(err_low[:3] >= 0) and np.all(err_high[:3] >= 0)
    assert np.isnan(eff[3]) and np.isnan(err_low[3]) and np.isnan(err_high[3])


def test_filling_keeps_out_of_range_entries_in_the_flows_with_their_variance():
    w = np.array([2.0, 1.0, 3.0, 0.5, 4.0])
    h = ab.hist1d(np.array([-1.0, 0.5, 0.5, 9.0, 1.5]), weights=w,
                  bins=np.array([0.0, 1.0, 2.0]), overflow=True)
    np.testing.assert_array_equal(h.values(flow=True), [2.0, 4.0, 4.0, 0.5])
    np.testing.assert_array_equal(h.variances(flow=True), [4.0, 10.0, 16.0, 0.25])

    g = ab.hist2d(np.array([-1.0, 0.5, 0.5, 9.0]), np.array([0.5, 0.5, 2.5, -4.0]),
                  bins=(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])),
                  weights=np.array([2.0, 1.0, 3.0, 0.5]))
    expected = np.zeros((4, 4))
    expected[0, 1], expected[1, 1], expected[1, 3], expected[3, 0] = 2.0, 1.0, 3.0, 0.5
    np.testing.assert_array_equal(g.values(flow=True), expected)
    np.testing.assert_array_equal(g.variances(flow=True), expected ** 2)
    np.testing.assert_array_equal(g.counts, [[1.0, 0.0], [0.0, 0.0]])
    assert g.values(flow=True).sum() == 6.5


def test_weighted_flows_without_an_error_are_refused():
    with np.testing.assert_raises_regex(ValueError, "overflow_error"):
        ab.Hist1D([1.0], [0.0, 1.0], errors=[1.0], overflow=2.0, is_weighted=True)
    unweighted = ab.Hist1D([4.0], [0.0, 1.0], overflow=9.0)
    assert unweighted.overflow_error == 3.0
