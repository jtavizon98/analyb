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
        is_weighted=True,
    )
    density = ab.divide_by_bin_width(hist)

    np.testing.assert_allclose(density.counts, [4.0, 6.0, 8.0])
    np.testing.assert_allclose(density.errors, [2.0, 3.0, 4.0])
    assert density.is_density
    assert density.bin_width is None
    assert density.overflow == 7.0 and density.underflow == 5.0
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
    with np.testing.assert_raises_regex(ValueError, "raw-count and density"):
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
        with np.testing.assert_raises_regex(ValueError, "bin edges differ"):
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
        overflow=5.0, underflow=6.0, is_weighted=True,
    )
    target = [0.0, 2.0 + 1e-15, 8.0]
    merged = ab.rebin_to_edges(hist, target)

    assert np.array_equal(merged.edges, target)
    np.testing.assert_allclose(merged.counts, [3.0, 7.0])
    np.testing.assert_allclose(merged.errors ** 2, [0.25 + 1.0, 2.25 + 4.0])
    assert merged.overflow == 5.0 and merged.underflow == 6.0
    assert merged.is_weighted
    np.testing.assert_allclose(hist.counts, [1.0, 2.0, 3.0, 4.0])

    density = ab.rebin_to_edges(ab.divide_by_bin_width(hist), target)
    assert density.is_density
    np.testing.assert_allclose(density.counts, [3.0 / 2.0, 7.0 / 6.0])
    np.testing.assert_allclose(density.integral(), hist.integral())
    np.testing.assert_allclose(density.integral_error(), hist.integral_error())


def test_rebin_to_edges_matches_edges_in_linear_memory():
    # A full source x target distance matrix for 200k edges would need ~320 GB.
    edges = np.linspace(0.0, 1.0, 200_001)
    hist = ab.Hist1D(np.ones(edges.size - 1), edges)
    identity = ab.rebin_to_edges(hist, edges)
    np.testing.assert_array_equal(identity.counts, hist.counts)
    merged = ab.rebin_to_edges(hist, edges[::1000])
    np.testing.assert_allclose(merged.counts, 1000.0)


def test_rebin_to_edges_rejects_split_bins_and_range_changes():
    hist = ab.Hist1D([1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])
    for target, message in (
        ([0.0, 1.5, 3.0], "align with source edges"),
        ([0.0, 1.0, 1.0 + 1e-6, 3.0], "align with source edges"),
        ([0.0, 2.0, 4.0], "align with source edges"),
        ([-1.0, 1.0, 3.0], "align with source edges"),
        ([1.0, 3.0], "span exactly the source range"),
        ([0.0, 2.0], "span exactly the source range"),
    ):
        with np.testing.assert_raises_regex(ValueError, message):
            ab.rebin_to_edges(hist, target)


def test_data_mc_plots_reject_misaligned_bins():
    mc = ab.Hist1D([4.0, 4.0], [0.0, 1.0, 2.0], errors=[1.0, 1.0])
    other = ab.Hist1D([4.0, 4.0], [0.0, 1.5, 2.0], errors=[1.0, 1.0])
    with np.testing.assert_raises_regex(ValueError, "identical bin edges"):
        ab.hist1d_with_ratio(other, mc)
    with np.testing.assert_raises_regex(ValueError, "identical bin edges"):
        ab.hist1d_stacked({"MC": mc}, data_hist=other)
    with np.testing.assert_raises_regex(ValueError, "identical bin edges"):
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
