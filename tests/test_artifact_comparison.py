import matplotlib.pyplot as plt
import numpy as np
import pytest

import analyb as ab
from analyb import comparison as artifact_comparison


def _axis(name, edges, title=""):
    from uproot.writing.identify import to_TAxis

    edges = np.asarray(edges, dtype=np.float64)
    return to_TAxis(
        name, title, len(edges) - 1, edges[0], edges[-1], fXbins=edges,
    )


def _weighted_th1(values_with_flow, variances_with_flow, edges):
    from uproot.writing.identify import to_TH1x

    values = np.asarray(values_with_flow, dtype=np.float64)
    variances = np.asarray(variances_with_flow, dtype=np.float64)
    return to_TH1x(
        "h", "weighted", values,
        fEntries=float(np.sum(values[1:-1])),
        fTsumw=float(np.sum(values)),
        fTsumw2=float(np.sum(variances)),
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fSumw2=variances,
        fXaxis=_axis("xaxis", edges, "Observable [GeV]"),
    )


def _weighted_th2(values_with_flow, variances_with_flow, x_edges, y_edges):
    from uproot.writing.identify import to_TH2x

    values = np.asarray(values_with_flow, dtype=np.float64)
    variances = np.asarray(variances_with_flow, dtype=np.float64)
    return to_TH2x(
        "h2", "weighted 2D", values.T.reshape(-1),
        fEntries=float(np.sum(values[1:-1, 1:-1])),
        fTsumw=float(np.sum(values)),
        fTsumw2=float(np.sum(variances)),
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fTsumwy=0.0,
        fTsumwy2=0.0,
        fTsumwxy=0.0,
        fSumw2=variances.T.reshape(-1),
        fXaxis=_axis("xaxis", x_edges, "Reco"),
        fYaxis=_axis("yaxis", y_edges, "Particle"),
    )


def test_external_weighted_th1_preserves_variances_signed_bins_and_flows(tmp_path):
    uproot = pytest.importorskip("uproot")
    path = tmp_path / "weighted.root"
    values = np.array([3.0, -2.0, 0.0, 5.0, -1.0])
    variances = np.array([0.25, 4.0, 9.0, 16.0, 1.0])
    edges = np.array([0.0, 1.0, 3.0, 6.0])
    with uproot.recreate(path) as root_file:
        root_file["dir/h"] = _weighted_th1(values, variances, edges)

    loaded = ab.load_root_histogram(path, "dir/h")

    assert loaded.has_stored_variances
    assert loaded.classname == "TH1D"
    np.testing.assert_array_equal(loaded.values_with_flow, values)
    np.testing.assert_array_equal(loaded.variances_with_flow, variances)
    np.testing.assert_array_equal(loaded.histogram.counts, values[1:-1])
    np.testing.assert_array_equal(loaded.histogram.errors, np.sqrt(variances[1:-1]))
    np.testing.assert_array_equal(loaded.histogram.edges, edges)
    assert loaded.histogram.underflow == 3.0
    assert loaded.histogram.overflow == -1.0
    assert loaded.axis_titles == ("Observable [GeV]",)

    target = np.array([1.0, 6.0])
    rebinned = ab.rebin_root_histogram(loaded, (target,))
    np.testing.assert_array_equal(rebinned.values_with_flow, [1.0, 5.0, -1.0])
    np.testing.assert_array_equal(rebinned.variances_with_flow, [4.25, 25.0, 1.0])
    assert rebinned.histogram.underflow == 1.0
    assert rebinned.histogram.overflow == -1.0
    np.testing.assert_array_equal(rebinned.histogram.errors, [5.0])
    np.testing.assert_array_equal(loaded.values_with_flow, values)
    np.testing.assert_array_equal(loaded.variances_with_flow, variances)
    np.testing.assert_array_equal(loaded.histogram.edges, edges)
    target[0] = 0.0
    np.testing.assert_array_equal(rebinned.histogram.edges, [1.0, 6.0])


def test_external_loader_requires_stored_variances_by_default(tmp_path):
    uproot = pytest.importorskip("uproot")
    path = tmp_path / "plain.root"
    with uproot.recreate(path) as root_file:
        root_file["h"] = np.array([4.0, 9.0]), np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="no stored variances"):
        ab.load_root_histogram(path, "h")

    loaded = ab.load_root_histogram(path, "h", require_variances=False)
    assert not loaded.has_stored_variances
    np.testing.assert_array_equal(loaded.histogram.errors, [2.0, 3.0])


def test_external_nonsquare_th2_preserves_xy_orientation_and_flows(tmp_path):
    uproot = pytest.importorskip("uproot")
    path = tmp_path / "response.root"
    x_edges = np.array([0.0, 1.0, 4.0])
    y_edges = np.array([-2.0, 0.0, 3.0, 8.0])
    values = np.arange(20.0).reshape(4, 5)
    variances = np.arange(1.0, 21.0).reshape(4, 5)
    with uproot.recreate(path) as root_file:
        root_file["response"] = _weighted_th2(
            values, variances, x_edges, y_edges,
        )

    loaded = ab.load_root_histogram(path, "response")

    assert loaded.classname == "TH2D"
    assert loaded.histogram.counts.shape == (2, 3)
    np.testing.assert_array_equal(loaded.values_with_flow, values)
    np.testing.assert_array_equal(loaded.variances_with_flow, variances)
    np.testing.assert_array_equal(loaded.histogram.counts, values[1:-1, 1:-1])
    np.testing.assert_array_equal(
        loaded.histogram.errors, np.sqrt(variances[1:-1, 1:-1]),
    )
    np.testing.assert_array_equal(loaded.histogram.x_edges, x_edges)
    np.testing.assert_array_equal(loaded.histogram.y_edges, y_edges)

    rebinned = ab.rebin_root_histogram(loaded, ([1.0, 4.0], [0.0, 8.0]))
    np.testing.assert_array_equal(rebinned.values_with_flow, [
        [12, 20, 13], [21, 25, 14], [31, 35, 19],
    ])
    np.testing.assert_array_equal(rebinned.variances_with_flow, [
        [16, 24, 15], [23, 27, 15], [33, 37, 20],
    ])
    np.testing.assert_array_equal(rebinned.histogram.counts, [[25]])
    np.testing.assert_array_equal(rebinned.histogram.errors, [[np.sqrt(27)]])
    assert rebinned.values_with_flow.sum() == values.sum()
    assert rebinned.variances_with_flow.sum() == variances.sum()
    np.testing.assert_array_equal(loaded.values_with_flow, values)
    np.testing.assert_array_equal(loaded.variances_with_flow, variances)


def test_pairwise_diagnostics_use_numerator_reference_orientation_and_mask_zeros():
    numerator = ab.Hist1D(
        [4.0, -2.0, 3.0], [0.0, 1.0, 3.0, 6.0],
        errors=[2.0, 1.0, 1.5], is_weighted=True,
    )
    reference = ab.Hist1D(
        [2.0, -1.0, 0.0], [0.0, 1.0, 3.0, 6.0],
        errors=[1.0, 0.5, 2.0], is_weighted=True,
    )

    comparison = ab.compare_histograms(numerator, reference)

    np.testing.assert_allclose(comparison.ratio[:2], [2.0, 2.0])
    assert np.isnan(comparison.ratio[2])
    np.testing.assert_array_equal(comparison.ratio_valid, [True, True, False])
    np.testing.assert_allclose(comparison.difference, [2.0, -1.0, 3.0])
    assert comparison.numerator_yield == 5.0
    assert comparison.reference_yield == 1.0
    assert comparison.yield_ratio == 5.0
    assert comparison.shape_ndof == 2
    np.testing.assert_allclose(
        np.ones(3) @ comparison.shape_covariance,
        np.zeros(3), atol=1e-12,
    )


def test_pairwise_diagnostics_reject_mismatched_nonuniform_edges():
    numerator = ab.Hist1D([1.0, 2.0], [0.0, 1.0, 3.0], errors=[1.0, 1.0])
    reference = ab.Hist1D([1.0, 2.0], [0.0, 2.0, 3.0], errors=[1.0, 1.0])

    with pytest.raises(ValueError, match="identical edges"):
        ab.compare_histograms(numerator, reference)


def test_zero_total_signed_histogram_retains_raw_diagnostics():
    numerator = ab.Hist1D([2.0, -2.0], [0.0, 1.0, 2.0], errors=[1.0, 1.0])
    reference = ab.Hist1D([1.0, 3.0], [0.0, 1.0, 2.0], errors=[0.5, 1.5])

    comparison = ab.compare_histograms(numerator, reference)

    assert comparison.numerator_yield == 0.0
    assert comparison.reference_yield == 4.0
    assert comparison.yield_ratio == 0.0
    np.testing.assert_allclose(comparison.difference, [1.0, -5.0])
    np.testing.assert_allclose(comparison.ratio, [2.0, -2.0 / 3.0])
    assert not comparison.shape_available
    assert comparison.normalized_numerator is None
    assert comparison.shape_difference is None
    assert comparison.shape_chi2 is None
    assert comparison.shape_ndof is None

    fig, _, _ = ab.artifact_comparison_1d(numerator, reference)
    plt.close(fig)
    with pytest.raises(ValueError, match="shape plot is unavailable"):
        ab.artifact_comparison_1d(numerator, reference, shape=True)


def test_shape_chi2_is_infinite_for_mismatch_in_zero_variance_direction():
    numerator = ab.Hist1D(
        [1.0, 2.0, 1.0], [0.0, 1.0, 2.0, 3.0], errors=[0.0, 0.0, 0.0],
    )
    reference = ab.Hist1D(
        [1.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0], errors=[0.0, 0.0, 0.0],
    )

    comparison = ab.compare_histograms(numerator, reference)

    assert comparison.shape_available
    assert not comparison.shape_chi2_supported
    assert np.isinf(comparison.shape_chi2)
    assert comparison.shape_ndof == 0


def test_pairwise_density_yields_use_bin_widths():
    numerator = ab.divide_by_bin_width(ab.Hist1D(
        [2.0, 6.0], [0.0, 1.0, 4.0], errors=[1.0, 3.0],
    ))
    reference = ab.divide_by_bin_width(ab.Hist1D(
        [1.0, 3.0], [0.0, 1.0, 4.0], errors=[0.5, 1.5],
    ))

    comparison = ab.compare_histograms(numerator, reference)

    assert comparison.numerator_yield == 8.0
    assert comparison.reference_yield == 4.0
    assert comparison.yield_ratio == 2.0
    np.testing.assert_allclose(comparison.normalized_numerator, [0.25, 0.25])
    np.testing.assert_allclose(
        numerator.widths @ comparison.normalized_numerator_covariance,
        np.zeros(2), atol=1e-12,
    )


@pytest.mark.parametrize("points", ["numerator", "reference"])
def test_artifact_1d_shape_plot_ratios_normalized_shapes(points):
    numerator = ab.Hist1D([2.0, 6.0], [0.0, 1.0, 2.0], errors=[1.0, 2.0])
    reference = ab.Hist1D([2.0, 2.0], [0.0, 1.0, 2.0], errors=[1.0, 1.0])

    fig, (main_axis, ratio_axis), _ = ab.artifact_comparison_1d(
        numerator, reference, shape=True, points=points,
    )

    np.testing.assert_allclose(main_axis.lines[0].get_ydata(),
                               [0.25, 0.75] if points == "numerator" else [0.5, 0.5])
    np.testing.assert_allclose(ratio_axis.lines[0].get_ydata(), [0.5, 1.5])
    # Combined normalized-bin errors, independent of the drawing role.
    bars = ratio_axis.containers[0].lines[2][0].get_segments()
    np.testing.assert_allclose([(bar[1, 1] - bar[0, 1]) / 2 for bar in bars],
                               np.sqrt([21., 85.]) / 16)
    plt.close(fig)


@pytest.mark.parametrize("shape", [False, True])
def test_artifact_1d_density_display_preserves_raw_diagnostics(shape):
    edges = np.array([0.0, 1.0, 3.0, 6.0])
    widths = np.diff(edges)
    numerator = ab.Hist1D(
        [4.0, 6.0, 12.0], edges, errors=[2.0, 3.0, 6.0],
        label="Observable", unit=r"\GeV", is_weighted=True,
    )
    reference = ab.Hist1D(
        [2.0, 3.0, 6.0], edges, errors=[0.5, 1.5, 1.0],
        label="Observable", unit=r"\GeV", is_weighted=True,
    )
    original = tuple(
        array.copy() for array in
        (numerator.counts, numerator.errors, reference.counts, reference.errors)
    )
    plain_fig, (_, plain_ratio_axis), plain = ab.artifact_comparison_1d(
        numerator, reference, shape=shape,
    )

    fig, (main_axis, ratio_axis), diagnostics = ab.artifact_comparison_1d(
        numerator, reference, shape=shape, density=True,
    )

    if shape:
        total = 22.0
        counts = np.array([4.0, 6.0, 12.0])
        variances = np.array([4.0, 9.0, 36.0])
        expected_values = counts / total / widths
        expected_variances = (
            (total - counts) ** 2 * variances
            + counts ** 2 * (variances.sum() - variances)
        ) / total ** 4
        expected_errors = np.sqrt(expected_variances) / widths
        reference_total = 11.0
        reference_counts = np.array([2.0, 3.0, 6.0])
        reference_variances = np.array([0.25, 2.25, 1.0])
        expected_model_values = reference_counts / reference_total / widths
        expected_model_variances = (
            (reference_total - reference_counts) ** 2 * reference_variances
            + reference_counts ** 2
            * (reference_variances.sum() - reference_variances)
        ) / reference_total ** 4
        expected_model_errors = np.sqrt(expected_model_variances) / widths
        assert np.dot(main_axis.lines[0].get_ydata(), widths) == pytest.approx(1.0)
        assert main_axis.get_ylabel() == r"Normalized $\sum w$ / $\mathrm{GeV}$"
    else:
        expected_values = numerator.counts / widths
        expected_errors = numerator.errors / widths
        expected_model_values = reference.counts / widths
        expected_model_errors = reference.errors / widths
        assert np.dot(main_axis.lines[0].get_ydata(), widths) == numerator.integral()
        assert main_axis.get_ylabel() == r"$\sum w$ / $\mathrm{GeV}$"
    np.testing.assert_allclose(main_axis.lines[0].get_ydata(), expected_values)
    np.testing.assert_allclose(
        main_axis.patches[0].get_data().values, expected_model_values,
    )
    bars = main_axis.containers[0].lines[2][0].get_segments()
    np.testing.assert_allclose(
        [(bar[1, 1] - bar[0, 1]) / 2 for bar in bars], expected_errors,
    )
    band_y = main_axis.collections[0].get_paths()[0].vertices[:, 1]
    for boundary in np.r_[
        expected_model_values - expected_model_errors,
        expected_model_values + expected_model_errors,
    ]:
        assert np.any(np.isclose(band_y, boundary))
    np.testing.assert_array_equal(
        ratio_axis.lines[0].get_ydata(), plain_ratio_axis.lines[0].get_ydata(),
    )
    np.testing.assert_array_equal(diagnostics.ratio, plain.ratio)
    assert diagnostics.numerator_yield == 22.0
    assert diagnostics.reference_yield == 11.0
    for histogram, counts_before, errors_before in (
        (numerator, original[0], original[1]),
        (reference, original[2], original[3]),
    ):
        np.testing.assert_array_equal(histogram.counts, counts_before)
        np.testing.assert_array_equal(histogram.errors, errors_before)
        assert not histogram.is_density
    plt.close(fig)
    plt.close(plain_fig)


def test_raw_large_bin_1d_plot_skips_shape_work_and_default_stays_enabled(monkeypatch):
    small_numerator = ab.Hist1D([2.0, 3.0], [0.0, 1.0, 2.0], errors=[1.0, 1.0])
    small_reference = ab.Hist1D([1.0, 4.0], [0.0, 1.0, 2.0], errors=[1.0, 1.0])
    assert ab.compare_histograms(small_numerator, small_reference).shape_available

    size = 3000
    edges = np.arange(size + 1, dtype=np.float64)
    numerator = ab.Hist1D(np.full(size, 2.0), edges, errors=np.ones(size))
    reference = ab.Hist1D(np.ones(size), edges, errors=np.ones(size))

    def fail_shape_allocation(*args, **kwargs):
        raise AssertionError("raw 1D plot entered dense shape allocation")

    monkeypatch.setattr(artifact_comparison, "_normalized", fail_shape_allocation)
    fig, _, diagnostics = ab.artifact_comparison_1d(
        numerator, reference, shape=False,
    )

    assert diagnostics.numerator_yield == 6000.0
    assert diagnostics.reference_yield == 3000.0
    assert not diagnostics.shape_available
    assert diagnostics.shape_covariance is None
    plt.close(fig)


def test_artifact_2d_plot_keeps_nonsquare_xy_orientation():
    counts = np.arange(6.0).reshape(2, 3)
    numerator = ab.Hist2D(
        counts, [0.0, 1.0, 4.0], [-2.0, 0.0, 3.0, 8.0],
        errors=np.ones((2, 3)),
    )
    reference = ab.Hist2D(
        counts + 1.0, numerator.x_edges, numerator.y_edges,
        errors=np.ones((2, 3)),
    )

    fig, axes, comparison = ab.artifact_comparison_2d(numerator, reference)

    np.testing.assert_array_equal(axes[0, 0].collections[0].get_array(), counts.T)
    np.testing.assert_array_equal(axes[1, 0].collections[0].get_array(), (counts + 1).T)
    np.testing.assert_array_equal(comparison.difference, -np.ones((2, 3)))
    plt.close(fig)


def test_raw_200_by_200_comparison_skips_dense_shape_covariance(monkeypatch):
    size = 200
    edges = np.arange(size + 1, dtype=np.float64)
    numerator = ab.Hist2D(
        np.full((size, size), 2.0), edges, edges,
        errors=np.ones((size, size)),
    )
    reference = ab.Hist2D(
        np.ones((size, size)), edges, edges,
        errors=np.ones((size, size)),
    )

    def fail_shape_allocation(*args, **kwargs):
        raise AssertionError("raw comparison entered dense shape allocation")

    monkeypatch.setattr(artifact_comparison, "_normalized", fail_shape_allocation)
    diagnostics = ab.compare_histograms(numerator, reference, shape=False)

    assert diagnostics.numerator_yield == 80000.0
    assert diagnostics.reference_yield == 40000.0
    np.testing.assert_array_equal(diagnostics.ratio, np.full((size, size), 2.0))
    np.testing.assert_array_equal(diagnostics.difference, np.ones((size, size)))
    assert not diagnostics.shape_available
    assert diagnostics.shape_covariance is None

    fig, _, plotted = ab.artifact_comparison_2d(numerator, reference, shape=False)
    assert plotted.shape_covariance is None
    plt.close(fig)


def test_rebin_alignment_accepts_roundoff_but_rejects_splits_and_wider_ranges():
    values = np.array([1., 2., 3., 4., 5.])
    histogram = ab.Hist1D(values[1:-1], [-3., -1., 1., 3.], errors=[1., 1., 1.])
    loaded = ab.RootHistogram(histogram, values, np.ones(5), True, "TH1D", ("eta",))
    target = np.array([-3., np.nextafter(-1., 0.), 1., 3.])
    rebinned = ab.rebin_root_histogram(loaded, (target,))
    np.testing.assert_array_equal(rebinned.histogram.edges, target)
    np.testing.assert_array_equal(rebinned.values_with_flow, values)
    for edges in ([-3., 0., 3.], [-4., 3.], [-3., 4.]):
        with pytest.raises(ValueError, match="align"):
            ab.rebin_root_histogram(loaded, (edges,))

    # Observed eta-axis cancellation at the boundary immediately above zero.
    source_edges = [-5., 0., 0.05000000000000071, 0.1, 5.]
    target_edges = [-5., 0., 0.04999999999999982, 0.1, 5.]
    eta = ab.Hist1D(np.ones(4), source_edges, errors=np.ones(4))
    loaded = ab.RootHistogram(eta, np.ones(6), np.ones(6), True, "TH1D", ("eta",))
    rebinned = ab.rebin_root_histogram(loaded, (target_edges,))
    np.testing.assert_array_equal(rebinned.histogram.edges, target_edges)
    np.testing.assert_array_equal(rebinned.values_with_flow, np.ones(6))

    offset = ab.Hist1D([2., 3.], [1e16, 1e16+4, 1e16+8], errors=[1., 1.])
    loaded = ab.RootHistogram(offset, np.ones(4), np.ones(4), True, "TH1D", ("",))
    with pytest.raises(ValueError, match="align"):
        ab.rebin_root_histogram(loaded, ([1e16, 1e16+6, 1e16+8],))

    density = ab.divide_by_bin_width(histogram)
    loaded = ab.RootHistogram(density, values, np.ones(5), True, "TH1D", ("eta",))
    with pytest.raises(ValueError, match="raw counts"):
        ab.rebin_root_histogram(loaded, (histogram.edges,))


def _argmin_edge_match(original, target):
    """Independent nearest-edge oracle for the aligned-bin matching regression."""
    widths = np.diff(original)
    local_width = np.minimum(np.r_[widths[0], widths], np.r_[widths, widths[-1]])
    matched = np.array([np.argmin(np.abs(original - edge)) for edge in target])
    rounding_scale = np.maximum(np.abs(original[matched]), original[-1] - original[0])
    tolerance = np.minimum(
        64 * np.finfo(np.float64).eps * rounding_scale,
        1e-10 * local_width[matched],
    )
    if (np.any(np.abs(original[matched] - target) > tolerance)
            or np.any(np.diff(matched) <= 0)):
        raise ValueError("align")
    return matched


def _random_alignment_case(rng):
    """Source edges and a target near a subset of them, from roundoff to material shifts."""
    nbins = int(rng.integers(1, 13))
    offset = rng.choice([0.0, 1.0, -1.0]) * 10.0 ** rng.uniform(-3, 17)
    widths = 10.0 ** rng.uniform(-3, 3, size=nbins) * (10.0 ** rng.uniform(-2, 2))
    source = offset + np.r_[0.0, np.cumsum(widths)]
    if np.any(np.diff(source) <= 0):  # widths lost to the offset's precision
        source = np.r_[0.0, np.cumsum(widths)]
    size = int(rng.integers(2, source.size + 1)) if source.size > 2 else 2
    target = source[np.sort(rng.choice(source.size, size=size, replace=False))].copy()
    local = np.min(np.diff(source))
    scale = max(np.max(np.abs(source)), source[-1] - source[0])
    shifts = np.array([
        0.0, np.spacing(scale), 32 * np.finfo(float).eps * scale,
        64 * np.finfo(float).eps * scale, 65 * np.finfo(float).eps * scale,
        0.5e-10 * local, 1e-10 * local, 1.5e-10 * local, 0.3 * local, 0.5 * local, 2 * local,
    ])
    for index in np.flatnonzero(rng.random(target.size) < 0.4):
        target[index] += rng.choice([-1.0, 1.0]) * rng.choice(shifts)
    if rng.random() < 0.1:
        target = target[::-1].copy()  # nonincreasing target
    return source, target


def test_shared_edge_matcher_reproduces_previous_rebin_matching():
    # Decision C: reuse histogram._aligned_edge_indices without changing which
    # source edge each target edge selects, or which targets are rejected.
    from analyb.histogram import _aligned_edge_indices

    rng = np.random.default_rng(20261002)
    outcomes = {"accepted": 0, "rejected": 0}
    for _ in range(5000):
        source, target = _random_alignment_case(rng)
        try:
            expected = _argmin_edge_match(source, target)
        except ValueError:
            with pytest.raises(ValueError, match="align"):
                _aligned_edge_indices(source, target)
            outcomes["rejected"] += 1
            continue
        np.testing.assert_array_equal(_aligned_edge_indices(source, target), expected)
        outcomes["accepted"] += 1
    assert min(outcomes.values()) > 1000


def test_rebin_2d_conserves_flows_keeps_inputs_and_rejects_misalignment():
    x_edges = np.array([0.0, 1.0, 2.0, 4.0])
    y_edges = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    rng = np.random.default_rng(7)
    values = rng.normal(2.0, 3.0, size=(5, 6))  # signed weighted contents
    variances = rng.uniform(0.1, 4.0, size=(5, 6))
    histogram = ab.Hist2D(values[1:-1, 1:-1], x_edges, y_edges,
                          errors=np.sqrt(variances[1:-1, 1:-1]))
    loaded = ab.RootHistogram(histogram, values.copy(), variances.copy(), True,
                              "TH2D", ("x", "y"))

    rebinned = ab.rebin_root_histogram(loaded, ([1.0, 4.0], [-1.0, 2.0]))
    # x bins [0, 1) and y bins [-3, -1), [2, 5) move into the flows.
    assert rebinned.values_with_flow.shape == (3, 3)
    assert rebinned.values_with_flow.sum() == pytest.approx(values.sum(), rel=1e-12)
    assert rebinned.variances_with_flow.sum() == pytest.approx(variances.sum(), rel=1e-12)
    np.testing.assert_allclose(rebinned.histogram.counts, [[values[2:4, 2:4].sum()]], rtol=1e-12)
    np.testing.assert_allclose(rebinned.values_with_flow[0, 0],
                               values[:2, :2].sum(), rtol=1e-12)
    np.testing.assert_allclose(rebinned.variances_with_flow[2, 2],
                               variances[4:, 4:].sum(), rtol=1e-12)
    np.testing.assert_array_equal(loaded.values_with_flow, values)
    np.testing.assert_array_equal(loaded.variances_with_flow, variances)
    np.testing.assert_array_equal(loaded.histogram.x_edges, x_edges)
    np.testing.assert_array_equal(loaded.histogram.y_edges, y_edges)

    for target in (([0.0, 1.5, 4.0], y_edges), (x_edges, [-3.0, 0.5, 5.0]),
                   ([-1.0, 4.0], y_edges), (x_edges, [-3.0, 6.0])):
        with pytest.raises(ValueError, match="align"):
            ab.rebin_root_histogram(loaded, target)


def test_two_bin_shape_covariance_and_chi2_have_hand_calculated_values():
    numerator = ab.Hist1D([2., 6.], [0., 1., 2.], errors=[1., 2.])
    reference = ab.Hist1D([2., 2.], [0., 1., 2.], errors=[1., 1.])
    result = ab.compare_histograms(numerator, reference)
    # Var(n1/(n1+n2)) = (n2^2 Var(n1) + n1^2 Var(n2))/(n1+n2)^4.
    np.testing.assert_allclose(result.normalized_numerator_covariance,
                               [[13/1024, -13/1024], [-13/1024, 13/1024]])
    np.testing.assert_allclose(result.shape_covariance,
                               [[45/1024, -45/1024], [-45/1024, 45/1024]])
    assert result.shape_chi2 == pytest.approx(64/45)
    assert result.shape_ndof == 1


def test_shared_color_normalization_covers_both_inputs_without_mutating_caller():
    from matplotlib.colors import SymLogNorm

    numerator = ab.Hist2D([[1., 2.], [3., 4.]], [0., 1., 2.], [0., 1., 2.])
    reference = ab.Hist2D([[10., 20.], [30., 40.]], [0., 1., 2.], [0., 1., 2.])
    norm = SymLogNorm(linthresh=1.)
    fig, axes, _ = ab.artifact_comparison_2d(numerator, reference, norm=norm)
    # Previously the first panel alone set 1..4, saturating the reference.
    for ax in axes[:, 0]:
        assert ax.collections[0].norm.vmin == 1.
        assert ax.collections[0].norm.vmax == 40.
    assert norm.vmin is None and norm.vmax is None
    plt.close(fig)


@pytest.mark.parametrize("shape", [False, True])
def test_2d_signed_ratio_masks_zero_and_preserves_inputs_and_explicit_scale(shape):
    from matplotlib.colors import TwoSlopeNorm

    counts = np.array([[-2., 0., 6.], [4., -3., 2.]])
    reference_counts = np.array([[1., 0., -2.], [2., 3., 1.]])
    numerator = ab.Hist2D(counts, [0., 1., 4.], [-2., 0., 3., 8.], errors=np.ones((2, 3)))
    reference = ab.Hist2D(reference_counts, numerator.x_edges, numerator.y_edges,
                         errors=np.ones((2, 3)))
    ratio_norm = TwoSlopeNorm(vmin=0.5, vcenter=1., vmax=1.5)
    cmap = ab.catppuccin_colorbars["coolwarm"]
    original_bad = cmap.get_bad().copy()
    fig, axes, diagnostics = ab.artifact_comparison_2d(
        numerator, reference, shape=shape, ratio_norm=ratio_norm, ratio_cmap=cmap,
    )
    raw_ratio = np.array([[-2., np.nan, -3.], [2., -1., 2.]])
    expected = raw_ratio * (5 / 7 if shape else 1)
    np.testing.assert_allclose(axes[0, 1].collections[0].get_array().filled(np.nan),
                               expected.T, equal_nan=True)
    np.testing.assert_allclose(diagnostics.ratio, raw_ratio, equal_nan=True)
    expected_difference = counts / 7 - reference_counts / 5 if shape else counts - reference_counts
    np.testing.assert_allclose(axes[1, 1].collections[0].get_array(), expected_difference.T)
    assert axes[0, 1].collections[0].colorbar.extend == ("min" if shape else "both")
    assert "undefined (gray): 1" in axes[0, 1].texts[0].get_text()
    assert ratio_norm.vmin == 0.5 and ratio_norm.vmax == 1.5
    np.testing.assert_array_equal(cmap.get_bad(), original_bad)
    np.testing.assert_array_equal(numerator.counts, counts)
    np.testing.assert_array_equal(reference.counts, reference_counts)
    plt.close(fig)
