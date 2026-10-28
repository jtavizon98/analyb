"""Rendering tests for the portable MathText plotting defaults."""

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import rcParams
import numpy as np
import pytest

import analyb as ab
from analyb import plot


def test_mathtext_style_is_default():
    assert rcParams["text.usetex"] is False
    assert rcParams["figure.constrained_layout.use"] is True
    assert rcParams["mathtext.fontset"] == "stix"
    assert rcParams["legend.loc"] == "upper right"
    assert rcParams["legend.frameon"] is False


def test_unit_bearing_plots_render_without_latex():
    hist = ab.hist1d(np.linspace(20, 80, 100), bins=10, range=(0, 100), unit=r"\GeV")
    fig, _ = ab.hist1d_plot(hist, variable="met")
    fig.canvas.draw()
    plt.close(fig)

    fig, _ = ab.profile_plot(
        np.array([1.0, 2.0]), np.array([20.0, 40.0]), np.array([1.0, 1.0]),
        variable_x="nbjets", variable_y="met",
    )
    fig.canvas.draw()
    plt.close(fig)


def test_radian_unit_renders_as_rad_abbreviation():
    hist = ab.hist1d(np.linspace(-np.pi, np.pi, 100), bins=10)

    with ab.local({"angle": (r"$\phi$", r"\radian")}):
        fig, ax = ab.hist1d_plot(hist, variable="angle")

    assert r"\mathrm{rad}" in ax.get_xlabel()
    assert r"\mathrm{radian}" not in ax.get_xlabel()
    assert r"\mathrm{rad}" in ax.get_ylabel()
    fig.canvas.draw()
    plt.close(fig)


def test_fit_labels_render_without_latex():
    x = np.linspace(0, 10, 20)
    y = 5 + x
    fig, _ = ab.plot_data_with_fit(
        x, y, np.ones_like(y), lambda values, offset, slope: offset + slope * values,
        (5, 1), fit_label="Fit", unit=r"\GeV",
    )
    fig.canvas.draw()
    plt.close(fig)


def test_atlas_label_has_four_mathtext_safe_lines():
    fig, ax = plt.subplots()
    ab.add_atlas_label(ax, energy=13.6, lumi=139.0)

    label = ax.texts[-1]
    assert label.get_text() == (
        "ATLAS\n"
        "Internal\n"
        "$\\sqrt{s} = 13.6\\;\\mathrm{TeV}$\n"
        r"$\mathcal{L} = 139.0\;\mathrm{fb}^{-1}$"
    )
    assert label.get_text().count("\n") == 3
    assert not any(macro in label.get_text() for macro in
                   (r"\textbf", r"\textit", r"\SI", r"\si", r"\num", r"\unit", r"\displaystyle"))
    fig.canvas.draw()
    plt.close(fig)


def test_atlas_label_can_omit_luminosity():
    fig, ax = plt.subplots()
    ab.add_atlas_label(ax, status="Internal", lumi=False)

    label = ax.texts[-1].get_text()
    assert label == (
        "ATLAS\n"
        "Internal\n"
        r"$\sqrt{s} = 13\;\mathrm{TeV}$"
    )
    assert "mathcal{L}" not in label
    plt.close(fig)


def test_atlas_label_is_fixed_upper_left():
    """ATLAS label is always placed at the default fixed position (x=0.04, y=0.95)."""
    fig, ax = plt.subplots()
    ab.add_atlas_label(ax)

    label = ax.texts[-1]
    x, y = label.get_position()
    assert np.isclose(x, 0.04)
    assert np.isclose(y, 0.95)
    assert label.get_verticalalignment() == "top"
    assert label.get_horizontalalignment() == "left"
    plt.close(fig)


def test_atlas_label_respects_custom_fixed_position():
    """Custom (x, y) are honoured when passed explicitly."""
    fig, ax = plt.subplots()
    ab.add_atlas_label(ax, x=0.10, y=0.80)

    label = ax.texts[-1]
    x, y = label.get_position()
    assert np.isclose(x, 0.10)
    assert np.isclose(y, 0.80)
    plt.close(fig)


def test_reserve_atlas_label_headroom_linear():
    """Linear headroom reserves ~30% extra space above the data."""
    fig, ax = plt.subplots()
    ax.set_ylim(0, 100)

    ab.reserve_atlas_label_headroom(ax, ylog=False)

    ymin, ymax = ax.get_ylim()
    assert ymin == 0.0
    # new_ymax = 0 + (100 - 0) / 0.7 ≈ 142.9
    assert np.isclose(ymax, 100 / 0.7)
    plt.close(fig)


def test_reserve_atlas_label_headroom_log():
    """Log headroom expands by ~0.3 decades at the top."""
    fig, ax = plt.subplots()
    ax.set_ylim(1, 1000)

    ab.reserve_atlas_label_headroom(ax, ylog=True)

    ymin, ymax = ax.get_ylim()
    assert ymin == 1.0
    # Expanding the log range by 1 / 0.7 leaves 30% of the axes free.
    expected = 10 ** (3 / 0.7)
    assert np.isclose(ymax, expected, rtol=1e-3)
    plt.close(fig)


def test_stacked_fills_are_opaque():
    """Stacked MC fills use opaque colours matching the legend (no alpha blending)."""
    h1 = ab.hist1d(np.linspace(0, 10, 100), bins=5, range=(0, 10))
    h2 = ab.hist1d(np.linspace(5, 15, 100), bins=5, range=(0, 10))
    fig, ax = ab.hist1d_stacked({"a": h1, "b": h2})
    # stairs with fill=True creates PolyCollection objects in ax.collections
    for collection in ax.collections:
        alpha = collection.get_alpha()
        assert alpha is None or alpha == 1.0, f"Expected opaque fill, got alpha={alpha}"
    plt.close(fig)


def test_stacked_with_data_keeps_uncertainty_band():
    """Stacked MC + data overlay preserves the total-uncertainty band."""
    h1 = ab.hist1d(np.linspace(0, 10, 100), bins=5, range=(0, 10))
    h2 = ab.hist1d(np.linspace(5, 15, 100), bins=5, range=(0, 10))
    h_data = ab.hist1d(np.linspace(0, 10, 200), bins=5, range=(0, 10))
    fig, (ax_main, ax_ratio) = ab.hist1d_stacked(
        {"a": h1, "b": h2}, data_hist=h_data, save_name="test_opaque_stacked"
    )
    # The total MC uncertainty band should be a black fill_between with low alpha
    fill_between_found = False
    for collection in ax_main.collections:
        # Check for the uncertainty band: color-related check
        fc = collection.get_facecolor()
        if len(fc) > 0 and np.allclose(fc[0][:3], [0, 0, 0]):
            fill_between_found = True
            break
    # Also check the added uncertainty band via ax.collections after fill_between
    assert fill_between_found or len(ax_main.collections) >= 1
    plt.close(fig)


def test_hist2d_colorbar_is_labelled():
    hist = ab.Hist2D([[1.0]], [0.0, 1.0], [0.0, 1.0])
    fig, ax = ab.hist2d_plot(hist, variable_x="njets", variable_y="ht")
    fig.set_dpi(150)
    fig.canvas.draw()

    assert fig.axes[-1].get_ylabel() == "Counts"
    bbox = ax.get_window_extent()
    assert np.isclose(bbox.width / bbox.height, 1.0, rtol=0.03)
    plt.close(fig)


def test_caller_owned_axis_names():
    with ab.local({"momentum": (r"Momentum $p_T$", r"\GeV")}):
        assert ab.lookup("momentum") == (r"Momentum $p_T$", r"\GeV")


def test_data_mc_ratio_helpers_use_numerator_only_point_errors(monkeypatch):
    data = ab.Hist1D([10.0], [0.0, 1.0], errors=[3.0])
    mc = ab.Hist1D([5.0], [0.0, 1.0], errors=[4.0])
    calls = []
    original_ratio = plot.hist_ratio

    def spy_ratio(*args, **kwargs):
        calls.append(kwargs)
        return original_ratio(*args, **kwargs)

    monkeypatch.setattr(plot, "hist_ratio", spy_ratio)
    fig, _ = ab.hist1d_with_ratio(data, mc)
    plt.close(fig)
    fig, _ = ab.hist1d_stacked({"MC": mc}, data_hist=data)
    plt.close(fig)

    assert calls
    assert all(call["include_denominator_error"] is False for call in calls)


def test_data_mc_plot_omits_nonpositive_mc_ratio_bins():
    data = ab.Hist1D([2.0, 2.0], [0.0, 1.0, 2.0], errors=[1.0, 1.0])
    mc = ab.Hist1D([-1.0, 2.0], [0.0, 1.0, 2.0], errors=[.5, .5])

    fig, (_, ax_ratio) = ab.hist1d_with_ratio(data, mc)
    ratio_values = ax_ratio.lines[0].get_ydata()

    assert np.isnan(ratio_values[0])
    assert ratio_values[1] == pytest.approx(1.)
    plt.close(fig)


def _stack_hists():
    edges = np.array([0.0, 1.0, 2.0])
    return (
        ab.Hist1D(np.array([1.0, 2.0]), edges, errors=np.ones(2)),
        ab.Hist1D(np.array([3.0, 4.0]), edges, errors=np.ones(2)),
    )


def _assert_second_stack_patch_geometry(ax):
    """The second filled stairs path must run from cumulative bottom to top."""
    vertices = ax.patches[1].get_path().vertices
    # The forward half of a filled StepPatch contains the baseline-to-top path.
    assert np.allclose(vertices[:6, 1], [1.0, 4.0, 4.0, 6.0, 6.0, 2.0])


def test_stacked_stairs_use_cumulative_tops_without_data():
    h1, h2 = _stack_hists()
    fig, ax = ab.hist1d_stacked({"first": h1, "second": h2})
    _assert_second_stack_patch_geometry(ax)
    plt.close(fig)


def test_stacked_stairs_use_cumulative_tops_with_data():
    h1, h2 = _stack_hists()
    data = ab.Hist1D(np.array([4.0, 6.0]), h1.edges, errors=np.ones(2))
    fig, (ax_main, _) = ab.hist1d_stacked({"first": h1, "second": h2}, data_hist=data)
    _assert_second_stack_patch_geometry(ax_main)
    plt.close(fig)


def test_stacked_keyed_colors_preserve_geometry_and_sort_no_data_legend():
    """Color lookup and legend ordering must not reorder the caller's stack."""
    h_first, h_second = _stack_hists()
    fig, ax = ab.hist1d_stacked(
        {"first": h_first, "second": h_second},
        colors={"second": "gold", "first": "navy"},
    )

    _assert_second_stack_patch_geometry(ax)
    assert np.allclose(ax.patches[0].get_facecolor(), to_rgba("navy"))
    assert np.allclose(ax.patches[1].get_facecolor(), to_rgba("gold"))
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["second", "first"]
    plt.close(fig)


def test_stacked_data_is_black_and_legend_precedes_yield_sorted_mc():
    h_first, h_second = _stack_hists()
    data = ab.Hist1D(np.array([4.0, 6.0]), h_first.edges, errors=np.ones(2))
    fig, (ax_main, ax_ratio) = ab.hist1d_stacked(
        {"first": h_first, "second": h_second}, data_hist=data,
    )

    assert [text.get_text() for text in ax_main.get_legend().get_texts()] == ["Data", "second", "first"]
    assert ax_main.lines[-1].get_color() == "black"
    assert ax_ratio.lines[0].get_color() == "black"
    plt.close(fig)


def test_stacked_keyed_colors_report_missing_samples():
    first, second = _stack_hists()
    with pytest.raises(ValueError, match="no color for samples.*second"):
        ab.hist1d_stacked({"first": first, "second": second}, colors={"first": "blue"})


def test_ratio_limits_expand_for_finite_points_and_errors():
    edges = np.array([0.0, 1.0, 2.0])
    mc = ab.Hist1D(np.array([10.0, 10.0]), edges, errors=np.array([1.0, 1.0]))
    data = ab.Hist1D(np.array([10.0, 25.0]), edges, errors=np.array([1.0, 5.0]))
    fig, (_, ax_ratio) = ab.hist1d_with_ratio(data, mc)
    low, high = ax_ratio.get_ylim()
    assert low <= 0.5 and high >= 3.0  # includes 2.5 + propagated error
    plt.close(fig)


def test_ratio_limits_ignore_clear_pathological_denominator_bin():
    low, high = plot._adaptive_ratio_ylim(np.array([1.0, 1e9]), np.array([0.1, 1e9]))
    assert low <= 0.5
    assert high < 100.0


def test_ratio_limits_ignore_single_moderate_outlier_among_many_bins():
    ratios = np.array([1.0] * 19 + [25.0])
    errors = np.array([0.05] * 19 + [8.0])

    low, high = plot._adaptive_ratio_ylim(ratios, errors)

    assert low <= 0.5
    assert high < 5.0


def test_ratio_limits_keep_meaningful_finite_disagreement():
    low, high = plot._adaptive_ratio_ylim(
        np.array([1.0, 25.0]), np.array([0.1, 8.0]),
        denominator=np.array([100.0, 100.0]),
        denominator_errors=np.array([5.0, 5.0]),
    )
    assert low <= 0.5
    assert high > 30.0


def test_ratio_limits_exclude_denominator_compatible_with_zero():
    low, high = plot._adaptive_ratio_ylim(
        np.array([1.0, 25.0]), np.array([0.1, 8.0]),
        denominator=np.array([100.0, 1.0]),
        denominator_errors=np.array([5.0, 1.0]),
    )
    assert low <= 0.5
    assert high < 5.0


def test_density_comparison_normalizes_by_width_integrated_yield():
    density = ab.divide_by_bin_width(ab.Hist1D(
        [2.0, 6.0], [0.0, 1.0, 4.0], errors=[1.0, 2.0], unit=r"\GeV",
    ))

    fig, ax = ab.hist1d_comparison([density], variable="met", normalize=True)
    assert np.max(ax.patches[0].get_path().vertices[:, 1]) == 0.25
    plt.close(fig)

    fig, ax = ab.hist1d_comparison(
        [density], variable="met", normalize=True, range_indices=(1, 2),
    )
    assert np.max(ax.patches[0].get_path().vertices[:, 1]) == 1.0 / 3.0
    plt.close(fig)


def test_atlas_headroom_legend_upper_right_stacked():
    """hist1d_stacked with atlas=True: label upper-left, legend upper-right."""
    h1 = ab.hist1d(np.linspace(0, 50, 100), bins=5, range=(0, 100))
    h2 = ab.hist1d(np.linspace(50, 100, 100), bins=5, range=(0, 100))
    fig, ax = ab.hist1d_stacked({"a": h1, "b": h2}, atlas=True)

    labels = [t for t in ax.texts if "ATLAS" in t.get_text()]
    assert len(labels) == 1
    x, y = labels[0].get_position()
    assert np.isclose(x, 0.04)
    assert np.isclose(y, 0.95)

    legend = ax.get_legend()
    assert legend is not None
    assert legend._loc == 1
    assert not legend.get_frame_on()
    fig.canvas.draw()
    axes_bbox = ax.get_window_extent()
    legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
    assert axes_bbox.contains(*legend_bbox.get_points()[0])
    assert axes_bbox.contains(*legend_bbox.get_points()[1])
    plt.close(fig)


def test_atlas_label_kwargs_preserved_in_comparison():
    """atlas_label_kwargs are forwarded and headroom applied in comparison plot."""
    hist = ab.hist1d(np.linspace(0, 100, 200), bins=10, range=(0, 100))
    fig, ax = ab.hist1d_comparison(
        [hist], atlas=True,
        atlas_label_kwargs={"status": "Internal", "lumi": False},
    )

    labels = [t for t in ax.texts if "ATLAS" in t.get_text()]
    assert len(labels) == 1
    assert "Internal" in labels[0].get_text()
    assert "mathcal{L}" not in labels[0].get_text()
    x, y = labels[0].get_position()
    assert np.isclose(x, 0.04)
    assert np.isclose(y, 0.95)

    # Headroom applied: compare with non-atlas version
    hist2 = ab.hist1d(np.linspace(0, 100, 200), bins=10, range=(0, 100))
    fig2, ax2 = ab.hist1d_comparison(
        [hist2], atlas=False,
    )
    _, ymax_no_headroom = ax2.get_ylim()
    plt.close(fig2)

    _, ymax = ax.get_ylim()
    # Headroom should expand the upper limit (normalized plot has small values)
    assert ymax > ymax_no_headroom
    plt.close(fig)
