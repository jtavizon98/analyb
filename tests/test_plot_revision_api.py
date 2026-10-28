"""Synthetic coverage for the narrow artifact plot revision API."""

import matplotlib as mpl
import matplotlib
import matplotlib.collections
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pytest

import analyb as ab


def _histograms():
    numerator = ab.Hist1D([6.0, 2.0, 3.0], [0.0, 1.0, 3.0, 4.0], errors=[2.0, 1.0, 1.0])
    reference = ab.Hist1D([3.0, 4.0, 0.0], [0.0, 1.0, 3.0, 4.0], errors=[1.0, 2.0, 0.0])
    return numerator, reference


def _band_contains(axis, expected):
    vertices = axis.collections[0].get_paths()[0].vertices[:, 1]
    return all(np.any(np.isclose(vertices, value)) for value in expected)


def test_separate_ratio_errors_draw_accepted_ff_ir_split():
    numerator, reference = _histograms()

    fig, (main, ratio_axis), diagnostics = ab.artifact_comparison_1d(
        numerator, reference, points="reference", separate_ratio_errors=True,
    )

    # Diagnostics keep the independent combined-error ratio, including the
    # undefined zero-reference bin.
    np.testing.assert_allclose(diagnostics.ratio[:2], [2.0, 0.5])
    assert np.isnan(diagnostics.ratio[2])
    np.testing.assert_allclose(ratio_axis.patches[0].get_data().values[:2], [2.0, 0.5])
    assert np.isnan(ratio_axis.patches[0].get_data().values[2])
    # Red numerator-only band: sigma(FF)/abs(IR).
    assert _band_contains(ratio_axis, [4 / 3, 8 / 3, 0.25, 0.75])
    # Black unity points use sigma(IR)/abs(IR).
    segments = ratio_axis.containers[0].lines[2][0].get_segments()
    np.testing.assert_allclose(segments[0][:, 1], [2 / 3, 4 / 3])
    np.testing.assert_allclose(segments[1][:, 1], [0.5, 1.5])
    assert not main.get_title()
    plt.close(fig)


def test_multiple_ff_lines_share_ir_unity_points_and_pairwise_geometry():
    first, reference = _histograms()
    second = ab.Hist1D([9.0, 1.0, 5.0], reference.edges,
                       errors=[3.0, 0.5, 1.0])
    fig, (main, ratio), diagnostics = ab.artifact_comparison_many_1d(
        {"Accepted": first, "Without cut": second}, reference,
        ratio_ylim=(0.0, 4.0), atlas=True, atlas_label_kwargs={"lumi": False},
    )
    assert list(main.get_legend().get_texts())[0].get_text() == "Accepted"
    assert main.get_legend()._loc == 1  # top-right, inside the axis
    assert tuple(fig.get_size_inches()) == (4.5, 5.4)
    assert len(ratio.containers) == 1  # reference at unity only once
    segments = ratio.containers[0].lines[2][0].get_segments()
    np.testing.assert_allclose(segments[0][:, 1], [2 / 3, 4 / 3])
    np.testing.assert_allclose(segments[1][:, 1], [0.5, 1.5])
    np.testing.assert_allclose(ratio.patches[1].get_data().values[:2], [3.0, 0.25])
    assert np.isnan(ratio.patches[1].get_data().values[2])
    assert any(np.isclose(vertex, 4.0) for path in ratio.collections[2].get_paths()
               for vertex in path.vertices[:, 1])
    np.testing.assert_allclose(diagnostics.ratio[:2], [2.0, 0.5])
    fig.canvas.draw()
    frame = main.get_legend().get_window_extent()
    bounds = main.get_window_extent()
    assert bounds.x0 < frame.x0 < frame.x1 < bounds.x1
    assert bounds.y0 < frame.y0 < frame.y1 < bounds.y1
    plt.close(fig)


def test_many_comparison_marks_every_prediction_outside_ratio_window():
    first, reference = _histograms()
    second = ab.Hist1D([30.0, 0.1, 5.0], reference.edges, errors=[3.0, 0.1, 1.0])
    fig, (_, ratio), _ = ab.artifact_comparison_many_1d(
        {"Accepted": first, "Variation": second}, reference,
        colors={"Variation": "tab:blue"}, ratio_ylim=(0.5, 4.0), range_text=False)
    marks = {(line.get_marker(), matplotlib.colors.to_hex(line.get_color())): line.get_xydata()
             for line in ratio.lines if line.get_marker() in ("^", "v") and len(line.get_xdata())}
    # second/reference = [10, 0.025, undefined]: one bin above, one below, in its colour
    np.testing.assert_allclose(marks[("^", matplotlib.colors.to_hex("tab:blue"))], [[0.5, 4.0 - 0.14]])
    np.testing.assert_allclose(marks[("v", matplotlib.colors.to_hex("tab:blue"))], [[2.0, 0.5 + 0.14]])
    triangles = [line for line in ratio.lines if line.get_marker() in ("^", "v") and len(line.get_xdata())]
    assert all(line.get_markerfacecolor() == line.get_color() for line in triangles
               if matplotlib.colors.to_hex(line.get_color()) == matplotlib.colors.to_hex("tab:blue"))
    # first/reference = [2, 0.5, undefined] stays inside: no first-prediction triangles
    assert len(marks) == 2
    plt.close(fig)


def test_many_comparison_single_prediction_matches_pairwise_markers():
    first, reference = _histograms()
    kwargs = dict(ratio_ylim=(0.0, 1.5), range_text=False)
    fig_many, (_, many), _ = ab.artifact_comparison_many_1d({"Accepted": first}, reference, **kwargs)
    fig_pair, (_, pair), _ = ab.artifact_comparison_1d(
        first, reference, numerator_label="Accepted", points="reference",
        separate_ratio_errors=True, model_color=ab.catppuccin_colors["red"], **kwargs)
    shape = lambda ax: [(line.get_marker(), line.get_xydata().tolist(), matplotlib.colors.to_hex(line.get_color()))
                        for line in ax.lines if line.get_marker() in ("^", "v")]
    assert shape(many) == shape(pair)
    assert any(marker == "^" and xy for marker, xy, _ in shape(many))
    plt.close(fig_many)
    plt.close(fig_pair)


@pytest.mark.parametrize("edges,density,label", [
    ([0, 1, 2], False, "Counts / 1.0"),
    ([0, 1, 3], True, "Counts / unit"),
])
def test_flavour_overlay_preserves_ratios_and_density_integral(edges, density, label):
    refs = {c: ab.Hist1D([2., 4.], edges, errors=[1., 2.]) for c in ("l", "e", "m")}
    sources = {arm: {c: ab.Hist1D([4., 2.], edges, errors=[2., 1.])
                     for c in refs} for arm in ("HIST", "MOD")}
    colors = {name: {c: color for c in refs} for name, color in
              (("HIST", "red"), ("MOD", "blue"), ("IR", "black"))}
    fig, (main, ratio), diagnostics = ab.artifact_comparison_flavours_1d(
        sources, refs, colors=colors, shape=False, density=density)
    assert main.get_ylabel() == label
    np.testing.assert_allclose(diagnostics["HIST", "e"].ratio, [2., .5])
    np.testing.assert_allclose(ratio.patches[0].get_data().values, [2., .5])
    expected = np.array([4., 2.]) / (np.diff(edges) if density else 1)
    np.testing.assert_allclose(main.patches[0].get_data().values, expected)
    assert np.isclose(np.sum(main.patches[0].get_data().values *
                             (np.diff(edges) if density else 1)), 6.)
    plt.close(fig)


def test_flavour_shape_density_is_unit_area_without_ratio_change():
    edges = [0, 1, 3]
    refs = {c: ab.Hist1D([2., 4.], edges, errors=[1., 2.]) for c in ("l", "e", "m")}
    sources = {"HIST": {c: ab.Hist1D([4., 2.], edges, errors=[2., 1.]) for c in refs}}
    colors = {name: {c: color for c in refs} for name, color in
              (("HIST", "red"), ("IR", "black"))}
    fig, (main, ratio), diagnostics = ab.artifact_comparison_flavours_1d(
        sources, refs, colors=colors, shape=True, density=True)
    assert main.get_ylabel() == "Normalized counts / unit"
    np.testing.assert_allclose(main.patches[0].get_data().values * np.diff(edges), [2/3, 1/3])
    np.testing.assert_allclose(ratio.patches[0].get_data().values, [2., .5])
    assert diagnostics["HIST", "e"].shape_available
    plt.close(fig)


def test_flavour_matrix_uses_solid_unmarked_sources_and_filled_reference_points():
    edges = [0., 1., 2.]
    refs = {c: ab.Hist1D([2., 4.], edges, errors=[1., 2.]) for c in ("l", "e", "m")}
    sources = {arm: {c: ab.Hist1D([4., 2.], edges, errors=[2., 1.]) for c in refs}
               for arm in ("HIST", "MOD")}
    colors = {"HIST": {"l": "red", "e": "pink", "m": "orange"},
              "MOD": {"l": "blue", "e": "cyan", "m": "green"},
              "IR": {"l": "black", "e": "purple", "m": "magenta"}}
    fig, (main, ratio), _ = ab.artifact_comparison_flavours_1d(
        sources, refs, colors=colors, ratio_bands=False)
    legend = main.get_legend()
    # Legend text is supplied column-major so the rendered rows are HIST/MOD/IR.
    assert [text.get_text() for text in legend.get_texts()] == [
        "HIST combined", "MOD combined", "IR combined", "e", "e", "e", "μ", "μ", "μ"]
    assert all(patch.get_linestyle() in ("-", "solid") for patch in main.patches)
    assert all(patch.get_linestyle() in ("-", "solid") for patch in ratio.patches)
    assert len(main.lines) == len(refs)  # only the three IR errorbar point series
    assert all(container.lines[0].get_marker() == "o" for container in main.containers)
    for ax in (main, ratio):
        for container in ax.containers:
            marker = container.lines[0]
            assert marker.get_markersize() == 4
            assert marker.get_markerfacecolor() == marker.get_color()
            assert marker.get_markeredgecolor() == marker.get_color()
    assert all(handle.get_markersize() == 4 and
               handle.get_markerfacecolor() == handle.get_color()
               for handle in legend.legend_handles if handle.get_marker() == "o")
    assert not any(isinstance(coll, mpl.collections.PolyCollection) for coll in ratio.collections)
    plt.close(fig)


@pytest.mark.parametrize("shape", [False, True])
def test_shared_ir_combined_reference_keeps_flavour_main_and_only_one_unity(shape):
    edges = [0., 1., 3., 4.]
    refs = {"l": ab.Hist1D([2., 4., 0.], edges, errors=[1., 2., 0.]),
            "e": ab.Hist1D([1., 1., 2.], edges, errors=[1., 1., 1.]),
            "m": ab.Hist1D([1., 3., 2.], edges, errors=[1., 1., 1.])}
    sources = {"HIST": {
        "l": ab.Hist1D([4., 2., 0.], edges, errors=[2., 1., 0.]),
        "e": ab.Hist1D([1., 2., 4.], edges, errors=[.2, .4, .6]),
        "m": ab.Hist1D([2., 1., 2.], edges, errors=[.3, .4, .5])}}
    colors = {name: {c: color for c in refs} for name, color in
              (("HIST", "red"), ("IR", "black"))}
    fig, (main, ratio), diagnostics = ab.artifact_comparison_flavours_1d(
        sources, refs, colors=colors, shape=shape, density=True,
        ratio_reference_flavour="l")
    assert len(main.containers) == 3 and len(ratio.containers) == 1
    assert [container.lines[0].get_markerfacecolor() for container in main.containers] == ["black"] * 3
    np.testing.assert_allclose(main.containers[1].lines[0].get_ydata(),
                               ([.25, .125, .5] if shape else [1., .5, 2.]))
    np.testing.assert_allclose(main.patches[1].get_data().values,
                               ([1/7, 1/7, 4/7] if shape else [1., 1., 4.]))
    observed = ratio.patches[1].get_data().values
    if shape:
        np.testing.assert_allclose(observed[:2], [3/7, 3/7])  # e unit area / IR combined unit area
    else:
        np.testing.assert_allclose(observed[:2], [.5, .5])  # e raw / IR combined raw
    assert np.isnan(observed[2])
    expected = ab.compare_histograms(sources["HIST"]["e"], refs["l"], shape=shape)
    if shape:
        np.testing.assert_allclose(diagnostics["HIST", "e"].normalized_numerator,
                                   expected.normalized_numerator)
        np.testing.assert_allclose(diagnostics["HIST", "e"].normalized_reference,
                                   expected.normalized_reference)
    else:
        np.testing.assert_allclose(diagnostics["HIST", "e"].ratio[:2], expected.ratio[:2])
    plt.close(fig)


def test_shared_ratio_reference_must_be_first_flavour():
    refs = {c: ab.Hist1D([2., 4.], [0., 1., 2.], errors=[1., 2.]) for c in ("l", "e", "m")}
    with pytest.raises(ValueError, match="first flavour"):
        ab.artifact_comparison_flavours_1d(
            {"HIST": refs}, refs, colors={name: {c: "black" for c in refs}
                                           for name in ("HIST", "IR")},
            ratio_reference_flavour="e")


def test_reference_flavours_share_denominator_and_caller_covariance():
    refs = {"l": ab.Hist1D([5., 5., 0.], [0., 1., 2., 3.], errors=[2., 2., 0.]),
            "e": ab.Hist1D([2., 4., 0.], [0., 1., 2., 3.], errors=[1., 1., 0.]),
            "m": ab.Hist1D([3., 1., 0.], [0., 1., 2., 3.], errors=[1., 1., 0.])}
    colors = {a: {c: color for c in refs} for a, color in (("IR", "black"), ("FF", "red"))}
    fig, (main, ratio), _ = ab.artifact_comparison_flavours_1d(
        {"FF": refs}, refs, colors=colors, shape=False,
        ratio_reference_flavour="l", show_reference_ratios=True,
        reference_ratio_errors={"e": [.13, .21, np.nan], "m": [.19, .11, np.nan]})
    assert len(main.containers) == 3 and len(ratio.containers) == 3
    np.testing.assert_allclose(ratio.containers[1].lines[0].get_ydata(), [.4, .8])
    np.testing.assert_allclose(ratio.containers[2].lines[0].get_ydata(), [.6, .2])
    np.testing.assert_allclose(ratio.containers[1].lines[2][0].get_segments()[0][:, 1],
                               [.27, .53])
    assert np.isnan(ratio.patches[0].get_data().values[2])
    plt.close(fig)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ab.artifact_comparison_flavours_1d(
            {"FF": refs}, refs, colors=colors, ratio_reference_flavour="l",
            show_reference_ratios=True, reference_ratio_errors={"e": [-1., 0., 0.], "m": None})


def test_separate_flavour_page_shows_combined_and_matching_ir_in_both_panels():
    combined = ab.Hist1D([5., 5., 0.], [0., 1., 2., 3.], errors=[2., 2., 0.])
    electron = ab.Hist1D([2., 4., 0.], combined.edges, errors=[1., 1., 0.])
    fig, (main, ratio), _ = ab.artifact_comparison_many_1d(
        {"Accepted": electron}, combined, reference_label="IR combined",
        reference_variation=electron, variation_label="IR e",
        variation_color="purple", variation_ratio_errors=[.2, .1, np.nan],
        atlas=False)
    assert len(main.containers) == 2 and len(ratio.containers) == 2
    np.testing.assert_allclose(ratio.containers[1].lines[0].get_ydata(), [.4, .8])
    assert all(container.lines[0].get_markerfacecolor() == container.lines[0].get_color()
               and container.lines[0].get_markersize() == 4 for container in main.containers)
    assert [t.get_text() for t in main.get_legend().get_texts()] == ["Accepted", "IR combined", "IR e"]
    plt.close(fig)


def test_cutflow_measurements_distinguish_event_and_nonsequential_objects(tmp_path):
    chart = ab.CutflowData(["total", "named bin"], [20, 25], errors=[0, 0])
    for measurement, expected in (("events", "Events"), ("raw_events", "Raw events"),
                                  ("raw_objects", "Raw objects"),
                                  ("objects_first_bin", "Objects / first bin"),
                                  ("survival", "Survival"),
                                  ("milestone_retention", "Milestone retention"),
                                  ("cut_efficiency", "Cut efficiency"),
                                  ("object_bin_ratio", "Object-bin ratio")):
        fig, ax = ab.cutflow_chart(chart, measurement=measurement, verbose=False,
                                   save_path=str(tmp_path), save_name=measurement)
        assert ax.get_ylabel() == expected
        plt.close(fig)


def test_separate_ratio_errors_requires_reference_pseudodata():
    numerator, reference = _histograms()

    with pytest.raises(ValueError, match="reference pseudodata"):
        ab.artifact_comparison_1d(numerator, reference, separate_ratio_errors=True)
    with pytest.raises(ValueError, match="not both"):
        ab.artifact_comparison_1d(
            numerator, reference, points="reference", separate_ratio_errors=True,
            ratio_presentation=ab.RatioPresentation(
                values=[1.0, 1.0, 1.0], errors=[0.1, 0.1, 0.1], valid=[True, True, False],
            ),
        )


def test_ratio_presentation_renders_split_and_preserves_diagnostics():
    numerator, reference = _histograms()
    presentation = ab.RatioPresentation(
        values=[2.0, 0.5, 9.0], errors=[0.2, 0.1, 0.3],
        valid=[True, True, False], unity_errors=[0.1, 0.2, 0.0],
    )

    fig, (main, ratio_axis), diagnostics = ab.artifact_comparison_1d(
        numerator, reference, ratio_presentation=presentation,
    )
    plain_fig, _, plain = ab.artifact_comparison_1d(numerator, reference)

    np.testing.assert_array_equal(diagnostics.ratio, plain.ratio)
    np.testing.assert_array_equal(diagnostics.ratio_error, plain.ratio_error)
    np.testing.assert_array_equal(diagnostics.ratio_valid, plain.ratio_valid)
    patch = ratio_axis.patches[0].get_data().values
    np.testing.assert_allclose(patch[:2], [2.0, 0.5])
    assert np.isnan(patch[2])
    assert _band_contains(ratio_axis, [1.8, 2.2, 0.4, 0.6])
    segments = ratio_axis.containers[0].lines[2][0].get_segments()
    np.testing.assert_allclose(segments[0][:, 1], [0.9, 1.1])
    np.testing.assert_allclose(segments[1][:, 1], [0.8, 1.2])
    assert not main.get_title()
    plt.close(fig)
    plt.close(plain_fig)


def test_unknown_reference_uncertainty_still_marks_defined_unity():
    numerator, reference = _histograms()
    presentation = ab.RatioPresentation(
        values=[2.0, 0.5, np.nan], errors=[0.2, 0.1, np.nan],
        valid=[True, True, False], unity_points=True,
    )
    fig, (_, ratio_axis), _ = ab.artifact_comparison_1d(
        numerator, reference, ratio_presentation=presentation,
        points="reference",
    )
    assert presentation.unity_errors is None
    assert len(ratio_axis.containers) == 1
    np.testing.assert_allclose(ratio_axis.containers[0].lines[0].get_ydata()[:2],
                               [1.0, 1.0])
    assert np.isnan(ratio_axis.containers[0].lines[0].get_ydata()[2])
    assert ratio_axis.containers[0].lines[2] == ()
    plt.close(fig)


def test_ratio_presentation_marks_out_of_range_and_undefined():
    numerator, reference = _histograms()
    presentation = ab.RatioPresentation(
        values=[2.0, 0.5, np.nan], errors=[0.2, 0.1, 0.3],
        valid=[True, True, False], unity_errors=[0.1, 0.1, 0.1],
    )

    fig, (_, ratio_axis), _ = ab.artifact_comparison_1d(
        numerator, reference, ratio_presentation=presentation, ratio_ylim=(0.5, 1.5),
    )

    text = ratio_axis.texts[0].get_text()
    assert "Below: 0" in text
    assert "above: 1" in text
    assert "undefined: 1" in text
    assert any(line.get_marker() == "^" for line in ratio_axis.lines)
    plt.close(fig)


def test_ratio_presentation_bar_errors_replace_the_band():
    numerator, reference = _histograms()
    presentation = ab.RatioPresentation(
        values=[2.0, 0.5, np.nan], errors=[0.2, 0.1, np.nan],
        valid=[True, True, False], error_style="bars",
    )
    fig, (_, ratio_axis), _ = ab.artifact_comparison_1d(
        numerator, reference, ratio_presentation=presentation, model_color="#123456",
    )
    # No band: the only collection is the error-bar line collection.
    assert not any(isinstance(c, matplotlib.collections.PolyCollection)
                   for c in ratio_axis.collections)
    bars = ratio_axis.containers[0]
    segments = bars.lines[2][0].get_segments()
    np.testing.assert_allclose(segments[0][:, 1], [1.8, 2.2])
    np.testing.assert_allclose(segments[1][:, 1], [0.4, 0.6])
    assert matplotlib.colors.to_hex(bars.lines[2][0].get_colors()[0]) == "#123456"
    plt.close(fig)


def test_ratio_lines_do_not_close_to_zero_at_the_outer_edges():
    numerator, reference = _histograms()
    for kwargs in ({"points": "reference", "separate_ratio_errors": True},
                   {"ratio_presentation": ab.RatioPresentation(
                       values=[2.0, 0.5, np.nan], errors=[0.2, 0.1, np.nan],
                       valid=[True, True, False])}):
        fig, (_, ratio_axis), _ = ab.artifact_comparison_1d(numerator, reference, **kwargs)
        steps = [p for p in ratio_axis.patches if isinstance(p, matplotlib.patches.StepPatch)]
        assert steps and all(step.get_data().baseline is None for step in steps)
        plt.close(fig)


def test_ratio_presentation_stacks_signed_components_from_one():
    numerator, reference = _histograms()
    presentation = ab.RatioPresentation(
        values=[2.0, 0.5, np.nan], errors=[0.2, 0.1, np.nan], valid=[True, True, False],
        error_style="bars",
        components={"$W$": [0.6, -0.2, 9.0], "$G$": [0.4, -0.3, 9.0], "$L$": [0.0, 0.0, 9.0]},
        component_colors={"$W$": "#E69F00", "$G$": "#009E73", "$L$": "#D55E00"},
    )
    fig, (main, ratio_axis), diagnostics = ab.artifact_comparison_1d(
        numerator, reference, ratio_presentation=presentation, legend_ncol=2,
    )
    rects = [r for r in ratio_axis.patches if isinstance(r, matplotlib.patches.Rectangle)]
    spans = {(round(r.get_x(), 9), round(r.get_y(), 9), round(r.get_y() + r.get_height(), 9))
             for r in rects if r.get_height() != 0}
    # Bin 0: W from 1 to 1.6, G from 1.6 to 2.0; bin 1: W from 1 to 0.8, G from 0.8 to 0.5.
    assert spans == {(0.0, 1.0, 1.6), (0.0, 1.6, 2.0), (1.0, 1.0, 0.8), (1.0, 0.8, 0.5)}
    assert not any(r.get_x() == 3.0 and r.get_height() != 0 for r in rects)  # invalid bin
    positive, negative = presentation.component_stacks()
    np.testing.assert_allclose(positive, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(negative, [0.0, -0.5, 0.0])
    labels = [t.get_text() for t in main.get_legend().get_texts()]
    assert labels[-3:] == ["$W$", "$G$", "$L$"]
    assert main.get_legend()._ncols == 2
    low, high = ratio_axis.get_ylim()
    assert low < 0.5 and high > 2.2
    np.testing.assert_array_equal(diagnostics.ratio_valid, [True, True, False])
    plt.close(fig)


def test_ratio_presentation_counts_stacks_beyond_the_window():
    numerator, reference = _histograms()
    presentation = ab.RatioPresentation(
        values=[1.1, 0.9, np.nan], errors=[0.05, 0.05, np.nan], valid=[True, True, False],
        components={"up": [0.6, 0.0, 0.0], "down": [-0.5, -0.1, 0.0]},
        component_colors={"up": "red", "down": "blue"},
    )
    fig, (_, ratio_axis), _ = ab.artifact_comparison_1d(
        numerator, reference, ratio_presentation=presentation, ratio_ylim=(0.8, 1.2),
    )
    text = ratio_axis.texts[0].get_text()
    assert "Below: 0; above: 0" in text and "stacks beyond: 1" in text
    markers = {line.get_marker() for line in ratio_axis.lines if len(line.get_xdata())}
    assert {"^", "v"} <= markers
    plt.close(fig)


def test_range_text_can_be_left_to_the_caller():
    numerator, reference = _histograms()
    presentation = ab.RatioPresentation(
        values=[1.1, 0.9, np.nan], errors=[0.05, 0.05, np.nan], valid=[True, True, False],
        components={"up": [0.6, 0.0, 0.0], "down": [-0.5, -0.1, 0.0]},
        component_colors={"up": "red", "down": "blue"},
    )
    fig, (_, ratio_axis), _ = ab.artifact_comparison_1d(
        numerator, reference, ratio_presentation=presentation, ratio_ylim=(0.8, 1.2),
        range_text=False,
    )
    assert len(ratio_axis.texts) == 0
    assert any(line.get_marker() == "^" for line in ratio_axis.lines)  # markers stay
    counts = ab.ratio_range_counts(presentation.values, (0.8, 1.2), valid=presentation.valid,
                                   stacks=presentation.component_stacks())
    assert counts == {"below": 0, "above": 0, "undefined": 1, "stacks_beyond": 1}
    assert ab.ratio_range_counts([2.0, 0.5, np.nan], (0.5, 1.5)) == {
        "below": 0, "above": 1, "undefined": 1, "stacks_beyond": 0}
    plt.close(fig)


def test_model_baseline_opens_the_main_panel_line():
    numerator, reference = _histograms()

    def main_step(**kwargs):
        fig, (main_axis, _), _ = ab.artifact_comparison_1d(numerator, reference, **kwargs)
        steps = [p for p in main_axis.patches if isinstance(p, matplotlib.patches.StepPatch)]
        plt.close(fig)
        return steps[0].get_data()

    assert main_step().baseline == 0  # count histograms keep the closed outline
    assert main_step(model_baseline=None).baseline is None
    efficiency = ab.Hist1D([0.9, 0.5, 0.25], [1.0, 2.0, 3.0, 4.0], errors=[0.1, 0.1, 0.1])
    for baseline, reaches_zero in ((0.0, True), (None, False)):
        fig, (main_axis, _), _ = ab.artifact_comparison_1d(
            efficiency, efficiency, model_baseline=baseline, points="reference")
        step = [p for p in main_axis.patches if isinstance(p, matplotlib.patches.StepPatch)][0]
        assert np.any(np.isclose(step.get_path().vertices[:, 1], 0.0)) == reaches_zero
        plt.close(fig)


def test_ratio_presentation_rejects_invalid_components():
    base = dict(values=[1.0], errors=[0.1], valid=[True])
    with pytest.raises(ValueError, match="error_style"):
        ab.RatioPresentation(**base, error_style="shade")
    with pytest.raises(ValueError, match="non-empty mapping"):
        ab.RatioPresentation(**base, components=[0.1], component_colors={})
    with pytest.raises(ValueError, match="map every component"):
        ab.RatioPresentation(**base, components={"a": [0.1]}, component_colors={"b": "red"})
    with pytest.raises(ValueError, match="match values"):
        ab.RatioPresentation(**base, components={"a": [0.1, 0.2]}, component_colors={"a": "red"})
    with pytest.raises(ValueError, match="components must be finite"):
        ab.RatioPresentation(**base, components={"a": [np.nan]}, component_colors={"a": "red"})
    with pytest.raises(ValueError, match="colour"):
        ab.RatioPresentation(**base, components={"a": [0.1]}, component_colors={"a": "nocolour"})
    with pytest.raises(ValueError, match="require components"):
        ab.RatioPresentation(**base, component_colors={"a": "red"})


def test_ratio_presentation_rejects_invalid_fields():
    with pytest.raises(ValueError, match="one-dimensional"):
        ab.RatioPresentation(values=[[1.0]], errors=[0.1], valid=[True])
    with pytest.raises(ValueError, match="share one shape"):
        ab.RatioPresentation(values=[1.0], errors=[0.1, 0.2], valid=[True])
    with pytest.raises(ValueError, match="finite"):
        ab.RatioPresentation(values=[np.nan], errors=[0.1], valid=[True])
    with pytest.raises(ValueError, match="nonnegative"):
        ab.RatioPresentation(values=[1.0], errors=[-0.1], valid=[True])
    with pytest.raises(ValueError, match="unity_errors must match"):
        ab.RatioPresentation(values=[1.0], errors=[0.1], valid=[True], unity_errors=[0.1, 0.2])
    # Invalid bins may carry placeholders because they are never drawn.
    presentation = ab.RatioPresentation(
        values=[np.nan], errors=[np.nan], valid=[False],
    )
    assert np.isnan(presentation.values[0])

    numerator, reference = _histograms()
    with pytest.raises(ValueError, match="match the histogram bins"):
        ab.artifact_comparison_1d(
            numerator, reference,
            ratio_presentation=ab.RatioPresentation(
                values=[1.0, 1.0], errors=[0.1, 0.1], valid=[True, True],
            ),
        )
    with pytest.raises(TypeError, match="RatioPresentation"):
        ab.artifact_comparison_1d(numerator, reference, ratio_presentation=[1.0, 1.0, 1.0])


def _closure_histograms():
    edges = np.array([0.0, 1.0, 2.0, 4.0])
    baseline = ab.Hist1D([5.0, 4.0, 3.0], edges, errors=[1.0, 1.0, 1.0], label="All")
    retained = ab.Hist1D([3.0, 3.0, 2.0], edges, errors=[1.0, 1.0, 1.0], label="Retained")
    removed = ab.Hist1D([2.0, 1.0, 1.0], edges, errors=[0.5, 0.5, 0.5], label="Removed")
    return baseline, {"Retained": retained, "Removed": removed}


def test_artifact_closure_1d_renders_stairs_and_residual():
    baseline, components = _closure_histograms()

    fig, (main, residual) = ab.artifact_closure_1d(
        baseline, components,
        residual_values=[0.0, 0.0, 1.0], residual_errors=[0.0, 0.0, 0.1],
        residual_valid=[True, True, False], colors={"Removed": "peach"},
        atlas=True, atlas_label_kwargs={"lumi": False}, ysymlog_linthresh=1.0,
    )

    assert main.get_yscale() == "symlog"
    np.testing.assert_allclose(main.patches[0].get_data().values, components["Retained"].counts)
    np.testing.assert_allclose(main.patches[1].get_data().values, components["Removed"].counts)
    np.testing.assert_allclose(main.patches[2].get_data().values, baseline.counts)
    assert [patch.get_linestyle() for patch in main.patches] == ["-", "-", "--"]
    assert [patch.get_linewidth() for patch in main.patches] == [1.0, 1.0, 1.4]
    legend = [text.get_text() for text in main.get_legend().get_texts()]
    assert legend == ["Retained", "Removed", "Baseline"]
    assert "ATLAS" in main.texts[0].get_text()
    panel = residual.patches[0].get_data().values
    np.testing.assert_allclose(panel[:2], [0.0, 0.0])
    assert np.isnan(panel[2])
    np.testing.assert_allclose(residual.lines[0].get_ydata(), [0.0, 0.0])
    plt.close(fig)


def test_artifact_closure_1d_marks_out_of_range_residual():
    baseline, components = _closure_histograms()

    fig, (_, residual) = ab.artifact_closure_1d(
        baseline, components,
        residual_values=[2.0, 0.0, np.nan], residual_errors=[0.1, 0.1, 0.1],
        residual_valid=[True, True, False], residual_ylim=(-1.0, 1.0),
    )

    text = residual.texts[0].get_text()
    assert "above: 1" in text
    assert "undefined: 1" in text
    assert any(line.get_marker() == "^" for line in residual.lines)
    plt.close(fig)


def test_artifact_closure_1d_linestyles_keep_baseline_distinguishable():
    baseline, components = _closure_histograms()

    fig, (main, _) = ab.artifact_closure_1d(
        baseline, components,
        residual_values=[0.0, 0.0, 0.0], residual_errors=[0.0, 0.0, 0.0],
        baseline_linestyle=":", component_linestyles={"Removed": "-."},
        baseline_linewidth=2.0, component_linewidths={"Removed": 0.5},
    )

    assert [patch.get_linestyle() for patch in main.patches] == ["-", "-.", ":"]
    assert [patch.get_linewidth() for patch in main.patches] == [1.0, 0.5, 2.0]
    plt.close(fig)


def test_artifact_closure_1d_rejects_invalid_inputs():
    baseline, components = _closure_histograms()
    valid_residual = dict(residual_values=[0.0, 0.0, 0.0], residual_errors=[0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="non-empty"):
        ab.artifact_closure_1d(baseline, {}, **valid_residual)
    with pytest.raises(TypeError, match="Hist1D baseline"):
        ab.artifact_closure_1d("baseline", components, **valid_residual)
    mismatched = ab.Hist1D([1.0, 1.0, 1.0], [0.0, 1.0, 3.0, 4.0], errors=[0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="share the baseline edges"):
        ab.artifact_closure_1d(baseline, {"Other": mismatched}, **valid_residual)
    with pytest.raises(ValueError, match="raw-count and density"):
        ab.artifact_closure_1d(
            ab.divide_by_bin_width(baseline), components, **valid_residual,
        )
    with pytest.raises(ValueError, match="match the baseline bins"):
        ab.artifact_closure_1d(
            baseline, components, residual_values=[0.0, 0.0], residual_errors=[0.0, 0.0],
        )
    with pytest.raises(ValueError, match="nonnegative"):
        ab.artifact_closure_1d(
            baseline, components, residual_values=[0.0, 0.0, 0.0],
            residual_errors=[0.0, -1.0, 0.0],
        )
    with pytest.raises(ValueError, match="invalid Matplotlib color"):
        ab.artifact_closure_1d(
            baseline, components, colors={"Removed": "not-a-colour"}, **valid_residual,
        )
    with pytest.raises(ValueError, match="unknown labels"):
        ab.artifact_closure_1d(
            baseline, components, colors={"Missing": "red"}, **valid_residual,
        )
    with pytest.raises(ValueError, match="component_linestyles has unknown labels"):
        ab.artifact_closure_1d(
            baseline, components, component_linestyles={"Missing": ":"}, **valid_residual,
        )
    with pytest.raises(ValueError, match="invalid Matplotlib linestyle"):
        ab.artifact_closure_1d(
            baseline, components, baseline_linestyle="bogus", **valid_residual,
        )
    with pytest.raises(TypeError, match="component_linestyles must be a mapping"):
        ab.artifact_closure_1d(
            baseline, components, component_linestyles=[":"], **valid_residual,
        )
    with pytest.raises(ValueError, match="component_linewidths has unknown labels"):
        ab.artifact_closure_1d(
            baseline, components, component_linewidths={"Missing": 2.0}, **valid_residual,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        ab.artifact_closure_1d(
            baseline, components, baseline_linewidth=0.0, **valid_residual,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        ab.artifact_closure_1d(
            baseline, components, component_linewidths={"Removed": np.inf}, **valid_residual,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        ab.artifact_closure_1d(
            baseline, components, component_linewidths={"Removed": "thick"}, **valid_residual,
        )
    with pytest.raises(TypeError, match="component_linewidths must be a mapping"):
        ab.artifact_closure_1d(
            baseline, components, component_linewidths=[1.0], **valid_residual,
        )
    with pytest.raises(ValueError, match="increasing finite"):
        ab.artifact_closure_1d(baseline, components, residual_ylim=(1.0, 0.5), **valid_residual)
    with pytest.raises(ValueError, match="exclusive of ylog"):
        ab.artifact_closure_1d(
            baseline, components, ylog=True, ysymlog_linthresh=1.0, **valid_residual,
        )
    negative = ab.Hist1D([-1.0, 2.0, 3.0], baseline.edges, errors=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="negative"):
        ab.artifact_closure_1d(negative, components, ylog=True, **valid_residual)


def test_write_compact_plotbook_composition(tmp_path):
    book = tmp_path / "book.tex"
    ab.write_compact_plotbook(
        book,
        {
            "Particle": [(f"{index}.pdf", f"Jet {index}") for index in range(6)],
            "Reco": [("r.pdf", "Reco")],
        },
    )
    text = book.read_text()
    assert text.count("\\clearpage") == 2
    assert text.count("\\includegraphics") == 7
    assert "Jet 0" in text and "Reco" in text

    dense = tmp_path / "dense.tex"
    ab.write_compact_plotbook(
        dense,
        {
            "Plots": [(f"{index}.pdf", "x") for index in range(8)],
            "Native": [(f"n{index}.pdf", "x") for index in range(6)],
        },
        rows_per_page=3, section_rows={"Native": 2},
    )
    dense_text = dense.read_text()
    assert dense_text.count("\\clearpage") == 3
    assert dense_text.count("height=0.27") == 8
    assert dense_text.count("height=0.405") == 6

    with pytest.raises(ValueError, match="two or three"):
        ab.write_compact_plotbook(tmp_path / "bad.tex", {"Plots": []}, rows_per_page=4)


def test_plotbook_context_sets_and_restores_readable_type():
    before = mpl.rcParams["font.size"]
    assert before != 16

    with ab.plotbook_context():
        assert mpl.rcParams["font.size"] == 16
        assert mpl.rcParams["axes.labelsize"] == 16
        assert mpl.rcParams["xtick.labelsize"] == 13
        assert mpl.rcParams["legend.fontsize"] == 13

    assert mpl.rcParams["font.size"] == before
