"""Render-time figure lint on synthetic figures with known defects."""

import json
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import analyb as ab
from analyb import config


@pytest.fixture(autouse=True)
def _analyb_style_on_matplotlib_defaults():
    """Measure against analyb's style alone, not a host's personal matplotlibrc."""
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        ab.style.apply()
        yield
    config.set_figure_check(None)
    config.set_figure_final_width_cm(None)
    plt.close("all")


def _source_width_cm(fig):
    return fig.get_figwidth() * 2.54


def _by_check(findings, check, severity=None):
    return [f for f in findings if f.check == check and (severity is None or f.severity == severity)]


def _histograms():
    rng = np.random.default_rng(7)
    edges = np.linspace(0, 200, 21)
    mc = ab.hist1d(rng.exponential(50, 5000), bins=edges, unit=r"\GeV")
    data = ab.hist1d(rng.exponential(50, 5000), bins=edges, unit=r"\GeV")
    return mc, data


def _bare_axes(figsize=(4, 3)):
    """Axes without a layout engine, so defects are not repaired by layout."""
    fig, ax = plt.subplots(figsize=figsize, layout="none")
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    # Keep the first ticks off the corner, where x and y tick labels meet.
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    return fig, ax


# --------------------------------------------------------------------------
# Standard analyb plots
# --------------------------------------------------------------------------

def _standard_plots():
    mc, data = _histograms()
    yield ab.hist1d_with_ratio(data, mc, xlabel=r"$p_{\mathrm{T}}$ [GeV]", atlas=True)[0]
    yield ab.hist1d_stacked(
        {"A": mc, "B": data}, data_hist=data + mc, xlabel=r"$m_{bb}$ [GeV]", atlas=True,
    )[0]


def test_standard_plots_have_no_geometry_failures():
    for fig in _standard_plots():
        findings = ab.check_figure(fig)
        assert [f for f in findings if f.severity == "fail"] == []


def test_standard_plots_meet_font_floor_at_source_size():
    for fig in _standard_plots():
        findings = ab.check_figure(fig, final_width_cm=_source_width_cm(fig))
        assert _by_check(findings, "font-below-floor") == []


def test_standard_plot_x_small_legends_are_flagged_at_source_size():
    # fontsize="x-small" is 6.94 pt with the default 10 pt font.size.
    for fig in _standard_plots():
        small = ab.check_figure(fig, final_width_cm=_source_width_cm(fig))
        assert small and all(f.check == "font-below-advisory" for f in small)
        assert all(" legend text " in f.where for f in small)
        assert all(f.value == pytest.approx(6.94, abs=0.01) for f in small)


def test_clean_bare_figure_has_no_findings():
    fig, ax = _bare_axes()
    ax.plot([0, 1], [0, 1], label="line")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right")
    ax.text(0.5, 0.5, "", transform=ax.transAxes)  # empty strings are ignored
    hidden = ax.text(0.05, 0.9, "hidden", transform=ax.transAxes)
    hidden.set_visible(False)

    assert ab.check_figure(fig) == []


# --------------------------------------------------------------------------
# Known defects
# --------------------------------------------------------------------------


def test_axis_with_tick_labels_but_no_label_fails():
    fig, ax = _bare_axes()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x")
    findings = _by_check(ab.check_figure(fig), "axis-unlabelled", "fail")
    assert [f.where for f in findings] == ["axes[0] y axis"]


def test_shared_x_without_tick_labels_needs_no_label_and_hidden_axes_are_skipped():
    fig, (top, bottom, key) = plt.subplots(3, 1, sharex=True, layout="constrained")
    for ax in (top, bottom):
        ax.plot([0, 1], [0, 1])
        ax.set_ylabel("y")
    bottom.set_xlabel("x")
    key.axis("off")
    assert _by_check(ab.check_figure(fig), "axis-unlabelled") == []


def test_sidecar_lists_the_checks_run(tmp_path):
    fig, ax = _bare_axes()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ab.save_figure(fig, "checks", script_id="t", save_path=str(tmp_path), figure_check="warn")
    record = json.loads((tmp_path / "t-checks.figcheck.json").read_text())
    assert "axis-unlabelled" in record["checks"]
    assert not any(c.startswith("font-below") for c in record["checks"])


def test_legend_anchored_outside_canvas_fails():
    fig, ax = _bare_axes()
    ax.plot([0, 1], [0, 1], label="a long legend entry")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    clipped = _by_check(ab.check_figure(fig), "text-outside-canvas")

    assert any(f.where == "axes[0] legend" for f in clipped)
    assert any("a long legend entry" in f.where for f in clipped)
    assert all(f.severity == "fail" and f.value > 0 for f in clipped)
    assert all("right" in f.message for f in clipped)


def test_figure_legend_clipped_at_top_fails():
    fig, ax = _bare_axes()
    ax.plot([0, 1], [0, 1], label="model")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06))

    clipped = _by_check(ab.check_figure(fig), "text-outside-canvas")

    assert any(f.where == "figure legend" for f in clipped)
    assert all("top" in f.message for f in clipped)
    x0, y0, x1, y1 = next(f.bbox for f in clipped if f.where == "figure legend")
    assert y1 > fig.get_figheight()  # bbox is in figure inches


def test_last_tick_label_truncated_at_right_edge_fails():
    fig, ax = _bare_axes()
    fig.subplots_adjust(right=0.99)
    ax.plot([0, 1000], [0, 1])
    ax.set_xlim(0, 1000)
    ax.set_xticks([0, 500, 1000])

    clipped = _by_check(ab.check_figure(fig), "text-outside-canvas")

    assert [f.where for f in clipped] == ["axes[0] xtick label '1000'"]


def test_tick_labels_outside_view_limits_are_ignored():
    fig, ax = _bare_axes()
    fig.subplots_adjust(right=0.99)
    ax.plot([0, 1000], [0, 1])
    ax.set_xticks([0, 500, 1000, 1500])
    ax.set_xlim(0, 900)

    assert _by_check(ab.check_figure(fig), "text-outside-canvas") == []


def test_overlapping_tick_labels_fail():
    fig, ax = _bare_axes(figsize=(3, 2))
    ticks = np.arange(0, 1001, 50)
    ax.set_xticks(ticks, [f"{t:,}.00" for t in ticks])
    ax.set_xlim(0, 1000)

    overlaps = _by_check(ab.check_figure(fig), "text-overlap")

    assert any("xtick label '50.00' x axes[0] xtick label '100.00'" in f.where for f in overlaps)


def test_rotated_tick_labels_overlap_only_when_the_rotated_text_does():
    stages = ["initial", "stage A", "synthetic selection", "synthetic threshold", "final filter"]
    fig, ax = _bare_axes(figsize=(5, 3))
    fig.subplots_adjust(bottom=0.35)
    ax.set_xticks(range(len(stages)), stages, rotation=28, ha="right")
    ax.set_xlim(-0.5, len(stages) - 0.5)
    assert _by_check(ab.check_figure(fig), "text-overlap") == []

    ax.set_xticks(np.arange(len(stages)) * 0.2, stages, rotation=28, ha="right")
    overlaps = _by_check(ab.check_figure(fig), "text-overlap")
    assert any("'stage A'" in f.where and "'synthetic selection'" in f.where for f in overlaps)
    assert all("rotated" in f.message for f in overlaps if "xtick" in f.where)
    assert ax.get_xticklabels()[0].get_rotation() == 28  # measurement restores rotation


def test_tick_labels_of_stacked_panels_collide():
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(4, 4), layout="none")
    fig.subplots_adjust(hspace=0.0)
    top.set_ylim(0, 10)
    bottom.set_ylim(0.5, 1.5)
    top.set_yticks([0, 5, 10])
    bottom.set_yticks([0.5, 1.0, 1.5])

    overlaps = _by_check(ab.check_figure(fig), "text-overlap")

    assert any(
        "axes[0] ytick label '0'" in f.where and "axes[1] ytick label '1.5'" in f.where
        for f in overlaps
    )


def test_two_overlapping_texts_fail_and_text_is_never_paired_with_itself():
    fig, ax = _bare_axes()
    ax.text(0.3, 0.5, "first label", transform=ax.transAxes)
    ax.text(0.32, 0.5, "second label", transform=ax.transAxes)

    overlaps = _by_check(ab.check_figure(fig), "text-overlap")

    assert len(overlaps) == 1
    assert "first label" in overlaps[0].where and "second label" in overlaps[0].where
    assert overlaps[0].value > 0


def test_legend_entries_are_not_paired_with_their_own_legend():
    fig, ax = _bare_axes()
    ax.plot([0, 1], [0, 0], label="one")
    ax.plot([0, 1], [0.1, 0.1], label="two")
    ax.set_ylim(0, 2)
    ax.legend(loc="upper right", frameon=True)

    assert _by_check(ab.check_figure(fig), "text-overlap") == []


def test_overlapping_entries_inside_one_legend_fail():
    fig, ax = _bare_axes()
    ax.plot([0, 1], [0, 0], label="first entry")
    ax.plot([0, 1], [0.1, 0.1], label="second entry")
    ax.legend(loc="center", labelspacing=-1.5)

    overlaps = _by_check(ab.check_figure(fig), "text-overlap")

    assert len(overlaps) == 1
    assert overlaps[0].where == (
        "axes[0] legend text 'first entry' x axes[0] legend text 'second entry'"
    )


def test_legend_overlapping_atlas_label_fails():
    fig, ax = _bare_axes()
    ax.plot([0, 1], [0, 0], label="a very long legend entry label")
    ax.set_ylim(0, 1)
    ab.add_atlas_label(ax, lumi=False)
    ax.legend(loc="upper left")

    overlaps = _by_check(ab.check_figure(fig), "text-overlap")

    assert any("axes[0] legend" in f.where and "ATLAS" in f.where for f in overlaps)


def test_legend_sitting_on_a_line_fails_with_vertex_count():
    fig, ax = _bare_axes()
    x = np.linspace(0, 1, 101)
    ax.plot(x, np.full_like(x, 0.5), label="flat model")
    ax.set_ylim(0, 1)
    ax.legend(loc="center")

    covered = _by_check(ab.check_figure(fig), "text-over-data")

    assert len(covered) == 1
    assert covered[0].severity == "fail"
    assert covered[0].where == "axes[0] legend"
    assert covered[0].value > 0
    assert "flat model" in covered[0].message


def test_line_segment_crossing_a_legend_without_vertices_fails():
    fig, ax = _bare_axes()
    ax.plot([0, 1], [0.5, 0.5], label="two-point line")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="center")

    covered = _by_check(ab.check_figure(fig), "text-over-data", "fail")

    assert len(covered) == 1
    assert covered[0].value == 0
    assert "crossed by 1" in covered[0].message


def test_atlas_label_over_markers_fails():
    fig, ax = _bare_axes()
    x = np.linspace(0.02, 0.4, 10)
    y = np.full_like(x, 9.0)
    ax.errorbar(x, y, yerr=0.3, fmt="o", color="black", label="Data")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    ab.add_atlas_label(ax, lumi=False)

    covered = _by_check(ab.check_figure(fig), "text-over-data", "fail")

    assert len(covered) == 1
    assert "ATLAS" in covered[0].where
    assert covered[0].value >= 1
    assert "Data" in covered[0].message  # errorbar container label


@pytest.mark.parametrize("draw", [
    lambda ax: ax.plot([0, 1], [0.5, 0.5], alpha=0),
    lambda ax: ax.plot([0, 1], [0.5, 0.5], color=(0, 0, 0, 0)),
    lambda ax: ax.plot([0, 1], [0.5, 0.5], linewidth=0),
    lambda ax: ax.plot([0, 1], [0.5, 0.5], linestyle="none", marker="o", markersize=0),
    lambda ax: ax.plot([0, 1], [0.5, 0.5], linestyle="none", marker="o",
                       markerfacecolor="none", markeredgewidth=0),
    lambda ax: ax.plot([0, 1], [0.5, 0.5])[0].set_visible(False),
    lambda ax: ax.scatter([0.5], [0.5], s=0),
    lambda ax: ax.scatter([0.5], [0.5], alpha=0),
    lambda ax: ax.scatter([0.5], [0.5], facecolors="none", edgecolors="none"),
    lambda ax: ax.fill_between([0, 1], [0.4, 0.4], [0.6, 0.6], alpha=0),
    lambda ax: ax.fill_between([0, 1], [0.4, 0.4], [0.6, 0.6], facecolor="none",
                               edgecolor="none"),
    lambda ax: ax.vlines([0.5], 0, 1, linewidth=0),
    lambda ax: ax.stairs([0.6], [0, 1], fill=True, alpha=0),
    lambda ax: ax.stairs([0.6], [0, 1], linewidth=0),
], ids=[
    "line-alpha0", "line-transparent-color", "line-zero-width", "marker-size0",
    "marker-no-face-no-edge", "line-invisible", "scatter-s0", "scatter-alpha0",
    "scatter-no-face-no-edge", "band-alpha0", "band-transparent", "vlines-zero-width",
    "stairs-filled-alpha0", "stairs-zero-width",
])
def test_artists_that_paint_nothing_are_not_covered(draw):
    fig, ax = _bare_axes()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    draw(ax)
    ax.text(0.5, 0.5, "label over invisible data", ha="center", va="center")

    assert _by_check(ab.check_figure(fig), "text-over-data") == []


@pytest.mark.parametrize("draw", [
    lambda ax: ax.plot([0, 1], [0.5, 0.5], color=(0, 0, 0, 0), alpha=1),
    lambda ax: ax.plot([0.5], [0.5], linestyle="none", marker="o", markersize=8,
                       markerfacecolor=(0, 0, 0, 0), markeredgewidth=0, alpha=1),
], ids=["line-transparent-color-alpha1", "marker-transparent-face-alpha1"])
def test_explicit_alpha_overrides_transparent_colour(draw):
    # matplotlib draws these opaque: the artist alpha replaces the RGBA alpha.
    fig, ax = _bare_axes()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    draw(ax)
    ax.text(0.5, 0.5, "label over painted data", ha="center", va="center")

    assert len(_by_check(ab.check_figure(fig), "text-over-data", "fail")) == 1


def test_none_marker_face_stays_unpainted_with_explicit_alpha():
    fig, ax = _bare_axes()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0.5], [0.5], linestyle="none", marker="o", markerfacecolor="none",
            markeredgewidth=0, alpha=1)
    ax.text(0.5, 0.5, "label", ha="center", va="center")

    assert _by_check(ab.check_figure(fig), "text-over-data") == []


def test_per_point_scatter_alpha_counts_only_opaque_points():
    fig, ax = _bare_axes()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.scatter([0.45, 0.55], [0.5, 0.5], alpha=np.array([0.0, 1.0]))
    ax.text(0.5, 0.5, "label over two points", ha="center", va="center")

    (covered,) = _by_check(ab.check_figure(fig), "text-over-data")
    assert covered.value == 1


def test_partly_painted_scatter_counts_only_painted_points():
    fig, ax = _bare_axes()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.scatter([0.45, 0.5, 0.55], [0.5, 0.5, 0.5], s=[0, 20, 0])
    ax.text(0.5, 0.5, "label over one point", ha="center", va="center")

    (covered,) = _by_check(ab.check_figure(fig), "text-over-data")
    assert covered.value == 1


def test_data_clipped_outside_the_axes_is_not_covered():
    fig, ax = _bare_axes()
    ax.plot([0, 1], [5, 5], marker="o", label="off-axis")
    ax.set_ylim(0, 1)
    ax.text(0.5, 1.02, "caption above axes", transform=ax.transAxes)

    assert _by_check(ab.check_figure(fig), "text-over-data") == []


def test_legend_over_filled_area_is_only_flagged():
    fig, ax = _bare_axes()
    ax.stairs([1.0, 1.0, 1.0], [0, 1, 2, 3], fill=True, label="filled step")
    ax.fill_between([0, 3], [0, 0], [1, 1], alpha=0.2)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.5)
    ax.legend(loc="center")

    covered = _by_check(ab.check_figure(fig), "text-over-data")

    assert [f.severity for f in covered] == ["flag"]
    assert "filled step" in covered[0].message


def test_font_below_floor_at_final_width_fails():
    fig, ax = _bare_axes(figsize=(6, 4))
    ax.set_xlabel("observable", fontsize=10)

    def xlabel_findings(width):
        return [f for f in ab.check_figure(fig, final_width_cm=width) if "xlabel" in f.where]

    source = 6 * 2.54
    (fail,) = xlabel_findings(0.4 * source)  # 4 pt
    assert (fail.check, fail.severity) == ("font-below-floor", "fail")
    assert fail.value == pytest.approx(4.0)
    (flag,) = xlabel_findings(0.6 * source)  # 6 pt
    assert (flag.check, flag.severity) == ("font-below-advisory", "flag")
    assert flag.value == pytest.approx(6.0)
    assert xlabel_findings(0.7 * source) == []  # 7 pt
    assert [f for f in ab.check_figure(fig) if f.check.startswith("font-")] == []


def test_mathtext_label_is_measured_at_its_base_size():
    fig, ax = _bare_axes(figsize=(6, 4))
    ax.set_xlabel(r"$p_{\mathrm{T}}^{\ell_{1}}$ [GeV]", fontsize=10)
    ax.tick_params(labelsize=12)

    source = 6 * 2.54
    at_75_percent = ab.check_figure(fig, final_width_cm=0.75 * source)
    assert [f for f in at_75_percent if f.check.startswith("font-")] == []

    at_40_percent = ab.check_figure(fig, final_width_cm=0.4 * source)
    xlabel = [f for f in at_40_percent if "xlabel" in f.where]
    assert [f.check for f in xlabel] == ["font-below-floor"]
    assert xlabel[0].value == pytest.approx(4.0)


def test_invalid_arguments_are_rejected():
    fig, _ = _bare_axes()
    with pytest.raises(ValueError):
        ab.check_figure(fig, final_width_cm=0)
    with pytest.raises(ValueError):
        ab.check_figure(fig, min_font_pt=-1)
    with pytest.raises(ValueError):
        ab.check_figure(fig, min_font_pt=7.0, flag_font_pt=5.0)
    with pytest.raises(ValueError):
        ab.set_figure_check("strict")
    with pytest.raises(ValueError):
        ab.set_figure_final_width_cm(float("nan"))


# --------------------------------------------------------------------------
# Save hook
# --------------------------------------------------------------------------

def _defective_figure():
    fig, ax = _bare_axes()
    ax.plot([0, 1], [0.5, 0.5], label="model")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_xlabel("x", fontsize=8)
    return fig


def test_save_hook_is_off_by_default(tmp_path):
    config.set_save_png(False)
    try:
        ab.save_figure(_defective_figure(), "off", script_id="t", save_path=str(tmp_path))
    finally:
        config.set_save_png(True)

    assert (tmp_path / "t-off.pdf").exists()
    assert not (tmp_path / "t-off.figcheck.json").exists()


def test_save_hook_warn_logs_and_writes_sidecar(tmp_path, caplog):
    config.set_figure_check("warn")
    config.set_figure_final_width_cm(3.0)
    fig = _defective_figure()
    with caplog.at_level(logging.WARNING, logger="analyb.figcheck"):
        ab.save_figure(fig, "warn", script_id="t", save_path=str(tmp_path))

    assert not plt.fignum_exists(fig.number)
    record = json.loads((tmp_path / "t-warn.figcheck.json").read_text())
    assert record["pdf"] == "t-warn.pdf"
    assert len(record["pdf_sha256"]) == 64
    assert record["figure_size_in"] == [4.0, 3.0]
    assert record["final_width_cm"] == 3.0
    assert record["min_font_pt"] == 5.0
    assert record["flag_font_pt"] == 7.0
    assert record["checks_skipped"] == []
    checks = {f["check"] for f in record["findings"]}
    assert {"text-outside-canvas", "font-below-floor"} <= checks
    assert set(record["findings"][0]) == {"check", "severity", "where", "bbox", "value", "message"}
    assert len(caplog.records) == len(record["findings"])


def test_save_hook_raise_saves_first_then_raises(tmp_path):
    config.set_figure_check("raise")
    fig = _defective_figure()
    with pytest.raises(ab.FigureCheckError) as excinfo:
        ab.save_figure(fig, "raise", script_id="t", save_path=str(tmp_path))

    assert (tmp_path / "t-raise.pdf").exists()
    assert (tmp_path / "t-raise.figcheck.json").exists()
    assert not plt.fignum_exists(fig.number)
    assert "text-outside-canvas" in str(excinfo.value)
    assert all(f.severity in {"fail", "flag"} for f in excinfo.value.findings)
    record = json.loads((tmp_path / "t-raise.figcheck.json").read_text())
    assert record["final_width_cm"] is None
    assert record["checks_skipped"] == ["font-below-floor", "font-below-advisory"]


def test_save_hook_raise_passes_clean_figure_and_per_call_override(tmp_path):
    config.set_figure_check("raise")
    fig, ax = _bare_axes()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ab.save_figure(fig, "clean", script_id="t", save_path=str(tmp_path))
    assert json.loads((tmp_path / "t-clean.figcheck.json").read_text())["findings"] == []

    ab.save_figure(_defective_figure(), "skip", script_id="t", save_path=str(tmp_path),
                   figure_check=False)
    assert not (tmp_path / "t-skip.figcheck.json").exists()

    config.set_figure_check(None)
    ab.save_figure(_defective_figure(), "percall", script_id="t", save_path=str(tmp_path),
                   figure_check="warn", final_width_cm=4.0)
    record = json.loads((tmp_path / "t-percall.figcheck.json").read_text())
    assert record["final_width_cm"] == 4.0


def test_plot_helper_save_uses_configured_mode(tmp_path):
    config.set_figure_check("warn")
    mc, data = _histograms()
    ab.hist1d_with_ratio(data, mc, xlabel="x [GeV]", atlas=True, save_name="ratio",
                         script_id="t", save_path=str(tmp_path))

    record = json.loads((tmp_path / "t-ratio.figcheck.json").read_text())
    assert [f for f in record["findings"] if f["severity"] == "fail"] == []


def test_save_hook_raise_ignores_flags(tmp_path):
    fig, ax = _bare_axes()
    ax.stairs([1.0, 1.0, 1.0], [0, 1, 2, 3], fill=True, label="filled step")
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(-0.1, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="center")

    ab.save_figure(fig, "flag", script_id="t", save_path=str(tmp_path), figure_check="raise")

    record = json.loads((tmp_path / "t-flag.figcheck.json").read_text())
    assert [f["severity"] for f in record["findings"]] == ["flag"]


# write_compact_plotbook placement, measured with pdflatex for its preamble:
# 0.485\textwidth = 256.67 TeX pt, \textheight = 776.76 TeX pt (72.27 per inch).
_PLOTBOOK_WIDTH_IN = 256.67264 / 72.27
_PLOTBOOK_HEIGHT_CAP_IN = {2: 0.405 * 776.76025 / 72.27, 3: 0.27 * 776.76025 / 72.27}


@pytest.mark.parametrize("rows", [2, 3])
def test_native_cutflow_labels_meet_floor_at_compact_plotbook_placement(rows):
    from analyb.cutflow import native_cutflow_chart

    stages = ["synthetic_initial_quality", "synthetic_long_stage_label_one",
              "synthetic_long_stage_label_two", "synthetic_candidate_requirement",
              "synthetic_final_stage_requirement"]
    with ab.plotbook_context():
        fig, ax = native_cutflow_chart(stages, np.linspace(1e6, 1e5, len(stages)), label="FF")
        width, height = fig.get_size_inches()
        scale = min(_PLOTBOOK_WIDTH_IN / width, _PLOTBOOK_HEIGHT_CAP_IN[rows] / height)
        findings = ab.check_figure(fig, final_width_cm=width * scale * 2.54)

    assert ax.get_yticklabels()[0].get_fontsize() == 13  # plotbook ytick.labelsize
    assert [f for f in findings if f.severity == "fail"] == []
