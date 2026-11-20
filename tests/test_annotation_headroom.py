"""Regression tests for final annotation headroom and artist geometry."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import analyb as ab

def _content_fraction(ax, value):
    """Fractional height of ``value`` in the axis, measured after the transform."""
    lower, upper = ax.get_ylim()
    transform = ax.yaxis.get_transform()
    lo, hi, v = transform.transform([lower, upper, value])
    return float((v - lo) / (hi - lo))


def _tall_left_hist(n=12):
    """Counts that peak under the label's column, so the label needs headroom."""
    edges = np.linspace(0.0, 120.0, n + 1)
    counts = np.linspace(900.0, 30.0, n)
    return ab.Hist1D(counts, edges, errors=np.sqrt(counts))


@pytest.mark.parametrize("ylog", [False, True])
@pytest.mark.parametrize("atlas", [False, True])
def test_plotter_fits_the_top_to_its_label_and_legend(ylog, atlas):
    fig, ax = ab.hist1d_plot(_tall_left_hist(), "x [GeV]", label="sample", atlas=atlas, ylog=ylog)
    findings = ab.check_figure(fig)
    assert not [f for f in findings if f.check in ("text-over-data", "text-overlap")], findings
    limits = ax.get_ylim()
    ab.fit_overlay_headroom(ax)   # the plotter's fit is final: fitting again moves nothing
    assert ax.get_ylim() == pytest.approx(limits, rel=1e-6)
    plt.close(fig)


def test_symlog_plotter_keeps_its_label_clear_of_the_data():
    hist = _tall_left_hist()
    fig, axes, _ = ab.artifact_comparison_1d(hist, hist.copy(), atlas=True, ysymlog_linthresh=10.0)
    main = axes[0]
    assert main.get_yscale() == "symlog"
    _, label_y0, _, _ = ab.artist_axes_bounds(main.texts[0], main)
    assert _content_fraction(main, float((hist.counts + hist.errors).max())) < label_y0
    plt.close(fig)


def test_artist_axes_bounds_reports_text_and_legend():
    hist = ab.hist1d(np.linspace(0, 100, 200), bins=10, range=(0, 100))
    fig, ax = ab.hist1d_plot(hist, variable="met", label="Data", atlas=True)
    fig.canvas.draw()

    label = next(t for t in ax.texts if "ATLAS" in t.get_text())
    text_bounds = ab.artist_axes_bounds(label)
    legend_bounds = ab.artist_axes_bounds(ax.get_legend(), ax)

    assert text_bounds[0] == pytest.approx(0.04, abs=0.02)
    assert text_bounds[3] <= 1.0
    assert 0.0 <= legend_bounds[0] < legend_bounds[2] <= 1.0
    assert 0.0 <= legend_bounds[1] < legend_bounds[3] <= 1.0
    plt.close(fig)


def test_artist_axes_bounds_defaults_to_artist_axes():
    fig, ax = plt.subplots()
    ab.add_atlas_label(ax, lumi=False)
    fig.canvas.draw()

    from_axes = ab.artist_axes_bounds(ax.texts[-1])
    explicit = ab.artist_axes_bounds(ax.texts[-1], ax)

    assert from_axes == pytest.approx(explicit)
    plt.close(fig)


def test_artist_axes_bounds_works_on_pdf_and_svg_canvases():
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    for canvas_cls in (FigureCanvasPdf, FigureCanvasSVG):
        fig = Figure(figsize=(4, 3))
        canvas_cls(fig)
        ax = fig.add_subplot()
        ab.add_atlas_label(ax, lumi=False)

        bounds = ab.artist_axes_bounds(ax.texts[-1], ax)

        assert bounds is not None
        assert bounds[0] == pytest.approx(0.04, abs=0.02)
        assert bounds[3] <= 1.0


def test_artist_axes_bounds_tracks_layout_after_size_change():
    """Geometry is recomputed from the drawn layout, not cached from a first draw."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ab.add_atlas_label(ax, lumi=False)

    first = ab.artist_axes_bounds(ax.texts[-1], ax)
    fig.set_size_inches(8, 3)
    second = ab.artist_axes_bounds(ax.texts[-1], ax)

    assert first != second
    assert second[0] == pytest.approx(0.04, abs=0.02)
    plt.close(fig)


def test_caller_limits_stand_unless_the_label_would_cover_data():
    centers = np.array([0.5, 1.5, 2.5, 3.5])
    eff = np.array([0.98, 0.97, 0.6, 0.5])
    err = np.full(4, 0.01)
    fig, (main, _) = ab.efficiency_plot(centers, eff, err, err, reference=(centers, eff, err, err),
                                        ylim=(0.0, 1.05), atlas=True)
    assert main.get_ylim()[1] > 1.05   # raised: the label sits over the efficiency plateau
    assert not [f for f in ab.check_figure(fig) if f.check == "text-over-data"]
    plt.close(fig)
    fig, (main, _) = ab.efficiency_plot(centers, eff * 0.5, err, err, reference=(centers, eff * 0.5, err, err),
                                        ylim=(0.0, 1.05), atlas=True)
    assert main.get_ylim() == (0.0, 1.05)   # nothing covered: the caller's range stands
    plt.close(fig)
