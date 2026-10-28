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


def test_reserve_annotation_headroom_linear_places_content_at_seventy_percent():
    fig, ax = plt.subplots()
    values = np.array([10.0, 100.0])
    errors = np.array([1.0, 5.0])
    ax.set_ylim(0, 100)

    assert ab.reserve_annotation_headroom(ax, (values, errors)) is True

    # Top of the explicit envelope (105) sits 30% below the new ceiling.
    assert _content_fraction(ax, 105.0) == pytest.approx(0.70)
    assert ax.get_ylim()[0] == 0.0
    plt.close(fig)


def test_late_limit_change_overwrites_headroom_and_helper_restores_it():
    """Reproduce the demonstrated last-limit-overwrite ordering failure."""
    fig, ax = plt.subplots()
    values = np.array([100.0])
    errors = np.array([0.0])
    ax.set_ylim(0, 100)

    ab.reserve_atlas_label_headroom(ax)
    assert ax.get_ylim()[1] == pytest.approx(100 / 0.7)

    # A later overlay or final scale/bounds call replaces the reserved limits.
    ax.set_ylim(0, 100)
    assert _content_fraction(ax, 100.0) == pytest.approx(1.0)

    # Re-reserving after that last limit is what keeps the annotation clear.
    assert ab.reserve_annotation_headroom(ax, (values, errors)) is True
    assert _content_fraction(ax, 100.0) == pytest.approx(0.70)
    plt.close(fig)


def test_reserve_annotation_headroom_log_scale_reserves_decades():
    fig, ax = plt.subplots()
    values = np.array([10.0, 100.0])
    errors = np.array([0.0, 0.0])
    ax.set_yscale("log")
    ax.set_ylim(1, 100)

    assert ab.reserve_annotation_headroom(ax, (values, errors)) is True

    lower, upper = ax.get_ylim()
    assert lower == 1.0
    assert np.log10(upper) == pytest.approx(np.log10(100.0) / 0.7)
    assert _content_fraction(ax, 100.0) == pytest.approx(0.70)
    plt.close(fig)


def test_reserve_annotation_headroom_symlog_scale_uses_transform():
    fig, ax = plt.subplots()
    values = np.array([-1.0, 50.0])
    errors = np.array([0.0, 0.0])
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-10, 50)

    assert ab.reserve_annotation_headroom(ax, (values, errors)) is True

    assert _content_fraction(ax, 50.0) == pytest.approx(0.70)
    assert ax.get_ylim()[0] == -10.0
    plt.close(fig)


def test_reserve_annotation_headroom_keeps_limits_when_content_clears():
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1000)

    assert ab.reserve_annotation_headroom(
        ax, (np.array([50.0]), np.array([0.0])),
    ) is False

    assert ax.get_ylim() == (0, 1000)
    plt.close(fig)


def test_reserve_annotation_headroom_uses_envelope_not_artists():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1e6])  # artist far above the declared envelope
    ax.set_ylim(0, 100)

    ab.reserve_annotation_headroom(ax, (np.array([100.0]), np.array([0.0])))

    assert ax.get_ylim()[1] == pytest.approx(100 / 0.7)
    assert _content_fraction(ax, 100.0) == pytest.approx(0.70)
    plt.close(fig)


def test_reserve_annotation_headroom_combines_series_and_ignores_nonfinite():
    fig, ax = plt.subplots()
    ax.set_ylim(0, 4)

    assert ab.reserve_annotation_headroom(
        ax,
        (np.array([1.0, np.nan]), np.array([0.0, 0.0])),
        (np.array([3.0, 2.0]), np.array([1.0, np.inf])),
    ) is True

    # Finite content top is 4 (3 + 1); NaN and inf entries are ignored.
    assert _content_fraction(ax, 4.0) == pytest.approx(0.70)
    plt.close(fig)


def test_reserve_annotation_headroom_returns_false_for_no_finite_envelope():
    fig, ax = plt.subplots()
    ax.set_ylim(0, 100)

    assert ab.reserve_annotation_headroom(
        ax, (np.array([np.nan]), np.array([np.inf])),
    ) is False
    assert ax.get_ylim() == (0, 100)
    plt.close(fig)


def test_reserve_annotation_headroom_rejects_invalid_fraction():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="fraction"):
        ab.reserve_annotation_headroom(
            ax, (np.array([1.0]), np.array([0.0])), fraction=1.0,
        )
    plt.close(fig)


def test_reserve_annotation_headroom_rejects_mismatched_error_shape():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="match values shape"):
        ab.reserve_annotation_headroom(ax, (np.ones(10), np.ones((10, 1))))
    plt.close(fig)


def test_reserve_annotation_headroom_rejects_negative_errors():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="non-negative"):
        ab.reserve_annotation_headroom(
            ax, (np.ones(3), np.array([1.0, -1.0, 1.0])),
        )
    plt.close(fig)


def test_reserve_annotation_headroom_overflowing_log_top_leaves_limits():
    """A nonfinite inverse top is reported as no change, not passed to set_ylim."""
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e6)
    before = ax.get_ylim()

    changed = ab.reserve_annotation_headroom(
        ax, (np.array([1e6]), np.array([0.0])), fraction=0.999,
    )

    assert changed is False
    assert ax.get_ylim() == before
    plt.close(fig)


def test_reserve_annotation_headroom_accepts_scalar_envelope():
    fig, ax = plt.subplots()
    ax.set_ylim(0, 6)

    assert ab.reserve_annotation_headroom(ax, (5.0, 1.0)) is True

    assert _content_fraction(ax, 6.0) == pytest.approx(0.70)
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


def test_reserved_headroom_clears_atlas_label_text():
    fig, ax = plt.subplots(figsize=(5, 4))
    values = np.array([100.0])
    errors = np.array([0.0])
    ax.errorbar([0], values, yerr=errors)
    ax.set_ylim(0, 100)

    assert ab.reserve_annotation_headroom(ax, (values, errors)) is True
    ab.add_atlas_label(ax, lumi=False)
    fig.canvas.draw()

    _, label_y0, _, _ = ab.artist_axes_bounds(ax.texts[-1], ax)
    assert _content_fraction(ax, 100.0) < label_y0
    plt.close(fig)
