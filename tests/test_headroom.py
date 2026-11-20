"""fit_overlay_headroom and overlay_need: the fit and the log-headroom check share one rule."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest

import analyb as ab
from analyb import config


@pytest.fixture(autouse=True)
def _close_and_reset():
    yield
    plt.close("all")
    config.set_figure_layout_rules()


def _spectrum(seed=0, n=30):
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, 300, n + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    mean, width = rng.uniform(40, 260), rng.uniform(15, 60)
    y = 10 ** rng.uniform(2, 6) * np.exp(-0.5 * ((centres - mean) / width) ** 2) + 10 ** rng.uniform(0, 2)
    return edges, y * rng.uniform(0.8, 1.2, n)


def _figure(edges, y, *, log=True, loc="upper right", entries=2, axvline=False, label=True):
    fig, ax = plt.subplots(figsize=(4, 3), layout="none")
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.16, top=0.95)
    for i in range(entries):
        ax.stairs(y * (1 + 0.02 * i), edges, label=f"Sample {i}")
    if axvline:
        ax.axvline(edges[len(edges) // 3], color="k", ls="--")
    if log:
        ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("x [GeV]")
    ax.set_ylabel("Events")
    if label:
        ax.text(0.04, 0.95, "ATLAS Simulation Internal", transform=ax.transAxes, va="top")
    ax.legend(frameon=False, loc=loc)
    ax.set_ylim(0.5 * y.min() if log else 0.0, y.max() * 1.02 ** (entries - 1))
    return fig, ax


def _clearance(fig, ax, edges, y):
    """Independent of analyb: the smallest gap, in axes fraction, between a text or legend in the upper half of the
    frame and the steps under it (negative: the box covers data); None without such a box over data."""
    fig.draw_without_rendering()
    frame = ax.get_window_extent()
    boxes = [t.get_window_extent() for t in ax.texts]
    leg = ax.get_legend()
    pad = leg.borderpad * leg.prop.get_size_in_points() * fig.dpi / 72
    boxes.append(leg.get_window_extent().padded(-pad))
    gaps = []
    for box in boxes:
        if (box.y0 + box.y1) / 2 < frame.y0 + frame.height / 2:
            continue
        x0, x1 = ax.transData.inverted().transform([(box.x0, 0), (box.x1, 0)])[:, 0]
        under = (edges[1:] >= x0) & (edges[:-1] <= x1)
        if under.any():
            gaps.append((box.y0 - ax.transData.transform((0, y[under].max()))[1]) / frame.height)
    return min(gaps) if gaps else None


def _gap_decades(fig, ax, edges, y):
    """Independent of analyb: the smallest room, in decades, between the axis top and the top each box over data needs
    (box at axes fraction f, clearance c over the data, log span S: room = c * S / f)."""
    fig.draw_without_rendering()
    frame = ax.get_window_extent()
    lo, hi = ax.get_ylim()
    span = np.log10(hi / lo)
    boxes = [t.get_window_extent() for t in ax.texts]
    leg = ax.get_legend()
    pad = leg.borderpad * leg.prop.get_size_in_points() * fig.dpi / 72
    boxes.append(leg.get_window_extent().padded(-pad))
    rooms = []
    for box in boxes:
        if (box.y0 + box.y1) / 2 < frame.y0 + frame.height / 2:
            continue
        x0, x1 = ax.transData.inverted().transform([(box.x0, 0), (box.x1, 0)])[:, 0]
        under = (edges[1:] >= x0) & (edges[:-1] <= x1)
        if under.any():
            f = (box.y0 - frame.y0) / frame.height
            c = (box.y0 - ax.transData.transform((0, y[under].max()))[1]) / frame.height
            rooms.append(c * span / f)
    return min(rooms) if rooms else None


def _boxes_clear(fig, ax, edges, y):
    gap = _clearance(fig, ax, edges, y)
    return gap is None or gap > 0


def _range_findings(fig, cap=2.5):
    return [(f.check, f.message) for f in ab.check_figure(fig, log_headroom_cap=cap)
            if f.check in ("log-headroom", "log-range")]


@pytest.mark.parametrize("seed,loc,axvline", list(itertools.product(
    range(6), ("upper right", "upper left", "upper center", "lower center"), (False, True))))
def test_a_fitted_log_axis_passes_the_check(seed, loc, axvline):
    edges, y = _spectrum(seed)
    fig, ax = _figure(edges, y, loc=loc, axvline=axvline)
    ax.set_ylim(1e-3, None)   # a floor far under the core: the fit raises it
    ab.fit_overlay_headroom(ax, cap=2.5)
    assert _range_findings(fig) == []
    assert _boxes_clear(fig, ax, edges, y * 1.02)
    room = _gap_decades(fig, ax, edges, y * 1.02)   # every box over data keeps the gap, also one under the data top
    assert room is None or room >= ab.headroom.GAP_DECADES - 0.01


def test_a_second_fit_changes_nothing():
    edges, y = _spectrum(3)
    fig, ax = _figure(edges, y)
    ab.fit_overlay_headroom(ax)
    first = ax.get_ylim()
    ab.fit_overlay_headroom(ax)
    assert ax.get_ylim() == pytest.approx(first, rel=1e-9)


def test_the_returned_need_is_the_top_the_check_accepts():
    edges, y = _spectrum(1)
    fig, ax = _figure(edges, y)
    need = ab.overlay_need(ax)
    assert need.scale == "log" and need.binding is not None and need.upper > need.top
    lo = ax.get_ylim()[0]
    ax.set_ylim(lo, need.upper * 10 ** 0.05)
    assert [c for c, _ in _range_findings(fig, cap=10.0) if c == "log-headroom"] == []
    ax.set_ylim(lo, need.upper * 10 ** -0.2)
    assert any("reaches down into the data" in m for c, m in _range_findings(fig, cap=10.0))
    ax.set_ylim(lo, need.upper * 10 ** 0.5)
    assert any("beyond it" in m for c, m in _range_findings(fig, cap=10.0))


def test_a_reference_line_spanning_the_frame_does_not_move_the_fit():
    edges, y = _spectrum(2)
    fig, ax = _figure(edges, y, loc="upper left")
    ab.fit_overlay_headroom(ax)
    plain = ax.get_ylim()
    fig, ax = _figure(edges, y, loc="upper left", axvline=True)
    ab.fit_overlay_headroom(ax)
    assert ax.get_ylim() == pytest.approx(plain, rel=1e-9)


def test_the_cap_comes_from_the_layout_rules_and_a_larger_need_is_kept():
    edges = np.linspace(0, 300, 31)
    y = 1e5 * np.exp(-0.5 * ((0.5 * (edges[1:] + edges[:-1]) - 150) / 60) ** 2) + 10
    fig, ax = _figure(edges, y, entries=6, loc="upper center")   # a tall legend over the peak needs 1.6 decades
    ab.set_figure_layout_rules(log_headroom_cap=1.0)
    need = ab.fit_overlay_headroom(ax)
    assert np.log10(need.upper / need.top) > 1.0
    assert _clearance(fig, ax, edges, y * 1.02 ** 5) > -0.002   # over the cap the gap goes, the legend still clears the data
    assert any("exceed the cap" in m for _, m in _range_findings(fig, cap=1.0))   # and the check says so


def test_the_fit_does_not_depend_on_the_figure_dpi():
    edges, y = _spectrum(5)
    tops = []
    for dpi in (72, 600):
        fig, ax = _figure(edges, y)
        fig.set_dpi(dpi)
        ab.fit_overlay_headroom(ax)
        assert fig.dpi == dpi
        tops.append(ax.get_ylim())
    assert tops[0] == pytest.approx(tops[1], rel=1e-9)


def test_a_linear_axis_keeps_the_gap_above_the_data():
    edges, y = _spectrum(0)
    fig, ax = _figure(edges, y, log=False, loc="upper left")
    need = ab.fit_overlay_headroom(ax, floor=0.0)
    assert need.scale == "linear" and ax.get_ylim()[0] == 0.0
    assert _clearance(fig, ax, edges, y * 1.02) == pytest.approx(0.02, abs=0.005)


def test_axes_that_cannot_be_fitted_are_left_alone():
    edges, y = _spectrum(0)
    fig, ax = _figure(edges, y)
    ax.set_ylim(1e7, 1.0)   # inverted
    assert ab.fit_overlay_headroom(ax) is None and ax.get_ylim() == (1e7, 1.0)
    assert ab.overlay_need(ax) is None
    fig, ax = _figure(edges, y)
    with pytest.raises(ValueError):
        ab.fit_overlay_headroom(ax, floor=0.0)
    with pytest.raises(ValueError):
        ab.fit_overlay_headroom(ax, gap=0.3)
    with pytest.raises(ValueError):
        ab.fit_overlay_headroom(ax, cap=float("nan"))


def test_a_marker_drawn_up_from_the_floor_is_not_the_core():
    # r23 daod-polish event-ratio figures: vlines(1, floor, top) marks the reference; its end on the floor is not data,
    # so the floor far under the populated core is still raised (and the check still sees the empty decades)
    edges = np.linspace(0, 300, 31)
    y = 2e3 * np.exp(-0.5 * ((0.5 * (edges[1:] + edges[:-1]) - 100) / 30) ** 2) + 20   # core 20, top 2e3
    fig, ax = _figure(edges, y, loc="upper right")
    floor = 0.5   # above 1e-4 of the top: a value there would count as core; 1.6 decades under the real core
    ax.vlines(150.0, floor, y.max(), color="k", ls=":")
    ax.set_ylim(floor, y.max())
    assert any(c == "log-range" for c, _ in _range_findings(fig, cap=None))
    need = ab.fit_overlay_headroom(ax)
    assert need.core == pytest.approx(y[y >= 1e-4 * y.max()].min(), rel=1e-6)
    assert np.log10(need.core / ax.get_ylim()[0]) == pytest.approx(0.3)
    assert _range_findings(fig) == []


def test_a_box_needing_just_under_the_data_top_still_gets_the_gap():
    # r23 daod-polish iter2 eta figure: the label's need lay 0.01 decades under the peak beside it; the gap still applies
    edges = np.linspace(0, 300, 31)
    centres = 0.5 * (edges[1:] + edges[:-1])
    peak = 1e4
    y = np.where(centres > 260, peak, 1.0)
    fig, ax = _figure(edges, y, loc="lower center", entries=1)
    ax.set_ylim(1.0, peak)
    fig.draw_without_rendering()
    label = ax.texts[0].get_window_extent().transformed(ax.transAxes.inverted())
    assert label.x1 < 0.85   # the label hangs over the flat part only
    level = 10 ** (label.y0 * (np.log10(peak) - 0.03))   # the label needs 0.03 decades under the peak
    y = np.where(centres > 260, peak, level)
    fig, ax = _figure(edges, y, loc="lower center", entries=1)
    ab.fit_overlay_headroom(ax, floor=1.0, raise_floor=False)
    assert _gap_decades(fig, ax, edges, y) >= ab.headroom.GAP_DECADES - 0.01
