"""Fit the upper y limit of an axes to the legends and texts hung from its top.

``fit_overlay_headroom`` sets the limit the ``log-headroom`` figure check accepts: it measures with the check's own
rule (:func:`analyb.figcheck.overlay_need`), so a fitted figure and the check cannot disagree on the geometry. On a log
axis it also raises a floor that sits far under the lowest populated value, never above it, inside the same loop, so a floor raised after the
fit cannot stretch the headroom.
"""

import numpy as np
from matplotlib.ticker import NullFormatter

from . import config
from .figcheck import LOG_HEADROOM_SLACK, _all_axes, _collect_items, _layout_dpi, _overlay_need, _scale_maps

GAP_DECADES = 0.1      # log: decades added above the need, so each box clears the data under it by about a tenth of its height
GAP_FRACTION = 0.02    # linear: axes fraction each box keeps above the data under it
DATA_MARGIN = 0.03     # log: decades kept above the highest data value, so it lies inside the axes
DATA_FRACTION = 0.02   # linear, symlog: axes fraction kept above the highest data value, for the same reason
FLOOR_BELOW_MAX = 1.2  # log: a floor farther than this many decades under the lowest populated value is raised (log-range flags above 1.5)
FLOOR_MARGIN = 0.3     # log: decades between a raised floor and the lowest populated value
TOP_TICK_ROOM = 0.06   # share of the span under the top kept free of tick labels
_PASSES = 5
_RULES = object()

assert GAP_DECADES < LOG_HEADROOM_SLACK, "a fitted axis must pass the check's slack"


def _upper(ax, need, box, log, gap, cap):
    """Upper limit from a measured need and the largest box need ``box`` (data units, None without a box over data):
    the box need plus the gap, even when it lies under the data top, and never under the data margin; on a log axis,
    over the cap the gap is dropped, and a need beyond the cap is kept (the check then fails it: shrink the legend).
    On a linear or symlog axis the box need already holds the gap (measured with ``pad``)."""
    if not log:
        to_scale, from_scale = _scale_maps(ax)
        s_lo = to_scale(need.floor)
        upper = from_scale(s_lo + (to_scale(need.top) - s_lo) / (1 - DATA_FRACTION))
        return max(upper, box) if box is not None else upper
    top = np.log10(need.top)
    upper = top + DATA_MARGIN
    if box is not None:
        fit = np.log10(box)
        if cap is None or fit + gap - top <= cap:
            upper = max(upper, fit + gap)
        else:
            upper = max(upper, fit, top + cap)
    return 10 ** upper


def _set(ax, lo, hi, log, prune_ticks):
    ax.set_ylim(lo, hi)
    if prune_ticks:
        to_scale, from_scale = _scale_maps(ax)
        edge = from_scale(to_scale(hi) - TOP_TICK_ROOM * (to_scale(hi) - to_scale(lo)))
        if log:
            ax.yaxis.set_minor_formatter(NullFormatter())   # a narrow log range prints minor labels that collide
        ax.set_yticks([t for t in ax.get_yticks() if lo <= t <= edge])   # no tick label at the top edge
        ax.set_ylim(lo, hi)


def fit_overlay_headroom(ax, *, floor=None, cap=_RULES, gap=None, raise_floor=True, prune_ticks=True):
    """Set the upper y limit of ``ax`` so that every legend and text hung from its top clears the data under it.

    The need is the one the ``log-headroom`` check measures (:func:`analyb.figcheck.overlay_need`): boxes whose centre
    lies in the upper half of the frame, keeping their place in axes fraction. The loop measures, sets the limits and
    measures again on the new layout (tick labels move the axes) until the limits stop moving, at ``LAYOUT_DPI``; the
    figure's dpi is restored. Calling it again changes nothing, so call it after anything that moves the layout.

    Parameters
    ----------
    floor : float, optional
        Lower limit to set (default: keep the current one).
    cap : float or None
        Log axes: most empty decades allowed above the data. Default: the ``log_headroom_cap`` of
        :func:`analyb.set_figure_layout_rules`, the value the save-time check uses. A need over the cap is kept, and
        the check fails it.
    gap : float, optional
        Room above the need: decades on a log axis (default ``GAP_DECADES``), axes fraction on a linear or symlog one
        (default ``GAP_FRACTION``).
    raise_floor : bool
        Log axes: raise a floor more than ``FLOOR_BELOW_MAX`` decades under the lowest populated value (``OverlayNeed.core``,
        the reference of the ``log-range`` check) to ``FLOOR_MARGIN`` under it. A floor above a populated value is left
        alone (the check flags it).
    prune_ticks : bool
        Drop tick labels within ``TOP_TICK_ROOM`` of the span under the top; on a log axis hide minor tick labels.

    Returns
    -------
    OverlayNeed or None
        The need at the final limits; None (limits untouched) for an axis neither linear, log nor symlog, an inverted
        axis or one without painted data.
    """
    scale = ax.get_yscale()
    log = scale == "log"
    if cap is _RULES:
        cap = config.get_figure_layout_rules()["log_headroom_cap"]
    if cap is not None and not (np.isfinite(cap) and cap > 0):
        raise ValueError("cap must be a positive finite number of decades or None")
    if gap is None:
        gap = GAP_DECADES if log else GAP_FRACTION
    if not (np.isfinite(gap) and gap >= 0) or log and gap >= LOG_HEADROOM_SLACK or not log and gap >= 0.5:
        raise ValueError("gap must be at least 0 and under the check's slack (log) or 0.5 (linear)")
    lo = ax.get_ylim()[0] if floor is None else float(floor)
    if log and not lo > 0:
        raise ValueError("a log axis needs a positive floor")
    if scale not in ("log", "linear", "symlog") or ax.get_ylim()[0] > ax.get_ylim()[1]:
        return None
    fig = ax.get_figure(root=True)
    need = None
    with _layout_dpi(fig):
        for _ in range(_PASSES):
            before = ax.get_ylim()
            ax.set_ylim(lo, before[1])
            fig.draw_without_rendering()
            items = _collect_items(fig, _all_axes(fig))
            need, box = _overlay_need(ax, items, pad=0.0 if log else gap, boxes=True)
            if need is None:
                ax.set_ylim(*before)
                break
            if log and raise_floor and need.core and np.log10(need.core / lo) > FLOOR_BELOW_MAX:
                lo = need.core * 10 ** (-FLOOR_MARGIN)
                need, box = _overlay_need(ax, items, floor=lo, boxes=True)   # boxes hold; the next pass measures again
            hi = _upper(ax, need, box, log, gap, cap)
            _set(ax, lo, hi, log, prune_ticks)
            if np.allclose(ax.get_ylim(), before, rtol=1e-6, atol=0):
                break
    return need
