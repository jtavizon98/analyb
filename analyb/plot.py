"""Plotting: 1D (ATLAS default), data/MC ratio, stacked, comparison, 2D, fits, profile, efficiency."""

import logging
import os
import re
from collections.abc import Mapping
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from . import config, figcheck, variables
from .colors import PALETTES, catppuccin_colors, current_colorbar


def _siunitx_to_mathtext(unit_macro):
    """Convert siunitx macro (e.g. \\GeV) to mathtext-safe form (\\mathrm{GeV})."""
    if unit_macro is None:
        return None
    if unit_macro == r"\radian":
        return r"\mathrm{rad}"
    return re.sub(r'\\([A-Za-z]+)', r'\\mathrm{\1}', unit_macro)


def _fmt_si(value_str, unit_str):
    """Format a value and unit using portable MathText."""
    u = _siunitx_to_mathtext(unit_str)
    return value_str + r"\;" + u


def _fmt_num(value_str):
    """Format a number without TeX-only macros."""
    return value_str


def _fmt_siunit(unit_str):
    """Format a unit using portable MathText."""
    return r"$" + _siunitx_to_mathtext(unit_str) + r"$"
from .histogram import (
    Hist1D, Hist2D,
    ratio as hist_ratio,
    normalize as hist_normalize,
    _edges_match,
)
from .units import format_bin_width


# --------------------------------------------------------------------------
# Figure save logic (carried over & centralised)
# --------------------------------------------------------------------------

def save_figure(
    fig, save_name, script_id=None, save_path=None, svg=False, verbose=True,
    *, figure_check=None, final_width_cm=None,
):
    """Save ``fig`` as ``{script_id}-{save_name}.pdf`` (plus optional SVG/PNG).

    ``figure_check`` overrides ``config.get_figure_check()`` for this call:
    ``"warn"`` logs each :func:`analyb.figcheck.check_figure` finding and
    writes ``<pdf stem>.figcheck.json`` beside the PDF; ``"raise"`` does the
    same and then raises ``FigureCheckError`` if any finding fails; ``False``
    disables the check; ``None`` inherits the configured mode (default off).
    ``final_width_cm`` overrides ``config.get_figure_final_width_cm()``; the
    font-floor check runs only when one of them is set.
    """
    return _save_figure(
        fig, save_name, script_id, save_path, svg, verbose=verbose,
        figure_check=figure_check, final_width_cm=final_width_cm,
    )


def _save_figure(
    fig, save_name, script_id=None, save_path=None, svg=False, verbose=True,
    *, figure_check=None, final_width_cm=None,
):
    mode = config.get_figure_check() if figure_check is None else figure_check
    if mode not in (False, "warn", "raise", None):
        raise ValueError(f"figure_check must be None, False, 'warn' or 'raise', got {mode!r}")
    if final_width_cm is None:
        final_width_cm = config.get_figure_final_width_cm()
    # Measure before saving: savefig restores a figure without a layout engine
    # to the rc default (constrained here), so the post-save state can differ
    # from what was written.
    findings = figcheck.check_figure(fig, final_width_cm=final_width_cm) if mode else []
    sid = script_id or config.get_script_id()
    base = save_path if save_path is not None else (config.get_save_path() or os.path.join(os.getcwd(), "plots"))
    os.makedirs(base, exist_ok=True)
    pdf_path = os.path.join(base, f"{sid}-{save_name}.pdf")
    fig.savefig(pdf_path)
    if svg:
        svg_dir = os.path.join(base, "svg")
        os.makedirs(svg_dir, exist_ok=True)
        fig.savefig(os.path.join(svg_dir, f"{sid}-{save_name}.svg"))
    if config.get_save_png():
        png_dir = os.path.join(base, "png")
        os.makedirs(png_dir, exist_ok=True)
        fig.savefig(os.path.join(png_dir, f"{sid}-{save_name}.png"),
                    dpi=config.get_save_png_dpi())
    if verbose:
        logger.info("Saved plot: %s", pdf_path)
    if mode:
        figcheck.write_sidecar(fig, pdf_path, findings, final_width_cm=final_width_cm)
    plt.close(fig)
    if mode == "raise" and any(f.severity == "fail" for f in findings):
        raise figcheck.FigureCheckError(pdf_path, findings)


# --------------------------------------------------------------------------
# ATLAS label helpers
# --------------------------------------------------------------------------

_DEFAULT_LABEL_ENERGY = 13
_DEFAULT_LABEL_LUMI = 140


def _add_legend(ax, *args, **kwargs):
    """Place a frameless legend inside the fixed upper-right header region."""
    kwargs.setdefault("loc", "upper right")
    kwargs.setdefault("bbox_to_anchor", (0.98, 0.98))
    kwargs.setdefault("borderaxespad", 0)
    kwargs.setdefault("frameon", False)
    legend = ax.legend(*args, **kwargs)
    legend.set_in_layout(False)
    return legend


def reserve_atlas_label_headroom(ax, ylog=None):
    """Expand y-axis limits to reserve ~30% headroom for the fixed ATLAS label.

    Called before ``add_atlas_label`` on 1D plotting axes.  2D and scatter axes
    should not use this.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ylog : bool or None
        If True, expand log-space range.  If None, detect from ``ax.get_yscale()``.
    """
    if ylog is None:
        ylog = ax.get_yscale() == "log"
    ymin, ymax = ax.get_ylim()
    if ymax <= ymin:
        return
    if ax.get_yscale() == "symlog":
        transform = ax.yaxis.get_transform()
        lower, upper = transform.transform([ymin, ymax])
        new_ymax = transform.inverted().transform(lower + (upper - lower) / 0.7)
        ax.set_ylim(bottom=ymin, top=float(new_ymax))
    elif ylog and ymin > 0:
        log_ymin = np.log10(ymin)
        log_range = np.log10(ymax) - log_ymin
        new_log_ymax = log_ymin + log_range / 0.7
        ax.set_ylim(bottom=ymin, top=10 ** new_log_ymax)
    else:
        new_ymax = ymin + (ymax - ymin) / 0.7
        ax.set_ylim(bottom=ymin, top=new_ymax)


def _expand_top_to_fraction(ax, content_top, fraction):
    """Return the y-top leaving ``fraction`` of the axis above ``content_top``.

    Measured in axis-transform coordinates, so linear, log10 and symlog axes
    reserve the same visible fraction. Returns ``None`` when the current limits,
    the transform, or the resulting top cannot express a finite reservation; the
    caller then leaves the axes untouched.
    """
    bottom, top = ax.get_ylim()
    if not (np.isfinite(bottom) and np.isfinite(top)) or top <= bottom:
        return None
    transform = ax.yaxis.get_transform()

    def _forward(value):
        return np.asarray(transform.transform(value), dtype=float).reshape(-1)[0]

    try:
        lower_t = _forward(bottom)
        upper_t = _forward(top)
        content_t = _forward(content_top)
    except (ValueError, OverflowError):
        return None
    if not np.all(np.isfinite([lower_t, upper_t, content_t])):
        return None
    if content_t <= lower_t or upper_t <= lower_t:
        return None
    if content_t <= lower_t + (1 - fraction) * (upper_t - lower_t):
        return None
    needed = lower_t + (content_t - lower_t) / (1 - fraction)
    if not np.isfinite(needed):
        return None
    # A wide log range with a large fraction can invert to a nonfinite top;
    # treat that as unsupported instead of passing inf to ``set_ylim``.
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            new_top = np.asarray(
                transform.inverted().transform(needed), dtype=float,
            ).reshape(-1)[0]
        except (ValueError, OverflowError, FloatingPointError):
            return None
    if not np.isfinite(new_top) or new_top <= bottom:
        return None
    return float(new_top)


def reserve_annotation_headroom(ax, envelope, *additional_envelopes, fraction=0.30):
    """Reserve the top ``fraction`` of a data axis for in-axes annotations.

    A companion to :func:`reserve_atlas_label_headroom` for callers that keep
    drawing after analyb returns. It takes the plotted data/error envelope
    explicitly instead of reading artists or the current limits, and it measures
    in axis-transform coordinates, so linear, log and symlog axes reserve the
    same visible fraction. Only the upper limit changes.

    Call it **last**: after every overlay, after the final ``set_yscale`` and
    ``set_ylim``, and before saving. Any later ``set_ylim``, ``set_yscale`` or
    autoscale call replaces the reserved limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to expand.
    envelope, *additional_envelopes : (values, errors) pairs
        Explicit data/error envelope of each plotted series, one pair per
        positional argument. ``values`` and ``errors`` are array-like; each
        ``errors`` must match its ``values`` shape or be a scalar broadcast
        over it. Negative finite errors (non-magnitudes) are rejected. Only
        finite ``values + errors`` points set the content top. Nothing is
        inferred from artists already on the axes.
    fraction : float
        Visible fraction of the axis reserved above the highest content point.
        Default ``0.30`` matches the ATLAS label allowance. Must be in
        ``[0, 1)``.

    Returns
    -------
    bool
        ``True`` when the upper limit was expanded. ``False`` when the
        reservation was unnecessary, unsupported for the current scale/limits,
        or not representable as a finite top; the axes are then left unchanged.

    Notes
    -----
    The helper does not validate scientific admissibility: it does not require
    the envelope or the lower limit to be positive on a log axis, and it never
    adjusts the lower limit. The caller owns those checks.

    Examples
    --------
    ::

        ax.set_yscale("log")
        ax.set_ylim(low, high)                       # caller's final bounds
        ab.reserve_annotation_headroom(
            ax, (model_values, model_errors), (data_values, data_errors),
        )
        ab.add_atlas_label(ax, lumi=False)
    """
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    content_top = None
    for values, errors in (envelope, *additional_envelopes):
        values_arr = np.asarray(values, dtype=float)
        errors_arr = np.asarray(errors, dtype=float)
        if errors_arr.size != 1 and errors_arr.shape != values_arr.shape:
            raise ValueError(
                "errors must match values shape or be scalar: "
                f"{errors_arr.shape} != {values_arr.shape}"
            )
        if np.any(np.isfinite(errors_arr) & (errors_arr < 0)):
            raise ValueError("errors must be non-negative magnitudes")
        upper = np.atleast_1d(values_arr + errors_arr)
        upper = upper[np.isfinite(upper)]
        if upper.size:
            candidate = float(upper.max())
            content_top = candidate if content_top is None else max(content_top, candidate)
    if content_top is None:
        return False
    new_top = _expand_top_to_fraction(ax, content_top, fraction)
    if new_top is None:
        return False
    ax.set_ylim(top=new_top)
    return True


def artist_axes_bounds(artist, ax=None):
    """Return an artist's bounding box in axes-fraction coordinates.

    Intended for checking that legends and text labels stay inside the axes and
    clear of the plotted content after headroom is reserved. No collision
    detection is performed.

    The figure is laid out with ``Figure.draw_without_rendering()`` and the
    artist is measured afterwards, so the result is the final post-layout
    geometry on any backend (including PDF and SVG) without writing output and
    without reusing a stale canvas renderer.

    Parameters
    ----------
    artist : matplotlib artist
        Text, Legend, or any artist exposing ``get_window_extent``.
    ax : matplotlib.axes.Axes, optional
        Axes providing the transform. Defaults to ``artist.axes``.

    Returns
    -------
    tuple or None
        ``(x0, y0, x1, y1)`` in axes coordinates, where ``(0, 0)`` is the
        lower-left and ``(1, 1)`` the upper-right of the axes. ``None`` when
        the artist has no finite extent.
    """
    ax = ax if ax is not None else getattr(artist, "axes", None)
    if ax is None:
        raise ValueError("ax is required when the artist has no axes")
    fig = ax.figure
    fig.draw_without_rendering()
    bbox = artist.get_window_extent()
    if bbox is None or not np.all(np.isfinite(bbox.extents)):
        return None
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(
        [[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]],
    )
    return float(x0), float(y0), float(x1), float(y1)


def add_atlas_label(
    ax,
    *,
    status="Internal",
    lumi=_DEFAULT_LABEL_LUMI,
    energy=_DEFAULT_LABEL_ENERGY,
    channel=None,
    category=None,
    x=0.04,
    y=0.95,
    fontsize="small",
):
    """Draw an ATLAS-style label at fixed upper-left inside the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    status : str
        ATLAS status text (default "Internal").
    lumi : float, None, or False
        Luminosity in fb⁻¹ (default 140). ``None`` uses the default; ``False``
        omits the luminosity line.
    energy : float, None, or False
        Centre-of-mass energy in TeV (default 13). ``None`` uses the default;
        ``False`` omits the energy line.
    channel, category : str or None
        Optional channel and category labels.
    x, y : float
        Fixed position in axes coordinates (default x=0.04, y=0.95).
    fontsize : str or float
        Font size for the label.
    """
    energy = _DEFAULT_LABEL_ENERGY if energy is None else energy
    lumi = _DEFAULT_LABEL_LUMI if lumi is None else lumi
    status_details = [status]
    if channel is not None:
        status_details.append(channel)
    if category is not None:
        status_details.append(category)
    lines = ["ATLAS", ", ".join(status_details)]
    if energy is not False:
        lines.append(r"$\sqrt{s} = " + _fmt_si(str(energy), r"\TeV") + r"$")
    if lumi is not False:
        lines.append(r"$\mathcal{L} = " + _fmt_si(str(lumi), r"\fb^{-1}") + r"$")
    text = "\n".join(lines)

    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment="top", horizontalalignment="left")


# --------------------------------------------------------------------------
# Axis label helpers
# --------------------------------------------------------------------------

def _ylabel_counts(bin_width, unit=None, scaling=1, is_density=False, weighted=False, normalized=False):
    """Format a MathText-safe y-axis label with SI prefix scaling.

    The plotted quantity is ``Counts`` for a raw histogram and ``$\\sum w$`` (the sum of
    event weights) for a weighted one (``Hist1D.is_weighted``); ``normalized``
    prefixes ``Normalized``."""
    quantity = r"$\sum w$" if weighted else "Counts"
    if normalized:
        quantity = "Normalized " + (quantity if weighted else "counts")
    if is_density:
        if unit is not None:
            return quantity + r" / $" + _siunitx_to_mathtext(unit) + "$"
        return quantity + " / unit"
    bw = (bin_width or 0) * scaling
    result = format_bin_width(bw, unit)
    if result is None:
        return quantity
    value_str, unit_str = result
    if unit_str is not None:
        return quantity + r" / $" + value_str + r"\;" + _siunitx_to_mathtext(unit_str) + "$"
    return quantity + " / " + value_str


def _resolve_label_unit(variable, xlabel, hist_unit=None):
    """Resolve (xlabel, unit) from variable registry, explicit xlabel, and hist override.

    The xlabel is returned with the unit appended in bracket notation
    (e.g. ``$E_T^{\\text{miss}}$ [\\si{\\GeV}]``) when a unit is available.
    """
    var_label, var_unit = variables.lookup(variable) if variable is not None else (xlabel, None)
    if xlabel is None:
        xlabel = var_label
    if hist_unit is not None:
        var_unit = hist_unit
    if var_unit is not None:
        xlabel = xlabel + r" [" + _fmt_siunit(var_unit) + r"]"
    return xlabel, var_unit


def _data_errorbar(ax, x, y, yerr=None, color="black", label=None, **kw):
    """Draw data points with ATLAS-style error bars and marker."""
    return ax.errorbar(
        x, y, yerr=yerr,
        fmt=".", markersize=4, markerfacecolor=color,
        markeredgecolor=color, color=color,
        elinewidth=0.8, capsize=0, label=label,
        **kw,
    )


def _adaptive_ratio_ylim(
    ratio_values,
    ratio_errors,
    denominator=None,
    denominator_errors=None,
    max_relative_denominator_error=1.0,
):
    """Return readable ratio limits while retaining normal finite error bars.

    Undefined denominators never contribute. When denominator uncertainties are
    supplied, bins compatible with zero (relative uncertainty at least
    ``max_relative_denominator_error``) are omitted from automatic limits, but
    remain plotted. Finite, statistically meaningful disagreements are kept.

    Without denominator information, retain the legacy robust-outlier fallback
    for direct callers.
    """
    r = np.asarray(ratio_values, dtype=float)
    err = np.asarray(ratio_errors, dtype=float)
    err = np.where(np.isfinite(err) & (err >= 0), err, 0.0)
    finite = np.isfinite(r)
    has_denominator = denominator is not None
    if has_denominator:
        den = np.asarray(denominator, dtype=float)
        if den.shape != r.shape:
            raise ValueError("denominator must have the same shape as ratio_values")
        if denominator_errors is None:
            den_err = np.zeros_like(den)
        else:
            den_err = np.asarray(denominator_errors, dtype=float)
            if den_err.shape != r.shape:
                raise ValueError("denominator_errors must match ratio_values")
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_den_err = den_err / np.abs(den)
        meaningful = ((den > 0) & np.isfinite(relative_den_err)
                      & (relative_den_err < max_relative_denominator_error))
        excluded = finite & ~meaningful
        if np.any(excluded):
            logger.warning("Excluding %d statistically meaningless ratio points from y limits",
                           np.count_nonzero(excluded))
        finite &= meaningful
    if not np.any(finite):
        return 0.5, 1.5

    finite_r, finite_err = r[finite], err[finite]
    center = np.median(finite_r)
    q25, q75 = np.percentile(finite_r, [25, 75])
    scale = max(q75 - q25, np.median(np.abs(finite_r - center)) * 1.4826, 0.1)
    if has_denominator:
        regular_point = np.ones(finite_r.size, dtype=bool)
    elif finite_r.size < 5:
        regular_point = (np.abs(finite_r) <= 100.0) | (finite_err <= 100.0)
    else:
        # With enough bins, a robust envelope distinguishes ordinary shape
        # disagreement from a nearly empty denominator without hiding the point.
        regular_point = np.abs(finite_r - center) <= max(1.5, 10.0 * scale)
    if not np.all(regular_point):
        logger.warning("Excluding %d pathological ratio points from y limits",
                       np.count_nonzero(~regular_point))
    bounds = np.concatenate((
        finite_r[regular_point] - finite_err[regular_point],
        finite_r[regular_point] + finite_err[regular_point],
    ))
    if bounds.size == 0:
        return 0.5, 1.5
    low, high = min(0.5, float(np.min(bounds))), max(1.5, float(np.max(bounds)))
    padding = 0.08 * max(high - low, 1.0)
    return low - padding, high + padding


# --------------------------------------------------------------------------
# 1D plots
# --------------------------------------------------------------------------

def hist1d_plot(
    hist,
    xlabel=None,
    *,
    variable=None,
    style="atlas",
    color=None,
    alpha=0.3,
    label=None,
    scaling=1,
    atlas=False,
    ylog=False,
    title=None,
    save_name=None,
    save_path=None,
    script_id=None,
    figsize=None,
    ax=None,
    verbose=True,
):
    """Plot a single 1D histogram.

    `style`:
      - "atlas"  (default): ATLAS data style -> error bars (points) with stat errors.
      - "stairs": filled step (your original convention).
      - "mc":     filled step with alpha, used for MC backgrounds.
    """
    if figsize is None:
        figsize = config.fig_size_square
    if color is None:
        color = catppuccin_colors["red"]
    xlabel, var_unit = _resolve_label_unit(variable, xlabel, hist.unit)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    centers = (hist.edges[:-1] + hist.edges[1:]) / 2
    if style == "atlas":
        _data_errorbar(ax, centers * scaling, hist.counts, yerr=hist.errors,
                       color=color, label=label)
    elif style == "mc":
        ax.stairs(
            hist.counts, hist.edges * scaling, baseline=0,
            fill=True, color=color, alpha=alpha, linewidth=0.8, label=label,
        )
    elif style == "stairs":
        ax.stairs(
            hist.counts, hist.edges * scaling,
            linestyle="-", linewidth=0.8, color=color, label=label,
        )
    else:
        raise ValueError(f"Unknown style {style!r}")

    ydescr = _ylabel_counts(
        hist.bin_width, unit=var_unit, scaling=scaling, is_density=hist.is_density,
        weighted=hist.is_weighted,
    )
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(ydescr, loc="top")
    if title is not None:
        ax.set_title(title, loc="right")
    if ylog:
        ax.set_yscale("log")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom=hist.counts[hist.counts > 0].min() * 0.3 if np.any(hist.counts > 0) else 1.0, top=top * 3)
    if label is not None:
        _add_legend(ax, fontsize="x-small")

    if atlas:
        reserve_atlas_label_headroom(ax, ylog=ylog)
        add_atlas_label(ax, status="Internal")

    if save_name is not None:
        _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
    return fig, ax


def _ratio_axes(figsize):
    return plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )


def _require_shared_edges(reference, others, operation):
    """Reject histograms whose bins differ; plots never redistribute counts."""
    if any(not _edges_match(reference.edges, other.edges) for other in others):
        raise ValueError(
            f"{operation} requires identical bin edges; merge aligned bins "
            "with rebin_to_edges() before plotting"
        )


def _draw_ratio_panel(
    ax, data_hist, mc_hist, *, color, mc_color, xlabel, label, band_alpha=0.25,
):
    """Draw the shared data/MC ratio and denominator uncertainty band."""
    centers = (mc_hist.edges[:-1] + mc_hist.edges[1:]) / 2
    values, errors = hist_ratio(
        data_hist, mc_hist, nan_to=np.nan, include_denominator_error=False,
    )
    valid_mc = mc_hist.counts > 0
    values = np.where(valid_mc, values, np.nan)
    errors = np.where(valid_mc, errors, np.nan)
    _data_errorbar(ax, centers, values, yerr=errors, color=color)
    with np.errstate(divide="ignore", invalid="ignore"):
        band_low = 1 - np.where(valid_mc, mc_hist.errors / mc_hist.counts, 0)
        band_high = 1 + np.where(valid_mc, mc_hist.errors / mc_hist.counts, 0)
    ax.fill_between(
        centers, band_low, band_high, step="mid",
        color=mc_color, alpha=band_alpha, linewidth=0,
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.6)
    ax.set_ylim(*_adaptive_ratio_ylim(
        values, errors, mc_hist.counts, mc_hist.errors,
    ))
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(label, loc="top")


def hist1d_with_ratio(
    data_hist,
    mc_hist,
    *,
    variable=None,
    xlabel=None,
    data_color=None,
    mc_color=None,
    data_label="Data",
    mc_label="MC",
    ratio_label="Data / MC",
    atlas=False,
    ylog=False,
    save_name=None,
    save_path=None,
    script_id=None,
    figsize=None,
    verbose=True,
):
    """Data vs MC plot with ratio panel below (ATLAS data/MC layout)."""
    if figsize is None:
        figsize = (config.fig_size_square[0], config.fig_size_square[1] + 1.4)
    if data_color is None:
        data_color = "black"
    if mc_color is None:
        mc_color = catppuccin_colors["red"]

    _require_shared_edges(mc_hist, [data_hist], "hist1d_with_ratio")

    fig, (ax_main, ax_ratio) = _ratio_axes(figsize)

    # MC band = filled step in main panel with error band as a shaded region.
    ax_main.stairs(
        mc_hist.counts, mc_hist.edges, baseline=0,
        fill=True, color=mc_color, alpha=0.25, edgecolor=mc_color, linewidth=0.8, label=mc_label,
    )
    band_lo = mc_hist.counts - mc_hist.errors
    band_hi = mc_hist.counts + mc_hist.errors
    centers = (mc_hist.edges[:-1] + mc_hist.edges[1:]) / 2
    ax_main.fill_between(
        centers, band_lo, band_hi, step="mid",
        color=mc_color, alpha=0.15, linewidth=0,
    )
    # Data points on top.
    _data_errorbar(ax_main, centers, data_hist.counts,
                   yerr=data_hist.errors, color=data_color, label=data_label)

    xlabel, var_unit = _resolve_label_unit(variable, xlabel, data_hist.unit)
    ydescr = _ylabel_counts(
        mc_hist.bin_width, unit=var_unit, is_density=mc_hist.is_density,
        weighted=mc_hist.is_weighted,
    )
    ax_main.set_ylabel(ydescr, loc="top")
    if ylog:
        ax_main.set_yscale("log")
        mc_pos = mc_hist.counts[mc_hist.counts > 0]
        ax_main.set_ylim(bottom=mc_pos.min() * 0.3 if mc_pos.size else 1.0)
    _add_legend(ax_main, fontsize="x-small")
    if atlas:
        reserve_atlas_label_headroom(ax_main, ylog=ylog)
        add_atlas_label(ax_main, status="Internal")

    _draw_ratio_panel(
        ax_ratio, data_hist, mc_hist, color=data_color, mc_color=mc_color,
        xlabel=xlabel, label=ratio_label,
    )


    if save_name is not None:
        _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
    return fig, (ax_main, ax_ratio)


def hist1d_stacked(
    samples,
    *,
    variable=None,
    xlabel=None,
    data_hist=None,
    normalize=False,
    atlas=False,
    ylog=False,
    save_name=None,
    save_path=None,
    script_id=None,
    figsize=None,
    colors=None,
    data_color="black",
    verbose=True,
):
    """Stack MC Hist1D samples, optionally overlaying data and a ratio panel.

    ``colors`` accepts either a sequence in stack draw order or a mapping from
    sample label to color.  A mapping keeps sample colors stable when callers
    choose a different stack order.  Draw order is always the input mapping's
    order; only the legend is yield-sorted.
    """
    labels = list(samples.keys())
    hists = list(samples.values())
    if not hists:
        raise ValueError("hist1d_stacked requires at least one histogram")
    if any(h.is_density != hists[0].is_density for h in hists[1:]):
        raise ValueError("hist1d_stacked cannot mix raw-count and density histograms")
    if data_hist is not None and data_hist.is_density != hists[0].is_density:
        raise ValueError("hist1d_stacked cannot mix raw-count and density histograms")
    _require_shared_edges(
        hists[0], hists[1:] + ([] if data_hist is None else [data_hist]), "hist1d_stacked",
    )
    if normalize:
        hists = [hist_normalize(h) for h in hists]
    if colors is None:
        palette = [catppuccin_colors[name] for name in PALETTES["stack"]]
        colors = [palette[i % len(palette)] for i in range(len(hists))]
    elif isinstance(colors, Mapping):
        missing = [label for label in labels if label not in colors]
        if missing:
            raise ValueError(f"hist1d_stacked has no color for samples: {missing}")
        colors = [colors[label] for label in labels]
    else:
        colors = list(colors)
        if len(colors) < len(hists):
            raise ValueError("hist1d_stacked requires one color per sample")

    legend_labels = [
        label for label, _ in sorted(
            zip(labels, hists), key=lambda item: (-item[1].integral(), item[0]),
        )
    ]

    def set_yield_sorted_legend(ax, include_data=False):
        handles, drawn_labels = ax.get_legend_handles_labels()
        handle_by_label = dict(zip(drawn_labels, handles))
        ordered_labels = (["Data"] if include_data else []) + legend_labels
        _add_legend(
            ax,
            [handle_by_label[label] for label in ordered_labels], ordered_labels,
            fontsize="x-small",
        )

    if data_hist is None:
        if figsize is None:
            figsize = config.fig_size_square
        fig, ax = plt.subplots(figsize=figsize)
        bottoms = np.zeros_like(hists[0].counts)
        for h, c, lab in zip(hists, colors, labels):
            top = bottoms + h.counts
            ax.stairs(
                top, h.edges, baseline=bottoms,
                fill=True, color=c, edgecolor=c, linewidth=0.8, label=lab,
            )
            bottoms = top
        xlabel, var_unit = _resolve_label_unit(variable, xlabel, hists[0].unit)
        ax.set_xlabel(xlabel, loc="right")
        ax.set_ylabel(_ylabel_counts(
            hists[0].bin_width, unit=var_unit, is_density=hists[0].is_density,
            weighted=any(h.is_weighted for h in hists),
        ), loc="top")
        set_yield_sorted_legend(ax)
        if ylog:
            ax.set_yscale("log")
            ax.set_ylim(bottom=bottoms[bottoms > 0].min() * 0.3 if np.any(bottoms > 0) else 1.0)
        if atlas:
            reserve_atlas_label_headroom(ax, ylog=ylog)
            add_atlas_label(ax, status="Internal")

        if save_name is not None:
            _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
        return fig, ax

    # With data overlay + ratio panel.
    stacked_counts = np.zeros_like(hists[0].counts)
    stacked_err2 = np.zeros_like(hists[0].counts)
    for h in hists:
        stacked_counts = stacked_counts + h.counts
        stacked_err2 = stacked_err2 + h.errors ** 2
    stacked_err = np.sqrt(stacked_err2)
    stacked_hist = Hist1D(
        counts=stacked_counts, edges=hists[0].edges, errors=stacked_err,
        label="+".join(labels), unit=hists[0].unit,
        is_density=hists[0].is_density,
    )
    ratio_figsize = (
        (config.fig_size_square[0], config.fig_size_square[1] + 1.4)
        if figsize is None else figsize
    )
    fig, (ax_main, ax_ratio) = _ratio_axes(ratio_figsize)
    bottoms = np.zeros_like(hists[0].counts)
    for h, c, lab in zip(hists, colors, labels):
        top = bottoms + h.counts
        ax_main.stairs(
            top, h.edges, baseline=bottoms,
            fill=True, color=c, edgecolor=c, linewidth=0.8, label=lab,
        )
        bottoms = top
    # Total MC error band.
    centers = (hists[0].edges[:-1] + hists[0].edges[1:]) / 2
    ax_main.fill_between(
        centers, stacked_counts - stacked_err, stacked_counts + stacked_err,
        step="mid", color="black", alpha=0.08, linewidth=0,
    )
    _data_errorbar(ax_main, centers, data_hist.counts,
                    yerr=data_hist.errors, color=data_color, label="Data")
    xlabel, var_unit = _resolve_label_unit(variable, xlabel, hists[0].unit)
    ax_main.set_ylabel(_ylabel_counts(
        hists[0].bin_width, unit=var_unit, is_density=hists[0].is_density,
        weighted=any(h.is_weighted for h in hists),
    ), loc="top")
    if ylog:
        ax_main.set_yscale("log")
        stacked_pos = stacked_counts[stacked_counts > 0]
        ax_main.set_ylim(bottom=stacked_pos.min() * 0.3 if stacked_pos.size else 1.0)
    set_yield_sorted_legend(ax_main, include_data=True)
    if atlas:
        reserve_atlas_label_headroom(ax_main, ylog=ylog)
        add_atlas_label(ax_main, status="Internal")
    _draw_ratio_panel(
        ax_ratio, data_hist, stacked_hist, color=data_color, mc_color="black",
        xlabel=xlabel, label="Data / MC", band_alpha=0.08,
    )

    if save_name is not None:
        _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
    return fig, (ax_main, ax_ratio)


def hist1d_comparison(
    hists,
    *,
    labels=None,
    variable=None,
    xlabel=None,
    normalize=True,
    range_indices=None,
    colors=None,
    atlas=False,
    atlas_label_kwargs=None,
    ylog=False,
    save_name=None,
    save_path=None,
    script_id=None,
    figsize=None,
    verbose=True,
):
    """Overlay several 1D histograms after optional normalization."""
    if isinstance(hists, dict):
        labels = list(hists.keys())
        hists_list = list(hists.values())
    else:
        hists_list = list(hists)
        if labels is None:
            labels = [getattr(h, "label", None) or f"Hist {i}" for i, h in enumerate(hists_list)]
    if colors is None:
        colors = [catppuccin_colors[name] for name in PALETTES["comparison"]][:len(hists_list)]
    if len(colors) < len(hists_list):
        colors += [catppuccin_colors["text"]] * (len(hists_list) - len(colors))
    if figsize is None:
        figsize = config.fig_size_square
    fig, ax = plt.subplots(figsize=figsize)
    plotted_counts = []
    for h, lab, c in zip(hists_list, labels, colors):
        counts = h.counts
        if normalize:
            denom = _comparison_normalization_denominator(h, range_indices)
            counts = counts / denom if denom else counts
        plotted_counts.append(counts)
        ax.stairs(counts, h.edges, linestyle="-", linewidth=0.8, color=c, label=lab)
    xlabel, var_unit = _resolve_label_unit(variable, xlabel, hists_list[0].unit)
    ydescr = "Normalized counts" if normalize else "Counts"
    ydescr += _ylabel_offset(hists_list[0], var_unit)
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(ydescr, loc="top")
    if ylog:
        ax.set_yscale("log")
        all_counts = np.concatenate(plotted_counts)
        pos = all_counts[all_counts > 0]
        ax.set_ylim(bottom=pos.min() * 0.3 if pos.size else 1.0)
    _add_legend(ax, fontsize="x-small")
    if atlas:
        label_kwargs = {"status": "Internal"}
        label_kwargs.update(atlas_label_kwargs or {})
        reserve_atlas_label_headroom(ax, ylog=ylog)
        add_atlas_label(ax, **label_kwargs)

    if save_name is not None:
        _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
    return fig, ax


def _comparison_normalization_denominator(hist, range_indices):
    """Return the yield used to normalize a comparison histogram.

    Density histograms store counts per x-unit, so their yield is the
    width-integrated density.  ``range_indices`` uses normal Python slice
    semantics and applies to both counts and widths.
    """
    if range_indices is None:
        selected = slice(None)
    else:
        if len(range_indices) != 2:
            raise ValueError("range_indices must be a (start, stop) pair")
        selected = slice(*range_indices)
    counts = hist.counts[selected]
    if hist.is_density:
        return float(np.sum(counts * hist.widths[selected]))
    return float(np.sum(counts))


def _ylabel_offset(hist, unit):
    if hist.is_density:
        if unit is not None:
            return r" / $" + _siunitx_to_mathtext(unit) + "$"
        return " / unit"
    result = format_bin_width(hist.bin_width, unit)
    if result is None:
        return ""
    value_str, unit_str = result
    if unit_str is not None:
        return r" / $" + value_str + r"\;" + _siunitx_to_mathtext(unit_str) + "$"
    return r" / " + value_str


def hist1d_overlay(*args, **kwargs):
    """Alias for hist1d_comparison with normalize=False by default."""
    kwargs.setdefault("normalize", False)
    return hist1d_comparison(*args, **kwargs)


# --------------------------------------------------------------------------
# 2D plots
# --------------------------------------------------------------------------

def hist2d_plot(
    hist,
    xlabel=None,
    ylabel=None,
    *,
    variable_x=None,
    variable_y=None,
    cmap=None,
    norm=None,
    equal_aspect=False,
    colorbar_label="Counts",
    title=None,
    atlas=False,
    save_name=None,
    save_path=None,
    script_id=None,
    figsize=None,
    verbose=True,
):
    """Render exact 2D bin edges (including nonuniform bins) with a colorbar."""
    if figsize is None:
        figsize = config.fig_size_2d
    if cmap is None:
        cmap = current_colorbar
    if isinstance(hist, Hist2D):
        counts, x_edges, y_edges = hist.counts, hist.x_edges, hist.y_edges
        xu, yu = hist.x_unit, hist.y_unit
        xl, yl = hist.x_label, hist.y_label
    else:
        counts, x_edges, y_edges = hist
        xu = yu = xl = yl = None
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False, which="both")
    image = ax.pcolormesh(
        x_edges, y_edges, np.ma.masked_invalid(counts.T),
        shading="flat", cmap=cmap, norm=norm,
        edgecolors="none", linewidth=0, rasterized=True,
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    xlabel, _ = _resolve_label_unit(variable_x, xlabel, xu)
    ylabel, _ = _resolve_label_unit(variable_y, ylabel, yu)
    ax.set_xlabel(xlabel or "", loc="right")
    ax.set_ylabel(ylabel or "", loc="top")
    if title is not None:
        ax.set_title(title, loc="right")
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    if atlas:
        add_atlas_label(ax, status="Internal")

    if save_name is not None:
        _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
    return fig, ax


# --------------------------------------------------------------------------
# Scatter plot
# --------------------------------------------------------------------------

def scatter_plot(
    x,
    y,
    *,
    variable_x=None,
    variable_y=None,
    xlabel=None,
    ylabel=None,
    color=None,
    alpha=None,
    markersize=4,
    atlas=False,
    save_name=None,
    save_path=None,
    script_id=None,
    figsize=None,
    verbose=True,
):
    """Scatter plot with density-dependent alpha transparency.

    Uses ``ax.plot`` for efficiency (single Line2D) instead of ``ax.scatter``.
    Alpha scales with the number of points so that dense regions appear darker
    via overlapping transparent markers.

    Parameters
    ----------
    x, y : array-like
        Data point coordinates.
    variable_x, variable_y : str or None
        Variable names for label/unit lookup in the registry.
    xlabel, ylabel : str or None
        Explicit axis labels (overrides registry lookup).
    color : color or None
        Marker colour (default ``catppuccin_colors["red"]``).
    alpha : float or None
        Marker transparency.  When ``None``, computed from the number of
        points as ``max(0.01, min(0.5, 50 / len(x)))``.
    markersize : float
        Marker size in points (default 4).
    atlas : bool
        If True, add the four-line ATLAS status, energy, and luminosity label.
    save_name, save_path, script_id : str or None
        Figure save controls (see :func:`save_figure`).
    figsize : tuple or None
        Figure size (default ``config.fig_size_square``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if color is None:
        color = catppuccin_colors["red"]
    if alpha is None:
        n = max(x.size, 1)
        alpha = max(0.01, min(0.25, 50.0 / n))
    if figsize is None:
        figsize = config.fig_size_square

    xlabel, _ = _resolve_label_unit(variable_x, xlabel)
    ylabel, _ = _resolve_label_unit(variable_y, ylabel)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, marker=".", markersize=markersize,
            markerfacecolor=color, markeredgecolor="none",
            linestyle=" ", alpha=alpha, rasterized=True)
    ax.set_xlabel(xlabel or "", loc="right")
    ax.set_ylabel(ylabel or "", loc="top")
    if atlas:
        add_atlas_label(ax, status="Internal")

    if save_name is not None:
        _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
    return fig, ax


# --------------------------------------------------------------------------
# Profile & efficiency plots
# --------------------------------------------------------------------------

def profile_plot(
    centers, means, errors,
    xlabel=None,
    ylabel="Mean",
    *,
    variable_x=None,
    variable_y=None,
    color=None,
    atlas=False,
    save_name=None,
    save_path=None,
    script_id=None,
    figsize=None,
    verbose=True,
):
    xlabel, _ = _resolve_label_unit(variable_x, xlabel)
    if variable_y is not None and ylabel == "Mean":
        var_label, var_unit = variables.lookup(variable_y)
        ylabel = f"Mean {var_label}"
        if var_unit is not None:
            ylabel += " [" + _fmt_siunit(var_unit) + "]"
    if figsize is None:
        figsize = config.fig_size_square
    if color is None:
        color = catppuccin_colors["red"]
    fig, ax = plt.subplots(figsize=figsize)
    _data_errorbar(ax, centers, means, yerr=errors, color=color)
    ax.set_xlabel(xlabel or "", loc="right")
    ax.set_ylabel(ylabel, loc="top")
    if atlas:
        reserve_atlas_label_headroom(ax)
        add_atlas_label(ax, status="Internal")

    if save_name is not None:
        _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
    return fig, ax


def efficiency_plot(
    centers, eff, err_low, err_high,
    xlabel=None,
    ylabel=r"Efficiency",
    *,
    variable=None,
    color=None,
    reference=None,
    label="Efficiency",
    reference_label="Reference",
    reference_color="black",
    reference_errors=False,
    ratio_label=r"Eff / Ref",
    ratio_errors=None,
    ylim=(0, 1.05),
    ratio_ylim=None,
    more_series=None,
    atlas=False,
    save_name=None,
    save_path=None,
    script_id=None,
    figsize=None,
    verbose=True,
):
    """Plot a binomial efficiency curve. If `reference` provided, add a ratio panel.

    ``reference`` is ``(centers, eff, err_low, err_high)``. By default it is a
    dashed line; ``reference_errors=True`` draws it as points with its
    asymmetric errors. ``label``/``reference_label`` name the two series and
    ``ratio_label`` the ratio axis. ``ratio_errors`` (caller-owned: one array
    for symmetric or ``(low, high)``) adds error bars to the ratio points; analyb
    does not propagate efficiency errors into the ratio because their
    correlation belongs to the caller. ``ylim``/``ratio_ylim`` set the ranges
    (``None`` keeps Matplotlib's automatic range).

    ``more_series`` (requires ``reference``) adds further curves against the same
    reference: an ordered mapping ``label -> dict(centers, eff, err_low,
    err_high, color, ratio_errors=None, marker=None)``. Each is drawn as points
    in the main panel and as ``eff / reference eff`` (by bin index, as the first
    series) in the ratio panel, with its caller-owned ``ratio_errors``.
    """
    if more_series and reference is None:
        raise ValueError("more_series needs a reference")
    xlabel, _ = _resolve_label_unit(variable, xlabel)
    if figsize is None:
        figsize = config.fig_size_square
    if color is None:
        color = catppuccin_colors["red"]
    if reference is None:
        fig, ax = plt.subplots(figsize=figsize)
        _data_errorbar(ax, centers, eff, yerr=[err_low, err_high], color=color)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel or "", loc="right")
        ax.set_ylabel(ylabel, loc="top")
        if atlas:
            reserve_atlas_label_headroom(ax)
            add_atlas_label(ax, status="Internal")

        if save_name is not None:
            _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
        return fig, ax
    # With reference -> ratio panel.
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(figsize[0], figsize[1] + 1.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    _data_errorbar(ax_main, centers, eff, yerr=[err_low, err_high], color=color, label=label)
    ref_centers, ref_eff, ref_lo, ref_hi = reference
    if reference_errors:
        _data_errorbar(ax_main, ref_centers, ref_eff, yerr=[ref_lo, ref_hi],
                       color=reference_color, label=reference_label)
    else:
        ax_main.plot(ref_centers, ref_eff, "--", color=reference_color, linewidth=0.8,
                     label=reference_label)
    more_series = dict(more_series or {})
    for name, extra in more_series.items():
        drawn = _data_errorbar(ax_main, extra["centers"], extra["eff"],
                               yerr=[extra["err_low"], extra["err_high"]], color=extra["color"],
                               label=name)
        if extra.get("marker"):
            drawn.lines[0].set_marker(extra["marker"])
    if ylim is not None:
        ax_main.set_ylim(*ylim)
    ax_main.set_ylabel(ylabel, loc="top")
    _add_legend(ax_main, fontsize="x-small")
    if atlas:
        reserve_atlas_label_headroom(ax_main)
        add_atlas_label(ax_main, status="Internal")

    def _ratio(values):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ref_eff > 0, np.asarray(values, dtype=float) / ref_eff, np.nan)

    def _errors(errors):
        if errors is None:
            return None
        return (np.asarray(errors, dtype=float) if np.ndim(errors) == 1
                else [np.asarray(e, dtype=float) for e in errors])

    _data_errorbar(ax_ratio, centers, _ratio(eff), yerr=_errors(ratio_errors), color=color)
    for name, extra in more_series.items():
        drawn = _data_errorbar(ax_ratio, extra["centers"], _ratio(extra["eff"]),
                               yerr=_errors(extra.get("ratio_errors")), color=extra["color"])
        if extra.get("marker"):
            drawn.lines[0].set_marker(extra["marker"])
    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=0.6)
    if ratio_ylim is not None:
        ax_ratio.set_ylim(*ratio_ylim)
    ax_ratio.set_ylabel(ratio_label, loc="top")
    ax_ratio.set_xlabel(xlabel or "", loc="right")

    if save_name is not None:
        _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
    return fig, (ax_main, ax_ratio)


# --------------------------------------------------------------------------
# Systematic uncertainty summary plot
# --------------------------------------------------------------------------

def systematic_summary_plot(
    nominal_hist,
    systematics,
    *,
    data_stat_hist=None,
    variable=None,
    xlabel=None,
    atlas=False,
    save_name=None,
    save_path=None,
    script_id=None,
    figsize=None,
    colors=None,
    verbose=True,
):
    """Fractional uncertainty decomposition plot.

    Parameters
    ----------
    nominal_hist : Hist1D
        Nominal MC prediction.
    systematics : dict[str, Hist1D | tuple(Hist1D, Hist1D)]
        Per-category systematic variations. Each entry can be:
        - A single Hist1D of fractional uncertainty (already as |delta|/nominal)
        - A tuple (up_hist, down_hist) from which the fractional RMS is computed.
    data_stat_hist : Hist1D, optional
        Data statistical uncertainty (used for expected stat. band).
    """
    if figsize is None:
        figsize = config.fig_size_wide
    centers = (nominal_hist.edges[:-1] + nominal_hist.edges[1:]) / 2
    nbins = nominal_hist.nbins
    nom_counts = nominal_hist.counts

    frac_categories = {}
    for name, var in systematics.items():
        if isinstance(var, tuple):
            up, down = var
            frac = np.sqrt((up.counts / nom_counts) ** 2 + (down.counts / nom_counts) ** 2)
        else:
            frac = var.counts
        frac_categories[name] = np.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)

    total_frac = np.sqrt(sum(f ** 2 for f in frac_categories.values()))

    if colors is None:
        cat_colors = [catppuccin_colors[name] for name in PALETTES["systematics"]]
        colors = {name: cat_colors[i % len(cat_colors)] for i, name in enumerate(frac_categories)}

    fig, ax = plt.subplots(figsize=figsize)

    # Stacked bands: iterate in reverse so first category appears on top of legend
    sorted_names = list(frac_categories.keys())
    bottoms = np.zeros(nbins)
    for name in sorted_names:
        vals = frac_categories[name]
        ax.fill_between(
            centers, bottoms, bottoms + vals,
            step="mid", color=colors[name], alpha=0.7, linewidth=0, label=name,
        )
        bottoms = bottoms + vals

    # Total uncertainty band (gray)
    ax.fill_between(
        centers, np.zeros_like(total_frac), total_frac,
        step="mid", color="gray", alpha=0.3, linewidth=0.8, edgecolor="black",
        label="Total",
    )

    # Data statistical uncertainty (dark band)
    if data_stat_hist is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            stat_frac = np.where(nom_counts > 0, data_stat_hist.errors / nom_counts, 0.0)
        ax.fill_between(
            centers, np.zeros_like(stat_frac), stat_frac,
            step="mid", color="black", alpha=0.5, linewidth=0, label="Stat.",
        )

    xlabel, _ = _resolve_label_unit(variable, xlabel)
    ax.set_xlabel(xlabel or "", loc="right")
    ax.set_ylabel(r"Fractional uncertainty", loc="top")
    _add_legend(ax, fontsize="x-small", ncol=2)
    if atlas:
        reserve_atlas_label_headroom(ax)
        add_atlas_label(ax, status="Internal")


    if save_name is not None:
        _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
    return fig, ax


# --------------------------------------------------------------------------
# Fit-related plots
# --------------------------------------------------------------------------

def plot_data_with_fit(
    xdata,
    ydata,
    yerr,
    fit_func,
    popt,
    *,
    xdescr="",
    ydescr="Counts",
    fit_label="",
    script_id=None,
    save_name=None,
    save_path=None,
    xscaling=1,
    yscaling=1,
    bins=None,
    unit=None,
    intervall=None,
    multiproc=False,
    fitting_method="chi_squared",
    atlas=False,
    figsize=None,
    verbose=True,
):
    from .stats import chi_squared, poisson_likelihood

    if figsize is None:
        figsize = config.fig_size_wide
    fig, ax = plt.subplots(figsize=figsize)
    label = None
    if bins is None:
        label = "Data (" + _fmt_num(f"{len(xdata)}") + " points)"
    ax.plot(
        np.array(xdata) * xscaling, np.array(ydata) * yscaling,
        marker=".", markerfacecolor=catppuccin_colors["red"],
        linestyle=" ", linewidth=0.8, label=label,
    )

    if intervall is None:
        x_fit = np.linspace(np.min(xdata), np.max(xdata), 200)
        if fitting_method == "chi_squared":
            chi2 = chi_squared(fit_func, xdata=xdata, ydata=ydata, sigma=yerr, params=popt)
            chi2_red = chi2 / (len(xdata) - len(popt))
            fit_label += r" $\chi_\text{red}^2 = " + _fmt_num(f"{chi2_red:.2f}") + r"$"
        elif fitting_method == "poisson":
            lik = poisson_likelihood(func=fit_func, xdata=xdata, ydata=ydata, params=popt)
            fit_label += r" $\mathcal{L} = " + _fmt_num(f"{lik:.3f}") + r"$"
    else:
        i0, i1 = intervall
        x_fit = np.linspace(np.min(xdata[i0:i1]), np.max(xdata[i0:i1]), 200)
        if fitting_method == "chi_squared":
            chi2 = chi_squared(fit_func, xdata=xdata[i0:i1], ydata=ydata[i0:i1], sigma=yerr[i0:i1], params=popt)
            chi2_red = chi2 / (len(xdata[i0:i1]) - len(popt))
            fit_label += r" $\chi_\text{red}^2 = " + _fmt_num(f"{chi2_red:.2f}") + r"$"
        elif fitting_method == "poisson":
            lik = poisson_likelihood(func=fit_func, xdata=xdata[i0:i1], ydata=ydata[i0:i1], params=popt)
            fit_label += r" $\mathcal{L} = " + _fmt_num(f"{lik:.3f}") + r"$"

    ax.plot(x_fit * xscaling, fit_func(x_fit, *popt) * yscaling,
            linestyle="-", color=catppuccin_colors["green"], label=fit_label)

    if bins is not None:
        bin_width = (xdata[1] - xdata[0]) * xscaling
        if unit is not None:
            ydescr += r"$/" + _fmt_si(f"{bin_width:.1f}", unit) + r"$"
        else:
            ydescr += r"$/" + _fmt_num(f"{bin_width:.1f}") + r"$"

    ax.set_xlabel(xdescr, loc="right")
    ax.set_ylabel(ydescr, loc="top")
    _add_legend(ax, fontsize="x-small")
    if atlas:
        reserve_atlas_label_headroom(ax)
        add_atlas_label(ax, status="Internal")

    if save_name is not None:
        _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
    return fig, ax
