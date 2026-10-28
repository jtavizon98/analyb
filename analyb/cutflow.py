"""Cutflow chart: events surviving each sequential Stage."""

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class CutflowData:
    """Per-stage yields, optional uncertainties and relative efficiencies.

    ``errors=None`` means the uncertainty is unknown; it is never drawn.
    ``errors="poisson"`` requests ``sqrt(counts)``, which is valid only for raw
    unweighted event counts.
    """

    def __init__(self, stages, counts, errors=None, label=None):
        self.stages = list(stages)
        self.counts = np.asarray(counts, dtype=float)
        if self.counts.shape != (len(self.stages),):
            raise ValueError("cutflow needs one count per stage")
        if isinstance(errors, str):
            if errors != "poisson":
                raise ValueError(f"errors must be an array, None or 'poisson', got {errors!r}")
            if np.any(self.counts < 0):
                raise ValueError("Poisson errors require nonnegative raw counts")
            errors = np.sqrt(self.counts)
        if errors is not None:
            errors = np.asarray(errors, dtype=float)
            if errors.shape != self.counts.shape:
                raise ValueError("cutflow errors must match counts")
        self.errors = errors
        self.label = label

    @property
    def eff_rel(self):
        """Each Stage vs previous: [1.0, N1/N0, N2/N1, ...]."""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.zeros_like(self.counts)
            out[0] = 1.0
            prev = self.counts[:-1]
            cur = self.counts[1:]
            out[1:] = np.where(prev > 0, cur / prev, np.nan)
        return out

    @property
    def eff_abs(self):
        """Yield relative to initial: [1.0, N1/N0, ..., Nk/N0]."""
        with np.errstate(divide="ignore", invalid="ignore"):
            base = self.counts[0]
            out = np.where(base > 0, self.counts / base, np.nan)
            out[0] = 1.0
            return out

    def __repr__(self):
        return f"CutflowData(stages={len(self.stages)}, initial={self.counts[0]:.3g}, final={self.counts[-1]:.3g})"


# Relative excess of |C| over its Cauchy-Schwarz bound sqrt(Q_A Q_B) accepted
# as accumulation roundoff and clipped; any larger excess is rejected.
_MOMENT_RTOL = 1e-12


def nested_efficiency(passed, total, passed_sumw2, total_sumw2, *, cross_sumw=None):
    """Ratio ``eps = A / B`` of nested weighted yields and its first-order variance.

    With ``A = passed`` (sum of stage-A weights over the passing events),
    ``B = total`` (sum of stage-B weights over the denominator events),
    ``Q_A``/``Q_B`` their sumw2 and ``C = sum_passed w_A w_B``, the
    delta-method variance is

        V = (Q_A + eps^2 Q_B - 2 eps C) / B^2.

    Model and caller obligations:

    - ``Q_A`` and ``Q_B`` are independent-event sumw2 (each event contributes
      its weight squared). ``V`` is a first-order (delta-method) statistical
      estimate; systematic uncertainties are not included.
    - The default ``cross_sumw=None`` sets ``C = Q_A``. This holds only when the
      passing events are a genuine subset of the denominator events and carry
      identical weights at both stages. It is evaluated as the equivalent
      ``V = ((1 - eps)^2 Q_A + eps^2 (Q_B - Q_A)) / B^2``.
    - If weights change between the stages, histogram sums cannot recover
      ``C``: pass the event-level ``cross_sumw`` or treat the uncertainty as
      unknown.

    Arguments are scalars or arrays of one common shape. Weights may be signed,
    so ``eps`` may lie outside [0, 1]; it is never clipped. A ``None`` or NaN
    second moment means unknown and gives ``var = NaN`` while ``eps`` is kept.
    Negative or infinite sumw2 and covariances violating ``|C| <= sqrt(Q_A Q_B)``
    (for the default, ``Q_A <= Q_B``) by more than roundoff raise ``ValueError``.

    Returns ``(eps, var, valid)``. ``valid`` marks a defined central ratio
    (finite ``A`` and ``B``, ``B != 0``); ``eps`` and ``var`` are NaN elsewhere.
    ``valid`` does not certify Gaussian coverage of ``eps +- sqrt(var)``.
    """
    a = np.asarray(passed, dtype=np.float64)
    b = np.asarray(total, dtype=np.float64)

    def supplied(value, name, signed=False):
        """Shape-checked moment; every non-NaN entry is validated on its own."""
        if value is None:
            return None
        value = np.asarray(value, dtype=np.float64)
        if value.shape != a.shape:
            raise ValueError("nested_efficiency arguments must have one common shape")
        given = value[~np.isnan(value)]
        if np.any(np.isinf(given)) or (not signed and np.any(given < 0)):
            kind = "finite" if signed else "finite and nonnegative"
            raise ValueError(f"{name} must be {kind}; NaN marks unknown")
        return value

    if b.shape != a.shape:
        raise ValueError("nested_efficiency arguments must have one common shape")
    q_a = supplied(passed_sumw2, "passed_sumw2")
    q_b = supplied(total_sumw2, "total_sumw2")
    cross = supplied(cross_sumw, "cross_sumw", signed=True)

    valid = np.isfinite(a) & np.isfinite(b) & (b != 0)
    eps = np.full(a.shape, np.nan)
    np.divide(a, b, out=eps, where=valid)
    var = np.full(a.shape, np.nan)
    if q_a is None or q_b is None:
        return eps[()], var[()], valid[()]

    # Covariance consistency needs every moment of an entry. Square roots keep
    # the Cauchy-Schwarz bound representable for extreme but finite moments.
    known = ~(np.isnan(q_a) | np.isnan(q_b))
    if cross is not None:
        known &= ~np.isnan(cross)
    root_a, root_b = np.sqrt(q_a[known]), np.sqrt(q_b[known])
    if cross is None:
        if np.any(q_a[known] - q_b[known] > _MOMENT_RTOL * q_b[known]):
            raise ValueError("passed_sumw2 exceeds total_sumw2: not a same-weight "
                             "subset of the denominator events; pass cross_sumw")
    elif np.any(np.abs(cross[known]) - root_a * root_b > _MOMENT_RTOL * root_a * root_b):
        raise ValueError("|cross_sumw| exceeds sqrt(passed_sumw2 * total_sumw2)")

    # Evaluate V from moments scaled by 1/B^2 without forming B^2 itself, so
    # that common-weight rescaling cannot overflow or underflow.
    use = valid & known
    e, scale = eps[use], np.abs(b[use])
    qa, qb = q_a[use], q_b[use]
    scaled_a = (np.sqrt(qa) / scale) ** 2
    if cross is None:
        failed_sumw2 = np.maximum(qb - qa, 0.0)  # clip accepted roundoff only
        var[use] = (1 - e) ** 2 * scaled_a + e ** 2 * (np.sqrt(failed_sumw2) / scale) ** 2
    else:
        limit = np.sqrt(qa) * np.sqrt(qb)
        scaled_c = np.clip(cross[use], -limit, limit) / scale / scale
        scaled_b = (np.sqrt(qb) / scale) ** 2
        # A positive semidefinite covariance gives V >= 0; clip cancellation only.
        var[use] = np.maximum(scaled_a + e ** 2 * scaled_b - 2 * e * scaled_c, 0.0)
    return eps[()], var[()], valid[()]


def cutflow_chart(
    cutflow,
    *,
    mode="absolute",
    script_id=None,
    save_name="cutflow",
    save_path=None,
    atlas=False,
    color=None,
    figsize=None,
    verbose=True,
    measurement=None,
):
    """Plot a cutflow chart.

    mode : "absolute" (yields per stage) | "relative" (efficiency vs previous stage)
    measurement : optional measured quantity for the y label. Explicit object-bin
        labels prevent nonsequential diagnostics being described as efficiencies.
    """
    from . import config
    from .plot import save_figure

    if plt is None:
        raise ImportError("matplotlib required")
    if figsize is None:
        figsize = config.fig_size_wide
    if color is None:
        from .colors import catppuccin_colors
        color = catppuccin_colors["blue"]
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(cutflow.stages))
    if mode == "absolute":
        y = cutflow.counts
        ylabel = cutflow_ylabel(measurement or "events")
    elif mode == "relative":
        y = cutflow.eff_rel * 100
        ylabel = cutflow_ylabel(measurement or "cut_efficiency") + " (%)"
    else:
        raise ValueError(f"mode must be 'absolute' or 'relative', got {mode!r}")

    ax.bar(x, y, color=color, alpha=0.85, edgecolor=color)
    # Relative efficiencies carry no uncertainty model; unknown errors stay undrawn.
    if mode == "absolute" and cutflow.errors is not None:
        ax.errorbar(x, y, yerr=cutflow.errors, fmt="none", color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cutflow.stages, rotation=20, ha="right", fontsize="x-small")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, loc="top")

    if atlas:
        from .plot import add_atlas_label, reserve_atlas_label_headroom
        reserve_atlas_label_headroom(ax)
        add_atlas_label(ax, status="Internal")

    pass  # constrained_layout enabled globally via matplotlibrc
    save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
    return fig, ax


def cutflow_ylabel(measurement):
    """Name a measured cutflow quantity without asserting nesting of object bins."""
    labels = {"events": "Events", "raw_events": "Raw events",
              "raw_objects": "Raw objects", "survival": "Survival",
              "cut_efficiency": "Cut efficiency",
              "milestone_retention": "Milestone retention",
              "objects_first_bin": "Objects / first bin",
              "object_bin_ratio": "Object-bin ratio"}
    try:
        return labels[measurement]
    except KeyError:
        raise ValueError(f"unknown cutflow measurement: {measurement!r}") from None


def cutflow_comparison(stages, numerator, reference, *, numerator_errors=None,
                       reference_errors=None, normalized=False, weighted=False,
                       numerator_label="FF", reference_label="IR", figsize=(7.2, 6.4)):
    """Compare explicitly aligned cumulative stages; absent errors stay absent.

    The caller owns stage correspondence and population definitions. Normalized
    counters use the initial central value as a fixed scale; errors are scaled
    likewise and are not efficiency uncertainties. Undefined denominators are
    masked. Reference errors and numerator-only ratio errors are drawn separately.
    """
    from .colors import catppuccin_colors
    from .plot import _data_errorbar, add_atlas_label
    a, b = np.asarray(numerator, float), np.asarray(reference, float)
    if a.shape != b.shape or a.shape != (len(stages),):
        raise ValueError("cutflow stages and arrays must have equal length")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("cutflow values must be finite")
    ae = None if numerator_errors is None else np.asarray(numerator_errors, float)
    be = None if reference_errors is None else np.asarray(reference_errors, float)
    for error in (ae, be):
        if error is not None and (error.shape != a.shape or not np.all(np.isfinite(error)) or np.any(error < 0)):
            raise ValueError("invalid cutflow errors")
    if normalized:
        if a[0] == 0 or b[0] == 0:
            raise ValueError("zero initial cutflow cannot be normalized")
        if ae is not None: ae = ae / abs(a[0])
        if be is not None: be = be / abs(b[0])
        a, b = a / a[0], b / b[0]
    fig, (main, ratio_ax) = plt.subplots(2, 1, sharex=True, figsize=figsize,
                                        gridspec_kw={'height_ratios': [3,1]})
    x = np.arange(len(stages));edges = np.arange(len(stages)+1)-.5
    red = catppuccin_colors['red']
    main.bar(x, a, width=1., color=red, alpha=.85, label=numerator_label)
    if ae is not None:
        main.errorbar(x,a,yerr=ae,fmt='none',color=red,linewidth=.7)
    _data_errorbar(main,x,b,yerr=be,color='black',label=reference_label)
    main.set_ylabel('Fraction of initial yield' if normalized else ('Weighted events' if weighted else 'Events'),loc='top')
    low=min(0.,float(min(a.min(),b.min())))
    high=max(float(a.max()),float(b.max()))
    main.set_ylim(low,high*1.45 if high>0 else 1.)
    main.legend(loc='upper right',fontsize='small')
    add_atlas_label(main,status='Internal',lumi=False)
    valid=b!=0;ratio=np.full_like(a,np.nan);ratio[valid]=a[valid]/b[valid]
    ratio_ax.stairs(ratio,edges,color=red,linewidth=1.2)
    if ae is not None:
        err=np.full_like(a,np.nan);err[valid]=ae[valid]/abs(b[valid])
        lo,hi=ratio-err,ratio+err
        ratio_ax.fill_between(edges,np.r_[lo,lo[-1]],np.r_[hi,hi[-1]],step='post',color=red,alpha=.2)
    unity=np.where(valid,1.,np.nan)
    berr=None if be is None else np.divide(be,abs(b),out=np.full_like(b,np.nan),where=valid)
    _data_errorbar(ratio_ax,x,unity,yerr=berr,color='black')
    ratio_ax.axhline(1,color='black',linestyle='--',linewidth=.7)
    finite=ratio[np.isfinite(ratio)]
    ratio_low=min(.5,float(finite.min())) if finite.size else .5
    ratio_high=max(1.5,float(finite.max())) if finite.size else 1.5
    ratio_padding=.08*(ratio_high-ratio_low)
    ratio_ax.set_ylim(ratio_low-ratio_padding,ratio_high+ratio_padding)
    ratio_ax.set_ylabel('FF / IR');ratio_ax.set_xlabel('Selection stage',loc='right')
    ratio_ax.set_xticks(x);ratio_ax.set_xticklabels(stages,rotation=28,ha='right',fontsize='x-small')
    for ax in (main,ratio_ax):
        ax.set_xlim(edges[0],edges[-1]);ax.grid(True,alpha=.5);ax.set_axisbelow(True)
    return fig,(main,ratio_ax)


def native_cutflow_chart(stages, values, *, errors=None, label=None, color=None,
                         xlim=None, figsize=(7.2, 6.4)):
    """Draw every source stage separately; absent uncertainties remain absent.

    Horizontal bars accommodate native labels without asserting correspondence
    to another source's selection sequence. Values are signed sums of weights.
    """
    import textwrap
    from .colors import catppuccin_colors
    values = np.asarray(values, float)
    if values.shape != (len(stages),) or not np.all(np.isfinite(values)):
        raise ValueError("one finite value is required per stage")
    if errors is not None:
        errors = np.asarray(errors, float)
        if errors.shape != values.shape or np.any(errors < 0) or not np.all(np.isfinite(errors)):
            raise ValueError("invalid cutflow errors")
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(stages))
    ax.barh(y, values, color=color or catppuccin_colors['red'], alpha=.85, height=.75)
    if errors is not None:
        ax.errorbar(values, y, xerr=errors, fmt='none', color='black', linewidth=.7)
    ax.set_yticks(y)
    # Inherit ytick.labelsize so plotbook_context() scales the stage labels.
    ax.set_yticklabels([textwrap.fill(str(s).replace('_', ' '), 25) for s in stages])
    ax.invert_yaxis()
    from matplotlib.ticker import FuncFormatter
    ax.set_xlabel(r'Weighted yield [$10^6$]', loc='right')
    ax.set_ylabel('Native cut', loc='top')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value / 1e6:g}'))
    if label:
        ax.set_title(label, loc='right')
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, axis='x', alpha=.5)
    ax.grid(False, axis='y')
    ax.set_axisbelow(True)
    return fig, ax
