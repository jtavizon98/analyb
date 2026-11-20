"""Histogram filling (numpy/scipy based) and Hist1D/Hist2D classes."""

from typing import NamedTuple, Optional

import numpy as np
from scipy import stats

from .binning import resolve_bins, _data_range


class Hist1D:
    """1D histogram with counts, edges, errors, optional overflow/underflow."""

    def __init__(
        self,
        counts,
        edges,
        errors=None,
        label=None,
        unit=None,
        overflow=0,
        underflow=0,
        is_weighted=False,
        bin_width=None,
        is_density=False,
    ):
        self.counts = np.asarray(counts, dtype=np.float64)
        self.edges = np.asarray(edges, dtype=np.float64)
        if self.edges.ndim != 1 or self.edges.size != self.counts.size + 1:
            raise ValueError("Hist1D edges must have length len(counts) + 1")
        if not np.all(np.isfinite(self.edges)) or np.any(np.diff(self.edges) <= 0):
            raise ValueError("Hist1D edges must be finite and strictly increasing")
        if errors is None:
            if is_weighted:
                raise ValueError(
                    "Hist1D with is_weighted=True requires explicit errors. "
                    "Use hist1d() which computes them correctly."
                )
            errors = np.sqrt(self.counts)
        self.errors = np.asarray(errors, dtype=np.float64)
        if self.errors.shape != self.counts.shape:
            raise ValueError("Hist1D errors must have the same shape as counts")
        self.label = label
        self.unit = unit
        self.overflow = float(overflow)
        self.underflow = float(underflow)
        self.is_weighted = is_weighted
        self.is_density = bool(is_density)
        if bin_width is None and self.is_uniform and not self.is_density:
            bin_width = float(self.edges[1] - self.edges[0])
        self.bin_width = None if self.is_density or not self.is_uniform else bin_width
        self.n_entries = self.integral()

    @property
    def nbins(self):
        return self.counts.size

    @property
    def centers(self):
        return (self.edges[:-1] + self.edges[1:]) / 2

    @property
    def widths(self):
        """Displayed bin widths."""
        return np.diff(self.edges)

    @property
    def is_uniform(self):
        """Whether all displayed bins have equal widths, within float precision."""
        widths = self.widths
        return widths.size <= 1 or np.allclose(widths, widths[0], rtol=1e-10, atol=1e-12)

    def integral(self):
        if self.is_density:
            return float(np.sum(self.counts * self.widths))
        return float(np.sum(self.counts))

    def integral_error(self):
        if self.is_density:
            return float(np.sqrt(np.sum((self.errors * self.widths) ** 2)))
        if self.is_weighted:
            return float(np.sqrt(np.sum(self.errors ** 2)))
        return float(np.sqrt(self.integral()))

    def copy(self):
        return Hist1D(
            self.counts.copy(),
            self.edges.copy(),
            self.errors.copy(),
            label=self.label,
            unit=self.unit,
            overflow=self.overflow,
            underflow=self.underflow,
            is_weighted=self.is_weighted,
            bin_width=self.bin_width,
            is_density=self.is_density,
        )

    def __repr__(self):
        return (
            f"Hist1D(nbins={self.nbins}, range=[{self.edges[0]:.3g}, "
            f"{self.edges[-1]:.3g}], integral={self.integral():.3g})"
        )

    def __add__(self, other):
        if not isinstance(other, Hist1D):
            return NotImplemented
        return sum_histograms([self, other])


class Hist2D:
    """2D histogram with counts, x_edges, y_edges, and optional errors."""

    def __init__(
        self,
        counts,
        x_edges,
        y_edges,
        errors=None,
        x_label=None,
        y_label=None,
        x_unit=None,
        y_unit=None,
    ):
        self.counts = np.asarray(counts, dtype=np.float64)
        self.x_edges = np.asarray(x_edges, dtype=np.float64)
        self.y_edges = np.asarray(y_edges, dtype=np.float64)
        if errors is None:
            errors = np.sqrt(self.counts)
        self.errors = np.asarray(errors, dtype=np.float64)
        if self.errors.shape != self.counts.shape:
            raise ValueError("Hist2D errors must have the same shape as counts")
        self.x_label = x_label
        self.y_label = y_label
        self.x_unit = x_unit
        self.y_unit = y_unit

    @property
    def shape(self):
        return self.counts.shape

    def integral(self):
        return float(np.sum(self.counts))

    def copy(self):
        return Hist2D(
            self.counts.copy(),
            self.x_edges.copy(),
            self.y_edges.copy(),
            errors=self.errors.copy(),
            x_label=self.x_label,
            y_label=self.y_label,
            x_unit=self.x_unit,
            y_unit=self.y_unit,
        )

    def __repr__(self):
        return (
            f"Hist2D(shape={self.shape}, integral={self.integral():.3g})"
        )


#--------------------------------------------------------------------------
# Filling
# --------------------------------------------------------------------------

def hist1d(
    values,
    weights=None,
    bins="scott",
    range=None,
    pct_range=None,
    overflow=True,
    merge_overflow=False,
    label=None,
    unit=None,
):
    """Fill a 1D histogram.

    Parameters
    ----------
    bins : str | int | array-like | float
        Binning rule ("scott", "sturges", "fd"), a fixed number of bins, an explicit
        edge array, or a bin width.
    range : (lo, hi) | None
        Explicit binning range (out-of-range values become overflow/underflow unless
        merge_overflow is set).
    pct_range : (lo_pct, hi_pct) | None
        Percentile-based range (used only when `range` is None).
    overflow : bool
        If True (default), count events outside [edges[0], edges[-1]] in
        `Hist1D.overflow`/`underflow`. If False, the numpy default discards them and
        keeps only the in-range counts.
    merge_overflow : bool
        Add overflow/underflow counts into the first/last bin (ATLAS common option).
    """
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        values = values[finite_mask]
        if weights is not None:
            weights = np.asarray(weights, dtype=float)[finite_mask]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.size != values.size:
            raise ValueError("weights and values must have the same length")

    edges = resolve_bins(bins, values, range=range, pct_range=pct_range)
    is_weighted = weights is not None

    if overflow:
        lo, hi = edges[0], edges[-1]
        in_range = (values >= lo) & (values < hi)
        uf = float(np.sum(weights[~in_range & (values < lo)])) if is_weighted \
            else float(np.sum(values < lo))
        of = float(np.sum(weights[~in_range & (values >= hi)])) if is_weighted \
            else float(np.sum(values >= hi))
        counts, _ = np.histogram(values[in_range], bins=edges, weights=weights[in_range] if is_weighted else None)
    else:
        uf = 0.0
        of = 0.0
        counts, _ = np.histogram(values, bins=edges, weights=weights)

    if is_weighted:
        # Compute sum-of-weights² for the error on each bin.
        w2 = weights ** 2
        if overflow:
            err, _ = np.histogram(
                values[in_range], bins=edges, weights=w2[in_range]
            )
            err = np.sqrt(err)
        else:
            err, _ = np.histogram(values, bins=edges, weights=w2)
            err = np.sqrt(err)
    else:
        err = np.sqrt(counts)

    if merge_overflow:
        counts[0] += uf
        counts[-1] += of
        if is_weighted:
            uf_w2 = float(np.sum(w2[~in_range & (values < lo)])) if overflow \
                else float(np.sum(w2[values < lo]))
            of_w2 = float(np.sum(w2[~in_range & (values >= hi)])) if overflow \
                else float(np.sum(w2[values >= hi]))
            err[0] = np.sqrt(err[0] ** 2 + uf_w2)
            err[-1] = np.sqrt(err[-1] ** 2 + of_w2)
        else:
            err[0] = np.sqrt(err[0] ** 2 + uf)
            err[-1] = np.sqrt(err[-1] ** 2 + of)
        uf = 0.0
        of = 0.0

    return Hist1D(
        counts=counts,
        edges=edges,
        errors=err,
        overflow=of,
        underflow=uf,
        is_weighted=is_weighted,
        label=label,
        unit=unit,
    )


def hist2d(
    x,
    y,
    bins=50,
    range=None,
    weights=None,
    x_label=None,
    y_label=None,
    x_unit=None,
    y_unit=None,
):
    """Fill a 2D histogram.

    When ``weights`` are given, errors are computed as sqrt of the
    sum-of-weights-squared per bin (Poisson-like for weighted data).
    Uses a single-pass np.bincount for both counts and error².
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[finite]
    if isinstance(bins, (list, tuple)) and len(bins) == 2:
        bins_x, bins_y = bins
    else:
        bins_x = bins_y = bins
    if range is not None:
        range_x, range_y = range
    else:
        range_x = range_y = None
    x_edges = resolve_bins(bins_x, x, range=range_x)
    y_edges = resolve_bins(bins_y, y, range=range_y)
    nx, ny = len(x_edges) - 1, len(y_edges) - 1

    ix = np.digitize(x, x_edges) - 1
    iy = np.digitize(y, y_edges) - 1
    in_range = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = ix[in_range] * ny + iy[in_range]

    if weights is not None:
        w = weights[in_range]
        counts_flat = np.bincount(flat, weights=w, minlength=nx * ny)
        err2_flat = np.bincount(flat, weights=w * w, minlength=nx * ny)
    else:
        counts_flat = np.bincount(flat, minlength=nx * ny)
        err2_flat = counts_flat.astype(np.float64)

    counts = counts_flat.reshape(nx, ny)
    errors = np.sqrt(err2_flat.reshape(nx, ny))

    return Hist2D(
        counts=counts,
        x_edges=x_edges,
        y_edges=y_edges,
        errors=errors,
        x_label=x_label,
        y_label=y_label,
        x_unit=x_unit,
        y_unit=y_unit,
    )


#--------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------

def _check_rebin_factor(factor, nbins, axis="", hint=True):
    if (isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, np.integer))
            or factor < 1):
        raise ValueError("rebin factor must be a positive integer")
    if nbins % factor:
        raise ValueError(
            f"rebin factor {factor} must exactly divide the {nbins} {axis}bins"
            + ("; use rebin_to_edges() for an explicit trailing bin" if hint else "")
        )


def rebin(hist, factor=2):
    """Merge `factor` adjacent bins. Returns a new Hist1D/Hist2D.

    The factor must divide the number of bins on each axis, so the summed
    counts, stored variances and flows are preserved up to roundoff; a
    trailing partial group is never dropped. Density histograms are merged
    through raw bin yields. Merged variances add the stored per-bin
    variances, which assumes bins are uncorrelated.
    """
    if isinstance(hist, Hist2D):
        fx, fy = factor if isinstance(factor, (list, tuple)) else (factor, factor)
        n, m = hist.counts.shape
        _check_rebin_factor(fx, n, "x ", hint=False)
        _check_rebin_factor(fy, m, "y ", hint=False)
        counts = hist.counts.reshape(n // fx, fx, m // fy, fy).sum(axis=(1, 3))
        err2 = (hist.errors ** 2).reshape(n // fx, fx, m // fy, fy).sum(axis=(1, 3))
        return Hist2D(
            counts,
            hist.x_edges[::fx],
            hist.y_edges[::fy],
            errors=np.sqrt(err2),
            x_label=hist.x_label, y_label=hist.y_label,
            x_unit=hist.x_unit, y_unit=hist.y_unit,
        )
    _check_rebin_factor(factor, hist.nbins)
    widths = hist.widths if hist.is_density else 1.0
    raw_counts = (hist.counts * widths).reshape(-1, factor).sum(axis=1)
    raw_err2 = ((hist.errors * widths) ** 2).reshape(-1, factor).sum(axis=1)
    edges = hist.edges[::factor]
    new_widths = np.diff(edges) if hist.is_density else 1.0
    return Hist1D(
        counts=raw_counts / new_widths,
        edges=edges,
        errors=np.sqrt(raw_err2) / new_widths,
        label=hist.label,
        unit=hist.unit,
        overflow=hist.overflow,
        underflow=hist.underflow,
        is_weighted=hist.is_weighted,
        is_density=hist.is_density,
    )


def normalize(hist, to=1.0):
    """Return a new Hist1D rescaled so that the integral equals `to`."""
    integral = hist.integral()
    if integral == 0:
        scale = 0.0
    else:
        scale = to / integral
    return Hist1D(
        counts=hist.counts * scale,
        edges=hist.edges.copy(),
        errors=hist.errors * scale,
        label=hist.label, unit=hist.unit,
        overflow=hist.overflow * scale,
        underflow=hist.underflow * scale,
        is_weighted=hist.is_weighted,
        is_density=hist.is_density,
    )


def ratio(num, den, nan_to=None, include_denominator_error=True):
    """Point-by-point ratio ``num / den`` with propagated errors.

    By default, independent numerator and denominator uncertainties are
    combined.  Set ``include_denominator_error=False`` when the denominator
    uncertainty is displayed separately (for example, as an MC ratio band).
    Bins where ``divide`` is undefined (zero denominator or a non-finite
    input) are NaN unless ``nan_to`` names an explicit fill value.  Both
    histograms must share their bin edges up to floating-point roundoff.
    """
    num = num if isinstance(num, Hist1D) else Hist1D(num, np.arange(len(num) + 1))
    den = den if isinstance(den, Hist1D) else Hist1D(den, np.arange(len(den) + 1))
    require_comparable(num, den, edges="roundoff", operation="ratio")
    r, r_err, _ = divide(num.counts, num.errors, den.counts,
                         den.errors if include_denominator_error else None)
    if nan_to is not None:
        nan_mask = ~np.isfinite(r)
        r = np.where(nan_mask, nan_to, r)
        r_err = np.where(nan_mask | ~np.isfinite(r_err), nan_to, r_err)
    return r, r_err


def sum_histograms(hists, label=None):
    """Sum a list or dict of Hist1D objects into one.

    Counts, underflow and overflow are summed; errors are propagated as
    sqrt(sum of squares) for weighted (or Poisson) uncertainties.

    Parameters
    ----------
    hists : list[Hist1D] | dict[str, Hist1D]
        Histograms to sum.  All must have the same edges up to roundoff
        (the rule of ``ratio``); the output takes the first one's edges.
    label : str or None
        Label for the output histogram.  If None and *hists* is a dict,
        the joined keys are used.

    Returns
    -------
    Hist1D
    """
    if isinstance(hists, dict):
        if label is None:
            label = " + ".join(hists.keys())
        values = list(hists.values())
    else:
        values = list(hists)

    if not values:
        raise ValueError("sum_histograms: need at least one histogram")

    ref = values[0]
    edges = ref.edges
    total_counts = np.zeros_like(ref.counts)
    total_err2 = np.zeros_like(ref.counts)
    overflow = underflow = 0.0

    require_comparable(ref, *values[1:], edges="roundoff", operation="sum_histograms")
    for h in values:
        total_counts += h.counts
        total_err2 += h.errors ** 2
        overflow += h.overflow
        underflow += h.underflow

    return Hist1D(
        counts=total_counts,
        edges=edges.copy(),
        errors=np.sqrt(total_err2),
        label=label,
        unit=ref.unit,
        overflow=overflow,
        underflow=underflow,
        is_weighted=True,
        is_density=ref.is_density,
    )


def sum_histograms2d(hists, x_label=None, y_label=None, x_unit=None, y_unit=None):
    """Sum a list or dict of Hist2D objects on one grid (roundoff edge rule).

    Counts are summed bin-wise; errors are propagated as
    sqrt(sum of squares).

    Parameters
    ----------
    hists : list[Hist2D] | dict[str, Hist2D]
        Histograms to sum.  All must have the same x_edges and y_edges up to
        roundoff; the output takes the first one's edges.
    x_label, y_label : str or None
        Labels for the output.  Fall back to the first histogram's labels.
    x_unit, y_unit : str or None
        Units for the output.  Fall back to the first histogram's units.

    Returns
    -------
    Hist2D
    """
    if isinstance(hists, dict):
        values = list(hists.values())
    else:
        values = list(hists)

    if not values:
        raise ValueError("sum_histograms2d: need at least one histogram")

    ref = values[0]
    require_comparable(ref, *values[1:], edges="roundoff", operation="sum_histograms2d")

    total_counts = np.zeros_like(ref.counts)
    total_err2 = np.zeros_like(ref.counts)

    for h in values:
        total_counts += h.counts
        total_err2 += h.errors ** 2

    return Hist2D(
        counts=total_counts,
        x_edges=ref.x_edges.copy(),
        y_edges=ref.y_edges.copy(),
        errors=np.sqrt(total_err2),
        x_label=x_label if x_label is not None else ref.x_label,
        y_label=y_label if y_label is not None else ref.y_label,
        x_unit=x_unit if x_unit is not None else ref.x_unit,
        y_unit=y_unit if y_unit is not None else ref.y_unit,
    )


#--------------------------------------------------------------------------
# MC luminosity scaling
# --------------------------------------------------------------------------

def scale_to_luminosity(hist, cross_section, luminosity, sum_of_weights):
    """Return a new Hist1D scaled so MC matches data luminosity.

    The scaling factor is ``σ · L / Σw``, where:
      - *cross_section*  — MC cross-section in pb
      - *luminosity*     — integrated luminosity in pb⁻¹
      - *sum_of_weights* — sum of event weights before selection
        (``sum(df['eventWeight'])`` over all events in the sample)

    Both counts and errors are scaled.
    """
    if sum_of_weights == 0:
        raise ValueError("sum_of_weights must be non-zero for luminosity scaling")
    scale = cross_section * luminosity / sum_of_weights
    return Hist1D(
        counts=hist.counts * scale,
        edges=hist.edges.copy(),
        errors=hist.errors * scale,
        label=hist.label,
        unit=hist.unit,
        overflow=hist.overflow * scale,
        underflow=hist.underflow * scale,
        is_weighted=hist.is_weighted,
        is_density=hist.is_density,
    )


#--------------------------------------------------------------------------
# Edge-matching / rebin to arbitrary edges
# --------------------------------------------------------------------------

def rebin_to_edges(hist, target_edges):
    """Merge whole bins of a Hist1D onto aligned *target_edges*.

    Every target edge must be a source edge (up to roundoff) and the target
    must span exactly the source range, so counts, variances, underflow and
    overflow are preserved up to roundoff.  Splitting a bin, widening or narrowing
    the range raises ``ValueError``: Hist1D flows carry no variance, so bins
    cannot be moved into them.  Density histograms are merged through raw bin
    yields.  The output adopts an exact copy of *target_edges*.
    """
    target = np.array(target_edges, dtype=np.float64)
    if (target.ndim != 1 or target.size < 2
            or not np.all(np.isfinite(target)) or np.any(np.diff(target) <= 0)):
        raise ValueError("target_edges must be finite and strictly increasing")
    matched = _aligned_edge_indices(hist.edges, target)
    if matched[0] != 0 or matched[-1] != hist.nbins:
        raise ValueError("target_edges must span exactly the source range")
    widths = hist.widths if hist.is_density else 1.0
    raw_counts = np.add.reduceat(hist.counts * widths, matched[:-1])
    raw_variances = np.add.reduceat((hist.errors * widths) ** 2, matched[:-1])
    new_widths = np.diff(target) if hist.is_density else 1.0
    return Hist1D(
        counts=raw_counts / new_widths, edges=target,
        errors=np.sqrt(raw_variances) / new_widths,
        label=hist.label, unit=hist.unit,
        overflow=hist.overflow, underflow=hist.underflow,
        is_weighted=hist.is_weighted,
        is_density=hist.is_density,
    )


def divide_by_bin_width(hist):
    """Return a differential copy of *hist*, divided by each displayed bin width.

    Underflow and overflow remain raw out-of-range yields because they have no
    finite displayed-bin width.  The returned histogram has ``is_density=True``.
    """
    if not isinstance(hist, Hist1D):
        raise TypeError("divide_by_bin_width expects a Hist1D")
    if hist.is_density:
        raise ValueError("histogram is already divided by bin width")
    widths = hist.widths
    return Hist1D(
        counts=hist.counts / widths,
        edges=hist.edges.copy(),
        errors=hist.errors / widths,
        label=hist.label,
        unit=hist.unit,
        overflow=hist.overflow,
        underflow=hist.underflow,
        is_weighted=hist.is_weighted,
        is_density=True,
    )


def _require_same_density(first, second, operation):
    if first.is_density != second.is_density:
        raise ValueError(
            f"{operation}: cannot mix raw-count and per-bin-width density histograms"
        )


def _edge_tolerance(edges, indices):
    """Roundoff accepted at ``edges[indices]``.

    Double-precision scale of the coordinate or the axis span, capped by a
    small fraction of the narrower neighbouring bin, so that no tolerance can
    absorb a material edge shift (same rule as ``rebin_root_histogram``).
    """
    widths = np.diff(edges)
    local_width = np.minimum(np.r_[widths[0], widths], np.r_[widths, widths[-1]])
    rounding_scale = np.maximum(np.abs(edges[indices]), edges[-1] - edges[0])
    return np.minimum(
        64 * np.finfo(np.float64).eps * rounding_scale,
        1e-10 * local_width[indices],
    )


def _aligned_edge_indices(source, target):
    """Index of the source edge matching each target edge.

    Raises unless every target edge coincides with a distinct source edge up
    to ``_edge_tolerance``; a target edge inside a bin or outside the range
    has no match.
    """
    # Nearest source edge per target edge in linear memory: compare the two
    # neighbours bracketing each target in the strictly increasing source.
    upper = np.clip(np.searchsorted(source, target), 1, source.size - 1)
    lower = upper - 1
    matched = np.where(
        np.abs(source[lower] - target) <= np.abs(source[upper] - target), lower, upper
    )
    if (np.any(np.abs(source[matched] - target) > _edge_tolerance(source, matched))
            or np.any(np.diff(matched) <= 0)):
        raise ValueError(
            "target edges must align with source edges; cannot split bins or widen range"
        )
    return matched


def _edges_match(first, second):
    """Whether two edge arrays describe the same bins up to roundoff."""
    if first.shape != second.shape:
        return False
    if first.size < 2:
        return bool(np.array_equal(first, second))
    return bool(np.all(np.abs(first - second) <= _edge_tolerance(first, np.arange(first.size))))


# --------------------------------------------------------------------------
# Comparability and bin-by-bin quotients: the one owner of both rules
# --------------------------------------------------------------------------

EDGE_RULES = ("exact", "roundoff")


def require_comparable(first, *others, edges, finite=False, operation="comparison"):
    """Raise unless ``others`` can be combined bin by bin with ``first``.

    All inputs must be the same histogram type (Hist1D or Hist2D), share the
    raw-count or density state, and share every axis grid under ``edges``:

    - ``"exact"``: identical edge values. Comparisons of external artifacts
      use this, so a regenerated axis cannot pass as the reference grid.
    - ``"roundoff"``: equal up to ``_edge_tolerance`` (double-precision scale,
      capped by a small fraction of the narrower neighbouring bin). Arithmetic
      inside analyb uses this.

    With ``finite=True`` the counts must also be finite and the errors finite
    and nonnegative.
    """
    if edges not in EDGE_RULES:
        raise ValueError(f"edges must be one of {EDGE_RULES}, got {edges!r}")
    if not isinstance(first, (Hist1D, Hist2D)):
        raise TypeError(f"{operation} expects Hist1D or Hist2D inputs")
    match = ((lambda a, b: a.shape == b.shape and bool(np.array_equal(a, b)))
             if edges == "exact" else _edges_match)
    for other in others:
        if type(other) is not type(first):
            raise TypeError(f"{operation} requires histograms of one type")
        if isinstance(first, Hist1D):
            _require_same_density(first, other, operation)
            axes = (("edges", first.edges, other.edges),)
        else:
            axes = (("x_edges", first.x_edges, other.x_edges),
                    ("y_edges", first.y_edges, other.y_edges))
        for name, a, b in axes:
            if not match(a, b):
                raise ValueError(f"{operation} requires identical {name} ({edges} rule)")
    if any(hist.errors.shape != hist.counts.shape for hist in (first, *others)):
        raise ValueError(f"{operation}: errors must match their counts")
    if isinstance(first, Hist2D):
        expected = (first.x_edges.size - 1, first.y_edges.size - 1)
        if any(hist.counts.shape != expected for hist in (first, *others)):
            raise ValueError(f"{operation}: Hist2D counts must match its x/y edges")
    if finite:
        for hist in (first, *others):
            if not np.all(np.isfinite(hist.counts)):
                raise ValueError(f"{operation}: counts must be finite")
            if not np.all(np.isfinite(hist.errors)) or np.any(hist.errors < 0):
                raise ValueError(f"{operation}: errors must be finite and nonnegative")


class Quotient(NamedTuple):
    """Bin-by-bin quotient; ``value`` and ``error`` are NaN where not ``valid``."""

    value: np.ndarray
    error: Optional[np.ndarray]
    valid: np.ndarray


def divide(numerator, numerator_error, denominator, denominator_error=None):
    """``numerator / denominator`` per bin with independent-error propagation.

    A bin is defined when every supplied input is finite and the denominator
    is nonzero; elsewhere value and error are NaN, never a fill value that
    could be read as a measurement. With ``denominator_error`` the error is
    ``hypot(sigma_n / |d|, |n| sigma_d / d**2)``, written without relative
    errors so a zero numerator keeps a finite error. Without it, the error is
    the numerator-only ``sigma_n / |d|``, for plots that draw the denominator
    uncertainty separately. ``numerator_error=None`` returns ``error=None``.
    Inputs are arrays or scalars of one shape; nothing is assumed about
    correlations between numerator and denominator.
    """
    n = np.asarray(numerator, dtype=np.float64)
    d = np.asarray(denominator, dtype=np.float64)
    supplied = [n, d] + [np.asarray(e, dtype=np.float64)
                         for e in (numerator_error, denominator_error) if e is not None]
    if any(a.shape != n.shape for a in supplied):
        raise ValueError("divide: inputs must share one shape")
    valid = np.logical_and.reduce([np.isfinite(a) for a in supplied]) & (d != 0)
    value = np.full(n.shape, np.nan)
    np.divide(n, d, out=value, where=valid)
    error = None
    if numerator_error is not None:
        n_err = supplied[2]
        error = np.full(n.shape, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator_term = n_err / np.abs(d)
            if denominator_error is None:
                propagated = numerator_term
            else:
                propagated = np.hypot(numerator_term, np.abs(n) * supplied[-1] / d ** 2)
        error[valid] = propagated[valid]
    return Quotient(value, error, valid)


#--------------------------------------------------------------------------
# Profile & efficiency
# --------------------------------------------------------------------------

def _weighted_binned_stats(x, y, edges, weights):
    """Weighted mean and weighted standard error per bin.

    Returns (centers, means, errs) — same shape for all three.
    Empty bins get NaN in means and errs.
    Uses np.bincount for a single-pass computation over all bins.
    """
    n_bins = len(edges) - 1
    bin_indices = np.digitize(x, edges) - 1
    in_range = (bin_indices >= 0) & (bin_indices < n_bins)
    idx = bin_indices[in_range]
    w = weights[in_range]
    y_in = y[in_range]
    if len(idx) == 0:
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, np.full(n_bins, np.nan), np.full(n_bins, np.nan)

    n_points = np.bincount(idx, minlength=n_bins)
    sum_w = np.bincount(idx, weights=w, minlength=n_bins)
    sum_wy = np.bincount(idx, weights=w * y_in, minlength=n_bins)
    sum_w2 = np.bincount(idx, weights=w * w, minlength=n_bins)
    sum_wy2 = np.bincount(idx, weights=w * y_in * y_in, minlength=n_bins)

    means = np.full(n_bins, np.nan)
    errs = np.full(n_bins, np.nan)

    nonempty = sum_w > 0
    means[nonempty] = sum_wy[nonempty] / sum_w[nonempty]

    n_eff = np.zeros(n_bins)
    n_eff[nonempty] = sum_w[nonempty] ** 2 / np.maximum(sum_w2[nonempty], 1e-300)

    mask = nonempty & (n_points > 1) & (sum_w2 > 0)
    mu_masked = means[mask]
    var_masked = (
        sum_wy2[mask]
        - 2 * mu_masked * sum_wy[mask]
        + mu_masked * mu_masked * sum_w[mask]
    ) / (sum_w[mask] - sum_w2[mask] / sum_w[mask])
    errs[mask] = np.sqrt(np.maximum(var_masked, 0) / n_eff[mask])

    centers = (edges[:-1] + edges[1:]) / 2
    return centers, means, errs


def profile_sufficient_statistics(x, y, bins="scott", range=None, weights=None):
    """Compute additive binned sufficient statistics for profile merging.

    Returns statistics that can be summed across chunks and then passed to
    :func:`profile_from_sufficient_statistics` to reconstruct the profile.

    Parameters
    ----------
    x, y : array-like
    bins, range : same as :func:`profile`.
    weights : array-like or None
        Per-event weights.  If None, unit weights are used.

    Returns
    -------
    n_points : ndarray of int
        Number of points per bin.
    sum_w : ndarray
        Sum of weights per bin.
    sum_w2 : ndarray
        Sum of weights-squared per bin.
    sum_wy : ndarray
        Sum of (weight * y) per bin.
    sum_wy2 : ndarray
        Sum of (weight * y²) per bin.
    edges : ndarray
        Bin edges (length ``nbins + 1``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    is_weighted = weights is not None
    if is_weighted:
        weights = np.asarray(weights, dtype=float)[finite]

    edges = resolve_bins(bins, x, range=range)
    n_bins = len(edges) - 1
    bin_indices = np.digitize(x, edges) - 1
    if not is_weighted:
        bin_indices[x == edges[-1]] = n_bins - 1
    in_range = (bin_indices >= 0) & (bin_indices < n_bins)
    idx = bin_indices[in_range]

    if is_weighted:
        w = weights[in_range]
    else:
        w = np.ones(len(idx), dtype=float)

    y_in = y[in_range]
    n_points = np.bincount(idx, minlength=n_bins)
    sum_w = np.bincount(idx, weights=w, minlength=n_bins)
    sum_w2 = np.bincount(idx, weights=w * w, minlength=n_bins)
    sum_wy = np.bincount(idx, weights=w * y_in, minlength=n_bins)
    sum_wy2 = np.bincount(idx, weights=w * y_in * y_in, minlength=n_bins)

    return n_points, sum_w, sum_w2, sum_wy, sum_wy2, edges


def profile_from_sufficient_statistics(
    n_points, sum_w, sum_w2, sum_wy, sum_wy2, edges, *, weighted=True,
):
    """Reconstruct profile centers, means, and standard errors from additive
    sufficient statistics (the output of :func:`profile_sufficient_statistics`).

    Parameters
    ----------
    n_points, sum_w, sum_w2, sum_wy, sum_wy2 : ndarray
        Per-bin statistics.  These may be summed across chunks before calling.
    edges : ndarray
        Bin edges (length ``nbins + 1``).

    Returns
    -------
    centers : ndarray
    means : ndarray
    stderr : ndarray
        Standard error of the weighted mean.  NaN where insufficient data.
    """
    n_bins = len(edges) - 1
    means = np.full(n_bins, np.nan)
    errs = np.full(n_bins, np.nan)

    nonempty = sum_w > 0
    means[nonempty] = sum_wy[nonempty] / sum_w[nonempty]

    if weighted:
        n_eff = np.zeros(n_bins)
        n_eff[nonempty] = sum_w[nonempty] ** 2 / np.maximum(sum_w2[nonempty], 1e-300)
        mask = nonempty & (n_points > 1) & (sum_w2 > 0)
        mu_masked = means[mask]
        denominator = sum_w[mask] - sum_w2[mask] / sum_w[mask]
        valid = denominator > 0
        variance = np.zeros(np.count_nonzero(mask))
        variance[valid] = (
            sum_wy2[mask][valid]
            - 2 * mu_masked[valid] * sum_wy[mask][valid]
            + mu_masked[valid] ** 2 * sum_w[mask][valid]
        ) / denominator[valid]
        masked_errors = np.full(np.count_nonzero(mask), np.nan)
        masked_errors[valid] = np.sqrt(
            np.maximum(variance[valid], 0) / n_eff[mask][valid]
        )
        errs[mask] = masked_errors
    else:
        mask = nonempty & (n_points > 0)
        variance = sum_wy2[mask] / sum_w[mask] - means[mask] ** 2
        errs[mask] = np.sqrt(np.maximum(variance, 0) / n_points[mask])

    centers = (edges[:-1] + edges[1:]) / 2
    return centers, means, errs


def profile(x, y, bins="scott", range=None, weights=None):
    """Mean of `y` per bin of `x` with symmetric standard errors (stderr).

    When ``weights`` are given, computes the weighted mean and the
    standard error of the weighted mean (using effective sample size).

    Uses :func:`profile_sufficient_statistics` and
    :func:`profile_from_sufficient_statistics` internally for the weighted
    path so that the same computation can be reproduced from chunked data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[finite]
        n, sw, sw2, swy, swy2, edges = profile_sufficient_statistics(
            x, y, bins=bins, range=range, weights=weights,
        )
        return profile_from_sufficient_statistics(
            n, sw, sw2, swy, swy2, edges, weighted=True,
        )
    # Unweighted fast path via scipy.
    edges = resolve_bins(bins, x, range=range)
    means, edges_out, _ = stats.binned_statistic(
        x, y, statistic="mean", bins=edges,
    )
    stds, _, _ = stats.binned_statistic(
        x, y, statistic="std", bins=edges,
    )
    counts, _, _ = stats.binned_statistic(
        x, y, statistic="count", bins=edges,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        errs = stds / np.sqrt(counts)
    centers = (edges_out[:-1] + edges_out[1:]) / 2
    return centers, means, errs


def efficiency(passed, total, bins, range=None, variable=None, cl=0.6827):
    r"""Binomial efficiency per bin with Bayesian central credible interval (Jeffreys prior).

    Parameters
    ----------
    passed, total : array-like
        Two modes:
        - Per-bin counts: arrays of equal length matching ``bins``.
        - Per-event bool/01 arrays: provide ``variable`` to bin them by the
          given ``bins`` and ``range``.
    bins : int | array-like
        If ``variable`` is provided, resolved as :func:`resolve_bins` to
        produce the kinematic bin edges.  Otherwise an array of edges or
        an integer for equal-width bins over the data range or a dummy axis.
    range : (lo, hi) or None
        Passed to :func:`resolve_bins` when ``variable`` and ``bins`` (int)
        are given.
    variable : array-like or None
        Per-event kinematic variable for binning ``passed`` / ``total``.
    cl : float
        Confidence level for the credible interval (default 0.6827 ≈ 1σ).

    Returns
    -------
    centers, eff, err_low, err_high : length ``n_bins`` arrays.
        Error lengths are nonnegative: the lower bound is 0 at ``k = 0`` and
        the upper bound is 1 at ``k = N``.  Empty bins are NaN.
    """
    passed = np.asarray(passed)
    total = np.asarray(total)
    is_pb = variable is not None
    if is_pb:
        variable = np.asarray(variable, dtype=float)
        finite = np.isfinite(variable)
        variable = variable[finite]
        passed = passed[finite]
        total = total[finite]
        edges = resolve_bins(bins, variable, range=range)
    else:
        if hasattr(bins, "__len__"):
            edges = np.asarray(bins, dtype=float)
        else:
            edges = resolve_bins(int(bins), np.arange(int(bins)) if range else None, range=range)
    centers = (edges[:-1] + edges[1:]) / 2
    if is_pb:
        n = np.histogram(variable, bins=edges, weights=total)[0]
        k = np.histogram(variable, bins=edges, weights=passed)[0]
    else:
        if len(passed) != len(centers):
            raise ValueError(
                f"efficiency: per-bin counts length ({len(passed)}) "
                f"does not match nbins ({len(centers)}). "
                f"Pass variable= for per-event arrays."
            )
        n = total.astype(float)
        k = passed.astype(float)
    alpha = 1 - cl
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = np.where(n > 0, k / n, np.nan)
        lo = stats.beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
        hi = stats.beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)
        # Jeffreys bounds at k = 0 and k = N exclude the observed boundary
        # value; extend them to it so error lengths stay nonnegative.
        lo = np.where(n > 0, np.where(k == 0, 0.0, lo), np.nan)
        hi = np.where(n > 0, np.where(k == n, 1.0, hi), np.nan)
    return centers, eff, (eff - lo), (hi - eff)
