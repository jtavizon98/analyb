"""Histogram filling (numpy/scipy based) and Hist1D/Hist2D classes."""

from typing import NamedTuple, Optional

import numpy as np
from scipy import stats

from .binning import resolve_bins, _data_range


def _flow_error(flow, error, is_weighted, name):
    """Error of one 1D flow bin: given, or sqrt(count) for unweighted entries."""
    if error is not None:
        return float(error)
    if flow == 0:
        return 0.0
    if is_weighted:
        raise ValueError(
            f"a weighted Hist1D with a nonzero {name} needs {name}_error; "
            "flows carry their variance like every bin"
        )
    return float(np.sqrt(flow))


def _validated_edges(edges, name):
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 1:
        raise ValueError(f"{name} must be one-dimensional and nonempty")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return edges


def _checked_errors(errors, name):
    errors = np.asarray(errors, dtype=np.float64)
    if np.any(errors < 0):
        raise ValueError(f"{name} errors must be nonnegative")
    return errors


class _FlowStorage:
    """Flow-inclusive values and variances; ``counts``/``errors`` are the displayed bins.

    The first and last entry of every axis are the underflow and overflow
    bins, laid out as ROOT and uproot's ``values(flow=True)``. Variances are
    stored, as ROOT's fSumw2 is, so file variances are summed exactly when
    bins merge; ``errors`` is their square root. An error given to the
    constructor comes back unchanged, since sqrt(e * e) == e in IEEE
    arithmetic whenever e * e is a normal float (e > ~1.5e-154).
    """

    _interior = (slice(1, -1),)

    @property
    def counts(self):
        return self._values[self._interior]

    @counts.setter
    def counts(self, value):
        self._set_interior(self._values, value, "counts")

    @property
    def errors(self):
        # Computed from the variances, so read-only: an element write would
        # otherwise be lost silently. Assign a whole array instead.
        errors = np.sqrt(self._variances[self._interior])
        errors.flags.writeable = False
        return errors

    @errors.setter
    def errors(self, value):
        value = _checked_errors(value, type(self).__name__)
        self._set_interior(self._variances, value * value, "errors")

    def _set_interior(self, target, value, name):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != target[self._interior].shape:
            raise ValueError(
                f"{type(self).__name__} {name} must have shape "
                f"{target[self._interior].shape}, got {value.shape}"
            )
        target[self._interior] = value

    def _set_variances(self, variances):
        if np.any(variances < 0):
            raise ValueError(f"{type(self).__name__} variances must be nonnegative")
        self._variances[...] = variances

    def values(self, flow=False):
        """Bin values; ``flow=True`` includes underflow and overflow bins."""
        return self._values.copy() if flow else self.counts.copy()

    def variances(self, flow=False):
        """Bin variances (squared errors); ``flow=True`` includes the flow bins."""
        return self._variances.copy() if flow else self._variances[self._interior].copy()


class Hist1D(_FlowStorage):
    """1D histogram: displayed bins plus underflow and overflow, each with an error."""

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
        underflow_error=None,
        overflow_error=None,
    ):
        counts = np.asarray(counts, dtype=np.float64)
        self._edges = np.asarray(edges, dtype=np.float64)
        if self.edges.ndim != 1 or counts.ndim != 1 or self.edges.size != counts.size + 1:
            raise ValueError("Hist1D edges must have length len(counts) + 1")
        if not np.all(np.isfinite(self.edges)) or np.any(np.diff(self.edges) <= 0):
            raise ValueError("Hist1D edges must be finite and strictly increasing")
        if errors is None:
            if is_weighted:
                raise ValueError(
                    "Hist1D with is_weighted=True requires explicit errors. "
                    "Use hist1d() which computes them correctly."
                )
            errors = np.sqrt(counts)
        errors = _checked_errors(errors, "Hist1D")
        if errors.shape != counts.shape:
            raise ValueError("Hist1D errors must have the same shape as counts")
        underflow, overflow = float(underflow), float(overflow)
        self._values = np.concatenate(([underflow], counts, [overflow]))
        flow_errors = _checked_errors([
            _flow_error(underflow, underflow_error, is_weighted, "underflow"),
            _flow_error(overflow, overflow_error, is_weighted, "overflow"),
        ], "Hist1D flow")
        self._variances = np.concatenate(([flow_errors[0]], errors, [flow_errors[1]])) ** 2
        self.label = label
        self.unit = unit
        self.is_weighted = is_weighted
        self.is_density = bool(is_density)
        if bin_width is None and self.is_uniform and not self.is_density:
            bin_width = float(self.edges[1] - self.edges[0])
        self.bin_width = None if self.is_density or not self.is_uniform else bin_width
        self.n_entries = self.integral()

    @classmethod
    def from_flow(cls, values, edges, variances, **kwargs):
        """Build from flow-inclusive arrays (underflow, bins..., overflow)."""
        values = np.asarray(values, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        if values.ndim != 1 or values.size < 2 or variances.shape != values.shape:
            raise ValueError("flow-inclusive values and variances must be 1D, same length, >= 2")
        hist = cls(values[1:-1], edges, np.zeros(values.size - 2),
                   underflow=values[0], overflow=values[-1],
                   underflow_error=0.0, overflow_error=0.0, **kwargs)
        hist._set_variances(variances)
        return hist

    underflow = property(lambda self: float(self._values[0]))
    overflow = property(lambda self: float(self._values[-1]))
    # Read-only, so counts and edges cannot disagree after construction.
    edges = property(lambda self: self._edges)
    underflow_error = property(lambda self: float(np.sqrt(self._variances[0])))
    overflow_error = property(lambda self: float(np.sqrt(self._variances[-1])))

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
        return Hist1D.from_flow(
            self._values, self.edges.copy(), self._variances,
            label=self.label, unit=self.unit, is_weighted=self.is_weighted,
            bin_width=self.bin_width, is_density=self.is_density,
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


class Hist2D(_FlowStorage):
    """2D histogram: displayed bins plus flow strips and corners, each with an error.

    ``values(flow=True)`` has shape ``(nx + 2, ny + 2)``, indexed ``[x, y]``.
    The plain constructor starts with empty flows; ``from_flow`` sets them.
    """

    _interior = (slice(1, -1), slice(1, -1))

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
        self._x_edges = _validated_edges(x_edges, "Hist2D x_edges")
        self._y_edges = _validated_edges(y_edges, "Hist2D y_edges")
        counts = np.asarray(counts, dtype=np.float64)
        shape = (self.x_edges.size - 1, self.y_edges.size - 1)
        if counts.shape != shape:
            raise ValueError(f"Hist2D counts must have shape {shape} from its edges, got {counts.shape}")
        errors = np.sqrt(counts) if errors is None else _checked_errors(errors, "Hist2D")
        if errors.shape != counts.shape:
            raise ValueError("Hist2D errors must have the same shape as counts")
        self._values = np.zeros((shape[0] + 2, shape[1] + 2))
        self._variances = np.zeros_like(self._values)
        self._values[self._interior] = counts
        self._variances[self._interior] = errors * errors
        self.x_label = x_label
        self.y_label = y_label
        self.x_unit = x_unit
        self.y_unit = y_unit

    # Read-only, so counts and edges cannot disagree after construction.
    x_edges = property(lambda self: self._x_edges)
    y_edges = property(lambda self: self._y_edges)

    @classmethod
    def from_flow(cls, values, x_edges, y_edges, variances, **kwargs):
        """Build from flow-inclusive ``(nx + 2, ny + 2)`` values and variances."""
        values = np.asarray(values, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        if values.ndim != 2 or variances.shape != values.shape:
            raise ValueError("flow-inclusive values and variances must be 2D with one shape")
        hist = cls(values[1:-1, 1:-1], x_edges, y_edges, np.zeros(values[1:-1, 1:-1].shape), **kwargs)
        hist._values[...] = values
        hist._set_variances(variances)
        return hist

    @property
    def shape(self):
        return self.counts.shape

    def integral(self):
        return float(np.sum(self.counts))

    def copy(self):
        return Hist2D.from_flow(
            self._values, self.x_edges.copy(), self.y_edges.copy(), self._variances,
            x_label=self.x_label, y_label=self.y_label,
            x_unit=self.x_unit, y_unit=self.y_unit,
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

    # Per-bin variance: sum of w² (weighted) or the count (unweighted); flows
    # carry the same quantity for the out-of-range entries.
    if is_weighted:
        w2 = weights ** 2
        sumw2, _ = np.histogram(values[in_range] if overflow else values, bins=edges,
                                weights=w2[in_range] if overflow else w2)
        uf_w2 = float(np.sum(w2[values < lo])) if overflow else 0.0
        of_w2 = float(np.sum(w2[values >= hi])) if overflow else 0.0
    else:
        sumw2 = counts.astype(np.float64)
        uf_w2, of_w2 = uf, of

    if merge_overflow:
        counts[0] += uf
        counts[-1] += of
        sumw2[0] += uf_w2
        sumw2[-1] += of_w2
        uf = of = uf_w2 = of_w2 = 0.0

    return Hist1D.from_flow(
        np.concatenate(([uf], counts, [of])),
        edges,
        np.concatenate(([uf_w2], sumw2, [of_w2])),
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
    Entries outside the grid go to the flow strips and corners (ROOT's
    convention: a value equal to the last edge is overflow); non-finite
    entries are dropped. Uses a single-pass np.bincount for both counts and
    error².
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

    # Flow-inclusive cell index: 0 is underflow, n + 1 overflow, per axis.
    fx = np.digitize(x, x_edges)
    fy = np.digitize(y, y_edges)
    flat = fx * (ny + 2) + fy
    cells = (nx + 2) * (ny + 2)

    if weights is not None:
        counts_flat = np.bincount(flat, weights=weights, minlength=cells)
        err2_flat = np.bincount(flat, weights=weights * weights, minlength=cells)
    else:
        counts_flat = np.bincount(flat, minlength=cells).astype(np.float64)
        err2_flat = counts_flat

    return Hist2D.from_flow(
        counts_flat.reshape(nx + 2, ny + 2),
        x_edges,
        y_edges,
        err2_flat.reshape(nx + 2, ny + 2),
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
            + ("; use rebin(hist, edges=...) for an explicit trailing bin" if hint else "")
        )


def rebin(hist, factor=None, *, edges=None):
    """Merge whole bins of a Hist1D or Hist2D; the one rebinning path.

    Give either ``factor`` (an integer, or one per axis for a Hist2D) that
    divides the bin count of each axis, or ``edges`` (an array for a Hist1D,
    an ``(x_edges, y_edges)`` pair for a Hist2D) drawn from the source edges
    up to roundoff. Values and variances of merged bins add, which assumes
    uncorrelated bins. A narrower range moves the bins left outside into the
    underflow and overflow bins with their variances, so nothing is lost;
    splitting a bin or widening the range raises ``ValueError``. Density
    histograms are merged through raw bin yields. The output adopts exact
    copies of the target edges.
    """
    if (factor is None) == (edges is None):
        raise ValueError("rebin takes either factor or edges")
    sources = (hist.x_edges, hist.y_edges) if isinstance(hist, Hist2D) else (hist.edges,)
    if factor is not None:
        factors = tuple(factor) if isinstance(factor, (list, tuple)) else (factor,) * len(sources)
        if len(factors) != len(sources):
            raise ValueError("give one rebin factor per axis")
        names = ("x ", "y ") if len(sources) == 2 else ("",)
        for f, source, name in zip(factors, sources, names):
            _check_rebin_factor(f, source.size - 1, name, hint=len(sources) == 1)
        targets = tuple(source[::f].copy() for f, source in zip(factors, sources))
    else:
        targets = (edges,) if len(sources) == 1 else tuple(edges)
        if len(targets) != len(sources):
            raise ValueError("give one target edge array per source axis")
        targets = tuple(np.array(t, dtype=np.float64, copy=True) for t in targets)
        for target in targets:
            if (target.ndim != 1 or target.size < 2
                    or not np.all(np.isfinite(target)) or np.any(np.diff(target) <= 0)):
                raise ValueError("target edges must be finite and strictly increasing")
    return _merge_bins(hist, targets)


def _merge_bins(hist, targets):
    """Sum flow-inclusive values and variances onto aligned target axes."""
    sources = (hist.x_edges, hist.y_edges) if isinstance(hist, Hist2D) else (hist.edges,)
    destinations = []
    for source, target in zip(sources, targets):
        # Roundoff rule shared with require_comparable(edges="roundoff").
        matched = _aligned_edge_indices(source, target)
        # Flow-inclusive source index 0 is underflow and n + 1 overflow; place
        # by boundary index so immaterial float differences cannot move a bin.
        boundaries = np.arange(source.size + 1) - 1
        destinations.append(np.searchsorted(matched, boundaries, side="right"))
    values = hist.values(flow=True)
    variances = hist.variances(flow=True)
    density = isinstance(hist, Hist1D) and hist.is_density
    if density:
        values[1:-1] = hist.counts * hist.widths
        variances[1:-1] = (hist.errors * hist.widths) ** 2
    merged = np.zeros(tuple(t.size + 1 for t in targets))
    merged_variances = np.zeros_like(merged)
    np.add.at(merged, np.ix_(*destinations), values)
    np.add.at(merged_variances, np.ix_(*destinations), variances)
    if isinstance(hist, Hist2D):
        return Hist2D.from_flow(
            merged, *targets, merged_variances,
            x_label=hist.x_label, y_label=hist.y_label,
            x_unit=hist.x_unit, y_unit=hist.y_unit,
        )
    if density:
        widths = np.diff(targets[0])
        merged[1:-1] /= widths
        merged_variances[1:-1] = (np.sqrt(merged_variances[1:-1]) / widths) ** 2
    return Hist1D.from_flow(
        merged, targets[0], merged_variances,
        label=hist.label, unit=hist.unit,
        is_weighted=hist.is_weighted, is_density=hist.is_density,
    )


def _scaled(hist, scale):
    """Hist1D times a constant: every bin, flows included, and every error."""
    return Hist1D.from_flow(
        hist.values(flow=True) * scale, hist.edges.copy(),
        (np.sqrt(hist.variances(flow=True)) * scale) ** 2,
        label=hist.label, unit=hist.unit,
        is_weighted=hist.is_weighted, is_density=hist.is_density,
    )


def normalize(hist, to=1.0):
    """Return a new Hist1D rescaled so that the integral equals `to`."""
    integral = hist.integral()
    if integral == 0:
        scale = 0.0
    else:
        scale = to / integral
    return _scaled(hist, scale)


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


def _summed_flow_arrays(hists):
    """Bin-wise sums of flow-inclusive values and squared errors."""
    total = np.zeros_like(hists[0]._values)
    total_err2 = np.zeros_like(total)
    for h in hists:
        total += h._values
        total_err2 += h._variances
    return total, total_err2


def sum_histograms(hists, label=None):
    """Sum a list or dict of Hist1D objects into one.

    Counts, underflow and overflow are summed; errors, flows included, are
    propagated as sqrt(sum of squares) for weighted (or Poisson) uncertainties.

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
    require_comparable(ref, *values[1:], edges="roundoff", operation="sum_histograms")
    total, total_err2 = _summed_flow_arrays(values)
    return Hist1D.from_flow(
        total, ref.edges.copy(), total_err2,
        label=label, unit=ref.unit, is_weighted=True, is_density=ref.is_density,
    )


def sum_histograms2d(hists, x_label=None, y_label=None, x_unit=None, y_unit=None):
    """Sum a list or dict of Hist2D objects on one grid (roundoff edge rule).

    Counts and flow cells are summed bin-wise; errors are propagated as
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

    total, total_err2 = _summed_flow_arrays(values)
    return Hist2D.from_flow(
        total, ref.x_edges.copy(), ref.y_edges.copy(), total_err2,
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
    return _scaled(hist, scale)


#--------------------------------------------------------------------------
# Edge-matching / rebin to arbitrary edges
# --------------------------------------------------------------------------

def divide_by_bin_width(hist):
    """Return a differential copy of *hist*, divided by each displayed bin width.

    Underflow and overflow, and their errors, remain raw out-of-range yields
    because they have no finite displayed-bin width.  The returned histogram has ``is_density=True``.
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
        overflow_error=hist.overflow_error,
        underflow_error=hist.underflow_error,
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
    absorb a material edge shift.
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
    if finite:
        for hist in (first, *others):
            if not np.all(np.isfinite(hist.counts)):
                raise ValueError(f"{operation}: counts must be finite")
            if not np.all(np.isfinite(hist.errors)) or np.any(hist.errors < 0):
                raise ValueError(f"{operation}: errors must be finite and nonnegative")


class Quotient(NamedTuple):
    """Bin-by-bin quotient; ``value`` and ``error`` are NaN where not ``valid``.

    ``valid`` means the central value is defined; an error can still be NaN
    or infinite in a valid bin when an input error was.
    """

    value: np.ndarray
    error: Optional[np.ndarray]
    valid: np.ndarray


def divide(numerator, numerator_error, denominator, denominator_error=None):
    """``numerator / denominator`` per bin with independent-error propagation.

    A bin is defined when numerator and denominator are finite and the
    denominator is nonzero; elsewhere value and error are NaN, never a fill
    value that could be read as a measurement. Errors never decide whether a
    value exists: a non-finite input error (``sqrt`` of a negative unweighted
    count, say) gives a non-finite error in that bin and leaves the value
    alone, so the value does not depend on which errors are drawn. With
    ``denominator_error`` the error is
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
    valid = np.isfinite(n) & np.isfinite(d) & (d != 0)
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
