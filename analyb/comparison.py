"""Pairwise numerical diagnostics and plots for immutable histogram artifacts."""

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, is_color_like
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from . import config, makelatex
from .colors import PALETTES, catppuccin_colorbars, catppuccin_colors, current_colorbar
from .histogram import Hist1D, Hist2D, _aligned_edge_indices
from .io import RootHistogram
from .plot import (
    _add_legend,
    _data_errorbar,
    _resolve_label_unit,
    _save_figure,
    _ylabel_counts,
    add_atlas_label,
    reserve_atlas_label_headroom,
)


_SHAPE_EIGENVALUE_RTOL = 1e-12
_SHAPE_NULL_RTOL = 1e-10

# Default component stair colors for artifact_closure_1d, in draw order.
_CLOSURE_COMPONENT_COLORS = PALETTES["closure"]


@dataclass(frozen=True)
class HistogramComparison:
    """Numerator/reference yield, shape, ratio, and difference diagnostics."""

    numerator_yield: float
    numerator_yield_error: float
    reference_yield: float
    reference_yield_error: float
    yield_ratio: float
    yield_ratio_error: float
    yield_difference: float
    yield_difference_error: float
    shape_available: bool
    normalized_numerator: object
    normalized_numerator_error: object
    normalized_numerator_covariance: object
    normalized_reference: object
    normalized_reference_error: object
    normalized_reference_covariance: object
    shape_difference: object
    shape_difference_error: object
    shape_covariance: object
    shape_chi2: Optional[float]
    shape_ndof: Optional[int]
    shape_chi2_supported: bool
    shape_eigenvalue_cutoff: Optional[float]
    ratio: object
    ratio_error: object
    ratio_valid: object
    difference: object
    difference_error: object


@dataclass(frozen=True)
class RatioPresentation:
    """Caller-computed ratio-panel values for ``artifact_comparison_1d``.

    ``values`` and ``errors`` are already in ratio units: ``errors`` are the
    half-width of the red stair band, and ``valid`` marks the bins the caller
    considers defined. Invalid bins stay undefined when drawn. ``unity_errors``
    is optional and, when given, draws black unity-reference points with those
    relative errors. Arrays are converted to float and copied on construction.

    Validation is fail-closed: valid ``values`` must be finite, and valid
    ``errors``/``unity_errors`` must be finite and nonnegative. Invalid bins may
    hold any placeholder because they are never drawn.

    ``error_style`` selects how ``errors`` are drawn: ``"band"`` (default, the
    translucent stair band) or ``"bars"`` (vertical error bars at the bin
    centres in the ratio line's colour, no band).

    ``components`` is an optional ordered mapping ``label -> per-bin signed
    values`` in ratio units, for example the terms of a bin decomposition of
    ``ratio - 1``. They are drawn as translucent bars stacked from one:
    positive parts upward and negative parts downward, in mapping order, with
    ``component_colors`` (a mapping of every label to a colour) and listed in
    the main-panel legend. The components are display input only; analyb never
    checks that they add up to the ratio. Valid component values must be
    finite.
    """

    values: object
    errors: object
    valid: object
    unity_errors: object = None
    unity_points: bool = False
    error_style: str = "band"
    components: object = None
    component_colors: object = None
    component_alpha: float = 0.45

    def __post_init__(self):
        values = np.array(self.values, dtype=np.float64, copy=True)
        errors = np.array(self.errors, dtype=np.float64, copy=True)
        valid = np.array(self.valid, dtype=bool, copy=True)
        if values.ndim != 1:
            raise ValueError("RatioPresentation values must be one-dimensional")
        if errors.shape != values.shape or valid.shape != values.shape:
            raise ValueError("RatioPresentation arrays must share one shape")
        if np.any(~np.isfinite(values[valid])):
            raise ValueError("valid RatioPresentation values must be finite")
        if np.any(~np.isfinite(errors[valid])) or np.any(errors[valid] < 0):
            raise ValueError("valid RatioPresentation errors must be finite and nonnegative")
        if self.unity_errors is None:
            unity_errors = None
        else:
            unity_errors = np.array(self.unity_errors, dtype=np.float64, copy=True)
            if unity_errors.shape != values.shape:
                raise ValueError("RatioPresentation unity_errors must match values")
            if (np.any(~np.isfinite(unity_errors[valid]))
                    or np.any(unity_errors[valid] < 0)):
                raise ValueError(
                    "valid RatioPresentation unity_errors must be finite and nonnegative"
                )
        if self.error_style not in {"band", "bars"}:
            raise ValueError("RatioPresentation error_style must be 'band' or 'bars'")
        components = None
        if self.components is not None:
            if not isinstance(self.components, Mapping) or not self.components:
                raise ValueError("RatioPresentation components must be a non-empty mapping")
            colors = self.component_colors
            if not isinstance(colors, Mapping) or set(colors) != set(self.components):
                raise ValueError("RatioPresentation component_colors must map every component")
            components = {}
            for label, component in self.components.items():
                component = np.array(component, dtype=np.float64, copy=True)
                if component.shape != values.shape:
                    raise ValueError("RatioPresentation components must match values")
                if np.any(~np.isfinite(component[valid])):
                    raise ValueError("valid RatioPresentation components must be finite")
                if not is_color_like(colors[label]):
                    raise ValueError(f"invalid component colour for {label!r}")
                components[label] = component
            object.__setattr__(self, "component_colors", dict(colors))
        elif self.component_colors is not None:
            raise ValueError("RatioPresentation component_colors require components")
        if not 0 < self.component_alpha <= 1:
            raise ValueError("RatioPresentation component_alpha must lie in (0, 1]")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "unity_errors", unity_errors)
        object.__setattr__(self, "components", components)

    def component_stacks(self):
        """Return the positive and negative stack extents from one (zero where undefined)."""
        positive = np.zeros_like(self.values)
        negative = np.zeros_like(self.values)
        for component in (self.components or {}).values():
            component = np.where(self.valid, component, 0.0)
            positive += np.clip(component, 0.0, None)
            negative += np.clip(component, None, 0.0)
        return positive, negative


def ratio_range_counts(values, ratio_ylim, *, valid=None, stacks=None):
    """Count ratio bins outside a displayed window, as the comparison panel reports them.

    ``values`` are the drawn ratios (NaN = undefined). Returns a dict with
    ``below``, ``above`` (central values outside ``ratio_ylim``), ``undefined``
    (non-finite values) and ``stacks_beyond`` (bins whose stacked components,
    ``stacks=(positive, negative)`` extents from one as returned by
    ``RatioPresentation.component_stacks()``, leave the window while the
    central value is inside; 0 without stacks).
    """
    values = np.asarray(values, dtype=float)
    if valid is not None:
        values = np.where(np.asarray(valid, dtype=bool), values, np.nan)
    low, high = ratio_ylim
    finite = np.isfinite(values)
    below = finite & (values < low)
    above = finite & (values > high)
    beyond = np.zeros(values.shape, dtype=bool)
    if stacks is not None:
        positive, negative = (np.asarray(x, dtype=float) for x in stacks)
        beyond = finite & ((1.0 + positive > high) | (1.0 + negative < low)) & ~below & ~above
    return {"below": int(np.count_nonzero(below)), "above": int(np.count_nonzero(above)),
            "undefined": int(np.count_nonzero(~finite)), "stacks_beyond": int(np.count_nonzero(beyond))}


def rebin_root_histogram(source, target_edges):
    """Aggregate raw ROOT bins onto aligned target edges (one array per axis).

    Whole bins and their variances are summed, including flow bins and TH2
    corners. A narrower range sends excluded bins into the corresponding flows.
    Wider ranges and edges splitting source bins are unavailable. Only rounding
    at double-precision scale, capped by local bin width, is accepted. Inputs
    are never modified; the output adopts exact copies of the target edges.
    """
    histogram = source.histogram
    if isinstance(histogram, Hist1D):
        if histogram.is_density:
            raise ValueError("rebinning requires raw counts, before density conversion")
        source_edges = (histogram.edges,)
    elif isinstance(histogram, Hist2D):
        source_edges = (histogram.x_edges, histogram.y_edges)
    else:
        raise TypeError("rebinning expects a RootHistogram containing Hist1D or Hist2D")
    if len(target_edges) != len(source_edges):
        raise ValueError("provide one target edge array per source axis")

    edges = tuple(np.array(axis, dtype=np.float64, copy=True) for axis in target_edges)
    destinations = []
    for original, target in zip(source_edges, edges):
        if (target.ndim != 1 or target.size < 2
                or not np.all(np.isfinite(target)) or np.any(np.diff(target) <= 0)):
            raise ValueError("target edges must be finite and strictly increasing")
        if not np.all(np.isfinite(original)) or np.any(np.diff(original) <= 0):
            raise ValueError("source edges must be finite and strictly increasing")
        # Shared with rebin_to_edges: roundoff at double-precision scale of the
        # coordinate or axis span, capped by the narrower neighboring bin width.
        matched = _aligned_edge_indices(original, target)
        # Source index 0 is underflow; n+1 is overflow. Search by source
        # boundary index so immaterial floating differences cannot move a bin.
        source_bins = np.arange(len(original) + 1) - 1
        destinations.append(np.searchsorted(matched, source_bins, side="right"))

    values = np.asarray(source.values_with_flow)
    variances = np.asarray(source.variances_with_flow)
    expected = tuple(len(axis) + 1 for axis in source_edges)
    if values.shape != expected or variances.shape != expected:
        raise ValueError("flow-inclusive arrays must match source axes")
    if (not np.all(np.isfinite(values)) or not np.all(np.isfinite(variances))
            or np.any(variances < 0)):
        raise ValueError("counts must be finite and variances finite and nonnegative")
    rebinned_values = np.zeros(tuple(len(axis) + 1 for axis in edges))
    rebinned_variances = np.zeros_like(rebinned_values)
    np.add.at(rebinned_values, np.ix_(*destinations), values)
    np.add.at(rebinned_variances, np.ix_(*destinations), variances)
    central = (slice(1, -1),) * len(edges)
    errors = np.sqrt(rebinned_variances[central])
    if isinstance(histogram, Hist1D):
        rebinned = Hist1D(
            rebinned_values[central], edges[0], errors=errors,
            underflow=rebinned_values[0], overflow=rebinned_values[-1],
            is_weighted=histogram.is_weighted, label=histogram.label, unit=histogram.unit,
        )
    else:
        rebinned = Hist2D(
            rebinned_values[central], *edges, errors=errors,
            x_label=histogram.x_label, y_label=histogram.y_label,
            x_unit=histogram.x_unit, y_unit=histogram.y_unit,
        )
    return RootHistogram(
        rebinned, rebinned_values, rebinned_variances,
        source.has_stored_variances, source.classname, source.axis_titles,
    )


def _validate_pair(numerator, reference):
    if type(numerator) is not type(reference):
        raise TypeError("histogram comparison requires two histograms of the same type")
    if isinstance(numerator, Hist1D):
        if numerator.is_density != reference.is_density:
            raise ValueError("comparison cannot mix raw-count and density histograms")
        edge_pairs = (("edges", numerator.edges, reference.edges),)
    elif isinstance(numerator, Hist2D):
        edge_pairs = (
            ("x_edges", numerator.x_edges, reference.x_edges),
            ("y_edges", numerator.y_edges, reference.y_edges),
        )
    else:
        raise TypeError("comparison expects Hist1D or Hist2D inputs")
    for name, first, second in edge_pairs:
        if first.shape != second.shape or not np.array_equal(first, second):
            raise ValueError(f"comparison requires identical {name}")
    if isinstance(numerator, Hist2D):
        expected_shape = (numerator.x_edges.size - 1, numerator.y_edges.size - 1)
        if numerator.counts.shape != expected_shape:
            raise ValueError("numerator counts must match its x/y edges")
        if reference.counts.shape != expected_shape:
            raise ValueError("reference counts must match its x/y edges")
    if numerator.counts.shape != numerator.errors.shape:
        raise ValueError("numerator errors must match its counts")
    if reference.counts.shape != reference.errors.shape:
        raise ValueError("reference errors must match its counts")
    for role, histogram in (("numerator", numerator), ("reference", reference)):
        if np.any(~np.isfinite(histogram.counts)):
            raise ValueError(f"{role} counts must be finite")
        if np.any(~np.isfinite(histogram.errors)) or np.any(histogram.errors < 0):
            raise ValueError(f"{role} errors must be finite and nonnegative")


def _yield_and_error(histogram):
    value = histogram.integral()
    error = float(np.sqrt(np.sum(np.asarray(histogram.errors) ** 2)))
    if isinstance(histogram, Hist1D) and histogram.is_density:
        error = float(np.sqrt(np.sum((histogram.errors * histogram.widths) ** 2)))
    return value, error


def _normalized(histogram, integral):
    counts = histogram.counts.reshape(-1)
    variances = histogram.errors.reshape(-1) ** 2
    weights = _normalization_weights(histogram)
    jacobian = np.eye(counts.size) / integral
    jacobian -= np.outer(counts, weights) / integral ** 2
    covariance = (jacobian * variances) @ jacobian.T
    normalized = (counts / integral).reshape(histogram.counts.shape)
    errors = np.sqrt(np.maximum(np.diag(covariance), 0)).reshape(histogram.counts.shape)
    return normalized, errors, covariance


def _normalization_weights(histogram):
    if isinstance(histogram, Hist1D) and histogram.is_density:
        return histogram.widths
    return np.ones(histogram.counts.size, dtype=np.float64)


def _shape_chi2(difference, covariance, normalization_weights):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    cutoff = _SHAPE_EIGENVALUE_RTOL * scale
    if np.any(eigenvalues < -cutoff):
        raise ValueError("shape covariance is not positive semidefinite")

    supported = eigenvalues > cutoff
    projections = eigenvectors.T @ difference
    chi2 = float(np.sum(projections[supported] ** 2 / eigenvalues[supported]))

    null_component = eigenvectors[:, ~supported] @ projections[~supported]
    normalization_direction = normalization_weights / np.linalg.norm(normalization_weights)
    unsupported_component = (
        null_component
        - normalization_direction * (normalization_direction @ null_component)
    )
    tolerance = _SHAPE_NULL_RTOL * max(1.0, float(np.linalg.norm(difference)))
    chi2_supported = bool(np.linalg.norm(unsupported_component) <= tolerance)
    if not chi2_supported:
        chi2 = np.inf
    return chi2, int(np.count_nonzero(supported)), chi2_supported, cutoff


def compare_histograms(numerator, reference, *, shape=True):
    """Compare exact-binned Hist1D or Hist2D inputs as numerator/reference.

    Flow bins are intentionally excluded from displayed yields and shapes; use
    :class:`analyb.io.RootHistogram` source arrays to report them separately.
    Set ``shape=False`` to compute only raw yield, ratio, and difference
    diagnostics without allocating normalized-shape covariance matrices.
    Shape uncertainties include the correlations induced by normalization.
    The shape chi-square inversion and degrees of freedom use the same explicit
    covariance-eigenvalue cutoff. Unsupported zero-variance mismatch directions
    produce an infinite chi-square rather than being silently discarded.
    """
    _validate_pair(numerator, reference)
    numerator_yield, numerator_yield_error = _yield_and_error(numerator)
    reference_yield, reference_yield_error = _yield_and_error(reference)

    if reference_yield == 0 or not np.isfinite(reference_yield):
        yield_ratio = yield_ratio_error = np.nan
    else:
        yield_ratio = numerator_yield / reference_yield
        yield_ratio_error = np.hypot(
            numerator_yield_error / abs(reference_yield),
            abs(numerator_yield) * reference_yield_error / reference_yield ** 2,
        )

    ratio_valid = (
        np.isfinite(numerator.counts) & np.isfinite(reference.counts)
        & np.isfinite(numerator.errors) & np.isfinite(reference.errors)
        & (reference.counts != 0)
    )
    ratio = np.full(numerator.counts.shape, np.nan, dtype=np.float64)
    ratio_error = np.full(numerator.counts.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio[ratio_valid] = numerator.counts[ratio_valid] / reference.counts[ratio_valid]
        ratio_error[ratio_valid] = np.hypot(
            numerator.errors[ratio_valid] / np.abs(reference.counts[ratio_valid]),
            np.abs(numerator.counts[ratio_valid]) * reference.errors[ratio_valid]
            / reference.counts[ratio_valid] ** 2,
        )

    difference = numerator.counts - reference.counts
    difference_error = np.hypot(numerator.errors, reference.errors)

    shape_available = bool(shape and numerator_yield != 0 and reference_yield != 0)
    if shape_available:
        (
            normalized_numerator,
            normalized_numerator_error,
            normalized_numerator_covariance,
        ) = _normalized(numerator, numerator_yield)
        (
            normalized_reference,
            normalized_reference_error,
            normalized_reference_covariance,
        ) = _normalized(reference, reference_yield)
        shape_difference = normalized_numerator - normalized_reference
        shape_covariance = (
            normalized_numerator_covariance + normalized_reference_covariance
        )
        shape_difference_error = np.sqrt(
            np.maximum(np.diag(shape_covariance), 0)
        ).reshape(numerator.counts.shape)
        (
            shape_chi2,
            shape_ndof,
            shape_chi2_supported,
            shape_eigenvalue_cutoff,
        ) = _shape_chi2(
            shape_difference.reshape(-1),
            shape_covariance,
            _normalization_weights(numerator),
        )
    else:
        normalized_numerator = None
        normalized_numerator_error = None
        normalized_numerator_covariance = None
        normalized_reference = None
        normalized_reference_error = None
        normalized_reference_covariance = None
        shape_difference = None
        shape_difference_error = None
        shape_covariance = None
        shape_chi2 = None
        shape_ndof = None
        shape_chi2_supported = False
        shape_eigenvalue_cutoff = None

    return HistogramComparison(
        numerator_yield=numerator_yield,
        numerator_yield_error=numerator_yield_error,
        reference_yield=reference_yield,
        reference_yield_error=reference_yield_error,
        yield_ratio=yield_ratio,
        yield_ratio_error=yield_ratio_error,
        yield_difference=numerator_yield - reference_yield,
        yield_difference_error=np.hypot(numerator_yield_error, reference_yield_error),
        shape_available=shape_available,
        normalized_numerator=normalized_numerator,
        normalized_numerator_error=normalized_numerator_error,
        normalized_numerator_covariance=normalized_numerator_covariance,
        normalized_reference=normalized_reference,
        normalized_reference_error=normalized_reference_error,
        normalized_reference_covariance=normalized_reference_covariance,
        shape_difference=shape_difference,
        shape_difference_error=shape_difference_error,
        shape_covariance=shape_covariance,
        shape_chi2=shape_chi2,
        shape_ndof=shape_ndof,
        shape_chi2_supported=shape_chi2_supported,
        shape_eigenvalue_cutoff=shape_eigenvalue_cutoff,
        ratio=ratio,
        ratio_error=ratio_error,
        ratio_valid=ratio_valid,
        difference=difference,
        difference_error=difference_error,
    )


def artifact_comparison_1d(
    numerator,
    reference,
    *,
    variable=None,
    xlabel=None,
    numerator_label="Numerator",
    reference_label="Reference",
    ratio_label="Numerator / Reference",
    points="numerator",
    separate_ratio_errors=False,
    ratio_presentation=None,
    shape=False,
    density=False,
    atlas=False,
    atlas_label_kwargs=None,
    ylog=False,
    ysymlog_linthresh=None,
    ratio_ylim=None,
    save_name=None,
    save_path=None,
    script_id=None,
    figsize=None,
    verbose=True,
    model_color=None,
    point_color="black",
    legend_ncol=1,
    range_text=True,
    model_baseline=0.0,
):
    """Plot an exact-binned overlay using the shared analyb style.

    ``points`` selects the main-panel errorbar series ("numerator" or
    "reference"); the other series uses stairs and an uncertainty band.
    This drawing role never changes numerator/reference diagnostics.
    ``separate_ratio_errors=True`` draws reference unity points with relative
    reference errors and a numerator/reference curve with numerator-only errors.
    This requires ``points="reference"``; stored diagnostics keep combined errors.

    ``ratio_presentation`` accepts a :class:`RatioPresentation` with caller-
    computed ratio values, band errors, a valid mask, and optional
    unity-reference errors. It draws the same red stair/band and black unity
    presentation as ``separate_ratio_errors`` but leaves the ratio arithmetic to
    the caller. The two modes are mutually exclusive, and returned diagnostics
    keep their independent numerator/reference definition. Its ``error_style``
    may draw the errors as vertical bars in the line colour instead of a band,
    and its ``components`` are stacked from one in the ratio panel and listed
    in the main legend (``legend_ncol`` columns).

    ``density=True`` divides only the main-panel values and errors by each bin
    width after computing the requested raw or normalized-shape diagnostics.
    Returned diagnostics and the ratio panel retain their original convention.

    ``ysymlog_linthresh`` enables a signed logarithmic main axis, independently
    of the linear ratio panel. ``ratio_ylim`` bounds only the displayed ratio:
    boundary markers identify out-of-view central values and a count reports
    undefined ratios. ``range_text=False`` omits that count text (the
    boundary markers stay); callers that report the counts elsewhere get them
    from :func:`ratio_range_counts`. Returned diagnostics always retain every bin.

    ``model_baseline`` is the main-panel stairs baseline: the default 0 closes
    a count histogram's outline at its outer edges; ``None`` leaves the line
    open, for quantities that are not counts (survival, efficiency), where an
    edge down to zero would read as a value.
    """
    if not isinstance(numerator, Hist1D):
        raise TypeError("artifact_comparison_1d expects Hist1D inputs")
    if density and numerator.is_density:
        raise ValueError("density display requires raw-count histogram inputs")
    if points not in {"numerator", "reference"}:
        raise ValueError("points must be numerator or reference")
    if separate_ratio_errors and points != "reference":
        raise ValueError("separate ratio errors require reference pseudodata")
    if separate_ratio_errors and ratio_presentation is not None:
        raise ValueError(
            "provide either separate_ratio_errors or ratio_presentation, not both"
        )
    if ratio_presentation is not None:
        if not isinstance(ratio_presentation, RatioPresentation):
            raise TypeError("ratio_presentation must be a RatioPresentation")
        if ratio_presentation.values.shape != numerator.counts.shape:
            raise ValueError("ratio_presentation must match the histogram bins")
    if ysymlog_linthresh is not None:
        if ylog or not np.isfinite(ysymlog_linthresh) or ysymlog_linthresh <= 0:
            raise ValueError("ysymlog_linthresh must be positive and exclusive of ylog")
    if ratio_ylim is not None:
        if len(ratio_ylim) != 2 or not np.all(np.isfinite(ratio_ylim)) or ratio_ylim[0] >= ratio_ylim[1]:
            raise ValueError("ratio_ylim must contain two increasing finite limits")
    diagnostics = compare_histograms(numerator, reference, shape=shape)
    if shape and not diagnostics.shape_available:
        raise ValueError("shape plot is unavailable for a zero-integral histogram")
    if figsize is None:
        figsize = (config.fig_size_square[0], config.fig_size_square[1] + 1.4)
    xlabel = xlabel or numerator.label
    xlabel, unit = _resolve_label_unit(variable, xlabel, numerator.unit)
    weighted = numerator.is_weighted or reference.is_weighted

    if shape:
        numerator_values = diagnostics.normalized_numerator
        numerator_errors = diagnostics.normalized_numerator_error
        reference_values = diagnostics.normalized_reference
        reference_errors = diagnostics.normalized_reference_error
        ylabel = _ylabel_counts(
            numerator.bin_width, unit=unit, is_density=numerator.is_density, weighted=weighted,
            normalized=True,
        )
    else:
        numerator_values = numerator.counts
        numerator_errors = numerator.errors
        reference_values = reference.counts
        reference_errors = reference.errors
        ylabel = _ylabel_counts(
            numerator.bin_width, unit=unit, is_density=numerator.is_density, weighted=weighted,
        )
    if shape:
        ratio = np.full(numerator.counts.shape, np.nan, dtype=np.float64)
        ratio_error = np.full(numerator.counts.shape, np.nan, dtype=np.float64)
        finite = diagnostics.ratio_valid & (reference_values != 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio[finite] = numerator_values[finite] / reference_values[finite]
            ratio_error[finite] = np.hypot(
                numerator_errors[finite] / np.abs(reference_values[finite]),
                np.abs(numerator_values[finite]) * reference_errors[finite]
                / reference_values[finite] ** 2,
            )
    else:
        ratio = diagnostics.ratio
        ratio_error = diagnostics.ratio_error
    if density:
        widths = numerator.widths
        numerator_values = numerator_values / widths
        numerator_errors = numerator_errors / widths
        reference_values = reference_values / widths
        reference_errors = reference_errors / widths
        ylabel = _ylabel_counts(None, unit=unit, is_density=True, weighted=weighted, normalized=shape)
    if ylog and (np.any(numerator_values < 0) or np.any(reference_values < 0)):
        raise ValueError("logarithmic comparison cannot display negative bins")

    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    centers = numerator.centers
    model_color = model_color or catppuccin_colors["red"]
    if points == "numerator":
        point_values, point_errors, point_label = numerator_values, numerator_errors, numerator_label
        model_values, model_errors, model_label = reference_values, reference_errors, reference_label
    else:
        point_values, point_errors, point_label = reference_values, reference_errors, reference_label
        model_values, model_errors, model_label = numerator_values, numerator_errors, numerator_label
    ax_main.stairs(
        model_values, numerator.edges, color=model_color,
        linewidth=1.0, label=model_label, baseline=model_baseline,
    )
    model_low = model_values - model_errors
    model_high = model_values + model_errors
    ax_main.fill_between(
        numerator.edges,
        np.r_[model_low, model_low[-1]],
        np.r_[model_high, model_high[-1]], step="post",
        color=model_color, alpha=0.18, linewidth=0,
    )
    _data_errorbar(
        ax_main, centers, point_values, yerr=point_errors,
        color=point_color, label=point_label,
    )
    ax_main.set_ylabel(ylabel, loc="top")
    if ylog:
        ax_main.set_yscale("log")
    elif ysymlog_linthresh is not None:
        ax_main.set_yscale("symlog", linthresh=ysymlog_linthresh)
    components = None if ratio_presentation is None else ratio_presentation.components
    if components:
        handles, labels = ax_main.get_legend_handles_labels()
        handles += [Patch(facecolor=ratio_presentation.component_colors[label],
                          alpha=ratio_presentation.component_alpha, linewidth=0)
                    for label in components]
        labels += list(components)
        _add_legend(ax_main, handles, labels, fontsize="x-small", ncol=legend_ncol)
    else:
        _add_legend(ax_main, fontsize="x-small", ncol=legend_ncol)
    if atlas:
        label_kwargs = {"status": "Internal"}
        label_kwargs.update(atlas_label_kwargs or {})
        reserve_atlas_label_headroom(ax_main, ylog=ylog)
        add_atlas_label(ax_main, **label_kwargs)

    split_ratio = separate_ratio_errors or ratio_presentation is not None
    stack_high = stack_low = None
    if ratio_presentation is not None:
        ratio = np.where(ratio_presentation.valid, ratio_presentation.values, np.nan)
        ratio_error = np.where(ratio_presentation.valid, ratio_presentation.errors, np.nan)
        if components:
            stack_low, stack_high = _stack_components(
                ax_ratio, numerator.edges[:-1], np.diff(numerator.edges), ratio_presentation)
        ax_ratio.stairs(ratio, numerator.edges, color=model_color, linewidth=1.2, baseline=None)
        if ratio_presentation.error_style == "bars":
            ax_ratio.errorbar(centers, ratio, yerr=ratio_error, fmt="none",
                              ecolor=model_color, elinewidth=1.0, capsize=0)
        else:
            lower, upper = ratio - ratio_error, ratio + ratio_error
            ax_ratio.fill_between(numerator.edges, np.r_[lower, lower[-1]],
                                  np.r_[upper, upper[-1]], step="post",
                                  color=model_color, alpha=0.18, linewidth=0)
        if ratio_presentation.unity_errors is not None or ratio_presentation.unity_points:
            unity_errors = (None if ratio_presentation.unity_errors is None else
                            np.where(ratio_presentation.valid,
                                     ratio_presentation.unity_errors, np.nan))
            unity = np.where(np.isfinite(ratio), 1.0, np.nan)
            _data_errorbar(ax_ratio, centers, unity, yerr=unity_errors, color=point_color)
    elif separate_ratio_errors:
        valid = (diagnostics.ratio_valid & np.isfinite(reference_values)
                 & np.isfinite(reference_errors) & (reference_values != 0))
        numerator_error = np.full_like(ratio, np.nan)
        reference_error = np.full_like(ratio, np.nan)
        unity = np.full_like(ratio, np.nan)
        numerator_error[valid] = numerator_errors[valid] / np.abs(reference_values[valid])
        reference_error[valid] = reference_errors[valid] / np.abs(reference_values[valid])
        unity[valid] = 1.0
        ax_ratio.stairs(ratio, numerator.edges, color=model_color, linewidth=1.2, baseline=None)
        lower, upper = ratio - numerator_error, ratio + numerator_error
        ax_ratio.fill_between(numerator.edges, np.r_[lower, lower[-1]],
                              np.r_[upper, upper[-1]], step="post",
                              color=model_color, alpha=0.18, linewidth=0)
        _data_errorbar(ax_ratio, centers, unity, yerr=reference_error, color=point_color)
    else:
        _data_errorbar(
            ax_ratio, centers, ratio, yerr=ratio_error,
            color=point_color,
        )
    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=0.6)
    finite = np.isfinite(ratio) & np.isfinite(ratio_error)
    if np.any(finite):
        low = min(0.5, float(np.min(ratio[finite] - ratio_error[finite])))
        high = max(1.5, float(np.max(ratio[finite] + ratio_error[finite])))
        if stack_high is not None:
            low = min(low, float(np.nanmin(stack_low, initial=low)))
            high = max(high, float(np.nanmax(stack_high, initial=high)))
        padding = 0.08 * max(high - low, 1.0)
        ax_ratio.set_ylim(low - padding, high + padding)
    else:
        ax_ratio.set_ylim(0.5, 1.5)
    if ratio_ylim is not None:
        low, high = ratio_ylim
        below = np.isfinite(ratio) & (ratio < low)
        above = np.isfinite(ratio) & (ratio > high)
        # Mark central values outside the window without changing the data.
        inset = 0.04 * (high - low)
        for mask, boundary, marker in ((below, low + inset, "v"), (above, high - inset, "^")):
            ax_ratio.plot(centers[mask], np.full(np.count_nonzero(mask), boundary),
                          linestyle="none", marker=marker, markersize=3,
                          color=model_color if split_ratio else point_color)
        stacks = ""
        if stack_high is not None:
            # Stacks leaving the window are marked where the central value is not.
            stacks_beyond = (np.isfinite(stack_high) & ((stack_high > high) | (stack_low < low))
                             & ~below & ~above)
            for mask, boundary, marker in ((stacks_beyond & (stack_low < low), low + inset, "v"),
                                           (stacks_beyond & (stack_high > high), high - inset, "^")):
                ax_ratio.plot(centers[mask], np.full(np.count_nonzero(mask), boundary),
                              linestyle="none", marker=marker, markersize=3, color="0.35")
            if np.any(stacks_beyond):
                stacks = f"; stacks beyond: {np.count_nonzero(stacks_beyond)}"
        ax_ratio.set_ylim(low, high)
        undefined = np.count_nonzero(~np.isfinite(ratio))
        if range_text and (np.any(below) or np.any(above) or undefined or stacks):
            ax_ratio.text(
                0.02, 0.04,
                f"Below: {np.count_nonzero(below)}; above: {np.count_nonzero(above)}; undefined: {undefined}"
                + stacks,
                transform=ax_ratio.transAxes, va="bottom", fontsize="xx-small",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1},
            )
    ax_ratio.set_xlim(numerator.edges[0], numerator.edges[-1])
    ax_ratio.set_xlabel(xlabel or "", loc="right")
    ax_ratio.set_ylabel(ratio_label, loc="top")

    if save_name is not None:
        _save_figure(
            fig, save_name, script_id=script_id, save_path=save_path,
            verbose=verbose,
        )
    return fig, (ax_main, ax_ratio), diagnostics


def _stack_components(ax, left, widths, presentation):
    """Draw a presentation's components as bars stacked from one; return (low, high) extents.

    Positive parts stack upward and negative parts downward, in mapping order,
    only in valid slots; the extents are NaN where the slot is invalid.
    """
    shown = presentation.valid
    positive = np.zeros_like(presentation.values)
    negative = np.zeros_like(presentation.values)
    for label, component in presentation.components.items():
        component = np.where(shown, component, 0.0)
        up, down = np.clip(component, 0.0, None), np.clip(component, None, 0.0)
        for height, base in ((up, positive), (down, negative)):
            ax.bar(left[shown], height[shown], widths[shown], bottom=1.0 + base[shown],
                   align="edge", linewidth=0, color=presentation.component_colors[label],
                   alpha=presentation.component_alpha)
        positive += up
        negative += down
    return np.where(shown, 1.0 + negative, np.nan), np.where(shown, 1.0 + positive, np.nan)


def ratio_summary_plot(categories, series, *, ylabel, xlabel, colors=None, markers=None,
                       ratio_ylim=None, separators=(), separator_ymax=1.0, figsize=None,
                       legend_ncol=1,
                       legend_title=None, save_name=None, save_path=None, script_id=None,
                       verbose=True, ax=None):
    """Categorical summary of ratios: one slot per category, one point per series.

    ``series`` is an ordered mapping ``label -> RatioPresentation`` whose arrays
    have one entry per category (caller-owned values, errors, valid mask and
    optional components, as in :func:`artifact_comparison_1d`). With ``n``
    series each category slot is split into ``n`` equal sub-slots in mapping
    order: the series' point and error bar sit at its sub-slot centre and its
    components are stacked from one as bars filling that sub-slot, so each
    point reads against its own stack. Errors are always drawn as bars.
    Component labels are listed once in the legend (first series that has
    them); a label shared by several series must keep one colour.

    ``ratio_ylim`` bounds the display: central values outside it get boundary
    triangles in the series colour and stacks leaving it grey ones, as in
    :func:`artifact_comparison_1d`; nothing is clipped from the data.
    ``separators`` are category indices before which a vertical line is drawn
    (for example before a final "total" slot), up to ``separator_ymax`` (axes
    fraction, so it can stop below the legend). ``xlabel`` and ``ylabel`` are
    required: categorical tick labels are not an axis label. ``ax`` draws into an
    existing axes (e.g. one panel of a multi-panel figure; ``figsize`` is then
    ignored) instead of a new figure.

    Returns ``(fig, ax)``.
    """
    if not isinstance(series, Mapping) or not series:
        raise ValueError("series must be a non-empty mapping of RatioPresentation")
    categories = [str(c) for c in categories]
    n_cat = len(categories)
    for label, presentation in series.items():
        if not isinstance(presentation, RatioPresentation):
            raise TypeError(f"series {label!r} must be a RatioPresentation")
        if presentation.values.shape != (n_cat,):
            raise ValueError(f"series {label!r} must have one value per category")
    if not xlabel or not ylabel:
        raise ValueError("ratio_summary_plot needs an x and a y label")
    if ratio_ylim is not None:
        if len(ratio_ylim) != 2 or not np.all(np.isfinite(ratio_ylim)) or ratio_ylim[0] >= ratio_ylim[1]:
            raise ValueError("ratio_ylim must contain two increasing finite limits")
    seen_colors = {}
    for presentation in series.values():
        for label, color in (presentation.component_colors or {}).items():
            if seen_colors.setdefault(label, color) != color:
                raise ValueError(f"component {label!r} has two colours")
    default_colors = ["black", catppuccin_colors["blue"], catppuccin_colors["red"],
                      catppuccin_colors["green"]]
    colors = dict(colors or {})
    markers = dict(markers or {})
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or config.fig_size_square)
    else:
        fig = ax.figure
    width = 1.0 / len(series)
    slots = np.arange(n_cat, dtype=float)
    handles, labels = [], []
    component_handles = {}
    inset = None if ratio_ylim is None else 0.04 * (ratio_ylim[1] - ratio_ylim[0])
    for k, (label, presentation) in enumerate(series.items()):
        color = colors.get(label, default_colors[k % len(default_colors)])
        left = slots - 0.5 + k * width
        centres = left + 0.5 * width
        low = high = None
        if presentation.components:
            low, high = _stack_components(ax, left, np.full(n_cat, width), presentation)
            for name in presentation.components:
                component_handles.setdefault(name, Patch(
                    facecolor=presentation.component_colors[name],
                    alpha=presentation.component_alpha, linewidth=0))
        values = np.where(presentation.valid, presentation.values, np.nan)
        errors = np.where(presentation.valid, presentation.errors, np.nan)
        handles.append(ax.errorbar(centres, values, yerr=errors, linestyle="none",
                                   marker=markers.get(label, "o"), markersize=4.5, color=color,
                                   markerfacecolor=color, markeredgecolor=color,
                                   elinewidth=1.0, capsize=0))
        labels.append(label)
        if ratio_ylim is not None:
            lo, hi = ratio_ylim
            finite = np.isfinite(values)
            for mask, boundary, marker in ((finite & (values < lo), lo + inset, "v"),
                                           (finite & (values > hi), hi - inset, "^")):
                ax.plot(centres[mask], np.full(np.count_nonzero(mask), boundary), linestyle="none",
                        marker=marker, markersize=3, color=color)
            if low is not None:
                inside = finite & (values >= lo) & (values <= hi)
                for mask, boundary, marker in ((inside & (low < lo), lo + inset, "v"),
                                               (inside & (high > hi), hi - inset, "^")):
                    ax.plot(centres[mask], np.full(np.count_nonzero(mask), boundary),
                            linestyle="none", marker=marker, markersize=3, color="0.35")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.6)
    for index in separators:
        ax.axvline(index - 0.5, ymax=separator_ymax, color="0.4", linewidth=0.8)
    ax.set_xlim(-0.5, n_cat - 0.5)
    ax.set_xticks(slots, labels=categories)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.grid(axis="x", visible=False)
    if ratio_ylim is not None:
        ax.set_ylim(*ratio_ylim)
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(ylabel, loc="top")
    handles += list(component_handles.values())
    labels += list(component_handles)
    _add_legend(ax, handles, labels, fontsize="x-small", ncol=legend_ncol, title=legend_title)
    if save_name is not None:
        _save_figure(fig, save_name, script_id=script_id, save_path=save_path, verbose=verbose)
    return fig, ax


def artifact_comparison_many_1d(numerators, reference, *, colors=None,
                                reference_label="Reference", shape=False,
                                density=False, ylog=False, atlas=False,
                                figsize=None, reference_variation=None,
                                variation_label=None, variation_color=None,
                                variation_ratio_errors=None, **kwargs):
    """Overlay named predictions against one point reference and its unity points.

    Reuse the pairwise comparison's axes, normalization, numerator-only ratio bands,
    reference errors, and legend geometry. ``numerators`` is an ordered mapping
    of labels to exact-binned raw Hist1D objects. ``colors`` maps labels to
    Matplotlib colors; the first prediction defaults to the pairwise red.
    Return the first prediction's independent-error diagnostics as in the
    pairwise helper. ``reference_variation`` adds another observed point series
    in both panels against the same reference. The caller supplies its ratio
    errors, including any numerator/reference covariance; ``None`` draws bare
    ratio points when the uncertainty is unknown. No covariance between
    predictions is implied. With ``ratio_ylim``, every prediction's central
    ratio values outside the window get boundary triangles in that
    prediction's colour (the first through the pairwise helper); the
    ``range_text`` count describes the first prediction only.
    """
    if not isinstance(numerators, Mapping) or not numerators:
        raise ValueError("numerators must be a non-empty ordered mapping")
    if colors is not None and (not isinstance(colors, Mapping)
                               or set(colors) - set(numerators)):
        raise ValueError("colors must map known numerator labels")
    colors = colors or {}
    names = list(numerators)
    for name, hist in numerators.items():
        if not isinstance(hist, Hist1D):
            raise TypeError(f"numerator {name!r} must be Hist1D")
        if hist.is_density != reference.is_density or not np.array_equal(hist.edges, reference.edges):
            raise ValueError(f"numerator {name!r} must match reference edges and density state")
        if ylog and np.any(hist.counts < 0):
            raise ValueError("logarithmic comparison cannot display negative bins")
    if reference_variation is not None:
        if (not isinstance(reference_variation, Hist1D) or
                reference_variation.is_density != reference.is_density or
                not np.array_equal(reference_variation.edges, reference.edges)):
            raise ValueError("reference variation must share the raw reference grid")
        if not is_color_like(variation_color):
            raise ValueError("reference variation needs a valid colour")
    fig, axes, diagnostics = artifact_comparison_1d(
        numerators[names[0]], reference, numerator_label=names[0],
        reference_label=reference_label, points="reference",
        separate_ratio_errors=True, shape=shape, density=density,
        ylog=ylog, atlas=atlas, figsize=figsize,
        model_color=colors.get(names[0], catppuccin_colors["red"]), **kwargs)
    ax_main, ax_ratio = axes
    palette_names = PALETTES["comparison_many"]
    palette = [colors.get(name, catppuccin_colors[palette_names[i % len(palette_names)]])
               for i, name in enumerate(names)]
    for color in palette:
        if not is_color_like(color):
            plt.close(fig)
            raise ValueError(f"invalid comparison color: {color!r}")
    for name, color in zip(names[1:], palette[1:]):
        hist = numerators[name]
        result = compare_histograms(hist, reference, shape=shape)
        if shape and not result.shape_available:
            plt.close(fig)
            raise ValueError(f"shape plot unavailable for {name!r}")
        values = result.normalized_numerator if shape else hist.counts
        errors = result.normalized_numerator_error if shape else hist.errors
        ref = result.normalized_reference if shape else reference.counts
        valid = (np.isfinite(values) & np.isfinite(errors) & np.isfinite(ref)
                 & (ref != 0))
        ratio = np.full_like(values, np.nan)
        ratio_error = np.full_like(values, np.nan)
        ratio[valid] = values[valid] / ref[valid]
        ratio_error[valid] = errors[valid] / np.abs(ref[valid])
        main_values = values / hist.widths if density else values
        main_errors = errors / hist.widths if density else errors
        ax_main.stairs(main_values, hist.edges, color=color, linewidth=1.0, label=name)
        ax_main.fill_between(hist.edges,
                             np.r_[main_values - main_errors, (main_values - main_errors)[-1]],
                             np.r_[main_values + main_errors, (main_values + main_errors)[-1]],
                             step="post", color=color, alpha=0.12, linewidth=0)
        ax_ratio.stairs(ratio, hist.edges, color=color, linewidth=1.0, baseline=None)
        ax_ratio.fill_between(hist.edges, np.r_[ratio - ratio_error, (ratio - ratio_error)[-1]],
                              np.r_[ratio + ratio_error, (ratio + ratio_error)[-1]],
                              step="post", color=color, alpha=0.12, linewidth=0)
        if kwargs.get("ratio_ylim") is not None:
            _mark_out_of_window(ax_ratio, hist.centers, ratio, kwargs["ratio_ylim"], color)
    if reference_variation is not None:
        for ax in axes:
            marker = ax.containers[0].lines[0]
            marker.set_marker("o")
            marker.set_markersize(4)
            marker.set_markerfacecolor("black")
            marker.set_markeredgecolor("black")
        comparison = compare_histograms(reference_variation, reference, shape=shape)
        if shape and not comparison.shape_available:
            plt.close(fig)
            raise ValueError("reference variation shape unavailable")
        values = (comparison.normalized_numerator if shape else reference_variation.counts)
        errors = (comparison.normalized_numerator_error if shape else reference_variation.errors)
        denominator = (comparison.normalized_reference if shape else reference.counts)
        valid = np.isfinite(values) & np.isfinite(denominator) & (denominator != 0)
        variation_ratio = np.full_like(values, np.nan)
        variation_ratio[valid] = values[valid] / denominator[valid]
        ratio_errors = _reference_ratio_errors(variation_ratio_errors, valid)
        ax_main.errorbar(reference.centers, values/reference.widths if density else values,
                         yerr=errors/reference.widths if density else errors,
                         fmt="o", ms=4, color=variation_color,
                         markerfacecolor=variation_color, markeredgecolor=variation_color,
                         elinewidth=.55, capsize=0, label=variation_label or "Reference variation")
        _reference_ratio_errorbar(ax_ratio, reference.centers, variation_ratio,
                                  ratio_errors, valid, variation_color)
    # The established legend remains inside the top-right axes, with an inset
    # border and the same label headroom as the pairwise comparison.
    ax_main.get_legend().remove()
    _add_legend(ax_main, fontsize="x-small", ncol=2 if len(names) > 1 else 1)
    return fig, axes, diagnostics


def _mark_out_of_window(ax, centers, ratio, ratio_ylim, color):
    """Boundary triangles for central ratio values outside ``ratio_ylim``, as the
    pairwise comparison draws them, filled in ``color``."""
    low, high = ratio_ylim
    inset = 0.04 * (high - low)
    finite = np.isfinite(ratio)
    for mask, boundary, marker in ((finite & (ratio < low), low + inset, "v"),
                                   (finite & (ratio > high), high - inset, "^")):
        ax.plot(centers[mask], np.full(np.count_nonzero(mask), boundary),
                linestyle="none", marker=marker, markersize=3, color=color,
                markerfacecolor=color, markeredgecolor=color)


def _reference_ratio_errors(errors, valid):
    if errors is None:
        return None
    errors = np.asarray(errors, dtype=float)
    if errors.shape != valid.shape or np.any(~np.isfinite(errors[valid])) or np.any(errors[valid] < 0):
        raise ValueError("reference ratio errors must be finite and nonnegative on defined bins")
    return errors[valid]


def _reference_ratio_errorbar(ax, centers, values, errors, valid, color):
    ax.errorbar(centers[valid], values[valid], yerr=errors,
                fmt="o", ms=4, color=color, markerfacecolor=color,
                markeredgecolor=color, elinewidth=.55, capsize=0)


def artifact_comparison_flavours_1d(sources, references, *, colors,
                                     shape=False, density=False, ratio_ylim=None,
                                     reference_label="IR", ratio_bands=True,
                                     ratio_reference_flavour=None,
                                     show_reference_ratios=False,
                                     reference_ratio_errors=None, **kwargs):
    """Compare source flavours to their own or one shared reference flavour.

    ``sources`` maps source names to flavour -> raw Hist1D; ``references`` maps
    flavour -> raw Hist1D. All grids coincide. With ``ratio_reference_flavour``
    every source ratio uses that one reference, including its unit-area shape
    in shape view. ``show_reference_ratios`` adds each other reference as a
    point series divided by the shared reference. Supply caller-computed
    ``reference_ratio_errors`` where subset covariance is known; otherwise
    those points have no uncertainty bars. Ratio bands carry only
    numerator errors, and the reference unity error is drawn where available.
    Main-panel density changes no ratio or returned diagnostics.
    """
    if not sources or not references or set(references) != set(next(iter(sources.values()))):
        raise ValueError("sources and references must contain the same flavours")
    if any(set(series) != set(references) for series in sources.values()):
        raise ValueError("each source must contain all reference flavours")
    flavours = list(references)
    baseline = references[flavours[0]]
    if ratio_reference_flavour is not None and ratio_reference_flavour != flavours[0]:
        raise ValueError("the shared ratio reference must be the first flavour")
    if show_reference_ratios and ratio_reference_flavour is None:
        raise ValueError("reference flavour ratios require a shared reference")
    if reference_ratio_errors is not None and set(reference_ratio_errors) != set(flavours[1:]):
        raise ValueError("reference ratio errors must cover non-baseline flavours")
    for hist in [*references.values(), *(h for series in sources.values() for h in series.values())]:
        if (not isinstance(hist, Hist1D) or hist.is_density or
                not np.array_equal(hist.edges, baseline.edges)):
            raise ValueError("flavour comparisons require matching raw Hist1D grids")
    names = list(sources)
    first = {name: series[flavours[0]] for name, series in sources.items()}
    fig, (main, ratio_ax), first_diag = artifact_comparison_many_1d(
        first, baseline, colors={name: colors[name][flavours[0]] for name in names},
        reference_label=f"{reference_label} {flavours[0]}", shape=shape, density=density,
        ratio_ylim=ratio_ylim, **kwargs)
    for ax in (main, ratio_ax):
        marker = ax.containers[0].lines[0]
        marker.set_marker("o")
        marker.set_markersize(4)
        marker.set_markerfacecolor(colors[reference_label][flavours[0]])
        marker.set_markeredgecolor(colors[reference_label][flavours[0]])
    diagnostics = {(name, flavours[0]): compare_histograms(hist, baseline, shape=shape)
                   for name, hist in first.items()}
    # Use colour alone for flavour/source; every current curve is solid and
    # unmarked, while every observed reference is a filled round point.
    for flavour in flavours[1:]:
        reference = references[flavour]
        ref_diag = compare_histograms(reference, reference, shape=shape)
        ref_values = ref_diag.normalized_reference if shape else reference.counts
        ref_errors = ref_diag.normalized_reference_error if shape else reference.errors
        x = reference.centers
        main.errorbar(x, ref_values / reference.widths if density else ref_values,
                       yerr=ref_errors / reference.widths if density else ref_errors,
                        fmt="o", ms=4, color=colors[reference_label][flavour],
                        markerfacecolor=colors[reference_label][flavour],
                        markeredgecolor=colors[reference_label][flavour],
                        elinewidth=0.55, capsize=0, label=f"{reference_label} {flavour}")
        if show_reference_ratios:
            ref_ratio = compare_histograms(reference, baseline, shape=shape)
            denominator = ref_ratio.normalized_reference if shape else baseline.counts
            valid_ref = np.isfinite(ref_values) & np.isfinite(denominator) & (denominator != 0)
            values = np.full_like(ref_values, np.nan)
            values[valid_ref] = ref_values[valid_ref] / denominator[valid_ref]
            errors = _reference_ratio_errors(
                None if reference_ratio_errors is None else reference_ratio_errors[flavour],
                valid_ref)
            _reference_ratio_errorbar(ratio_ax, x, values, errors, valid_ref,
                                      colors[reference_label][flavour])
        elif ratio_reference_flavour is None:
            valid_ref = np.isfinite(ref_values) & (ref_values != 0)
            ratio_ax.errorbar(x[valid_ref], np.ones(np.count_nonzero(valid_ref)),
                              yerr=ref_errors[valid_ref] / np.abs(ref_values[valid_ref]),
                              fmt="o", ms=4, color=colors[reference_label][flavour],
                              markerfacecolor=colors[reference_label][flavour],
                              markeredgecolor=colors[reference_label][flavour],
                              elinewidth=0.55, capsize=0)
        for name, series in sources.items():
            hist = series[flavour]
            result = compare_histograms(
                hist, baseline if ratio_reference_flavour is not None else reference,
                shape=shape)
            if shape and not result.shape_available:
                plt.close(fig)
                raise ValueError(f"shape unavailable for {name}/{flavour}")
            diagnostics[name, flavour] = result
            values = result.normalized_numerator if shape else hist.counts
            errors = result.normalized_numerator_error if shape else hist.errors
            main_values = values / hist.widths if density else values
            main_errors = errors / hist.widths if density else errors
            color = colors[name][flavour]
            main.stairs(main_values, hist.edges, color=color, linestyle="-",
                         linewidth=1.05, label=f"{name} {flavour}")
            main.fill_between(hist.edges, np.r_[main_values-main_errors, (main_values-main_errors)[-1]],
                              np.r_[main_values+main_errors, (main_values+main_errors)[-1]],
                              step="post", color=color, alpha=0.08, linewidth=0)
            ref_ratio = (result.normalized_reference if shape else
                         (baseline if ratio_reference_flavour is not None else reference).counts)
            valid = np.isfinite(values) & np.isfinite(errors) & np.isfinite(ref_ratio) & (ref_ratio != 0)
            ratio = np.full_like(values, np.nan)
            error = np.full_like(values, np.nan)
            ratio[valid] = values[valid] / ref_ratio[valid]
            error[valid] = errors[valid] / np.abs(ref_ratio[valid])
            ratio_ax.stairs(ratio, hist.edges, color=color, linestyle="-",
                             linewidth=1.05, baseline=None)
            ratio_ax.fill_between(hist.edges, np.r_[ratio-error, (ratio-error)[-1]],
                                  np.r_[ratio+error, (ratio+error)[-1]],
                                  step="post", color=color, alpha=0.08, linewidth=0)
    if not ratio_bands:
        from matplotlib.collections import PolyCollection
        for collection in list(ratio_ax.collections):
            if isinstance(collection, PolyCollection):
                collection.remove()
    if main.get_legend() is not None:
        main.get_legend().remove()
    # Matplotlib fills columns first. Supply flavour columns in source-row
    # order; print the source in column one and keep the other cells compact.
    flavour_matrix_legend(main, colors, [*names, reference_label],
                          flavours=flavours, point_rows={reference_label})
    return fig, (main, ratio_ax), diagnostics


def flavour_matrix_legend(ax, colors, rows, *, flavours=("l", "e", "m"),
                          point_rows=()):
    """Fixed row-by-source, column-by-flavour legend in plotting colours."""
    labels = {"l": "combined", "e": "e", "m": "μ"}
    handles = [Line2D([], [], color=colors[name][flavour], linestyle="none" if name in point_rows else "-",
                      marker="o" if name in point_rows else None,
                      markerfacecolor=colors[name][flavour],
                      markeredgecolor=colors[name][flavour], markersize=4,
                      linewidth=1.1)
               for flavour in flavours for name in rows]
    entries = [f"{name} {labels.get(flavour, flavour)}" if flavour == flavours[0] else labels.get(flavour, flavour)
               for flavour in flavours for name in rows]
    return ax.legend(handles, entries, ncol=len(flavours), fontsize="xx-small",
                     loc="upper right", frameon=False, columnspacing=0.7,
                     handlelength=1.35, borderaxespad=0.25)


def artifact_comparison_2d(
    numerator,
    reference,
    *,
    variable_x=None,
    variable_y=None,
    xlabel=None,
    ylabel=None,
    numerator_label="Numerator",
    reference_label="Reference",
    difference_label="Numerator - Reference",
    ratio_label="Numerator / Reference",
    shape=False,
    atlas=False,
    atlas_label_kwargs=None,
    cmap=None,
    difference_cmap=None,
    ratio_cmap=None,
    norm=None,
    difference_norm=None,
    ratio_norm=None,
    save_name=None,
    save_path=None,
    script_id=None,
    figsize=None,
    verbose=True,
):
    """Plot inputs, ratio and difference on a 2-by-2 exact-binned grid.

    Returned axes have shape (2, 2): numerator and ratio on the top row,
    reference and difference below. ``norm`` is shared by both inputs;
    ``ratio_norm`` and ``difference_norm`` control the derived panels.
    Defaults center the ratio on one and the difference on zero, using the
    custom analyb coolwarm colormap. For signed log differences supply a
    symmetric ``SymLogNorm``. Caller norms and colormaps are not modified.

    Undefined cells are gray. Colorbar extensions and counts identify values
    outside explicit limits without changing arrays. Meshes are rasterized
    without cell edges or axis grids; labels and axes remain vector graphics.
    With ``shape=True`` both derived panels use the normalized input arrays;
    returned diagnostics retain their usual raw ratio and shape fields.
    """
    if not isinstance(numerator, Hist2D):
        raise TypeError("artifact_comparison_2d expects Hist2D inputs")
    diagnostics = compare_histograms(numerator, reference, shape=shape)
    if shape and not diagnostics.shape_available:
        raise ValueError("shape plot is unavailable for a zero-integral histogram")
    if cmap is None:
        cmap = current_colorbar
    if difference_cmap is None:
        difference_cmap = catppuccin_colorbars["coolwarm"]
    if ratio_cmap is None:
        ratio_cmap = catppuccin_colorbars["coolwarm"]
    if figsize is None:
        figsize = (2 * config.fig_size_square[0] + 1, 2 * config.fig_size_square[1])

    if shape:
        first = diagnostics.normalized_numerator
        second = diagnostics.normalized_reference
        difference = diagnostics.shape_difference
        colorbar_label = "Normalized counts"
    else:
        first = numerator.counts
        second = reference.counts
        difference = diagnostics.difference
        colorbar_label = "Counts"
    finite_values = np.concatenate((first[np.isfinite(first)], second[np.isfinite(second)]))
    if finite_values.size == 0:
        raise ValueError("2D comparison has no finite displayed bins")
    shared_min, shared_max = float(np.min(finite_values)), float(np.max(finite_values))
    if shared_min == shared_max:
        shared_max = shared_min + 1.0
    shared_norm = copy(norm) if norm is not None else Normalize(vmin=shared_min, vmax=shared_max)
    shared_norm.autoscale_None(finite_values)
    finite_difference = np.abs(difference[np.isfinite(difference)])
    difference_limit = float(np.max(finite_difference)) if finite_difference.size else 0.0
    difference_limit = difference_limit or 1.0
    if difference_norm is None:
        difference_norm = TwoSlopeNorm(
            vmin=-difference_limit, vcenter=0.0, vmax=difference_limit,
        )
    else:
        difference_norm = copy(difference_norm)
        difference_norm.autoscale_None([-difference_limit, difference_limit])

    ratio = np.full(first.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(first, second, out=ratio,
                  where=diagnostics.ratio_valid & (second != 0))
    finite_ratio = ratio[np.isfinite(ratio)]
    ratio_span = max(0.5, float(np.max(np.abs(finite_ratio - 1.0)))) if finite_ratio.size else 0.5
    if ratio_norm is None:
        ratio_norm = TwoSlopeNorm(vmin=1.0 - ratio_span, vcenter=1.0, vmax=1.0 + ratio_span)
    else:
        ratio_norm = copy(ratio_norm)
        ratio_norm.autoscale_None([1.0 - ratio_span, 1.0 + ratio_span])

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    x_edges, y_edges = numerator.x_edges, numerator.y_edges
    panels = (
        (first, numerator_label, cmap, shared_norm, colorbar_label),
        (ratio, ratio_label, ratio_cmap, ratio_norm, ratio_label),
        (second, reference_label, cmap, shared_norm, colorbar_label),
        (difference, difference_label, difference_cmap, difference_norm, difference_label),
    )
    xlabel = xlabel or numerator.x_label
    ylabel = ylabel or numerator.y_label
    xlabel, _ = _resolve_label_unit(variable_x, xlabel, numerator.x_unit)
    ylabel, _ = _resolve_label_unit(variable_y, ylabel, numerator.y_unit)
    for index, (ax, (values, title, panel_cmap, panel_norm, bar_label)) in enumerate(zip(axes.flat, panels)):
        finite = np.isfinite(values)
        below = np.count_nonzero(finite & (values < panel_norm.vmin))
        above = np.count_nonzero(finite & (values > panel_norm.vmax))
        undefined = np.count_nonzero(~finite)
        extend = "both" if below and above else "min" if below else "max" if above else "neither"
        # Copy even a caller-supplied map before assigning the undefined color.
        panel_cmap = plt.get_cmap(panel_cmap).with_extremes(bad="#9ca0b0")
        ax.grid(False, which="both")
        mesh = ax.pcolormesh(
            x_edges, y_edges, np.ma.masked_invalid(values.T),
            shading="flat", cmap=panel_cmap, norm=panel_norm,
            edgecolors="none", linewidth=0, antialiased=False, rasterized=True,
        )
        fig.colorbar(mesh, ax=ax, label=bar_label, extend=extend)
        ax.set_title(title, loc="right")
        ax.set_xlabel(xlabel or "", loc="right")
        ax.set_ylabel(ylabel or "", loc="top")
        if index == 1 or below or above or undefined:
            ax.text(
                0.02, 0.02,
                f"Below: {below}; above: {above}; undefined (gray): {undefined}",
                transform=ax.transAxes, va="bottom", fontsize="xx-small",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1},
            )
    if atlas:
        label_kwargs = {"status": "Internal"}
        label_kwargs.update(atlas_label_kwargs or {})
        add_atlas_label(axes[0, 0], **label_kwargs)

    if save_name is not None:
        _save_figure(
            fig, save_name, script_id=script_id, save_path=save_path,
            verbose=verbose,
        )
    return fig, axes, diagnostics


def artifact_closure_1d(
    baseline,
    components,
    *,
    residual_values,
    residual_errors,
    residual_valid=None,
    variable=None,
    xlabel=None,
    ylabel=None,
    baseline_label="Baseline",
    residual_label="Baseline - components",
    colors=None,
    baseline_linestyle="--",
    component_linestyles=None,
    baseline_linewidth=1.4,
    component_linewidths=None,
    residual_ylim=None,
    ylog=False,
    ysymlog_linthresh=None,
    atlas=False,
    atlas_label_kwargs=None,
    save_name=None,
    save_path=None,
    script_id=None,
    figsize=None,
    verbose=True,
):
    """Draw a baseline/component overlay with a caller-computed residual panel.

    ``baseline`` is one raw or density ``Hist1D`` and ``components`` is a
    non-empty ordered mapping of labels to ``Hist1D`` sharing its exact edges,
    density state, and bin count. Every series is drawn as stairs. The caller
    owns the closure formula and passes ``residual_values``, ``residual_errors``,
    and an optional boolean ``residual_valid`` mask in the same bin convention.

    Residual bins where ``residual_valid`` is false (default: non-finite values
    or errors) stay undefined. The residual panel draws an explicit zero line,
    and ``residual_ylim`` marks out-of-view central values with boundary
    triangles and reports below/above/undefined counts. The main panel supports
    ``ylog`` or a signed ``ysymlog_linthresh`` axis.

    ``colors`` optionally maps component labels (and ``baseline_label``) to any
    Matplotlib-compatible color; others use the Catppuccin palette. The baseline
    stair defaults to ``baseline_linestyle="--"`` and ``baseline_linewidth=1.4``
    so it stays distinguishable from overlapping components in grayscale.
    ``component_linestyles`` and ``component_linewidths`` optionally override a
    component's linestyle or linewidth; unknown labels are rejected. Returned
    values are ``(fig, (ax_main, ax_residual))``; no numerical diagnostics are
    computed here because the residual definition belongs to the caller.
    """
    if not isinstance(baseline, Hist1D):
        raise TypeError("artifact_closure_1d expects a Hist1D baseline")
    if not isinstance(components, Mapping) or not components:
        raise ValueError("components must be a non-empty ordered mapping")
    if baseline.errors.shape != baseline.counts.shape:
        raise ValueError("baseline errors must match counts")
    if (np.any(~np.isfinite(baseline.counts)) or np.any(~np.isfinite(baseline.errors))
            or np.any(baseline.errors < 0)):
        raise ValueError("baseline counts must be finite with finite nonnegative errors")
    for name, component in components.items():
        if not isinstance(component, Hist1D):
            raise TypeError("closure components must be Hist1D objects")
        if component.is_density != baseline.is_density:
            raise ValueError("closure cannot mix raw-count and density histograms")
        if component.counts.shape != baseline.counts.shape:
            raise ValueError("closure components must match the baseline bin count")
        if not np.array_equal(component.edges, baseline.edges):
            raise ValueError("closure components must share the baseline edges")
        if component.errors.shape != component.counts.shape:
            raise ValueError("component errors must match counts")
        if (np.any(~np.isfinite(component.counts))
                or np.any(~np.isfinite(component.errors))
                or np.any(component.errors < 0)):
            raise ValueError(
                "component counts must be finite with finite nonnegative errors"
            )

    values = np.asarray(residual_values, dtype=np.float64)
    errors = np.asarray(residual_errors, dtype=np.float64)
    if values.ndim != 1 or values.shape != baseline.counts.shape:
        raise ValueError("residual_values must be one-dimensional and match the baseline bins")
    if errors.shape != values.shape:
        raise ValueError("residual_errors must match residual_values")
    if residual_valid is None:
        valid = np.isfinite(values) & np.isfinite(errors)
    else:
        valid = np.asarray(residual_valid, dtype=bool)
        if valid.shape != values.shape:
            raise ValueError("residual_valid must match residual_values")
    if (np.any(~np.isfinite(values[valid])) or np.any(~np.isfinite(errors[valid]))
            or np.any(errors[valid] < 0)):
        raise ValueError("valid residuals must be finite with finite nonnegative errors")

    if colors is None:
        requested_colors = {}
    elif isinstance(colors, Mapping):
        requested_colors = dict(colors)
    else:
        raise TypeError("colors must be a mapping of labels to Matplotlib colors")
    known = set(components) | {baseline_label}
    unknown = set(requested_colors) - known
    if unknown:
        raise ValueError(f"colors has unknown labels: {sorted(unknown)}")
    for name, color in requested_colors.items():
        named = isinstance(color, str) and color in catppuccin_colors
        if not named and not is_color_like(color):
            raise ValueError(f"invalid Matplotlib color for {name!r}")
    resolved_colors = {}
    for index, name in enumerate(components):
        default = _CLOSURE_COMPONENT_COLORS[index % len(_CLOSURE_COMPONENT_COLORS)]
        color = requested_colors.get(name, catppuccin_colors[default])
        if isinstance(color, str) and color in catppuccin_colors:
            color = catppuccin_colors[color]
        resolved_colors[name] = color
    baseline_color = requested_colors.get(baseline_label, "black")
    if isinstance(baseline_color, str) and baseline_color in catppuccin_colors:
        baseline_color = catppuccin_colors[baseline_color]

    if component_linestyles is None:
        requested_linestyles = {}
    elif isinstance(component_linestyles, Mapping):
        requested_linestyles = dict(component_linestyles)
    else:
        raise TypeError(
            "component_linestyles must be a mapping of labels to linestyles"
        )
    unknown_styles = set(requested_linestyles) - set(components)
    if unknown_styles:
        raise ValueError(
            f"component_linestyles has unknown labels: {sorted(unknown_styles)}"
        )
    linestyle_probe = Line2D([], [])
    for name, linestyle in [
        (baseline_label, baseline_linestyle), *requested_linestyles.items(),
    ]:
        try:
            linestyle_probe.set_linestyle(linestyle)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid Matplotlib linestyle for {name!r}") from exc
    resolved_linestyles = {
        name: requested_linestyles.get(name, "-") for name in components
    }

    if component_linewidths is None:
        requested_widths = {}
    elif isinstance(component_linewidths, Mapping):
        requested_widths = dict(component_linewidths)
    else:
        raise TypeError(
            "component_linewidths must be a mapping of labels to linewidths"
        )
    unknown_widths = set(requested_widths) - set(components)
    if unknown_widths:
        raise ValueError(
            f"component_linewidths has unknown labels: {sorted(unknown_widths)}"
        )
    for name, linewidth in [
        (baseline_label, baseline_linewidth), *requested_widths.items(),
    ]:
        try:
            finite_positive = bool(np.isfinite(linewidth)) and linewidth > 0
        except TypeError:
            finite_positive = False
        if not finite_positive:
            raise ValueError(f"linewidth for {name!r} must be finite and positive")
    resolved_widths = {
        name: requested_widths.get(name, 1.0) for name in components
    }

    if ylog and (np.any(baseline.counts < 0)
                 or any(np.any(component.counts < 0) for component in components.values())):
        raise ValueError("logarithmic closure cannot display negative bins")
    if ysymlog_linthresh is not None:
        if ylog or not np.isfinite(ysymlog_linthresh) or ysymlog_linthresh <= 0:
            raise ValueError("ysymlog_linthresh must be positive and exclusive of ylog")
    if residual_ylim is not None:
        if (len(residual_ylim) != 2 or not np.all(np.isfinite(residual_ylim))
                or residual_ylim[0] >= residual_ylim[1]):
            raise ValueError("residual_ylim must contain two increasing finite limits")

    if figsize is None:
        figsize = (config.fig_size_square[0], config.fig_size_square[1] + 1.4)
    xlabel = xlabel or baseline.label
    xlabel, unit = _resolve_label_unit(variable, xlabel, baseline.unit)
    if ylabel is None:
        ylabel = _ylabel_counts(
            baseline.bin_width, unit=unit, is_density=baseline.is_density,
            weighted=baseline.is_weighted,
        )

    fig, (ax_main, ax_residual) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for name, component in components.items():
        ax_main.stairs(
            component.counts, component.edges, color=resolved_colors[name],
            linestyle=resolved_linestyles[name], linewidth=resolved_widths[name],
            label=name,
        )
    ax_main.stairs(
        baseline.counts, baseline.edges, color=baseline_color,
        linestyle=baseline_linestyle, linewidth=baseline_linewidth,
        label=baseline_label,
    )
    ax_main.set_ylabel(ylabel, loc="top")
    if ylog:
        ax_main.set_yscale("log")
    elif ysymlog_linthresh is not None:
        ax_main.set_yscale("symlog", linthresh=ysymlog_linthresh)
    _add_legend(ax_main, fontsize="x-small")
    if atlas:
        label_kwargs = {"status": "Internal"}
        label_kwargs.update(atlas_label_kwargs or {})
        reserve_atlas_label_headroom(ax_main, ylog=ylog)
        add_atlas_label(ax_main, **label_kwargs)

    centers = baseline.centers
    residual = np.where(valid, values, np.nan)
    residual_error = np.where(valid, errors, np.nan)
    ax_residual.stairs(residual, baseline.edges, color="black", linewidth=1.0)
    lower, upper = residual - residual_error, residual + residual_error
    ax_residual.fill_between(baseline.edges, np.r_[lower, lower[-1]],
                             np.r_[upper, upper[-1]], step="post",
                             color="black", alpha=0.18, linewidth=0)
    ax_residual.axhline(0.0, color="black", linestyle="--", linewidth=0.6)
    finite = np.isfinite(residual) & np.isfinite(residual_error)
    if np.any(finite):
        low = min(0.0, float(np.min(residual[finite] - residual_error[finite])))
        high = max(0.0, float(np.max(residual[finite] + residual_error[finite])))
        padding = 0.08 * (high - low) if high > low else 0.5
        ax_residual.set_ylim(low - padding, high + padding)
    else:
        ax_residual.set_ylim(-0.5, 0.5)
    if residual_ylim is not None:
        low, high = residual_ylim
        below = np.isfinite(residual) & (residual < low)
        above = np.isfinite(residual) & (residual > high)
        # Mark central values outside the window without changing the data.
        inset = 0.04 * (high - low)
        for mask, boundary, marker in ((below, low + inset, "v"), (above, high - inset, "^")):
            ax_residual.plot(centers[mask], np.full(np.count_nonzero(mask), boundary),
                             linestyle="none", marker=marker, markersize=3,
                             color="black")
        ax_residual.set_ylim(low, high)
        undefined = np.count_nonzero(~np.isfinite(residual))
        ax_residual.text(
            0.02, 0.04,
            f"Below: {np.count_nonzero(below)}; above: {np.count_nonzero(above)}; undefined: {undefined}",
            transform=ax_residual.transAxes, va="bottom", fontsize="xx-small",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1},
        )
    ax_residual.set_xlim(baseline.edges[0], baseline.edges[-1])
    ax_residual.set_xlabel(xlabel or "", loc="right")
    ax_residual.set_ylabel(residual_label, loc="top")

    if save_name is not None:
        _save_figure(
            fig, save_name, script_id=script_id, save_path=save_path,
            verbose=verbose,
        )
    return fig, (ax_main, ax_residual)


def register_comparison_plot(
    save_name, section, *, caption=None, script_id=None, width=0.45, label=None,
):
    """Register the exact PDF filename produced by a comparison plot."""
    resolved_script_id = script_id or config.get_script_id()
    filename = f"{resolved_script_id}-{save_name}.pdf"
    makelatex.register(
        resolved_script_id, section, filename, caption=caption,
        width=width, label=label,
    )
    return filename
