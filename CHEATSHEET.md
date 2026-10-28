# analyb cheat sheet

    import analyb as ab

## 0. First install

    python -m pip install -e ".[dev]"
    # Installs polars[rtcompat] for CPU-compatible execution.

    # Plot labels use portable MathText by default; TeX is not required.

## 1. Configuration

    ab.set_script_id("example")
    ab.set_save_path("plots")        # light files: plots, LaTeX, logs
    ab.set_data_path("output")       # heavy files: output ROOT, histogram cache
    # output root file: {data_path}/{script_id}-output.root
    # histogram cache:  {data_path}/{script_id}-cache.root
    ab.get_output_root()

    # PNG output (on by default, 150 DPI):
    ab.set_save_png(True)            # disable with ab.set_save_png(False)
    ab.set_save_png_dpi(200)         # override DPI

    # Figure lint at save time (off by default; see section 8):
    ab.set_figure_check("warn")      # or "raise"; None turns it off
    ab.set_figure_final_width_cm(7.5)  # enables the 5 pt floor / 7 pt advisory

## 2. Load TTrees (polars + uproot)

    # Recommended: set weight column once, then omit from branches list.
    ab.set_weight_column("eventWeight")

    df = ab.load_tree(
        "path/to/file.root",
        tree_path="events",
        branches=["met", "ptl1"],
        entry_stop=100000,         # remove for all events
    )
    # eventWeight is auto-appended from the config default above.

### Explicit weights_branch (overrides config default)

    df = ab.load_tree(..., weights_branch="myWeight")

### With cuts (polars Expr)

    import polars as pl
    df_cut = ab.load_tree(
        "path/to/file.root",
        "events",
        branches=["met", "ptl1"],
        cuts=pl.col("met") > 50,
    )
    # A list or insertion-ordered dict of expressions applies cuts sequentially.

### Batch load

    dfs = ab.load_trees({
        "sample_a": ("sample_a.root", "events"),
        "sample_b": ("sample_b.root", "events"),
    })

### Chunked iteration (bounded memory)

    for chunk in ab.iterate_tree(
        "file.root", "events",
        branches=["met", "ptl1"],
        weights_branch="eventWeight",
        entry_stop=500000,           # global exclusive entry index (or total when entry_start is None)
        entry_start=0,               # global inclusive entry index (optional, passed to uproot)
        step_size="100 MB",          # or an int like 50000
        workers=4,                    # uproot decompression/interpretation threads
    ):
        # chunk is a polars DataFrame — fill histograms, accumulate stats, etc.
        ...

Same weight-auto-append and cuts semantics as ``load_tree``.  ``entry_start``
and ``entry_stop`` are passed directly to ``uproot.iterate`` so disk I/O is
bounded *before* the chunk arrives; a defensive post-hoc slice still guards
against any unexpected extra entries.  ``entry_stop`` is a global exclusive
index — when ``entry_start`` is ``None`` it behaves as "total raw entries to
read".

## 3. Caller-owned observables

Define analysis observables in the caller, using Polars expressions or another
appropriate framework. No reconstruction catalogue is included. Register labels
and units explicitly (section 10); the registry starts empty.

## 4. Histogram filling

    wgt = df["eventWeight"].to_numpy()   # or df[ab.get_weight_column()].to_numpy()

    h = ab.hist1d(
        df["met"].to_numpy(),
        weights=wgt,                       # optional; if given, errors = sqrt(sum(w²))
        bins="scott",                      # "sturges" | "fd" | int | edges
        unit=r"\GeV",
        range=None,                        # or (lo, hi)
        pct_range=None,                    # or (5, 95) for robust bounds
        overflow=True,
        merge_overflow=False,
    )

    h2 = ab.hist2d(x, y, bins=(40, 40), range=None, weights=wgt)
    # Hist2D.errors is computed as sqrt(sum-of-weights²) when weighted

    h          # repr: nbins, range, integral
    h.counts   # array
    h.edges    # array
    h.errors   # array (sqrt(sum-w²) for weighted, sqrt(counts) otherwise)
    h.bin_width
    h.widths      # np.diff(h.edges)
    h.is_uniform
    h.centers
    h.integral()
    h.integral_error()                     # sqrt(Σ errors²) for weighted

### Binning rules

    ab.resolve_bins("scott", data)    # → edges
    ab.resolve_bins(50, data)         # → edges (50 equal bins)
    ab.resolve_bins(2.5, data)        # → edges (width 2.5)

Automatic ranges require at least one finite value. Pass an explicit `range=`
when intentionally creating empty histograms.

The y label names what is plotted: `Counts` for a raw histogram, `$\sum w$` for
a weighted one (`is_weighted=True`), e.g. `$\sum w$ / GeV` for a weighted
density; `shape=True` comparisons prefix `Normalized`.

Explicit nonuniform edge arrays are supported. Raw nonuniform histograms have
`bin_width is None` and plot with the y label `Counts`; uniform raw histograms
retain the existing numerical `Counts / <bin width>` label.

### Differential per-bin-width histograms

    density = ab.divide_by_bin_width(h)

This returns a new `Hist1D`, dividing every displayed-bin count and error by
its own width and marking it as a density (`density.is_density`). It preserves
edges, labels, units, weighted status, and raw underflow/overflow yields. Its
y label is `Counts / unit` (or `Counts / <unit>`), not a first-bin-width approximation. It is not a
unit-area normalization. Raw-count and density histograms cannot be summed or
ratioed together; density rebinning converts through raw bin yields.

## 5. Operations on histograms

    h_r   = ab.rebin(h, factor=2)
    h_n   = ab.normalize(h, to=1.0)
    r, e  = ab.ratio(h_num, h_den)       # → combined numerator/denominator error
    # Edges must match (roundoff only); zero-denominator bins are NaN by default.
    # Pass nan_to=<value> to fill them explicitly.
    r, e  = ab.ratio(h_num, h_den, include_denominator_error=False)
    # Use numerator-only point errors when the denominator uncertainty is a separate band.

### Sum multiple histograms (2D)

    hs = ab.sum_histograms([h1, h2, h3])
    hs = ab.sum_histograms({"ttbar": h_tt, "Wt": h_wt})
    hs = h1 + h2 + h3

    h2_sum = ab.sum_histograms2d([h2a, h2b, h2c])
    h2_sum = ab.sum_histograms2d({"chunk0": h2a, "chunk1": h2b})
    # Merges exact-matching 2D histograms bin-wise; errors = sqrt(Σ errors²)

### Chunked profile aggregation

When processing data in chunks (e.g. from ``ab.iterate_tree``), collect additive
sufficient statistics per chunk and merge them in one pass:

    # --- per chunk ---
    n, sum_w, sum_w2, sum_wy, sum_wy2, edges = ab.profile_sufficient_statistics(
        x, y, bins=20, range=(-5, 5), weights=w,
    )
    # accumulate: n_acc += n, sum_w_acc += sum_w, ... (all the same edges)

    # --- after all chunks ---
    centers, means, stderr = ab.profile_from_sufficient_statistics(
        n_acc, sum_w_acc, sum_w2_acc, sum_wy_acc, sum_wy2_acc, edges,
    )

``ab.profile`` (weighted path) uses these internally, so the chunk-merged
result matches the one-shot computation exactly.

### Merge bins onto aligned edges

    h_merged = ab.rebin_to_edges(h, target_edges)
    # sums whole source bins and their variances; flows are unchanged

Every target edge must be a source edge (roundoff only) and the target must
span exactly the source range. Splitting a bin, widening or narrowing the range
raises `ValueError`. Density histograms are merged through raw bin yields.

### Luminosity scaling helper

    scale = 139.0  # fb⁻¹
    cross_section = 2.0   # pb, synthetic example
    sum_w = float(df["eventWeight"].sum())
    h_scaled = ab.scale_to_luminosity(h, cross_section=cross_section,
                                      luminosity=scale * 1000,
                                      sum_of_weights=sum_w)
    # Formula: σ · L / Σw  (applied to both counts and errors)

Use this helper only for samples where the required
normalization inputs are known and the weights are not already normalized.

## 6. 1D plots

### Single histogram

    ab.hist1d_plot(h, variable="met", style="atlas", atlas=False, save_name="met")

`style`: "atlas" (errorbar+points), "stairs" (step outline), "mc" (filled step+alpha)

### Data vs MC with ratio panel

    ab.hist1d_with_ratio(h_data, h_mc, variable="met", atlas=False, save_name="met_ratio")

Data and MC must share bin edges (roundoff only); the plot raises instead of
redistributing counts. Merge aligned bins with `rebin_to_edges()` first if needed.
Data default to black points/errors and the MC model to Catppuccin red, including
its uncertainty band. Explicit `data_color` and `mc_color` override those defaults.

### Stacked MC (with optional data overlay + ratio)

    ab.hist1d_stacked(
        {"ttbar": h_tt, "Wt": h_wt}, variable="met", save_name="stack",
        colors={"ttbar": "tab:blue", "Wt": "tab:orange"}, data_color="black",
    )

All stack components and the optional data histogram must share bin edges.
Stacked fills use opaque colours (no alpha) so the legend colours match the
fills exactly.  Data points rendered via `_data_errorbar` (ATLAS-style:
``elinewidth=0.8``, ``capsize=0``, marker ``.`` size 4).
Each component runs from its cumulative bottom to its cumulative top, preventing
gaps and overlap. Ratio panels retain at least `[0.5, 1.5]` and expand for
finite points and error bars; clear near-zero-denominator outliers are omitted
from automatic limits.
Mapping colors are keyed by sample label and do not change stack geometry;
sequence colors continue to follow draw order. Legends are yield-descending
(with `Data` first for a data overlay), while geometry remains caller ordered.
Legends are frameless and inset inside the upper-right axes corner; ATLAS labels
occupy the upper-left of the same reserved header region. 2D histograms use a
wider default figure so the heatmap remains square beside the labelled colorbar.
Data/MC ratio points use data-only uncertainties because the MC uncertainty is
shown separately as the ratio band; direct `ab.ratio()` calls retain combined
uncertainties by default.

### Overlay comparison

    ab.hist1d_comparison({"ttbar": h_tt, "Wt": h_wt}, normalize=True, variable="met")
    ab.hist1d_overlay({"ttbar": h_tt, "Wt": h_wt}, variable="met")

### Systematic uncertainty summary (fractional breakdown)

    ab.systematic_summary_plot(
        nominal_hist,
        {"JES": (h_jes_up, h_jes_down),
         "b-tag": (h_btag_up, h_btag_down)},
        data_stat_hist=data_hist,
        variable="met",
        save_name="sys_summary_met",
    )
    # Accepts either (up, down) tuple or pre-computed fractional Hist1D per category.
    # Plots stacked colored bands per source + gray total + black stat band.

## 7. 2D plots

    ab.hist2d_plot(h2, variable_x="met", variable_y="metphi",
                   cmap=ab.current_colorbar,  # "superheat" default
                   norm=matplotlib.colors.LogNorm(),
                   save_name="2d_met")

### Scatter plot

    ab.scatter_plot(x, y, variable_x="met", variable_y="mtlep",
                    alpha=0.1, atlas=True, save_name="scat")

Uses `ax.plot` (single Line2D — fast for large datasets) with
density-dependent alpha transparency. When `alpha` is `None`,
it is automatically computed as `max(0.01, min(0.25, 50 / n))` so
dense regions appear darker via overlapping transparent markers.

### Axis label format

X- and y-axis labels automatically include the MathText unit in bracket notation:

    $E_T^{\mathrm{miss}}$ [$\mathrm{GeV}$]

Y-axis labels use portable MathText, e.g.
`Counts / $4.3\;\mathrm{GeV}$`; plotting does not require TeX or siunitx.
After whole-bin aggregation, `artifact_comparison_1d` and
`artifact_comparison_flavours_1d` label equal-width counts per display bin and,
with `density=True`, unequal-width counts per unit (including normalized
counts). Density divides only main-panel values and errors by displayed width;
ratios and diagnostics retain their original convention. The flavour helper
takes source -> flavour -> Hist1D, flavour -> reference Hist1D and source/flavour
colours; each ratio uses its own flavour's reference, with numerator-only bands and reference
unity points by default. Pass `ratio_reference_flavour="l"` to divide every
source flavour by the combined reference (for shape plots, its own unit-area
normalized histogram) while retaining separate main-panel reference flavours
and only one combined unity series below. With that shared reference, pass
`show_reference_ratios=True` to place the other reference flavours in the lower
panel as their ratio to combined. Supply `reference_ratio_errors={"e": array,
"m": array}` for caller-derived covariance-aware errors; omit them when the
reference uncertainties are unknown. The helper does not infer covariance
between a flavour subset and its combined reference. The numerator-only band
and reference-unity uncertainty are drawing conventions, not a total ratio uncertainty.
Source curves are solid and unmarked, references use explicit filled round
points of equal size, and the legend has source rows and flavour columns. Pass
`ratio_bands=False` for central-only ratio presentation (the returned
independent diagnostics are unchanged); `reference_label` names a different
observed reference. `flavour_matrix_legend(ax, colors, rows, point_rows=...)`
supplies the same ordered legend for a task-specific paired overview.
`artifact_comparison_1d(..., point_color=...)` changes point presentation
without changing the reference or ratio arithmetic. All histograms must share
raw whole-bin edges.

`cutflow_chart(..., measurement="raw_objects")` uses a measured-quantity label;
other choices are `events`, `raw_events`, `survival`, `cut_efficiency`,
`objects_first_bin`, `object_bin_ratio`, and `milestone_retention` (for other
mapped-retention diagnostics). Genuinely nested adjacent mapped event stages
may use `cut_efficiency` when compressed native cuts are qualified in the caption.
`cutflow_ylabel(measurement)`
also labels external ratio axes. Use object-bin labels for nonsequential bins;
neither first-bin scaling nor adjacent object-bin ratios are efficiencies.
Register angular variables with siunitx `\radian` when generating TeX;
rendered plot labels use the conventional abbreviation `rad`. This public
snapshot does not pre-register analysis variables.

## 8. ATLAS style

Toggle via `atlas=True` in any plot function:

    ab.hist1d_plot(h, variable="met", atlas=True, save_name="met")
    # adds the four-line ATLAS label at fixed upper-left inside the axes

Also callable manually:

    ab.add_atlas_label(ax, status="Internal", lumi=139, energy=13, channel=r"$e\mu$")

### Fixed layout (upper-left)

The ATLAS label is always placed at the upper-left corner of the axes
(``x=0.04, y=0.95`` in axes coordinates, top-aligned).  1D plotting functions
(``hist1d_plot``, ``hist1d_with_ratio``, ``hist1d_stacked``,
``hist1d_comparison``, ``profile_plot``, ``efficiency_plot``,
``systematic_summary_plot``, ``plot_data_with_fit``, ``cutflow_chart``)
automatically call ``reserve_atlas_label_headroom()`` to expand the y-axis
limits so that roughly the top 30% of the plot area is reserved for the label.
Legends are frameless and inset inside the upper-right corner.

### Final annotation headroom (after overlays and scale)

The plotting functions above already reserve label headroom before returning.
Callers that keep drawing afterwards (extra overlays, a final explicit
`set_ylim`, a log/symlog scale) must reserve again, because the last
`set_ylim`/`set_yscale`/autoscale call replaces whatever was reserved:

    ax.set_yscale("log")               # or "symlog", or leave linear
    ax.set_ylim(low, high)             # caller's final bounds, after all overlays
    ab.reserve_annotation_headroom(
        ax,
        (model_values, model_errors),  # one (values, errors) pair per series
        (data_values, data_errors),
    )
    ab.add_atlas_label(ax, lumi=False)

It expands only the upper limit so the highest finite `values + errors` point
occupies at most `1 - fraction` of the axis, measured in axis-transform
coordinates. The same visible `fraction` (default `0.30`) is therefore reserved
on linear, log and symlog axes. The envelope is explicit; artists already on
the axes are not inspected, so pass every series that must stay clear of the
annotation region. Each `errors` array must match its `values` shape or be a
scalar; negative finite errors are rejected. A single series is
`reserve_annotation_headroom(ax, (values, errors))`. The return value is `True`
when the upper limit changed and `False` when the reservation was unnecessary,
unsupported for the current scale/limits, or not representable as a finite top
(for example an extreme fraction on a very wide log range). On `False` the axes
are left untouched.

Limitations: it never lowers the top and never adjusts the bottom limit, so
callers still own fitting new overlaid series; it uses the axis transform of
the current scale, so set the final scale before calling it; it does not
validate scientific admissibility (positive envelope or positive lower limit on
a log axis); and it is a pure limit calculation with no collision engine.
`artist_axes_bounds(artist, ax=None)` returns an artist's `(x0, y0, x1, y1)`
bounding box in axes coordinates (0-1 across the axes) to check that a legend
or label actually fits. It lays the figure out with
`Figure.draw_without_rendering()` and measures the final post-layout geometry,
so it works on PDF/SVG backends without writing output; no overlap detection is
performed.

### Figure lint (mechanical geometry checks)

`check_figure(fig, *, final_width_cm=None, min_font_pt=5.0, flag_font_pt=7.0)`
lays the figure
out once with `Figure.draw_without_rendering()` and returns a list of
`Finding(check, severity, where, bbox, value, message)`. Like the headroom
helper, call it after the final limits, scales and overlays:

    findings = ab.check_figure(fig, final_width_cm=7.5)
    for f in findings:
        print(f.check, f.severity, f.where, f.value, f.message)

| `check` | severity | `value` |
| --- | --- | --- |
| `text-outside-canvas`: a text or legend box extends past the canvas (clipped in the saved PDF) | fail | overshoot, pt |
| `text-overlap`: two distinct texts intersect (a legend counts as one box against other text; its entries are compared with each other) | fail | smaller overlap side, pt |
| `text-over-data`: a legend or in-axes/figure text covers line or marker vertices, or a line segment crosses it | fail | covered vertices |
| `text-over-data` over a filled area (bars, bands, filled steps) | flag | outline vertices inside |
| `axis-unlabelled`: a visible x or y axis shows tick labels but has no axis label (a shared x axis with hidden tick labels needs none) | fail | `None` |
| `font-below-floor`: base font size scaled to `final_width_cm` is below `min_font_pt` (5 pt); skipped without a final width | fail | effective size, pt |
| `font-below-advisory`: scaled size at least `min_font_pt` but below `flag_font_pt` (7 pt) | flag | effective size, pt |

`where` names the axes index (`fig.axes` order) and the artist kind with a text
snippet, for example `axes[1] xtick label '1000'` or `axes[0] legend`. `bbox`
is `(x0, y0, x1, y1)` in figure inches from the canvas lower-left corner. Each
finding stands alone; there is no combined score, and an empty list does not
mean the figure reads well.

Save hook (default off). `save_figure(..., figure_check=None,
final_width_cm=None)` and every plot helper's `save_name=` path use the
configured mode; a per-call `figure_check` (`"warn"`, `"raise"`, or `False`
to skip) and `final_width_cm` override the configuration:

    ab.set_figure_check("raise")
    ab.save_figure(fig, "ptl1", final_width_cm=7.5)
    # writes {sid}-ptl1.pdf, then {sid}-ptl1.figcheck.json; raises
    # FigureCheckError (with .findings) after saving if any finding fails

Both modes log each finding at WARNING and write `<pdf stem>.figcheck.json`
beside the PDF with `schema`, `schema_version`, `pdf`, `pdf_sha256`,
`figure_size_in`, `figure_width_cm`, `final_width_cm`, `min_font_pt`,
`flag_font_pt`,
`bbox_units`, `checks_skipped`, `checks` (the checks that ran) and `findings`. The figure is measured
immediately before it is saved. Flags never raise.

Limitations: extents are Matplotlib layout boxes (ascent and descent
included), not glyph ink, so text that nearly touches is reported; rotated
text is compared as a rotated rectangle for overlaps but by its axis-aligned
box for the canvas and data checks; MathText
sub/superscripts are measured at their base size; only Line2D, scatter
offsets, line collections, filled patches and polygon collections that paint
something (visible, nonzero alpha, width or marker size) count as data (images, `QuadMesh` 2D histograms, and legend handles do not); text
deliberately placed on data (bar values, 2D cell labels) is reported; canvas
bounds assume the default `bbox_inches=None` save.

MC-only labels can omit luminosity:

    ab.add_atlas_label(ax, status="Internal", lumi=False)

### Label format

The ATLAS status text is placed on a new line:

    ATLAS
    Internal
    √s = 13 TeV
    L = 140 fb⁻¹

### Data-marker style

Data points (used by atlas-style histograms, ratio panels, profile plots, and
efficiency plots) are rendered as filled dots (`markerfacecolor` matches the
series colour, not white).

## 9. Unit prefix scaling

Y-axis bin-width labels automatically scale the SI prefix so numbers stay readable:

    0.0003\\,GeV  →  0.3\\,MeV        (was: "Counts/0.0\GeV")
    1500\\,GeV    →  1.5\\,TeV

This is handled by `ab.units`:

    ab.format_bin_width(0.0003, r"\GeV")   →  ("0.3", r"\MeV")
    ab.scale_value(0.0003, r"\GeV")        →  (0.3, r"\MeV")

### HEP unit families (predefined)

| Family | Members                                        |
| ------ | ---------------------------------------------- |
| energy | `\eV`, `\keV`, `\MeV`, `\GeV`, `\TeV`, `\PeV`  |
| length | `\pm`, `\nm`, `\um`, `\mm`, `\cm`, `\m`, `\km` |
| time   | `\fs`, `\ps`, `\ns`, `\us`, `\ms`, `\s`        |
| barn   | `\ab`, `\fb`, `\pb`, `\nb`, `\ub`, `\mb`, `\b` |

Units not in any family fall back to adaptive-precision formatting.

### Extending

    from analyb.units import UNIT_FAMILIES, UNIT_SCALES

    UNIT_FAMILIES["mass"] = [r"\ug", r"\mg", r"\g", r"\kg", r"\t"]
    for i, macro in enumerate(UNIT_FAMILIES["mass"]):
        UNIT_SCALES[macro] = 1000 ** i
    # → now mass units are auto-scaled too

### Submodule reference

    import analyb.units

    ab.units.SI_PREFIXES    # supported SI subset: {\\yocto: -24, ..., \\yotta: 24}
    ab.units.UNIT_FAMILIES  # dict of family name → macro list
    ab.units.UNIT_SCALES    # dict of macro → scale factor (relative to base)

## 10. Variable registry (labels + units)

The registry starts empty. Unknown variables use their supplied name and no
unit; `lookup(name, strict=True)` raises `KeyError` for unknown names.
Register your labels before drawing. MathText symbols and units are supported.

    ab.lookup("met")       # → ("met", None), before registration
    ab.register("met", r"$E_T^\mathrm{miss}$", r"\GeV")
    ab.register_many({"ptl1": (r"$p_T^{\ell 1}$", r"\GeV"), ...})

### Temporary overrides

    with ab.local({"met": (r"$E_T^{\mathrm{miss}}$", r"\GeV")}):
        label, unit = ab.lookup("met")

## 11. Profile & efficiency

    centers, means, errors = ab.profile(x, y, bins="scott", weights=w)
    # weights are optional; if provided, computes weighted mean and standard
    # error of the weighted mean (using effective sample size n_eff = (Σw)²/Σw²)
    ab.profile_plot(centers, means, errors, xlabel=r"\met", save_name="profile")

    centers, eff, err_lo, err_hi = ab.efficiency(passed, total, bins=50)
    # Both per-bin counts and per-event arrays are supported. For per-event mode
    # pass the kinematic variable: ab.efficiency(passed, total, bins=50, variable=x)
    ab.efficiency_plot(centers, eff, err_lo, err_hi, xlabel="var", save_name="eff")
    # With reference overlay + ratio panel:
    ab.efficiency_plot(centers, eff, err_lo, err_hi,
                       reference=(ref_centers, ref_eff, ref_lo, ref_hi))
    # Named series, reference points with errors, caller ratio errors, ranges:
    ab.efficiency_plot(centers, eff, err_lo, err_hi,
                       reference=(ref_centers, ref_eff, ref_lo, ref_hi),
                       label="Model", reference_label="Reference", reference_color="#0072B2",
                       reference_errors=True, ratio_label="Model / Reference",
                       ratio_errors=(ratio_lo, ratio_hi), ylim=(0, 1.3),
                       ratio_ylim=(0.5, 1.5))
    # analyb never propagates efficiency errors into the ratio: the caller owns
    # their correlation and passes ratio_errors (symmetric array or (low, high)).
    # Further curves against the same reference (points in both panels):
    ab.efficiency_plot(centers, eff, err_lo, err_hi, reference=(...),
                       more_series={"Alternative": dict(centers=c2, eff=e2, err_low=l2,
                                                     err_high=h2, color="#E69F00",
                                                     ratio_errors=r2, marker="s")})

`efficiency` uses the Jeffreys interval; at `k = 0` the lower bound is 0 and at
`k = N` the upper bound is 1, so both error lengths stay nonnegative.

    lower, upper = ab.clopper_pearson_interval(passed, total)  # cl=0.6827

Exact Clopper-Pearson bounds for genuine unweighted integer counts, at least
nominal coverage for every `N`; no method switching by `N` or `N_eff`. Bounds
are not a variance. Weighted, signed or non-integer inputs are rejected; for
weighted ratios use `nested_efficiency` (section 14), which has no coverage
guarantee. `total = 0` gives NaN bounds; counts above 2**53 are rejected.

## 12. Fits

    ab.plot_data_with_fit(x, y, yerr, fit_func, popt,
                          xdescr=r"$x$", ydescr="Counts",
                          fit_label="My fit", save_name="fit")

## 13. IO: save/load to root

    ab.save_hist_to_root("met_ttbar", h, file=ab.get_output_root(), mode="recreate")
    ab.save_hist2d_to_root("met_metphi", h2, file=..., mode="update")

    h_loaded = ab.load_hist_from_root(ab.get_output_root(), "met_ttbar")

Weighted 1D and 2D errors are persisted as sumw2 sidecars. `mode="update"`
creates a missing file, but an invalid existing ROOT file raises instead of
being silently replaced.

### Load an immutable external TH1 or TH2

    loaded = ab.load_root_histogram("artifact.root", "region/histogram")
    hist = loaded.histogram
    loaded.values_with_flow       # exact ROOT values, including all flow bins
    loaded.variances_with_flow    # exact ROOT variances, including all flow bins
    loaded.axis_titles
    loaded.classname

The external loader requires a stored ROOT `fSumw2` array by default. This is
the fail-closed path for weighted artifacts. A known unweighted histogram may
opt into ROOT's inferred Poisson variances with `require_variances=False`.
For TH1, `hist.underflow` and `hist.overflow` also expose the two flow yields.
For TH2, use the flow-inclusive source arrays; their shape is `(nx + 2, ny + 2)`
and their central slice is `[1:-1, 1:-1]`. No transpose, rebinning, or flow
merging is performed.

### Aggregate artifact bins before normalization

    source = ab.load_root_histogram("fine.root", "histogram")
    reference = ab.load_root_histogram("benchmark.root", "histogram")
    rebinned = ab.rebin_root_histogram(source, (reference.histogram.edges,))
    # TH2: target_edges=(reference.histogram.x_edges, reference.histogram.y_edges)

`rebin_root_histogram` takes a `RootHistogram` and one target edge array per
axis. It sums raw counts and variances of whole bins, returning another
`RootHistogram` with exact copies of the target edges and unchanged source
metadata. Source arrays are not modified. Narrower target ranges move excluded
bins into underflow/overflow, including TH2 edge strips and corners. Report
these flow-inclusive arrays separately from displayed-bin yields.

Target edges must align with existing edges, using the same matcher and
tolerance as `rebin_to_edges`. The tolerance is 64 double-precision
epsilons times the larger of the edge magnitude and full axis span, capped
at 1e-10 of the narrower adjacent source bin. This permits immaterial serialization
rounding but rejects material shifts even at large coordinate offsets. Splitting
bins or recovering a wider range from existing flow totals is unsupported.
Aggregate before shape normalization or density conversion; density inputs are
rejected. Variances are summed as independent bin sumw2; no event-level
information or covariance is inferred. Target binning belongs to the calling
analysis recipe.

### Pairwise artifact diagnostics

    result = ab.compare_histograms(numerator, reference)
    result.numerator_yield, result.reference_yield
    result.yield_ratio, result.yield_ratio_error
    result.normalized_numerator, result.normalized_reference
    result.shape_difference, result.shape_covariance
    result.shape_chi2, result.shape_ndof
    result.shape_available, result.shape_chi2_supported
    result.shape_eigenvalue_cutoff
    result.ratio, result.ratio_error, result.ratio_valid
    result.difference, result.difference_error

For raw-only diagnostics, especially histograms with many bins, disable shape
construction explicitly:

    result = ab.compare_histograms(numerator, reference, shape=False)
    assert result.shape_covariance is None

This sets `shape_available=False` because shape diagnostics were not requested.

The orientation is always `numerator / reference` and `numerator - reference`.
Zero-reference ratio bins are `NaN` with `ratio_valid=False`; signed nonzero
reference bins remain defined. Inputs must have exactly equal stored edges.
Displayed yields and shapes exclude flow bins so callers can report flows as a
separate artifact property. Shape normalization propagates the independent bin
variances through the normalization Jacobian. The shape chi-square uses the
combined covariance eigenspectrum with one relative cutoff for both inversion
and `shape_ndof`. A mismatch in an unsupported zero-variance direction reports
an infinite chi-square and `shape_chi2_supported=False`; the expected
normalization null mode is excluded. If either signed integral is zero, raw
yield, ratio, and difference diagnostics remain available while
`shape_available=False` and shape-valued fields are `None`.

    fig, axes, result = ab.artifact_comparison_1d(
        numerator, reference,
        numerator_label="Candidate", reference_label="Reference",
        ratio_label="Candidate / Reference", shape=False, density=False,
        variable="met", atlas=True, save_name="yield_met",
    )

The 1D helper passes its `shape` flag to `compare_histograms`. Raw yield plots
therefore skip normalized-shape covariance construction; shape plots request it.
Set `density=True` to divide only the main-panel values and errors by each bin
width after those diagnostics are computed. Yield-density integrals reproduce
the raw displayed yield, while normalized shape-density integrals are one.
The ratio panel and returned raw/shape diagnostics remain unchanged; density is
never normalized again by summing its displayed heights. Inputs must be raw
counts because this option owns the one display conversion.
Both comparison helpers use the shared analyb fonts, ticks, colors, and
label helpers. Heatmaps disable axis grids and rasterize only the mesh to avoid
cell seams; 1D plots retain the shared grid. Use `atlas_label_kwargs={"lumi": False}` to omit luminosity when
it is not applicable to the artifacts.

The 1D helper defaults to `points="numerator"`: black numerator points/errors and
Catppuccin red reference stairs/band. Set `points="reference"` for a reference drawn as
pseudodata: black reference points/errors and red numerator model stairs/band.
This changes only the main-panel drawing roles. Keep the arguments in the
scientific numerator/reference order: ratios and their combined independent
errors do not invert when drawing roles change. A pseudodata role does not
claim that the reference is observed collision data.

For a split-uncertainty presentation, pass `points="reference"` with
`separate_ratio_errors=True`. The ratio panel then draws the black reference
points at unity with their relative errors `sigma(reference)/abs(reference)`, and a red
`numerator/reference` stair with a band of numerator-only relative errors
`sigma(numerator)/abs(reference)`. The returned `diagnostics` keep the independent combined
errors unchanged; this is a display choice only. This mode requires
`points="reference"`.

For several predictions against the same reference, use
`artifact_comparison_many_1d({"Model": model, "Variation": variation}, reference,
colors={"Variation": catppuccin_colors["blue"]}, shape=False, density=False,
atlas=True)`. It preserves the pairwise panel geometry and inset legend,
draws reference points at unity once with their relative errors, and draws
each ratio line with numerator-only errors. Inputs must share exact raw histogram edges;
zero reference denominators remain undefined. The returned diagnostics describe the
first prediction only. With `ratio_ylim`, every prediction's central ratio
values outside the window get boundary triangles (`^`/`v`) in its own colour,
the first prediction's as in the pairwise helper; get the counts for a note
from `ab.ratio_range_counts(ratio, ratio_ylim)`.
To show an additional observed flavour against a combined reference on a
separate-flavour page, pass `reference_variation=reference_flavour`,
`variation_color=...` and caller-derived `variation_ratio_errors=...`. It is
drawn in both main and ratio panels; absent ratio errors draw bare points.

When the ratio values, errors, or defined-bin mask come from the caller, pass a
`RatioPresentation` instead:

    presentation = ab.RatioPresentation(
        values=paired_ratio,
        errors=paired_ratio_uncertainty,
        valid=defined_bins,
        unity_errors=optional_reference_relative_errors,
    )
    fig, axes, diagnostics = ab.artifact_comparison_1d(
        numerator, reference, ratio_presentation=presentation,
        ratio_ylim=(0.5, 1.5),
    )

`values` and `errors` are already in ratio units; `errors` is the red band
half-width. Invalid bins stay undefined when drawn. `unity_errors` is optional
and adds the black unity-reference errorbar. All fields are validated at
construction: shapes must agree, valid values must be finite, and valid
errors nonnegative. `ratio_presentation` and `separate_ratio_errors` are
mutually exclusive so one ratio definition is never applied twice.
For a reference with unknown uncertainty, use `unity_points=True` and leave
`unity_errors=None`; this draws defined unity markers without implying a
zero-variance measurement.
Out-of-range boundary markers and below/above/undefined counts use the
presentation values, and returned diagnostics remain independent.

`error_style="bars"` draws the presentation errors as vertical bars at the bin
centres in the ratio line's colour (`model_color`) instead of the translucent
band. To show the terms of a bin decomposition of `ratio - 1` inside the same
panel, pass them as signed per-bin `components` with caller colours:

    presentation = ab.RatioPresentation(
        values=q, errors=paired_sigma, valid=defined, error_style="bars",
        components={"$W$": w_rel, "$M^\\mathrm{in}$": in_rel,
                    "$M^\\mathrm{out}$": -out_rel, "$G$": g_rel, "$L$": -l_rel},
        component_colors={"$W$": "#E69F00", ...},
    )
    fig, axes, _ = ab.artifact_comparison_1d(
        candidate, reference, points="reference", ratio_presentation=presentation,
        ratio_ylim=(1 - half, 1 + half), legend_ncol=3,
    )

Each component is drawn as a translucent bar (`component_alpha`, default
0.45) stacked from one: positive parts upward, negative parts downward, in
mapping order. The component labels join the main-panel legend
(`legend_ncol` columns). analyb never checks that the components add up to
`values - 1`; that identity belongs to the caller. Without `ratio_ylim` the
automatic range covers the stacks. With it, a stack leaving the window gets a
grey boundary triangle where the central value is inside, and the panel text
adds `stacks beyond: n`. `component_stacks()` returns the positive and
negative stack extents for range choices.
`range_text=False` drops the in-panel below/above/undefined (and stacks
beyond) text while keeping the boundary markers; get the same numbers from
`ab.ratio_range_counts(values, ratio_ylim, valid=..., stacks=presentation.component_stacks())`
to print them elsewhere.
Ratio lines are open stairs (`baseline=None`): they do not drop to zero at the
outer edges, which would read as a ratio of zero.
`model_baseline=None` does the same for the main-panel line, for quantities
that are not counts (survival, cut efficiency); bins left out of the edges stay
empty. The default
`model_baseline=0` keeps a count histogram's closed outline.

Categorical ratio summary (one slot per category, e.g. the sub-steps of a chain
and a final total): each series is a `RatioPresentation` with one entry per
category, drawn as points with error bars in its own sub-slot, its components
stacked from one as bars filling that sub-slot:

    dev = ab.RatioPresentation(values=q_dev, errors=s_dev, valid=ok,
                               components={"$W$": w, "$G$": g, "$-L$": -l},
                               component_colors={"$W$": "#E69F00", "$G$": "#009E73",
                                                 "$-L$": "#D55E00"})
    fig, ax = ab.ratio_summary_plot(
        [r"R$\to$Hold", r"Hold$\to$HA", r"R$\to$H"], {"dev": dev, "val": val},
        ylabel=r"$q = Y_B/Y_A$", xlabel="Sub-step", colors={"val": "#0072B2"},
        markers={"val": "s"}, ratio_ylim=(0.8, 1.1), separators=(2,))

Central values beyond `ratio_ylim` get boundary triangles in the series colour,
stacks leaving it grey ones; `separators` draws a vertical line before the
given category indices, up to `separator_ymax` (axes fraction, default 1; lower
it to keep the line out of a legend). Both axis labels are required. Pass
`ax=` to draw into one panel of an existing figure (e.g. a zoomed second panel).

For a signed main-axis display, pass `ysymlog_linthresh=1.0` (in the displayed
count units); this is exclusive of `ylog=True`. Set `ratio_ylim=(0.5, 1.5)` for a
bounded ratio window. Boundary triangles mark out-of-view central values, and
the panel reports below/above/undefined bin counts. Error bars may extend outside
the window. Full x edges and returned diagnostics remain unchanged. Omit
`ratio_ylim` for the default unrestricted ratio extent.

    fig, axes, result = ab.artifact_comparison_2d(
        numerator_response, reference_response,
        numerator_label="Candidate", reference_label="Reference",
        difference_label="Candidate - Reference",
        ratio_label="Candidate / Reference",
        variable_x="reco", variable_y="particle",
        save_name="response_met",
    )

The 2D helper returns a `(2, 2)` axes array. The top row is numerator and
numerator/reference; the bottom row is reference and numerator minus reference.
Both inputs share one color normalization. The default ratio scale is centered
on one and covers all finite signed ratios; the difference scale is symmetric
about zero. The derived panels use the actual custom
`ab.catppuccin_colorbars["coolwarm"]` object, not Matplotlib's same-name builtin.
Input panels retain the shared `superheat` palette. `ratio_cmap` and
`difference_cmap` accept explicit colormap objects or names.

The mesh uses exact nonuniform x/y edges and keeps analyb's `(x, y)` array
orientation. Zero-reference ratios are masked gray and counted, distinct from
finite zero ratios. Explicit color limits show colorbar extensions and counts
for finite cells outside the range, without changing numerical values.

Its default `shape=False` passes through to `compare_histograms`, avoiding dense
shape covariance construction for raw responses. With `shape=True`, both ratio
and difference panels use the normalized arrays; the returned diagnostics keep
their documented raw ratio and separate shape fields. Only request normalized
response diagnostics when matrix dimensions are tractable.

Optional Matplotlib `norm`, `ratio_norm`, and `difference_norm` objects customize
the shared input, ratio, and difference color scales; they are copied before
initializing unset bounds. For a bounded ratio display use
`TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)`. For signed log differences use
`SymLogNorm(linthresh=1.0, vmin=-limit, vmax=limit)`. Keep the ratio centered at
one and difference at zero when choosing custom norms. Histogram arrays,
caller normalizations, and colormap objects remain unchanged.

When the closure residual is caller-computed, use the dedicated renderer:

    fig, (ax_main, ax_residual) = ab.artifact_closure_1d(
        baseline, {"Retained": retained, "Removed": removed},
        residual_values=baseline.counts - retained.counts - removed.counts,
        residual_errors=residual_sigma,
        residual_valid=defined_bins,
        baseline_label="Baseline", residual_label="Baseline - retained - removed",
        colors={"Removed": "peach"},
        baseline_linestyle="--", baseline_linewidth=1.6,
        component_linestyles={"Retained": "-"},
        residual_ylim=(-2.0, 2.0), ysymlog_linthresh=1.0,
        atlas=True, atlas_label_kwargs={"lumi": False},
    )

`components` is a non-empty ordered mapping of labels to `Hist1D`.
Every input must share the baseline's exact edges, density state, and bin
count, with finite contents and finite nonnegative errors. Baseline and
components are drawn as stairs on a shared-x 3:1 main/residual layout. The
baseline defaults to a dashed, slightly heavier stair (linestyle `--`, linewidth
`1.4`) and components to solid `1.0`, so an overlapping baseline stays
distinguishable in grayscale after compact-page reduction;
`baseline_linestyle`/`baseline_linewidth` and the optional
`component_linestyles`/`component_linewidths` mappings override them. Linewidths
must be finite and positive, and unknown mapping labels are rejected like
`colors`. The
caller owns the closure formula; the residual arrays must match the baseline
bins. Invalid residual bins stay undefined, the residual panel always draws a
zero line, and `residual_ylim` adds below/above triangles and an
undefined-bin count. `colors` accepts Matplotlib colors or Catppuccin palette
names and rejects unknown labels. The function returns only
`(fig, (ax_main, ax_residual))` because the formula is not analyb's.

Register the generated PDF with the existing LaTeX plotbook:

    ab.register_comparison_plot(
        "yield_met", "Reconstructed comparisons",
        caption="Candidate and reference missing transverse momentum",
    )

The returned filename is exactly `{script_id}-{save_name}.pdf`, matching the
comparison plot save path. Artifact comparisons are diagnostic only; analyb
does not define production, mappings, physics formulas, or acceptance.

### Save DataFrame as TTree

    ab.save_df_as_tree(df, tree_name="after_cuts", file=ab.get_output_root())
    # Use mode="update" to add to an existing file (default: "recreate")

### Extract a unit from a LaTeX label

    ab.extract_unit_from_latex_string(r"$p_T$ [\unit{\GeV}]")  # r"\GeV"
    ab.extract_unit_from_latex_string(r"Dimensionless")         # None

## 14. Cutflow

    cf = ab.CutflowData(stages, weighted_counts, errors=known_errors)
    # Supply stage labels, yields and uncertainties from your own data reader.
    raw = ab.CutflowData(stages, raw_counts, errors="poisson")  # sqrt(N), raw unweighted only

    ab.cutflow_chart(cf, mode="absolute", save_name="cutflow")
    # mode: "absolute" (yields) | "relative" (efficiencies)

`errors=None` means unknown: no error bars are drawn, and nothing is inferred
from the counts. The relative mode draws no error bars.

### Nested weighted efficiency

    eps, var, valid = ab.nested_efficiency(passed, total, passed_sumw2, total_sumw2)
    # stage weights changed between the stages: pass C = sum_passed w_A w_B
    eps, var, valid = ab.nested_efficiency(passed, total, passed_sumw2, total_sumw2,
                                           cross_sumw=cross)

First-order variance of `passed / total`,
`V = (Q_A + eps^2 Q_B - 2 eps C) / B^2`, from independent-event sumw2; no
systematic uncertainties. The default `C = Q_A` requires the passing events to
be a genuine subset of the denominator events with identical weights at both
stages; with changed weights pass `cross_sumw` or
treat the uncertainty as unknown. Scalars or arrays of one shape. `None`/NaN
sumw2 gives `var = NaN` with `eps` kept. Signed ratios outside [0, 1] are
never clipped. Inconsistent moments (`Q_A > Q_B` for the default,
`|C| > sqrt(Q_A Q_B)`) beyond 1e-12 relative roundoff raise. `valid` marks a
finite ratio with nonzero denominator; it does not certify coverage of
`eps +- sqrt(var)`.

## 15. Yield tables

    yt = ab.yield_table(
        {("ttbar", "SR"): h_tt, ("Wt", "SR"): h_wt},
        column="sample",
    )
    print(yt.to_text())
    print(yt.to_latex())
    yt.save("yields")

### Printed-value manifest

Record every number exactly as it is typeset, so a review gate can find it in
the final PDF. Off unless a manifest is active; `printed()` formats either way:

    from analyb import printed
    with printed.value_manifest("report/values.json", producer="generate_report.py",
                                sources={"evidence": "evidence.json"}):
        cell = printed.printed("yield_ratio/ptl1", ratio, ".6f",
                               context="ptl1", source="evidence", section="Yields")
        yt.save("yields-sr")   # records yields-sr/{row}/{column}/value|error

`printed(key, value, fmt, **fields)` returns `format(value, fmt)`;
`record_value(key, value, text, **fields)` records text formatted elsewhere.
Fields: `source` (a role in `sources`), `context` (a label on the same table
row or line, usually the row label), `section`, `page`. Keys are unique; an
identical re-record is a no-op, a conflicting one raises. A non-finite value
is stored as `null` with its placeholder text. `YieldTable.to_latex(key=...,
source=...)` records each value and error with the row label as `context`;
`save(base_name)` uses `base_name` as the key prefix. Output TeX is identical
with or without a manifest. The manifest is written only if the block
succeeds.

JSON: `schema` (`analyb.printed_values`), `schema_version` (4), `producer`,
`match_rule`, `sources` (`{role: {path, sha256}}`), `values` (ordered
`{key, value, text, source, context, section, page}`).

Matching rule v4 (`printed.MATCH_RULE`; `printed.find_printed(lines, text,
context=None)` implements it). Unary minus glyphs U+2212, U+FE63 and U+FF0D
fold to `-`, and Unicode decimal digits fold to ASCII. Dash glyphs U+2012,
U+2013, U+2010, U+2011 and U+2014 remain separators during scanning, so a
missing-cell dash before `3.2` does not invent `-3.2` and `0 – 5` does not
contain `-5`.

One maximal-candidate scanner (`printed.numeric_tokens`) handles both the
extracted line and the record's `text`. Supported numbers have an optional unary sign,
a mantissa (`digits[.digits]` or `.digits`), optional comma groups, an optional
`e`/`E` exponent and an optional suffix. Grouping commas allow whitespace on
both sides; the first group has one to three digits and later groups exactly
three. Exponents allow whitespace around `e`/`E` and their sign. Suffixes
allow any preceding whitespace: `%`, `GeV`, `MeV`, `TeV`, `keV`, `eV`, `fb-1`,
`pb-1`, `fb`, `pb`, `nb`, `mb`, `mm`, `cm`, `rad`, `mrad`, `ns`, `ps`.

A unary sign may be attached or followed by one whitespace character, excluding
a tab or line break. Longer gaps are ambiguous and reject the following number
as a whole. A sign directly after a word, `.`, `)` or another sign is invalid.
An attached sign directly after a numeric token is a compact range separator
(`0-5`); a spaced sign following a numeric token with at most one intervening
whitespace character is ambiguous and rejects its operand (`0 − 5`). A wider
cell gap before the sign allows a unary sign (`12    − 3.2`). ASCII dash runs
are placeholders. This conservative rule cannot resolve every typesetting
ambiguity; it rejects uncertain numeric presence rather than assigning columns.

Malformed grouping, repeated decimal fractions, incomplete exponents,
nondecimal numerals, unknown attached units (`12kg`) and word-adjacent numbers
are consumed and rejected whole; scanning never resumes inside them. A record
with any numeric character outside supported spans fails even if the same
literal text occurs on the line. An exactly one-token record matches an equal
canonical line token (whitespace removed): `.5` cannot match `-.5`, `04`
cannot match `.5e+04`, `234.5` cannot match `1 , 234.5`, and `12.5` cannot
match `12.5%`. Unicode decimal spellings compare with their ASCII equivalents.

Other supported text matches its characters in order with optional whitespace,
Unicode decimal boundaries, and every numeric token also present as a complete
line token. Dash glyphs fold to `-` only for this literal comparison, allowing
`0-5` to match `0–5`. `context` must appear on the same extracted line
(`pdftotext -layout`) as a whole token sequence: `ttbar` does not match
`ttbar_extra` or `ttbaré`; `_` and whitespace match each other because
pdflatex's default encoding extracts `\_` as a space. **Only row presence is
verified, not columns, value/error assignment or arithmetic**: swapping the
value and error in a row still passes.

## 16. Colors & colormaps

    ab.catppuccin_colors["red"]       # (0.8235, 0.0588, 0.2235)  RGB tuple
    ab.catppuccin_colors["blue"]
    ab.catppuccin_colors["green"]

    ab.current_colorbar              # superheat colormap (default for 2D plots)
    list(ab.catppuccin_colorbars)    # "light_coolwarm", "superheat", "nova", ...

    ab.colors.PALETTES               # categorical palettes the helpers draw together

Catppuccin palette values are MIT licensed; see `THIRD_PARTY_NOTICES.md`.
Optional colour-vision and perceptual-separation diagnostics are not included;
see `PUBLICATION.md`.

## 17. LaTeX auto-generation

    # Register each plot/table as you produce it.
    # filename must match the actual saved file (= {script_id}-{save_name}.pdf):
    ab.makelatex.register(
        script_id="example",
        section="Lepton kinematics",                   # becomes \subsection{...}
        filename="example-ptl1.pdf",                  # match _save_figure output
        caption=r"$p_T^{\ell 1}$ in ttbar",           # optional
    )

    # Register a table (e.g. from YieldTable):
    ab.makelatex.register_table(
        script_id="example",
        section="Yield tables",
        filename="example-yields.tex",
        caption="Event yields per sample and region",
        label="tab:yields",                           # optional; default: tab:{basename}
    )

    # Register a non-floating LaTex fragment (for example, a longtable).
    ab.makelatex.register_input(
        script_id="example",
        section="Method notes",
        filename="example-method-notes.tex",
        page_break_before=True,                       # optional
        page_break_after=True,                        # optional
    )

    # Emit per-script standalone .tex with guarded preamble.
    # Uses \ifdefined\analybmain around the preamble so the same file works
    # both as a standalone document and as a snippet \input{} from a master.
    ab.makelatex.write_tex(SCRIPT_ID)
    # → writes {save_path}/{script_id}.tex   (standalone compilable)
    #   Compile: cd {save_path} && pdflatex {script_id}.tex

    # Optional master document that \input{}s per-script .tex files.
    # Defines \analybmain so that per-script files skip their preamble.
    # Uses a hidden .plots_registry.json to accumulate script IDs across runs.
    # Each script should pass only its own ID; the registry handles the rest.
    # Running the same script multiple times is idempotent.
    ab.makelatex.write_master(
        ["example"],                           # only your own SCRIPT_ID
        title="Analysis Plots",
        save_path="plots",
    )
    # → writes plots/plots.tex  (defines \analybmain, \input{}s per-script files)

For a compact two-column benchmark book, pass ordered `(pdf, caption)` pairs:

    ab.makelatex.write_compact_plotbook(
        "book.tex",
        {"Particle": [("particle-yield.pdf", "Yield"), ("particle-shape.pdf", "Shape")]},
        rows_per_page=3, section_rows={"Native cutflows": 2},
    )

Each section maps a plain-text heading to ordered `(relative_pdf_path,
identifying_caption)` pairs. Default is two rows per page; `rows_per_page=3`
and per-section `section_rows` overrides select the denser layout. PDFs keep
their vector content and aspect ratio.

`page_map=True` also writes `book.pagemap.json` from the same pagination loop
(the TeX is unchanged): `schema` (`analyb.plotbook_page_map`),
`schema_version`, `tex`, `pdf`, `page_count`, and `pages`, each
`{page, section, rows, slots}` with ordered slots `{figure, caption, row,
column}` (1-based; a lone last figure is column 1). Each section starts a new
page and an empty section produces none. Pages are as intended by the writer;
compare them with the compiled PDF (`PTEX.FileName`) to catch an overflow.
`write_tex`/`write_master` use floats, so their pagination is LaTeX's and no
map is written.

    with ab.plotbook_context():
        ...  # draw 7-inch figures with 16 pt labels for two-across A4 pages

`plotbook_context()` is a Matplotlib `rc_context` that raises font sizes for
figures later scaled onto a compact page; it restores the previous rcParams on
exit.

## 18. Parallel batch processing

The `analyb.parallel` module lets you fill and plot multiple histograms in
parallel using `ProcessPoolExecutor` (fork mode). Workers inherit the
DataFrame copy-on-write, so no large data pickling happens.

    # 1D batch (replaces a manual for-loop over variables)
    results = ab.parallel.hist1d_batch(
        df,
        variables=["met", "ptl1", "ptb1"],   # flat list, or
        # variables=PLOT_PLAN,                # section-structured dict
        weights=wgt,
        ranges=VAR_RANGES,
        integer_vars={"njets", "nbjets"},
        ylog={"met", "ptl1"},                # True, set, or None
        max_workers=8,
        progress=True,                       # tqdm or simple counter
    )

    # 2D batch
    results = ab.parallel.hist2d_batch(
        df,
        pairs=[("met", "ptl1"), ("metphi", "phil1")],
        # pairs=PLAN_2D,                    # section dict also accepted
        weights=wgt, ranges=VAR_RANGES,
        log_norm=True, max_workers=8,
    )

    # Profile batch
    results = ab.parallel.profile_batch(
        df,
        pairs=[("met", "mtlep"), ("ptb1", "ptb2")],
        ranges=VAR_RANGES, max_workers=8,
    )

All batch functions return `list[TaskResult]` — inspect `.success`,
`.error`, and `.traceback` per task. ROOT histograms (including profiles
and per-sample compare histograms) and LaTeX captions are saved **serially
in the parent** after all workers finish, to ``{output_dir}/{script_id}-output.root``.
`TaskResult.root_data` is one of `Hist1DRecord`, `Hist2DRecord`,
`ProfileRecord`, or `CompareRecord`; tuple and marker-dict wire formats are not
accepted. Use `ab.hist1d_record(hist)` and `ab.hist2d_record(hist)` when
constructing a result outside the batch helpers.
An invalid record fails the caller instead of being skipped silently. Cache
replacement remains atomic, so a failed cache write retains its prior artifact.

### Data/MC comparison batch

The `compare_batch` function distributes per-variable data/MC comparison
plots (overlays, ratios, stacked) across workers.

    # Overlay (unit-area normalised MC comparison)
    ab.parallel.compare_batch(
        dfs_mc, None, variables, edges,
        plot_type="overlay", save_prefix="cmp_",
        section="MC shape comparisons",
        caption_fmt="Shape comparison: {var}, {samples}",
        ylog=set_of_log_vars,
    )

    # Data/MC ratio
    ab.parallel.compare_batch(
        dfs_mc, df_data, variables, edges,
        plot_type="ratio", save_prefix="data_mc_ratio_",
        section="Data/MC ratio panels",
        caption_fmt="{var} -- Data / MC ratio",
    )

    # Stacked MC + data
    ab.parallel.compare_batch(
        dfs_mc, df_data, variables, edges,
        plot_type="stacked", save_prefix="data_mc_stack_",
        section="Stacked MC + data",
        caption_fmt="{var} -- stacked MC ({samples}) with data",
        stack_colors={"ttbar": "tab:blue", "Wt": "tab:orange"},
        stack_data_color="black",
    )

Parameters:

- `dfs_mc` — dict of MC DataFrames (`{"ttbar": df_tt, "Wt": df_wt, ...}`)
- `df_data` — data DataFrame (`None` for MC-only overlays)
- `edges` — shared bin edges per variable (from `concat_edges`)
- `caption_fmt` — LaTeX caption with `{var}` (label) and `{samples}` (joined names)
- `stack_colors`, `stack_data_color` — optional stacked-only color controls;
  these are also accepted by `replot_compare_batch`

### Error handling

Each worker catches exceptions and returns a `TaskResult(success=False, …)`.
The parent summarises failures at the end:

    30 tasks: 28 succeeded, 2 failed
      FAILED: metphi  → ValueError: ...
      FAILED: dR2b   → OSError: ...

Set `fail_fast=True` to cancel remaining tasks on first failure.

### Progress display

- With tqdm installed: animated progress bar
- Without tqdm: simple `[done/total]` counter line
- Each indicator names its operation (for example `Fill 2D plots` or
  `Replot Comparison (stacked)`).

  pip install tqdm # optional, recommended

### Histogram cache and replot

The histogram cache stores all computed histograms (1D, 2D, profiles,
compare per-sample) in a ROOT file ``{output_dir}/{script_id}-cache.root``.

    # Save cache from batch results
    ab.save_histogram_cache(results)

    # Load cache back
    cache = ab.load_histogram_cache()
    cache["hist1d"]    # {var: Hist1D, ...}
    cache["hist2d"]    # {key: Hist2D, ...}
    cache["profiles"]  # {key: (centers, means, stderr), ...}
    cache["compare"]   # {var: {"mc": {name: Hist1D, ...}, "data": ...}}

### Replot from cache (no data loading)

    # Instead of hist1d_batch(df, ...)
    ab.parallel.replot_1d_batch(cache["hist1d"], PLOT_PLAN, ...)

    # Instead of hist2d_batch(df, ...)
    ab.parallel.replot_2d_batch(cache["hist2d"], PLAN_2D, ...)

    # Instead of profile_batch(df, ...)
    ab.parallel.replot_profile_batch(cache["profiles"], PROFILE_PLAN, ...)

    # Instead of compare_batch(dfs_mc, df_data, ...)
    ab.parallel.replot_compare_batch(cache["compare"], variables, ...)

All replot functions use the same ``ProcessPoolExecutor`` parallel
machinery.  The analysis script supports ``--replot`` as a shortcut:

    python example.py --replot

### Replot functions now register LaTeX

The four ``replot_*_batch`` functions (:func:`~analyb.parallel.replot_1d_batch`,
:func:`~analyb.parallel.replot_2d_batch`, :func:`~analyb.parallel.replot_profile_batch`,
:func:`~analyb.parallel.replot_compare_batch`) now also register LaTeX section entries
identical to their non-replot counterparts.  Only entries whose histograms are
actually **present** in the supplied cache dictionaries are registered — no
stale entries from the variable catalog.

### Table registration with ``register_table``

    ab.makelatex.register_table(script_id, section, filename, caption=None, label=None)

Tables are rendered in their own ``table[H]`` float (one per entry) with
``\centering`` and ``\input``.  They never share a row with figures.  The
preamble includes ``\usepackage{booktabs}`` and ``\usepackage{longtable}``.
``register_input`` entries are emitted directly under their subsection via
``\input`` without a float wrapper.  ``page_break_before=True`` places
``\clearpage`` before the subsection heading; ``page_break_after=True`` places
one after the included fragment.

### Yield table LaTeX escaping

:meth:`YieldTable.to_latex` now escapes LaTeX special characters (``_``, ``&``,
``%``, ``$``, etc.) in row and column labels, uses ``$\pm$`` for uncertainties,
and emits a correct all-``l`` column specification.

### Notes

- All batch functions (`hist1d_batch`, `hist2d_batch`, `profile_batch`,
  `compare_batch`) save their histograms to the output ROOT file and
  register LaTeX captions automatically.
- Replot functions do **not** save to ROOT (they only regenerate figures).
- `hist1d_batch` auto-registers LaTeX captions when variables are passed
  as a section-structured dict (the same format as `PLOT_PLAN`).
- The histogram cache is always written during normal runs (saved for all
  successful tasks).  It is a separate file from the output ROOT file.
  It is written locally first, then copied to a same-directory staging file and
  atomically replaced where supported; logs report write, copy, and total time.

### Compact benchmark presentation

Use `artifact_comparison_1d(..., points="reference",
separate_ratio_errors=True)` for pseudodata and separate ratio uncertainties.
Use `analyb.cutflow.cutflow_comparison(stages, ff, ir,
numerator_errors=ff_errors, reference_errors=ir_errors)` for explicitly aligned
cutflows. `None` omits unavailable errors. `normalized=True` scales each series
by its initial central yield. `native_cutflow_chart(stages, values,
errors=...)` draws every supplied stage separately, so nonsequential native
labels never assert correspondence to another source's sequence. Its stage
labels use the rc `ytick.labelsize`, so `plotbook_context()` keeps them at
least 5 pt in a compact plotbook.

`analyb.makelatex.write_compact_plotbook("book.tex",
{"Particle": [("yield.pdf", "Yield"), ("shape.pdf", "Shape")]})` emits up to
four PDF plots per page without splitting the attached ratio panels.

`hist2d_plot` renders the actual bin edges, including nonuniform bins, with
rasterized mesh contents and vector labels. Pass the same `norm` to multiple
plots to share their colour scale, and `colorbar_label` to name weighted
contents. Omit `cmap` to inherit the default sequential `superheat` palette
(`analyb.current_colorbar`); use an explicit diverging map only when the
plotted quantity calls for one.

## 19. Run provenance

```python
from analyb import provenance

# Locate the checkout the imported package really comes from (its file must be
# tracked at HEAD as analyb/__init__.py), refuse unclean sources early, and
# keep a drift baseline.
start = {"analysis": provenance.source_state(repo_dir),
         "analyb": provenance.source_state(provenance.package_checkout(analyb))}
...  # run the recipe and write its outputs into out_dir
sidecar = provenance.write_sidecar(
    out_dir,
    repos={role: state["path"] for role, state in start.items()},
    inputs={"benchmark": benchmark_root},
    configs={"mapping": mapping_json},
    parameters={"force": False},         # explicit safe recipe parameters
    seeds={"bootstrap": 12345},          # optional
    outputs=[out_dir / "evidence.json"], # recorded relative to out_dir
)
for state in start.values():
    provenance.verify_source(state)      # still clean at the same HEAD
provenance.verify_sidecar(out_dir, expected={
    "repos": {"analyb": {"head": pinned_revision}},
    "inputs": {"benchmark": {"sha256": accepted_sha256}},
})
```

`provenance.json` holds `schema`, `schema_version`, `created_utc`, `cwd`,
`parameters`, `repos` (per role: `path`, `head`, `branch`,
`head_in_remote_ref`), `inputs`/`configs` (per role: `path`, `size`,
`sha256`), `seeds`, `environment` (Python, executable, platform; versions and
locations of analyb, numpy, scipy, uproot, matplotlib) and `outputs` (`size`,
`sha256`). Raw argv and Git remotes are never recorded.

- Every recorded checkout must be clean: any `git status --porcelain=v2` entry
  (index or worktree change, staged-only change, mode change, untracked
  non-ignored file, nested repository), any submodule, any assume-unchanged or
  skip-worktree entry, and any committed symlink that is absolute or does not
  resolve inside the checkout to a tracked regular file raise
  `ProvenanceError`. Commit locally to explore; an unpushed HEAD is recorded
  with `head_in_remote_ref: false`.
- Publication holds `.provenance.lock` exclusively and renames the sidecar into
  place atomically; an existing sidecar is replaced only with `replace=True`.
- `verify_sidecar(path, expected=...)` re-hashes outputs and matches a nested
  subset of caller-owned accepted identities.
