# analyb

Small analysis + plotting library for TTree-based HEP work. Uses **polars** for dataframes and **matplotlib** for plots.

MIT licensed; see `LICENSE` and `THIRD_PARTY_NOTICES.md`. This public snapshot
contains reusable infrastructure, not an analysis definition. See
`PUBLICATION.md` for scope and compatibility changes.

## Dependencies

`numpy scipy matplotlib polars[rtcompat] pyarrow uproot`

## Installation

```bash
python -m pip install -e ".[dev]"
```

The `dev` extra installs `pytest`; omit it for a runtime-only installation.
`polars[rtcompat]` selects Polars' CPU-compatible runtime.

## Plot style

`analyb` applies its repository-owned, MathText-first Matplotlib style on import.
Routine plots do not require a TeX installation. Generated standalone `.tex`
reports can be compiled separately where TeX Live is available.
Histogram comparison helpers own bin-width/density y labels. The public
flavour comparison overlays separately normalized flavours against their own
reference by default or one selected shared reference, with numerator-only ratio
errors, solid colour-coded source curves, equally sized filled reference
points and a fixed source-by-flavour legend; cutflow measurements explicitly
distinguish nested events from nonsequential object bins (see `CHEATSHEET.md`).

`reserve_annotation_headroom()` lets callers that draw extra overlays or set a
final scale reserve label headroom last, after those changes. A later
`set_ylim`/`set_yscale` replaces it, so call it once all overlays and the final
linear/log/symlog scale are in place. See `CHEATSHEET.md` section 8.

`check_figure()` reports mechanical layout defects (text clipped by the
canvas, overlapping text, legends or labels over data, and text below a 5 pt
floor or a 7 pt advisory size at a given final width) as separate findings.
`set_figure_check("warn"|"raise")` runs it on every save and writes a
`.figcheck.json` sidecar beside each PDF. It is off by default and never
replaces visual review. See `CHEATSHEET.md` section 8.

## Local tests (no ROOT data needed)

```bash
# From this checkout, with the project environment activated:
python -m pytest tests/ -v
```

Scientific validation is data-led: run the intended artifacts, inspect numerical
summaries and rendered plots, and retain small independent regression examples
for numerical conventions and demonstrated failures. Avoid tests that freeze
imports, internal call structure, or other incidental implementation choices.

## Modules

| Module                 | Purpose                                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `binning.py`           | Scott, Sturges, Freedman-Diaconis rules                                                                                       |
| `colors.py`            | Catppuccin palette, custom colormaps                                                                                          |
| `comparison.py`        | Noncanonical whole-bin artifact aggregation, pairwise diagnostics, overlays, response differences, and plotbooks              |
| `config.py`            | Script ID, data/save paths, weight column                                                                                     |
| `cutflow.py`           | Caller-supplied cutflow data, nested weighted efficiency variance, charting, aligned cutflow comparison                       |
| `figcheck.py`          | Render-time figure lint: clipped, overlapping, data-covering and undersized text; optional save hook with JSON sidecar |
| `histogram.py`         | Hist1D/Hist2D, fill, rebin, normalize, ratio, profile                                                                         |
| `io.py`                | TTree loading, flow-aware external TH1/TH2 loading, and typed histogram-record ROOT persistence                               |
| `makelatex.py`         | LaTeX auto-generation for figure environments; compact plotbook with optional page map JSON                                   |
| `parallel.py`          | Forked fill/replot workers with parent-owned typed ROOT persistence                                                            |
| `plot.py`              | 1D/2D plots, ratio panels, stacked, fits, ATLAS label, final annotation headroom                                            |
| `printed.py`           | Printed-value manifest: each typeset number with key, value, text, source, context |
| `provenance.py`        | Run provenance sidecars: clean committed source revisions, input/output hashes, recipe parameters, verification               |
| `stats.py`             | Chi-squared, Poisson likelihood, Clopper-Pearson binomial intervals, Gaussian model functions                                 |
| `style.py`             | Import-time MathText-first Matplotlib style from `styles/analyb.mplstyle`; applied on `import analyb`                         |
| `units.py`             | SI prefix scaling, bin-width formatting                                                                                       |
| `variables.py`         | Empty, caller-extensible branch → (LaTeX label, unit) registry                                                               |
| `yieldtable.py`        | Yield tables (text/LaTeX export)                                                                                              |

See `CHEATSHEET.md` for detailed API reference.

## Compact artifact presentation

`artifact_comparison_1d(..., points="reference", separate_ratio_errors=True)`
draws reference pseudodata at unity with its own relative error and a
numerator/reference stair with numerator-only errors; numerical diagnostics
keep their combined-error definition. Density scaling still occurs after
normalization and cancels from ratios, and no histogram title is drawn. For
caller-computed paired ratios, pass a validated `RatioPresentation` (values,
errors, valid mask, optional unity errors) through `ratio_presentation=`. It is
mutually exclusive with `separate_ratio_errors`.

`artifact_closure_1d(baseline, {label: Hist1D}, residual_values=...,
residual_errors=..., residual_valid=...)` renders a stairs overlay and a
caller-defined residual panel on a shared-x 3:1 layout. It validates shared
edges, density state, shapes, finite nonnegative errors, and colors; the
closure formula stays in the caller. The baseline defaults to a dashed,
heavier stair (`baseline_linestyle="--"`, `baseline_linewidth=1.4`) and
components to solid `1.0`, with optional `component_linestyles` and
`component_linewidths` mappings, so overlapping series stay distinguishable in
grayscale after compact-page reduction.

`analyb.cutflow.cutflow_comparison` takes explicitly aligned cumulative arrays,
optional stored uncertainties, and caller-owned stage labels. Missing weighted
uncertainties remain absent. Initial-yield-normalized views use a fixed scale;
they are not binomial efficiency uncertainty estimates. `native_cutflow_chart`
draws every supplied weighted stage without grouping or inferring missing
errors.

These retained full-stack cutflow helpers preserve their original layout and
central-value range defaults, including a minimum 0.5–1.5 ratio span. For a new
presentation, finalize the returned axes from the displayed error envelopes and
the applicable plot contract; do not treat those defaults as a readability check.

`analyb.makelatex.write_compact_plotbook(path, sections)` writes a standalone
LaTeX book with two columns and four vector PDF figures per page. Each section
contains ordered `(relative_pdf_path, identifying_caption)` pairs; for
comparisons, register yield then shape to put two observables on each page. It
accepts `rows_per_page=3` and a `section_rows` mapping to retain two rows for
dense native labels, and `plotbook_context()` is a Matplotlib `rc_context` for
readable type in compact books. Compile with `pdflatex`. `page_map=True` also
writes `<stem>.pagemap.json` (page, section, ordered figure slots) for
coverage checks.

`analyb.printed` records each number a report prints (`key`, `value`, the
typeset `text`, `source`, row `context`) in a JSON manifest while a
`value_manifest` block is active; yield tables record theirs automatically.
The v4 matcher validates complete numeric spans, preserves placeholder/range
dashes and rejects ambiguous signs or unsupported numerals. Its contract is
row presence, not column assignment or arithmetic; see `CHEATSHEET.md` section 15.

`hist2d_plot` renders the actual bin edges, including nonuniform bins, with
rasterized mesh contents and vector labels. Pass the same `norm` to multiple
plots to share their colour scale, and `colorbar_label` to name weighted
contents. The default sequential 2D colormap is `superheat`, registered by
`analyb.colors` and exposed as `current_colorbar`; analysis recipes should omit
`cmap` to inherit it and use explicit diverging maps only when the plotted
quantity calls for one.
