# analyb Implementation Invariants

Use `README.md` for setup and module ownership and `CHEATSHEET.md` for
the public API. Keep this file limited to constraints that are easy to violate
without producing an immediate error.

- Keep raw `Hist1D` counts and per-bin-width densities distinct. Use
  `divide_by_bin_width()` for densities; never mix raw and density histograms
  in arithmetic. Explicit nonuniform edges are valid.
- `ratio()` propagates independent numerator and denominator errors by default.
  Disable denominator propagation only when plotting it as a separate band.
  It requires matching edges and leaves zero-denominator bins NaN; never fill
  them with a value that could be read as a measured ratio.
- Variable labels and units belong to the caller. Keep the default registry
  empty; unknown names fall back to `(name, None)` unless strict lookup is used.
- Direct ROOT and histogram-cache serialization must preserve weighted
  `Hist1D.errors` and `Hist2D.errors` as sumw2 so normal and replot outputs
  agree.
- Histogram persistence accepts only the typed records in `analyb.io`. An
  invalid existing ROOT file must fail an update; never recreate it silently.
  Any invalid record must fail the caller instead of being skipped silently;
  cache replacement remains atomic.
- External artifact loading must retain ROOT flow-inclusive values and stored
  variances without transposing, merging, or silently inferring weighted
  errors. Pairwise comparisons require exact edges, use numerator/reference
  orientation, and leave zero-reference ratios undefined.
- Artifact rebinning aggregates raw whole bins and stored variances before
  normalization, preserving flow accounting on both axes and immutable inputs.
  Target edges belong to the caller; reject material splits or wider ranges.
  Edge roundoff tolerance must be bounded by neighboring bin widths, never
  solely by the absolute coordinate magnitude. `rebin_root_histogram` and
  `rebin_to_edges` share `histogram._aligned_edge_indices`; ROOT validation,
  flow handling and aggregation stay in `comparison.py`.
- `rebin_to_edges` merges whole aligned bins only and keeps the source range;
  `hist1d_with_ratio` and `hist1d_stacked` require shared edges. Never
  redistribute counts by bin-overlap fraction.
- Artifact comparison in analyb is noncanonical and diagnostics-only. Keep
  sample identities, object mappings, physics formulas, production logic, and
  acceptance decisions in their owning analysis layer.
- Raw artifact comparisons must use `compare_histograms(..., shape=False)`;
  1D and 2D comparison plots must propagate their shape flag. Do not allocate
  normalized-shape covariance matrices unless shape diagnostics are requested.
- The optional 1D density display is applied only after raw/shape comparison.
  Divide main-panel values and errors by widths; keep returned diagnostics and
  ratio panels in their existing raw/shape convention. Never normalize density
  heights by their sum.
- Annotation headroom is a final limit operation. Reserve it after the last
  overlay and the final linear/log/symlog scale; a later `set_ylim`,
  `set_yscale`, or autoscale call silently replaces it. The reusable entry
  point is `reserve_annotation_headroom(ax, (values, errors), ...)`, which uses
  the explicit data/error envelope instead of inspecting artists.
- `separate_ratio_errors=True` is a display choice for `points="reference"`. It
  never changes returned diagnostics. `ratio_presentation` and
  `separate_ratio_errors` are mutually exclusive; a `RatioPresentation` carries
  caller-owned values/errors/valid mask and must never be recomputed from the
  histograms.
  Its `components` are caller-owned display terms stacked from one; analyb
  never derives them or enforces that they sum to `values - 1`.
- `artifact_closure_1d` never computes the closure formula, residual, or
  covariance; the caller passes them. All series must share exact baseline
  edges, density state, and bin count, and returned diagnostics are not
  produced because the definition belongs to the analysis layer.
- `CutflowData` errors default to unknown (`None`, never drawn);
  `sqrt(N)` requires `errors="poisson"` for raw unweighted counts.
- `nested_efficiency` assumes `Cov(A, B) = Q_A` only by default, for genuinely
  nested events with identical weights at both stages. Never use that default
  when stage weights change; the caller supplies `cross_sumw` or the variance
  stays unknown. Never clip signed ratios or substitute an interval for `var`.
- `clopper_pearson_interval` is for genuine unweighted integer counts only.
  Never switch interval methods automatically by `N` or `N_eff`, and never
  apply binomial intervals to weighted or signed ratios.
- `cutflow_comparison` and `native_cutflow_chart` never infer covariance or
  missing uncertainties. `cutflow_comparison` aligns caller-owned cumulative
  stages and draws numerator-only ratio errors plus separate reference points;
  `native_cutflow_chart` draws every native stage without asserting a shared
  selection sequence.
- `provenance` records clean committed sources only. Refuse any status entry
  (including staged-only and untracked non-ignored files), submodules, hidden
  index flags, and committed symlinks that are absolute or do not reach a
  tracked regular file inside the checkout. Never store source payloads, raw
  argv or Git remotes; callers pass explicit safe `parameters`. Publish
  exclusively under the lock and never overwrite without `replace=True`. Use
  `package_checkout` to identify an imported package's checkout; directory
  containment is not identity. A recorded revision is provenance, not
  acceptance: accepted pins and hashes stay with the caller and are checked
  through `verify_sidecar(expected=...)`.

- `figcheck` geometry checks flag mechanical defects only (clipped, colliding,
  data-covering or undersized text). They never certify readability and
  never combine findings into a score. The save hook stays off by default,
  measures the figure before `savefig` (which can switch an engine-less figure
  to the rc layout engine), and must not alter what is saved; `"raise"` raises
  only after the PDF and sidecar are written. The font limits at final
  placement are a decided contract: below 5 pt fails, below 7 pt is a flag.
  An axis showing tick labels without an axis label fails
  (`axis-unlabelled`: every axis is labelled; a shared x
  axis is labelled once, under the lowest panel).

- `printed` recording never changes typeset output, and `text` is the string
  a reader sees, not TeX source. Keep `MATCH_RULE` and `find_printed` in step
  and version the rule string when it changes (the metadata gate pins it);
  a manifest is written only when its block succeeds.
- `write_compact_plotbook(page_map=True)` derives the map from the same loop
  that emits the TeX; never compute pages separately.

After changes, run:

```bash
python -m pytest tests/ -v
```
