# Public snapshot scope

This MIT-licensed snapshot starts a new public history. It retains reusable
histogramming, statistics, ROOT IO, plotting, artifact comparison, parallel
execution, cutflow containers/charts, report generation and provenance helpers.
The original development history is not part of this public repository.

## Compatibility changes

- `analyb.derived`, `analyb.derived_reference` and `DERIVED_REFERENCE` are
  omitted. Analysis reconstruction definitions without verified public-paper
  provenance remain outside this snapshot, including their tests and examples.
- The variable registry starts empty. `lookup(name)` returns `(name, None)`
  for unknown names; strict lookup raises `KeyError`. Use `register`,
  `register_many` or `local` to supply labels and units before plotting.
- `load_cutflow` and its private sumw2 helper are omitted because the loader's
  source provenance is unresolved. Supply stages, yields and known uncertainties
  directly to `CutflowData`. Other cutflow and nested-efficiency helpers remain.
- The optional `analyb.palette` diagnostics and their reference-data tests are
  omitted pending redistribution clearance. The public Machado et al. (2009)
  project page supplies the matrices but no explicit redistribution license:
  https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html
  Sharma's CIEDE2000 page permits personal/research use of its programs with
  citation, but does not establish unrestricted MIT redistribution of the
  copied supplementary data:
  https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/
  Public availability is not treated as license clearance. Existing plot
  colours, Catppuccin values, colormaps and `colors.PALETTES` are unchanged.

Retained numerical implementations are unchanged by this extraction. Tests
use synthetic inputs; passing them is not publication clearance or validation
of any caller's physics model. Private machine paths and analysis-specific
examples have been replaced. The ATLAS label API remains a presentation helper,
not a claim of collaboration approval or permission to publish a result.

See `THIRD_PARTY_NOTICES.md` for the Catppuccin MIT notice and public units
references. Omitted capabilities require a separate provenance and reuse review
before reintroduction; the project license does not resolve their open rights.
