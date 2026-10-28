"""analyb: a small analysis+plotting library for TTree-based HEP work.

Public API is re-exported here so users can do:

    import analyb as ab
    ab.config.set_script_id("example")
    df = ab.io.load_tree(path, "events", branches=["met", "eventWeight"], entry_stop=10000)
    h = ab.histogram.hist1d(df["met"], weights=df["eventWeight"], bins="scott")
    ab.plot.hist1d_plot(h, variable="met", atlas=True, save_name="met")
"""

from . import style

style.apply()

from . import config, colors, binning, variables, io, histogram, comparison, stats, plot, cutflow, yieldtable, makelatex, units, parallel, provenance, figcheck, printed

try:
    from importlib.metadata import version as _metadata_version
    __version__ = _metadata_version("analyb")
except Exception:
    __version__ = "0.1.0"

# Convenience top-level re-exports (frequently used).
from .config import (
    set_script_id, get_script_id,
    set_save_path, get_save_path,
    set_data_path, get_data_path,
    set_output_root, get_output_root,
    set_weight_column, get_weight_column,
    set_save_png, get_save_png,
    set_save_png_dpi, get_save_png_dpi,
    set_figure_check, get_figure_check,
    set_figure_final_width_cm, get_figure_final_width_cm,
    fig_size_square, fig_size_2d, fig_size_wide,
)
from .io import (
    load_tree, load_trees, save_df_as_tree,
    iterate_tree,
    save_hist_to_root, save_hist2d_to_root, load_hist_from_root,
    extract_unit_from_latex_string,
    save_histogram_cache, load_histogram_cache,
    Hist1DRecord, Hist2DRecord, ProfileRecord, CompareRecord, RootHistogram,
    hist1d_record, hist2d_record,
    load_root_histogram,
)
from .histogram import (
    Hist1D, Hist2D,
    hist1d, hist2d, rebin, normalize, ratio,
    sum_histograms, sum_histograms2d,
    profile, efficiency, rebin_to_edges,
    profile_sufficient_statistics, profile_from_sufficient_statistics,
    scale_to_luminosity, divide_by_bin_width,
)
from .binning import resolve_bins, SCOTT, STURGES, FREEDMAN_DIACONIS
from .comparison import (
    HistogramComparison, compare_histograms, rebin_root_histogram,
    artifact_comparison_1d, artifact_comparison_many_1d, artifact_comparison_flavours_1d, flavour_matrix_legend, artifact_comparison_2d,
    artifact_closure_1d, RatioPresentation, ratio_range_counts, ratio_summary_plot,
    register_comparison_plot,
)
from .variables import register, register_many, lookup, local
from .units import scale_value, format_bin_width
from .plot import (
    hist1d_plot, hist1d_with_ratio, hist1d_stacked,
    hist1d_comparison, hist1d_overlay,
    hist2d_plot, scatter_plot, profile_plot, efficiency_plot,
    systematic_summary_plot,
    plot_data_with_fit, add_atlas_label, reserve_atlas_label_headroom,
    reserve_annotation_headroom, artist_axes_bounds,
    save_figure,
)
from .cutflow import (
    CutflowData, cutflow_chart, cutflow_ylabel,
    cutflow_comparison, native_cutflow_chart, nested_efficiency,
)
from .yieldtable import yield_table, YieldTable, YieldEntry
from .stats import (
    chi_squared, poisson_likelihood, poisson_likelihood_objective,
    clopper_pearson_interval,
    gaussian, gaussian_with_constant_background,
)
from .colors import catppuccin_colors, catppuccin_colorbars, current_colorbar
from .figcheck import check_figure, Finding, FigureCheckError
from .makelatex import write_compact_plotbook
from .style import plotbook_context

__all__ = [
    "__version__",
    "config", "colors", "binning", "variables", "io", "histogram", "stats",
    "plot", "comparison", "cutflow", "yieldtable", "parallel",
    "provenance", "figcheck", "printed",
    "set_script_id", "get_script_id", "set_save_path", "get_save_path",
    "set_data_path", "get_data_path",
    "set_output_root", "get_output_root",
    "set_weight_column", "get_weight_column",
    "set_save_png", "get_save_png",
    "set_save_png_dpi", "get_save_png_dpi",
    "set_figure_check", "get_figure_check",
    "set_figure_final_width_cm", "get_figure_final_width_cm",
    "fig_size_square", "fig_size_2d", "fig_size_wide",
    "load_tree", "load_trees", "save_df_as_tree",
    "iterate_tree",
    "save_hist_to_root", "save_hist2d_to_root", "load_hist_from_root",
    "load_root_histogram", "RootHistogram",
    "extract_unit_from_latex_string",
    "save_histogram_cache", "load_histogram_cache",
    "Hist1DRecord", "Hist2DRecord", "ProfileRecord", "CompareRecord",
    "hist1d_record", "hist2d_record",
    "Hist1D", "Hist2D", "hist1d", "hist2d", "rebin", "normalize", "ratio",
    "sum_histograms", "sum_histograms2d",
    "rebin_to_edges", "profile", "efficiency", "scale_to_luminosity", "divide_by_bin_width",
    "HistogramComparison", "compare_histograms", "rebin_root_histogram",
    "artifact_comparison_1d", "artifact_comparison_many_1d", "artifact_comparison_flavours_1d", "flavour_matrix_legend", "artifact_comparison_2d", "register_comparison_plot",
    "artifact_closure_1d", "RatioPresentation", "ratio_range_counts", "ratio_summary_plot",
    "write_compact_plotbook", "plotbook_context",
    "profile_sufficient_statistics", "profile_from_sufficient_statistics", "resolve_bins", "SCOTT", "STURGES", "FREEDMAN_DIACONIS",
    "register", "register_many", "lookup", "local",
    "hist1d_plot", "hist1d_with_ratio", "hist1d_stacked",
    "hist1d_comparison", "hist1d_overlay", "hist2d_plot", "scatter_plot",
    "profile_plot", "efficiency_plot", "systematic_summary_plot",
    "plot_data_with_fit", "add_atlas_label", "reserve_atlas_label_headroom",
    "reserve_annotation_headroom", "artist_axes_bounds",
    "save_figure", "check_figure", "Finding", "FigureCheckError",
    "CutflowData", "cutflow_chart", "cutflow_ylabel",
    "cutflow_comparison", "native_cutflow_chart", "nested_efficiency",
    "yield_table", "YieldTable", "YieldEntry",
    "chi_squared", "poisson_likelihood", "poisson_likelihood_objective",
    "clopper_pearson_interval",
    "gaussian", "gaussian_with_constant_background",
    "catppuccin_colors", "catppuccin_colorbars", "current_colorbar",
    "units", "makelatex", "style",
    "scale_value", "format_bin_width",
]
