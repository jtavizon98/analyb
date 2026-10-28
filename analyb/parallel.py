"""Parallel batch processing for histogram filling and plotting.

Provides functions to distribute independent fill → plot → save tasks
across a ``ProcessPoolExecutor``, with per-task error isolation, optional
tqdm progress bars, and summary reporting.

Design notes
------------
* Uses ``multiprocessing.get_context("fork")`` so child processes inherit
  the caller's DataFrames and numpy arrays copy-on-write.  Workers access
  these through module-level globals set before the pool is created.
* Matplotlib ``Agg`` backend is configured in each worker to avoid display
  issues on headless remote hosts.
* ROOT file writes happen **serially in the parent** after workers finish
  (``uproot.update()`` is not process-safe).
* LaTeX registration is deferred to the parent for the same reason.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from . import config, io as _io
from .histogram import Hist1D, Hist2D, hist1d, hist2d as _hist2d, sum_histograms as _sum_histograms
from .plot import (
    hist1d_plot as _hist1d_plot,
    hist2d_plot as _hist2d_plot,
    hist1d_comparison as _hist1d_comparison,
    hist1d_with_ratio as _hist1d_with_ratio,
    hist1d_stacked as _hist1d_stacked,
)
from .variables import lookup as _lookup

try:
    from tqdm import tqdm as _tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sensible default worker count (affinity-aware, capped at 4)
# ---------------------------------------------------------------------------

def _get_default_max_workers():
    """Return a sensible default worker count for parallel processing.

    Uses ``os.sched_getaffinity(0)`` on Linux (honours taskset/cgroup
    restrictions), falling back to ``os.cpu_count()``.  The result is
    capped at 4 and at least 1.
    """
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError, OSError):
        n = os.cpu_count() or 1
    return max(1, min(n, 4))


def default_max_workers():
    """Return the current affinity-aware worker limit, capped at four."""
    return _get_default_max_workers()


# ---------------------------------------------------------------------------
# Fork-inherited globals (set by batch functions before pool creation)
# ---------------------------------------------------------------------------

_WORKER_DF = None
_WORKER_WEIGHTS = None
_WORKER_RANGES: dict = {}
_WORKER_INT_VARS: set = set()
_WORKER_BINS: dict[str, Any] = {}
_WORKER_YL: set = set()
_WORKER_SAVE_DIR: str | None = None
_WORKER_SCRIPT_ID: str | None = None
_WORKER_ATLAS: bool = True
_WORKER_ATLAS_LABEL_KWARGS: dict = {}

# Multi-sample compare globals
_WORKER_DFS_MC: dict | None = None
_WORKER_DF_DATA = None
_WORKER_EDGES: dict = {}
_WORKER_STACK_COLORS = None
_WORKER_STACK_DATA_COLOR = "black"

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Outcome of a single parallel task.

    Attributes
    ----------
    name : str
        Variable or task label.
    success : bool
        Whether the task completed without exception.
    error : str or None
        Exception message, if any.
    traceback : str or None
        Full traceback string, if any.
    root_name : str or None
        Name under which to save the histogram in the ROOT file.
    root_data : histogram record or None
        Pickle-friendly `Hist1DRecord`, `Hist2DRecord`, `ProfileRecord`, or
        `CompareRecord` persistence payload.
    latex_args : tuple or None
        ``(script_id, section, filename, caption)`` for LaTeX registration.
    """
    name: str
    success: bool
    error: Optional[str] = None
    traceback: Optional[str] = None
    root_name: Optional[str] = None
    root_data: Optional[Any] = None
    latex_args: Optional[tuple] = None


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------


def _configure_worker() -> None:
    """Configure a worker process for safe parallel matplotlib."""
    os.environ.setdefault("POLARS_MAX_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import matplotlib

    matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Worker functions (must be top-level for pickling)
# ---------------------------------------------------------------------------


def _worker_fill_and_plot_1d(args: tuple) -> TaskResult:
    """Fill 1D histogram, plot it, save figure.  Returns TaskResult."""
    _configure_worker()

    var, style, kw = args

    try:
        import matplotlib.pyplot as plt

        # Fork-inherited globals
        df = _WORKER_DF
        wgt = _WORKER_WEIGHTS
        save_dir = _WORKER_SAVE_DIR
        script_id = _WORKER_SCRIPT_ID
        atlas = _WORKER_ATLAS
        int_vars = _WORKER_INT_VARS
        yl_set = _WORKER_YL
        ranges = _WORKER_RANGES
        bins_dict = _WORKER_BINS

        config.set_script_id(script_id)
        config.set_save_path(save_dir)

        vals = df[var].to_numpy()
        unit = _lookup(var)[1]
        is_int = var in int_vars

        bins_spec = bins_dict.get(var, "scott")
        range_ = ranges.get(var)

        if is_int and range_ is not None:
            vmin = int(np.floor(range_[0]))
            vmax = int(np.ceil(range_[1]))
            edges = np.arange(vmin - 0.5, vmax + 1.5, 1)
            h = hist1d(vals, weights=wgt, bins=edges, unit=unit)
        else:
            h = hist1d(vals, weights=wgt, bins=bins_spec, range=range_, unit=unit)

        plot_kw = dict(kw)
        plot_kw.setdefault("verbose", False)
        if var in yl_set:
            plot_kw.setdefault("ylog", True)

        _hist1d_plot(h, variable=var, style=style, atlas=atlas, save_name=var, **plot_kw)
        plt.close("all")

        root_data = _io.hist1d_record(h)

        return TaskResult(
            name=var,
            success=True,
            root_name=var,
            root_data=root_data,
        )
    except Exception as e:
        return TaskResult(
            name=var,
            success=False,
            error=str(e),
            traceback=traceback.format_exc(),
        )


def _worker_fill_and_plot_2d(args: tuple) -> TaskResult:
    """Fill 2D histogram, plot it, save figure.  Returns TaskResult."""
    _configure_worker()

    var_x, var_y, norm_log = args

    try:
        import matplotlib.pyplot as plt

        df = _WORKER_DF
        wgt = _WORKER_WEIGHTS
        save_dir = _WORKER_SAVE_DIR
        script_id = _WORKER_SCRIPT_ID
        ranges = _WORKER_RANGES

        config.set_script_id(script_id)
        config.set_save_path(save_dir)

        x_vals = df[var_x].to_numpy()
        y_vals = df[var_y].to_numpy()

        rx = ranges.get(var_x, (None, None))
        ry = ranges.get(var_y, (None, None))

        label_x, unit_x = _lookup(var_x)
        label_y, unit_y = _lookup(var_y)

        h2 = _hist2d(
            x_vals, y_vals,
            bins=(50, 50),
            weights=wgt,
            range=(rx, ry),
            x_label=label_x,
            y_label=label_y,
            x_unit=unit_x,
            y_unit=unit_y,
        )

        norm = None
        if norm_log:
            import matplotlib.colors as mcolors
            norm = mcolors.LogNorm()

        save_name = f"2d_{var_x}_{var_y}"
        _hist2d_plot(h2, variable_x=var_x, variable_y=var_y, save_name=save_name, norm=norm,
                     verbose=False)
        plt.close("all")

        root_data = _io.hist2d_record(h2)

        return TaskResult(
            name=f"{var_x}_vs_{var_y}",
            success=True,
            root_name=f"2d_{var_x}_{var_y}",
            root_data=root_data,
        )
    except Exception as e:
        return TaskResult(
            name=f"{var_x}_vs_{var_y}",
            success=False,
            error=str(e),
            traceback=traceback.format_exc(),
        )


def _sanitise_key(s: str) -> str:
    """Sanitise a section name for use as a ROOT key prefix."""
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def _flatten_1d_plan(variables, default_style=None):
    """Return ``(section, variable, style, options)`` in declared order."""
    sections = variables.items() if isinstance(variables, dict) else [(None, variables)]
    flat = []
    for section, entries in sections:
        for entry in entries:
            if isinstance(entry, str):
                var, style, options = entry, default_style, {}
            elif len(entry) == 3:
                var, style, options = entry
            elif len(entry) == 2:
                var, style = entry
                options = {}
            else:
                raise ValueError(f"invalid 1D plot-plan entry: {entry!r}")
            flat.append((section, var, style, options))
    return flat


def _resolve_ylog(flat, ylog):
    if ylog is True:
        return {var for _, var, _, _ in flat}
    return ylog if isinstance(ylog, set) else set()


def _worker_plot_profiles(args: tuple) -> TaskResult:
    """Compute profile and plot it.  Returns TaskResult."""
    _configure_worker()

    section, var_x, var_y = args

    try:
        import matplotlib.pyplot as plt

        from .histogram import profile as _profile
        from .plot import profile_plot as _profile_plot

        df = _WORKER_DF
        save_dir = _WORKER_SAVE_DIR
        script_id = _WORKER_SCRIPT_ID
        ranges = _WORKER_RANGES

        config.set_script_id(script_id)
        config.set_save_path(save_dir)

        x_vals = df[var_x].to_numpy()
        y_vals = df[var_y].to_numpy()

        _, unit_x = _lookup(var_x)
        _, unit_y = _lookup(var_y)

        centers, means, stderr = _profile(
            x_vals, y_vals,
            bins="scott",
            range=ranges.get(var_x),
        )

        core_name = f"prof_{var_y}_vs_{var_x}"
        save_name = f"{_sanitise_key(section)}_{core_name}" if section else core_name
        _profile_plot(
            centers, means, stderr,
            variable_x=var_x,
            variable_y=var_y,
            save_name=save_name,
            verbose=False,
        )
        plt.close("all")

        root_data = _io.ProfileRecord(centers, means, stderr, unit_x, unit_y)

        return TaskResult(
            name=core_name,
            success=True,
            root_name=save_name,
            root_data=root_data,
        )
    except Exception as e:
        return TaskResult(
            name=f"prof_{var_y}_vs_{var_x}",
            success=False,
            error=str(e),
            traceback=traceback.format_exc(),
        )


def _render_comparison(var, mc_hists, data_hist, plot_type, save_prefix, ylog):
    """Render one comparison identically for fill and replot workers."""
    save_name = f"{save_prefix}{var}"
    if plot_type == "overlay":
        _hist1d_comparison(
            mc_hists, variable=var, normalize=True,
            atlas=_WORKER_ATLAS, atlas_label_kwargs=_WORKER_ATLAS_LABEL_KWARGS,
            save_name=save_name, ylog=ylog, verbose=False,
        )
        return
    if plot_type == "ratio":
        if data_hist is None:
            raise ValueError(f"ratio plot for {var} requires data histogram")
        mc_total = _sum_histograms(
            list(mc_hists.values()), label=" + ".join(mc_hists.keys()),
        )
        _hist1d_with_ratio(
            data_hist, mc_total, variable=var, atlas=_WORKER_ATLAS,
            save_name=save_name, ylog=ylog, verbose=False,
        )
        return
    if plot_type == "stacked":
        _hist1d_stacked(
            mc_hists, variable=var, data_hist=data_hist,
            atlas=_WORKER_ATLAS, save_name=save_name, ylog=ylog,
            colors=_WORKER_STACK_COLORS, data_color=_WORKER_STACK_DATA_COLOR,
            verbose=False,
        )
        return
    raise ValueError(f"unknown plot_type {plot_type!r}")


def _worker_compare(args: tuple) -> TaskResult:
    r"""Build per-sample histograms and produce overlay / ratio / stacked plots.

    Fork-inherited globals:
        _WORKER_DFS_MC, _WORKER_DF_DATA, _WORKER_EDGES,
        _WORKER_SAVE_DIR, _WORKER_SCRIPT_ID, _WORKER_ATLAS
    """
    _configure_worker()

    var, plot_type, save_prefix, section, caption_fmt, ylog_flag = args

    try:
        import matplotlib.pyplot as plt

        dfs_mc = _WORKER_DFS_MC
        df_data = _WORKER_DF_DATA
        edges = _WORKER_EDGES
        save_dir = _WORKER_SAVE_DIR
        script_id = _WORKER_SCRIPT_ID

        config.set_script_id(script_id)
        config.set_save_path(save_dir)

        edge_arr = edges.get(var, "scott")

        # --- build per-sample MC histograms ---
        mc_hists = {}
        for name, df in dfs_mc.items():
            if var not in df.columns:
                logger.warning("compare_batch: %s missing column %s, skipping", name, var)
                continue
            vals = df[var].to_numpy()
            wgt = df[config.get_weight_column()].to_numpy()
            unit = _lookup(var)[1]
            mc_hists[name] = hist1d(vals, weights=wgt, bins=edge_arr, unit=unit)

        if not mc_hists:
            raise ValueError(f"no valid MC histograms for {var}")

        # --- build data histogram (if provided) ---
        data_hist = None
        if df_data is not None and var in df_data.columns:
            vals = df_data[var].to_numpy()
            unit = _lookup(var)[1]
            data_hist = hist1d(vals, bins=edge_arr, unit=unit)

        _render_comparison(
            var, mc_hists, data_hist, plot_type, save_prefix, ylog_flag,
        )

        plt.close("all")

        root_data = _io.CompareRecord(
            {name: _io.hist1d_record(hist) for name, hist in mc_hists.items()},
            _io.hist1d_record(data_hist) if data_hist is not None else None,
        )

        return TaskResult(
            name=var,
            success=True,
            root_name=var,
            root_data=root_data,
        )
    except Exception as e:
        return TaskResult(
            name=var,
            success=False,
            error=str(e),
            traceback=traceback.format_exc(),
        )


# ---------------------------------------------------------------------------
# Shared executor loop
# ---------------------------------------------------------------------------


def _run_batch(
    worker_func,
    task_args: list,
    total: int,
    *,
    desc: str,
    max_workers: int | None = None,
    progress: bool = True,
    fail_fast: bool = False,
) -> list[TaskResult]:
    """Submit *task_args* to a process pool, report progress, return results.

    Parameters
    ----------
    worker_func : callable
        Top-level picklable function that takes one ``args`` tuple.
    task_args : list
        List of argument tuples, one per task.
    total : int
        Total number of tasks.
    max_workers : int or None
        Passed to ``ProcessPoolExecutor``.  ``None`` resolves to
        :data:`default_max_workers` (capped at 4, affinity-aware).
    progress : bool
        Show a progress bar (tqdm) or simple counter.
    desc : str
        Human-readable operation shown beside the progress indicator.
    fail_fast : bool
        Cancel remaining futures on first failure.

    Returns
    -------
    list[TaskResult]
    """
    if total == 0:
        return []

    if max_workers is None:
        max_workers = default_max_workers()

    import multiprocessing

    context = multiprocessing.get_context("fork")

    results: list[TaskResult] = []
    failures: list[TaskResult] = []

    # progress bar / counter
    pbar = None
    if progress and HAS_TQDM:
        pbar = _tqdm(total=total, desc=desc, unit="plot", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
    elif progress:
        sys.stdout.write(f"  {desc}: 0/{total}")
        sys.stdout.flush()

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as ex:
        futures = {ex.submit(worker_func, t): i for i, t in enumerate(task_args)}

        for future in as_completed(futures):
            try:
                result = future.result(timeout=300)
            except Exception as exc:
                # The exception happened during result retrieval (unlikely if
                # the worker caught everything, but covers timeouts & crashes).
                idx = futures[future]
                fake_args = task_args[idx]
                name = fake_args[0] if isinstance(fake_args, tuple) else str(idx)
                result = TaskResult(
                    name=name,
                    success=False,
                    error=f"future.result() raised: {exc}",
                    traceback=traceback.format_exc(),
                )

            results.append(result)

            if result.success:
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(result.name, refresh=False)
                elif progress:
                    done = sum(1 for r in results if r.success)
                    sys.stdout.write(f"\r  {desc}: [{done}/{total}] {result.name}")
                    sys.stdout.flush()
            else:
                failures.append(result)
                logger.error("[FAILED] %s: %s\n%s", result.name, result.error, result.traceback)
                sys.stdout.write(f"\n  [{result.name}] FAILED: {result.error}\n")
                sys.stdout.flush()
                if fail_fast:
                    for f in futures:
                        f.cancel()
                    break

    if pbar:
        pbar.close()
    elif progress:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if failures:
        n_ok = total - len(failures)
        sys.stdout.write(
            f"  \u2500\u2500 {total} tasks: {n_ok} succeeded, {len(failures)} failed\n"
        )
        for f in failures:
            sys.stdout.write(f"    FAILED: {f.name}  \u2192 {f.error}\n")
        sys.stdout.flush()

    return results


# ---------------------------------------------------------------------------
# Public batch functions
# ---------------------------------------------------------------------------


def hist1d_batch(
    df,
    variables: Sequence[str] | dict,
    *,
    weights: np.ndarray | None = None,
    styles: str | dict | None = None,
    bins: str | dict = "scott",
    ranges: dict | None = None,
    atlas: bool = True,
    integer_vars: set | None = None,
    ylog: bool | set | None = None,
    max_workers: int | None = None,
    progress: bool = True,
    fail_fast: bool = False,
) -> list[TaskResult]:
    """Fill and plot 1D histograms for multiple variables in parallel.

    Parameters
    ----------
    df : polars.DataFrame
        Data source.
    variables : list[str] or dict
        Flat list of variable names, or a section-structured dict matching
        the ``PLOT_PLAN`` format: ``{section: [(var, style, kw), ...], ...}``.
    weights : numpy.ndarray or None
        Per-event weights column.
    styles : str or dict or None
        Plot style.  Single string for all variables, or dict mapping
        variable name → style string.  Default ``"atlas"``.
    bins : str or dict
        Binning rule.  Single string for all, or per-variable dict.
    ranges : dict or None
        ``{var: (lo, hi)}`` axis ranges.
    atlas : bool
        Add ATLAS label to plots.
    integer_vars : set[str] or None
        Discrete-integer variable names (histogrammed with unit bin width).
    ylog : bool or set[str] or None
        Enable log y-axis.  ``True`` for all, a set of names, or ``None``.
    max_workers : int or None
        Worker count.  ``None`` resolves to :data:`default_max_workers`
        (capped at 4, affinity-aware).
    progress : bool
        Show progress bar / counter.
    fail_fast : bool
        Cancel remaining tasks if one fails.

    Returns
    -------
    list[TaskResult]
    """
    global _WORKER_DF, _WORKER_WEIGHTS, _WORKER_RANGES, _WORKER_INT_VARS
    global _WORKER_BINS, _WORKER_YL, _WORKER_SAVE_DIR, _WORKER_SCRIPT_ID, _WORKER_ATLAS

    save_dir = config.get_save_path()
    script_id = config.get_script_id()

    if weights is None:
        weights = np.ones(df.height, dtype=np.float64)

    flat = _flatten_1d_plan(
        variables, styles if isinstance(styles, str) else None,
    )
    yl_set = _resolve_ylog(flat, ylog)

    # Build bin specs per variable
    bins_dict: dict = {}
    if isinstance(bins, str):
        for _, var, _, _ in flat:
            bins_dict[var] = bins
    else:
        bins_dict = dict(bins)

    # Set globals (inherited by fork)
    _WORKER_DF = df
    _WORKER_WEIGHTS = weights
    _WORKER_RANGES = ranges or {}
    _WORKER_INT_VARS = integer_vars or set()
    _WORKER_BINS = bins_dict
    _WORKER_YL = yl_set
    _WORKER_SAVE_DIR = save_dir
    _WORKER_SCRIPT_ID = script_id
    _WORKER_ATLAS = atlas

    # Build task args (small — only variable name, style, extra kwargs)
    task_args: list[tuple] = []
    latex_entries: list[tuple] = []
    for section, var, style, kw in flat:
        if var not in df.columns:
            logger.warning("hist1d_batch: skipping %s (not in DataFrame)", var)
            continue

        resolved_style = (
            style
            or (styles.get(var) if isinstance(styles, dict) else None)
            or (styles if isinstance(styles, str) else None)
            or "atlas"
        )

        task_args.append((var, resolved_style, kw))

        # Store LaTeX info for parent to register
        if section:
            label, _ = _lookup(var)
            filename = f"{script_id}-{var}.pdf"
            caption = f"{label} distribution in inclusive selection"
            latex_entries.append((script_id, section, filename, caption))

    results = _run_batch(
        _worker_fill_and_plot_1d,
        task_args,
        total=len(task_args),
        desc="Fill 1D plots",
        max_workers=max_workers,
        progress=progress,
        fail_fast=fail_fast,
    )

    # Save histograms to ROOT (serial, in parent)
    _save_results_to_root(results)

    # Register LaTeX captions (serial, in parent)
    for entry in latex_entries:
        from .makelatex import register as _register_latex
        _register_latex(*entry)

    return results


def hist2d_batch(
    df,
    pairs: Sequence[tuple[str, str]] | dict,
    *,
    weights: np.ndarray | None = None,
    ranges: dict | None = None,
    log_norm: bool = True,
    max_workers: int | None = None,
    progress: bool = True,
    fail_fast: bool = False,
) -> list[TaskResult]:
    """Fill and plot 2D histograms for multiple variable pairs in parallel.

    Parameters
    ----------
    df : polars.DataFrame
    pairs : list[tuple] or dict
        ``[(x_name, y_name), ...]`` or section-structured dict.
    weights : numpy.ndarray or None
    ranges : dict or None
        ``{var: (lo, hi)}``.
    log_norm : bool
        Use ``LogNorm`` colour mapping (default ``True``).
    max_workers, progress, fail_fast
        See :func:`hist1d_batch`.
    """
    global _WORKER_DF, _WORKER_WEIGHTS, _WORKER_RANGES
    global _WORKER_SAVE_DIR, _WORKER_SCRIPT_ID

    save_dir = config.get_save_path()
    script_id = config.get_script_id()

    # Normalise pairs
    flat: list[tuple] = []
    if isinstance(pairs, dict):
        for section, pair_list in pairs.items():
            for var_x, var_y in pair_list:
                flat.append((section, var_x, var_y))
    else:
        for var_x, var_y in pairs:
            flat.append((None, var_x, var_y))

    _WORKER_DF = df
    _WORKER_WEIGHTS = weights
    _WORKER_RANGES = ranges or {}
    _WORKER_SAVE_DIR = save_dir
    _WORKER_SCRIPT_ID = script_id

    task_args: list[tuple] = []
    latex_entries: list[tuple] = []
    for section, var_x, var_y in flat:
        task_args.append((var_x, var_y, log_norm))

        if section:
            label_x, _ = _lookup(var_x)
            label_y, _ = _lookup(var_y)
            filename = f"{script_id}-2d_{var_x}_{var_y}.pdf"
            caption = f"{label_x} vs {label_y}, inclusive selection"
            latex_entries.append((script_id, section, filename, caption))

    results = _run_batch(
        _worker_fill_and_plot_2d,
        task_args,
        total=len(task_args),
        desc="Fill 2D plots",
        max_workers=max_workers,
        progress=progress,
        fail_fast=fail_fast,
    )

    _save_results_to_root(results)

    from .makelatex import register as _register_latex
    for entry in latex_entries:
        _register_latex(*entry)

    return results


def profile_batch(
    df,
    pairs: Sequence[tuple[str, str]] | dict,
    *,
    ranges: dict | None = None,
    max_workers: int | None = None,
    progress: bool = True,
    fail_fast: bool = False,
) -> list[TaskResult]:
    """Compute and plot profiles for multiple variable pairs in parallel.

    Parameters
    ----------
    df : polars.DataFrame
    pairs : list[tuple] or dict
        ``[(x_name, y_name), ...]`` or section-structured dict.
    ranges, max_workers, progress, fail_fast
        See :func:`hist1d_batch`.
    """
    global _WORKER_DF, _WORKER_RANGES, _WORKER_SAVE_DIR, _WORKER_SCRIPT_ID

    save_dir = config.get_save_path()
    script_id = config.get_script_id()

    flat: list[tuple] = []
    if isinstance(pairs, dict):
        for section, pair_list in pairs.items():
            for var_x, var_y in pair_list:
                flat.append((section, var_x, var_y))
    else:
        for var_x, var_y in pairs:
            flat.append((None, var_x, var_y))

    _WORKER_DF = df
    _WORKER_RANGES = ranges or {}
    _WORKER_SAVE_DIR = save_dir
    _WORKER_SCRIPT_ID = script_id

    task_args: list[tuple] = []
    latex_entries: list[tuple] = []
    for section, var_x, var_y in flat:
        task_args.append((section, var_x, var_y))

        if section:
            label_x, _ = _lookup(var_x)
            label_y, _ = _lookup(var_y)
            core_name = f"prof_{var_y}_vs_{var_x}"
            save_name = f"{_sanitise_key(section)}_{core_name}"
            filename = f"{script_id}-{save_name}.pdf"
            caption = f"Profile of {label_y} vs {label_x}, inclusive selection"
            latex_entries.append((script_id, section, filename, caption))

    results = _run_batch(
        _worker_plot_profiles,
        task_args,
        total=len(task_args),
        desc="Plot profiles",
        max_workers=max_workers,
        progress=progress,
        fail_fast=fail_fast,
    )

    _save_results_to_root(results)

    from .makelatex import register as _register_latex
    for entry in latex_entries:
        _register_latex(*entry)

    return results


def compare_batch(
    dfs_mc,
    df_data=None,
    variables=None,
    edges=None,
    *,
    plot_type="overlay",
    save_prefix="cmp_",
    section="Comparison",
    caption_fmt="{var} distribution",
    atlas=True,
    atlas_label_kwargs=None,
    ylog=None,
    stack_colors=None,
    stack_data_color="black",
    progress=True,
    fail_fast=False,
    max_workers=None,
) -> list[TaskResult]:
    """Build and plot per-sample histograms in parallel for data/MC comparison.

    Parameters
    ----------
    dfs_mc : dict[str, polars.DataFrame]
        MC sample dataframes, keyed by sample name.
    df_data : polars.DataFrame or None
        Data dataframe (optional — required for ``"ratio"`` and ``"stacked"``).
    variables : list[str]
        Variable names to process.
    edges : dict[str, ndarray]
        Shared bin edges per variable (e.g. from ``concat_edges``).
    plot_type : str
        ``"overlay"`` — unit-area normalized comparison across MC samples.
        ``"ratio"`` — data / (sum of MC) with ratio panel.
        ``"stacked"`` — stacked MC with optional data overlay.
    save_prefix : str
        Prepended to filenames (e.g. ``"data_mc_ratio_"``).
    section : str
        LaTeX section name for registration.
    caption_fmt : str
        LaTeX caption template with ``{var}`` placeholder.
    atlas : bool
        Add an ATLAS label to plots.
    atlas_label_kwargs : dict or None
        Keyword arguments for the label on normalized MC overlays.
    ylog : bool or set[str] or None
        Enable log y-axis.
    stack_colors : mapping or sequence, optional
        Colors forwarded only to ``hist1d_stacked`` for stacked plots.
    stack_data_color : color, optional
        Data-marker color for stacked plots (default ``"black"``).
    progress, fail_fast, max_workers
        See :func:`hist1d_batch`.

    Returns
    -------
    list[TaskResult]
    """
    global _WORKER_DFS_MC, _WORKER_DF_DATA, _WORKER_EDGES
    global _WORKER_SAVE_DIR, _WORKER_SCRIPT_ID, _WORKER_ATLAS
    global _WORKER_ATLAS_LABEL_KWARGS
    global _WORKER_STACK_COLORS, _WORKER_STACK_DATA_COLOR

    save_dir = config.get_save_path()
    script_id = config.get_script_id()

    if not dfs_mc:
        logger.warning("compare_batch: no MC samples provided")
        return []

    if variables is None:
        raise ValueError("compare_batch requires a list of variables")

    # Resolve ylog to a set
    yl_set: set = set()
    if ylog is True:
        yl_set = set(variables)
    elif isinstance(ylog, set):
        yl_set = ylog

    # Set globals (inherited by fork)
    _WORKER_DFS_MC = dfs_mc
    _WORKER_DF_DATA = df_data
    _WORKER_EDGES = edges or {}
    _WORKER_SAVE_DIR = save_dir
    _WORKER_SCRIPT_ID = script_id
    _WORKER_ATLAS = atlas
    _WORKER_ATLAS_LABEL_KWARGS = dict(atlas_label_kwargs or {})
    _WORKER_STACK_COLORS = stack_colors
    _WORKER_STACK_DATA_COLOR = stack_data_color

    # Build task args + LaTeX entries
    task_args: list[tuple] = []
    latex_entries: list[tuple] = []
    for var in variables:
        ylog_flag = var in yl_set
        task_args.append((var, plot_type, save_prefix, section, caption_fmt, ylog_flag))

        label, _ = _lookup(var)
        samples_str = " + ".join(dfs_mc.keys())
        caption = caption_fmt.replace("{samples}", samples_str)
        caption = caption.replace("{var}", label)
        filename = f"{script_id}-{save_prefix}{var}.pdf"
        latex_entries.append((script_id, section, filename, caption))

    results = _run_batch(
        _worker_compare,
        task_args,
        total=len(task_args),
        desc=f"Data/MC {plot_type} plots",
        max_workers=max_workers,
        progress=progress,
        fail_fast=fail_fast,
    )

    _save_results_to_root(results)

    # Register LaTeX captions (serial, in parent)
    from .makelatex import register as _register_latex
    for entry in latex_entries:
        _register_latex(*entry)

    return results


# ---------------------------------------------------------------------------
# Result serialisation helpers
# ---------------------------------------------------------------------------


def _save_results_to_root(results: list[TaskResult]) -> None:
    """Save successful typed histogram records to the script output."""
    for r in results:
        if not r.success or r.root_data is None:
            continue
        _io.save_histogram_record(r.root_name, r.root_data, mode="update")


# ---------------------------------------------------------------------------
# Replot workers — take pre-computed histograms, only regenerate figures
# ---------------------------------------------------------------------------

_WORKER_HISTS_1D: dict | None = None
_WORKER_HISTS_2D: dict | None = None
_WORKER_PROFILES: dict | None = None
_WORKER_COMPARE: dict | None = None


def _worker_replot_1d(args: tuple) -> TaskResult:
    """Replot a 1D histogram from pre-computed data."""
    _configure_worker()

    var, style, kw = args

    try:
        import matplotlib.pyplot as plt

        h = _WORKER_HISTS_1D[var]
        save_dir = _WORKER_SAVE_DIR
        script_id = _WORKER_SCRIPT_ID

        config.set_script_id(script_id)
        config.set_save_path(save_dir)

        plot_kw = dict(kw)
        plot_kw.setdefault("verbose", False)
        if var in _WORKER_YL:
            plot_kw.setdefault("ylog", True)

        _hist1d_plot(h, variable=var, style=style, atlas=_WORKER_ATLAS,
                     save_name=var, **plot_kw)
        plt.close("all")

        return TaskResult(name=var, success=True)
    except Exception as e:
        return TaskResult(name=var, success=False, error=str(e),
                          traceback=traceback.format_exc())


def _worker_replot_2d(args: tuple) -> TaskResult:
    """Replot a 2D histogram from pre-computed data."""
    _configure_worker()

    var_x, var_y, norm_log = args

    try:
        import matplotlib.pyplot as plt

        key = f"2d_{var_x}_{var_y}"
        h2 = _WORKER_HISTS_2D[key]
        save_dir = _WORKER_SAVE_DIR
        script_id = _WORKER_SCRIPT_ID

        config.set_script_id(script_id)
        config.set_save_path(save_dir)

        norm = None
        if norm_log:
            import matplotlib.colors as mcolors
            norm = mcolors.LogNorm()

        save_name = key
        _hist2d_plot(h2, variable_x=var_x, variable_y=var_y,
                     save_name=save_name, norm=norm, verbose=False)
        plt.close("all")

        return TaskResult(name=key, success=True)
    except Exception as e:
        return TaskResult(name=key, success=False, error=str(e),
                          traceback=traceback.format_exc())


def _worker_replot_profile(args: tuple) -> TaskResult:
    """Replot a profile from pre-computed profile data."""
    _configure_worker()

    section, var_x, var_y = args

    try:
        import matplotlib.pyplot as plt

        from .plot import profile_plot as _profile_plot

        core_key = f"prof_{var_y}_vs_{var_x}"
        cache_key = f"{_sanitise_key(section)}_{core_key}" if section else core_key
        centers, means, stderr = _WORKER_PROFILES[cache_key]
        save_dir = _WORKER_SAVE_DIR
        script_id = _WORKER_SCRIPT_ID

        config.set_script_id(script_id)
        config.set_save_path(save_dir)

        _profile_plot(centers, means, stderr,
                      variable_x=var_x, variable_y=var_y,
                      save_name=cache_key, verbose=False)
        plt.close("all")

        return TaskResult(name=cache_key, success=True)
    except Exception as e:
        return TaskResult(name=cache_key, success=False, error=str(e),
                          traceback=traceback.format_exc())


def _worker_replot_compare(args: tuple) -> TaskResult:
    """Replot a comparison from pre-computed compare data."""
    _configure_worker()

    var, plot_type, save_prefix, section, caption_fmt, ylog_flag = args

    try:
        import matplotlib.pyplot as plt

        entry = _WORKER_COMPARE[var]
        mc_hists = entry["mc"]
        data_hist = entry.get("data")
        save_dir = _WORKER_SAVE_DIR
        script_id = _WORKER_SCRIPT_ID
        config.set_script_id(script_id)
        config.set_save_path(save_dir)

        _render_comparison(
            var, mc_hists, data_hist, plot_type, save_prefix, ylog_flag,
        )

        plt.close("all")

        return TaskResult(name=var, success=True)
    except Exception as e:
        return TaskResult(name=var, success=False, error=str(e),
                          traceback=traceback.format_exc())


# ---------------------------------------------------------------------------
# Public replot batch functions
# ---------------------------------------------------------------------------


def replot_1d_batch(
    hists: dict[str, Hist1D],
    variables: Sequence[str] | dict,
    *,
    atlas: bool = True,
    ylog: bool | set | None = None,
    max_workers: int | None = None,
    progress: bool = True,
    fail_fast: bool = False,
) -> list[TaskResult]:
    """Replot 1D histograms from pre-computed data (no data loading).

    Same signature as :func:`hist1d_batch` but takes pre-computed Hist1D
    objects instead of a DataFrame.  LaTeX entries are registered only for
    variables whose histograms are present in *hists*.
    """
    global _WORKER_HISTS_1D, _WORKER_SAVE_DIR, _WORKER_SCRIPT_ID, _WORKER_ATLAS, _WORKER_YL

    save_dir = config.get_save_path()
    script_id = config.get_script_id()

    flat = _flatten_1d_plan(variables)
    yl_set = _resolve_ylog(flat, ylog)

    _WORKER_HISTS_1D = hists
    _WORKER_SAVE_DIR = save_dir
    _WORKER_SCRIPT_ID = script_id
    _WORKER_ATLAS = atlas
    _WORKER_YL = yl_set

    task_args: list[tuple] = []
    latex_entries: list[tuple] = []
    for section, var, style, kw in flat:
        if var not in hists:
            logger.warning("replot_1d_batch: %s not in cache, skipping", var)
            continue
        resolved_style = style or "atlas"
        task_args.append((var, resolved_style, kw))

        # Register LaTeX only for entries present in the catalog
        if section:
            label, _ = _lookup(var)
            filename = f"{script_id}-{var}.pdf"
            caption = f"{label} distribution in inclusive selection"
            latex_entries.append((script_id, section, filename, caption))

    results = _run_batch(
        _worker_replot_1d,
        task_args,
        total=len(task_args),
        desc="Replot 1D histograms",
        max_workers=max_workers,
        progress=progress,
        fail_fast=fail_fast,
    )

    # Register LaTeX captions (serial, in parent)
    from .makelatex import register as _register_latex
    for entry in latex_entries:
        _register_latex(*entry)

    return results


def replot_2d_batch(
    hists: dict[str, Hist2D],
    pairs: Sequence[tuple[str, str]] | dict,
    *,
    log_norm: bool = True,
    max_workers: int | None = None,
    progress: bool = True,
    fail_fast: bool = False,
) -> list[TaskResult]:
    """Replot 2D histograms from pre-computed data.

    LaTeX entries are registered only for pairs whose histograms are present
    in *hists*.
    """
    global _WORKER_HISTS_2D, _WORKER_SAVE_DIR, _WORKER_SCRIPT_ID

    save_dir = config.get_save_path()
    script_id = config.get_script_id()

    flat: list[tuple] = []
    if isinstance(pairs, dict):
        for section, pair_list in pairs.items():
            for var_x, var_y in pair_list:
                flat.append((section, var_x, var_y))
    else:
        for var_x, var_y in pairs:
            flat.append((None, var_x, var_y))

    _WORKER_HISTS_2D = hists
    _WORKER_SAVE_DIR = save_dir
    _WORKER_SCRIPT_ID = script_id

    task_args: list[tuple] = []
    latex_entries: list[tuple] = []
    for section, var_x, var_y in flat:
        key = f"2d_{var_x}_{var_y}"
        if key not in hists:
            logger.warning("replot_2d_batch: %s not in cache, skipping", key)
            continue
        task_args.append((var_x, var_y, log_norm))

        if section:
            label_x, _ = _lookup(var_x)
            label_y, _ = _lookup(var_y)
            filename = f"{script_id}-2d_{var_x}_{var_y}.pdf"
            caption = f"{label_x} vs {label_y}, inclusive selection"
            latex_entries.append((script_id, section, filename, caption))

    results = _run_batch(
        _worker_replot_2d,
        task_args,
        total=len(task_args),
        desc="Replot 2D histograms",
        max_workers=max_workers,
        progress=progress,
        fail_fast=fail_fast,
    )

    from .makelatex import register as _register_latex
    for entry in latex_entries:
        _register_latex(*entry)

    return results


def replot_profile_batch(
    profiles: dict,
    pairs: Sequence[tuple[str, str]] | dict,
    *,
    max_workers: int | None = None,
    progress: bool = True,
    fail_fast: bool = False,
) -> list[TaskResult]:
    """Replot profiles from pre-computed data.

    *profiles* maps ``"prof_{y}_vs_{x}"`` → ``(centers, means, stderr)``.
    LaTeX entries are registered only for pairs whose profiles are present
    in *profiles*.
    """
    global _WORKER_PROFILES, _WORKER_SAVE_DIR, _WORKER_SCRIPT_ID

    save_dir = config.get_save_path()
    script_id = config.get_script_id()

    flat: list[tuple] = []
    if isinstance(pairs, dict):
        for section, pair_list in pairs.items():
            for var_x, var_y in pair_list:
                flat.append((section, var_x, var_y))
    else:
        for var_x, var_y in pairs:
            flat.append((None, var_x, var_y))

    _WORKER_PROFILES = profiles
    _WORKER_SAVE_DIR = save_dir
    _WORKER_SCRIPT_ID = script_id

    task_args: list[tuple] = []
    latex_entries: list[tuple] = []
    for section, var_x, var_y in flat:
        core_key = f"prof_{var_y}_vs_{var_x}"
        cache_key = f"{_sanitise_key(section)}_{core_key}" if section else core_key
        if cache_key not in profiles:
            logger.warning("replot_profile_batch: %s not in cache, skipping", cache_key)
            continue
        task_args.append((section, var_x, var_y))

        if section:
            label_x, _ = _lookup(var_x)
            label_y, _ = _lookup(var_y)
            filename = f"{script_id}-{cache_key}.pdf"
            caption = f"Profile of {label_y} vs {label_x}, inclusive selection"
            latex_entries.append((script_id, section, filename, caption))

    results = _run_batch(
        _worker_replot_profile,
        task_args,
        total=len(task_args),
        desc="Replot profiles",
        max_workers=max_workers,
        progress=progress,
        fail_fast=fail_fast,
    )

    from .makelatex import register as _register_latex
    for entry in latex_entries:
        _register_latex(*entry)

    return results


def replot_compare_batch(
    compare_data: dict,
    variables: list[str],
    *,
    plot_type="overlay",
    save_prefix="cmp_",
    section="Comparison",
    caption_fmt="{var} distribution",
    atlas=True,
    atlas_label_kwargs=None,
    ylog=None,
    stack_colors=None,
    stack_data_color="black",
    progress=True,
    fail_fast=False,
    max_workers=None,
) -> list[TaskResult]:
    """Replot comparison plots from pre-computed histogram data.

    *compare_data* maps ``"{save_prefix}{var}"`` → dict with keys ``"mc"``
    (dict of sample name → Hist1D) and optional ``"data"`` (Hist1D or None).
    LaTeX entries are registered only for variables present in *compare_data*.
    ``stack_colors`` and ``stack_data_color`` are forwarded only when
    ``plot_type="stacked"``. ``atlas_label_kwargs`` customizes normalized
    overlays; ratio labels retain their defaults.
    """
    global _WORKER_COMPARE, _WORKER_SAVE_DIR, _WORKER_SCRIPT_ID, _WORKER_ATLAS
    global _WORKER_ATLAS_LABEL_KWARGS
    global _WORKER_STACK_COLORS, _WORKER_STACK_DATA_COLOR

    save_dir = config.get_save_path()
    script_id = config.get_script_id()

    if not variables:
        return []

    yl_set: set = set()
    if ylog is True:
        yl_set = set(variables)
    elif isinstance(ylog, set):
        yl_set = ylog

    _WORKER_COMPARE = compare_data
    _WORKER_SAVE_DIR = save_dir
    _WORKER_SCRIPT_ID = script_id
    _WORKER_ATLAS = atlas
    _WORKER_ATLAS_LABEL_KWARGS = dict(atlas_label_kwargs or {})
    _WORKER_STACK_COLORS = stack_colors
    _WORKER_STACK_DATA_COLOR = stack_data_color

    task_args: list[tuple] = []
    latex_entries: list[tuple] = []
    for var in variables:
        if var not in compare_data:
            logger.warning("replot_compare_batch: %s not in cache, skipping", var)
            continue
        ylog_flag = var in yl_set
        task_args.append((var, plot_type, save_prefix, section, caption_fmt, ylog_flag))

        label, _ = _lookup(var)
        samples = " + ".join(compare_data[var].get("mc", {}).keys())
        caption = caption_fmt.replace("{samples}", samples)
        caption = caption.replace("{var}", label)
        filename = f"{script_id}-{save_prefix}{var}.pdf"
        latex_entries.append((script_id, section, filename, caption))

    results = _run_batch(
        _worker_replot_compare,
        task_args,
        total=len(task_args),
        desc=f"Replot {section or plot_type} ({plot_type})",
        max_workers=max_workers,
        progress=progress,
        fail_fast=fail_fast,
    )

    # Register LaTeX captions (serial, in parent)
    from .makelatex import register as _register_latex
    for entry in latex_entries:
        _register_latex(*entry)

    return results
