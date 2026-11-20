"""Input/output: TTree loading (polars), histogram persistence in ROOT files."""

import os
import re

try:
    import uproot
except ImportError:
    uproot = None

try:
    import polars as pl
except ImportError:
    pl = None


def _require_uproot():
    if uproot is None:
        raise ImportError("uproot is required for this operation")


def _require_polars():
    if pl is None:
        raise ImportError("polars is required for TTree loading")


def load_tree(
    path,
    tree_path,
    branches=None,
    cuts=None,
    weights_branch=None,
    entry_stop=None,
):
    """Load a TTree from a ROOT file into a polars DataFrame.

    Parameters
    ----------
    path : str
        Path to the .root file.
    tree_path : str
        TTree key, e.g. "events".
    branches : list[str] or None
        Branches to read. None reads all.
    cuts : polars.Expr | list[Expr] | dict[str, Expr] | None
        Filters applied after loading. A dict tracks per-cut names for cutflow.
    weights_branch : str or None
        If given, also loaded (handy to pass as weights to np.histogram).
        Falls back to `config.get_weight_column()` if None.
    entry_stop : int or None
        Limit number of entries read (handy for smoke tests over huge files).
    """
    _require_uproot()
    _require_polars()
    from . import config

    if weights_branch is None:
        weights_branch = config.get_weight_column()
    with uproot.open(path) as f:
        tree = f[tree_path]
        read_keys = list(branches) if branches is not None else None
        if weights_branch is not None and read_keys is not None:
            if weights_branch not in read_keys:
                read_keys.append(weights_branch)
        arrays = tree.arrays(read_keys, library="np", entry_stop=entry_stop)
    df = pl.DataFrame(arrays)
    if cuts is not None:
        if isinstance(cuts, dict):
            for _name, expr in cuts.items():
                df = df.filter(expr)
        elif isinstance(cuts, (list, tuple)):
            for expr in cuts:
                df = df.filter(expr)
        else:
            # Single polars expression.
            df = df.filter(cuts)
    return df


def load_trees(specs, branches=None, cuts=None, weights_branch=None, entry_stop=None):
    """Load multiple TTrees into a dict {label: polars.DataFrame}."""
    out = {}
    for label, (path, tree_path) in specs.items():
        out[label] = load_tree(
            path,
            tree_path,
            branches=branches,
            cuts=cuts,
            weights_branch=weights_branch,
            entry_stop=entry_stop,
        )
    return out


def iterate_tree(
    path,
    tree_path,
    branches=None,
    cuts=None,
    weights_branch=None,
    entry_start=None,
    entry_stop=None,
    step_size="100 MB",
    workers=1,
):
    """Iterate over chunks of a TTree, yielding polars DataFrames.

    Uses ``uproot.iterate`` for memory-efficient chunked reading.
    ``entry_start`` and ``entry_stop`` are passed directly to uproot so that
    I/O is bounded on disk (no large-read-then-slice).  A defensive post-hoc
    slice still guards against any extra entries returned.

    Parameters
    ----------
    path : str
        Path to the .root file.
    tree_path : str
        TTree key, e.g. "events".
    branches : list[str] or None
        Branches to read.  None reads all.
    cuts : polars.Expr | list[Expr] | dict[str, Expr] | None
        Filters applied to each chunk after loading.
    weights_branch : str or None
        Auto-appended to branches if not already present.
        Falls back to ``config.get_weight_column()`` if None.
    entry_start : int or None
        Global entry index to start reading from (0-based, passed to uproot).
    entry_stop : int or None
        Global entry index to stop reading at (exclusive, passed to uproot).
        When ``entry_start`` is None, this is the total raw entries read.
    step_size : str or int
        Chunk size passed to ``uproot.iterate`` (e.g. ``"100 MB"`` or ``50000``).
    workers : int
        Threads used by uproot for basket decompression and interpretation.

    Yields
    ------
    polars.DataFrame
    """
    _require_uproot()
    _require_polars()
    from . import config

    if weights_branch is None:
        weights_branch = config.get_weight_column()

    read_keys = list(branches) if branches is not None else None
    if weights_branch is not None and read_keys is not None:
        if weights_branch not in read_keys:
            read_keys.append(weights_branch)

    # Total raw entries that this call should yield (defensive guard).
    total_limit = None
    if entry_stop is not None:
        total_limit = entry_stop - (entry_start or 0)

    entries_processed = 0

    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        with uproot.open(path) as f:
            tree = f[tree_path]
            chunks = tree.iterate(
                expressions=read_keys,
                step_size=step_size,
                library="np",
                decompression_executor=executor,
                interpretation_executor=executor,
                entry_start=entry_start,
                entry_stop=entry_stop,
            )
            for arrays in chunks:
                if isinstance(arrays, dict):
                    chunk_size = len(next(iter(arrays.values()), ()))
                else:
                    chunk_size = len(arrays)

                # Defensive: guard against any extra entries beyond total_limit.
                if total_limit is not None:
                    remaining = total_limit - entries_processed
                    if remaining <= 0:
                        break
                    if chunk_size > remaining:
                        if isinstance(arrays, dict):
                            arrays = {name: values[:remaining] for name, values in arrays.items()}
                        else:
                            arrays = arrays[:remaining]
                        chunk_size = remaining

                df = pl.DataFrame(arrays)
                if cuts is not None:
                    if isinstance(cuts, dict):
                        for _name, expr in cuts.items():
                            df = df.filter(expr)
                    elif isinstance(cuts, (list, tuple)):
                        for expr in cuts:
                            df = df.filter(expr)
                    else:
                        df = df.filter(cuts)

                entries_processed += chunk_size
                yield df

                if total_limit is not None and entries_processed >= total_limit:
                    break


def save_df_as_tree(df, tree_name, file=None, mode="recreate"):
    """Write a polars/pandas DataFrame back as a TTree.

    By default writes to the per-script output root file (see config.get_output_root()).
    Set mode="update" to add a tree without recreating the file.
    """
    _require_uproot()
    if pl is not None and isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    else:
        pdf = df
    if file is None:
        from . import config

        file = config.get_output_root()
    os.makedirs(os.path.dirname(file) or ".", exist_ok=True)
    with _rootsavebook(file, mode) as f:
        f[tree_name] = pdf


# --------------------------------------------------------------------------
# Histogram round-trip with ROOT
# --------------------------------------------------------------------------

def _rootsavebook(file, mode):
    if mode == "recreate":
        return uproot.recreate(file)
    if mode == "update":
        if not os.path.exists(file):
            return uproot.recreate(file)
        return uproot.update(file)
    raise ValueError(f"mode must be 'update' or 'recreate', got {mode!r}")


def _write_density_metadata(root_file, name, is_density):
    """Persist Hist1D differential-normalization metadata beside its ROOT TH1."""
    root_file[f"{name}__analyb_density"] = (
        np.asarray([bool(is_density)], dtype=np.int8), np.asarray([0.0, 1.0]),
    )


def _read_density_metadata(root_file, name):
    key = f"{name}__analyb_density"
    if key not in root_file:
        return False
    values, _ = root_file[key].to_numpy()
    return bool(values[0])


def _axis(name, edges):
    from uproot.writing.identify import to_TAxis

    # Explicit edges round-trip exactly, uniform or not.
    return to_TAxis(name, "", edges.size - 1, float(edges[0]), float(edges[-1]), fXbins=edges)


def _write_native(root_file, name, hist, has_sumw2):
    """Write a Hist1D/Hist2D as a ROOT TH1D/TH2D with flow bins and fSumw2.

    Without ``has_sumw2`` no fSumw2 is stored, which ROOT and uproot read as
    unweighted (variance = value).
    """
    from uproot.writing.identify import to_TH1x, to_TH2x
    from .histogram import Hist2D

    values = hist.values(flow=True)
    sumw2 = hist.variances(flow=True)
    # load_histogram refuses these, so refuse them where they are written.
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(sumw2))):
        raise ValueError(f"{name!r} has non-finite values or variances (flow bins included)")
    interior = hist.counts
    interior_w2 = hist.errors ** 2 if has_sumw2 else interior
    stored_sumw2 = sumw2.ravel(order="F") if has_sumw2 else np.array([], dtype=np.float64)
    title = name.rsplit("/", 1)[-1]
    if isinstance(hist, Hist2D):
        cx = (hist.x_edges[:-1] + hist.x_edges[1:]) / 2
        cy = (hist.y_edges[:-1] + hist.y_edges[1:]) / 2
        wx, wy = interior.sum(axis=1), interior.sum(axis=0)
        root_file[name] = to_TH2x(
            title, "", values.ravel(order="F"), float(values.sum()),
            float(interior.sum()), float(interior_w2.sum()),
            float(wx @ cx), float(wx @ cx ** 2), float(wy @ cy), float(wy @ cy ** 2),
            float(cx @ interior @ cy), stored_sumw2,
            _axis("xaxis", hist.x_edges), _axis("yaxis", hist.y_edges),
        )
        return
    centers = hist.centers
    root_file[name] = to_TH1x(
        title, "", values, float(values.sum()),
        float(interior.sum()), float(interior_w2.sum()),
        float(interior @ centers), float(interior @ centers ** 2),
        stored_sumw2, _axis("xaxis", hist.edges),
    )


def _save_native(name, hist, has_sumw2, file, mode):
    _require_uproot()
    if file is None:
        from . import config

        file = config.get_output_root()
    os.makedirs(os.path.dirname(file) or ".", exist_ok=True)
    with _rootsavebook(file, mode) as root_file:
        _write_native(root_file, name, hist, has_sumw2)
        if not hasattr(hist, "x_edges"):
            _write_density_metadata(root_file, name, hist.is_density)


def save_hist_to_root(name, hist, file=None, mode="update"):
    """Save a Hist1D, or a ``(counts, edges)`` tuple, as a ROOT TH1D.

    A Hist1D is written with its flow bins and fSumw2, so ``load_histogram``
    reads it back exactly; a tuple is written without fSumw2 (unweighted).
    A density marker key records differential normalization.
    """
    from .histogram import Hist1D

    if isinstance(hist, Hist1D):
        _save_native(name, hist, True, file, mode)
    else:
        counts, edges = hist
        _save_native(name, Hist1D(counts, edges), False, file, mode)


def save_hist2d_to_root(name, hist, file=None, mode="update"):
    """Save a Hist2D, or a ``(counts, x_edges, y_edges)`` tuple, as a ROOT TH2D."""
    from .histogram import Hist2D

    if isinstance(hist, Hist2D):
        _save_native(name, hist, True, file, mode)
    else:
        counts, xe, ye = hist
        _save_native(name, Hist2D(counts, xe, ye), False, file, mode)


def _read_histogram(root_file, name, require_variances):
    """Read one TH1/TH2 from an open file into Hist1D/Hist2D, flows included."""
    from .histogram import Hist1D, Hist2D

    obj = root_file[name]
    classname = obj.classname
    if not (classname.startswith("TH1") or classname.startswith("TH2")):
        raise TypeError(f"{name!r} is {classname}, expected a ROOT TH1 or TH2")
    values = np.asarray(obj.values(flow=True), dtype=np.float64)
    has_stored_variances = np.asarray(obj.member("fSumw2")).size == values.size
    # The pre-flow writer stored every histogram without fSumw2 and kept sumw2
    # and flows in sidecars; reading only the main key would lose both.
    if not has_stored_variances and (
            f"{name}__sumw2" in root_file or f"{name}__flow" in root_file):
        raise ValueError(
            f"{name!r} keeps sumw2 or flows in sidecar keys: written by analyb "
            "before flow-carrying histograms; rewrite the file with the current analyb"
        )
    if require_variances and not has_stored_variances:
        raise ValueError(f"{name!r} has no stored variances (fSumw2)")
    variances = obj.variances(flow=True)
    if variances is None:
        raise ValueError(f"{name!r} has no recoverable variances")
    variances = np.asarray(variances, dtype=np.float64)
    if variances.shape != values.shape:
        raise ValueError(
            f"{name!r} variance shape {variances.shape} does not match value shape {values.shape}"
        )
    if np.any(~np.isfinite(variances)) or np.any(variances < 0):
        raise ValueError(f"{name!r} has invalid variances")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name!r} has non-finite values (flow bins included)")
    axes = tuple(obj.axes)
    edges = tuple(np.asarray(axis.edges(), dtype=np.float64) for axis in axes)
    titles = tuple(str(axis.member("fTitle")) or None for axis in axes)
    expected = tuple(len(axis) + 1 for axis in edges)
    if values.shape != expected:
        raise ValueError(
            f"{name!r} flow-inclusive shape {values.shape} does not match axis shape {expected}"
        )
    if classname.startswith("TH2"):
        return Hist2D.from_flow(values, edges[0], edges[1], variances,
                                x_label=titles[0], y_label=titles[1])
    return Hist1D.from_flow(
        values, edges[0], variances, is_weighted=has_stored_variances, label=titles[0],
        is_density=_read_density_metadata(root_file, name),
    )


def load_histogram(file, name, *, require_variances=True):
    """Load a ROOT TH1/TH2 into a Hist1D/Hist2D with its flow bins and errors.

    The one reader for external files and for files analyb wrote. ``fSumw2``
    must be present by default so weighted uncertainties cannot silently
    degrade to ``sqrt(counts)``; set ``require_variances=False`` only for a
    known unweighted input. Flow bins and their variances are kept:
    ``values(flow=True)`` and ``variances(flow=True)`` return them.
    """
    _require_uproot()
    with uproot.open(file) as root_file:
        return _read_histogram(root_file, name, require_variances)


# --------------------------------------------------------------------------
# Misc helpers
# --------------------------------------------------------------------------


def extract_unit_from_latex_string(input_string):
    match = re.search(r"\\unit\{([^}]*)\}", input_string)
    if match:
        return match.group(1)
    return None


# Module import guard for np usage kept local so users without numpy installed
# still get a clear ImportError on the first histogram operation.
import numpy as np  # noqa: E402
