"""Input/output: TTree loading (polars), histogram persistence in ROOT files."""

import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

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


def _write_sidecar_presence(root_file, name, sidecar, present):
    root_file[f"{name}__analyb_{sidecar}_present"] = (
        np.asarray([bool(present)], dtype=np.int8), np.asarray([0.0, 1.0]),
    )


def _sidecar_is_current(root_file, name, sidecar):
    """Read current-record metadata, falling back for pre-metadata files."""
    marker = f"{name}__analyb_{sidecar}_present"
    if marker not in root_file:
        return f"{name}__{sidecar}" in root_file
    values, _ = root_file[marker].to_numpy()
    return bool(values[0])


@dataclass(frozen=True)
class Hist1DRecord:
    counts: object
    edges: object
    errors: object = None
    unit: Optional[str] = None
    overflow: float = 0.0
    underflow: float = 0.0
    is_density: bool = False


@dataclass(frozen=True)
class Hist2DRecord:
    counts: object
    x_edges: object
    y_edges: object
    errors: object = None
    x_unit: Optional[str] = None
    y_unit: Optional[str] = None


@dataclass(frozen=True)
class ProfileRecord:
    centers: object
    means: object
    stderr: object = None
    x_unit: Optional[str] = None
    y_unit: Optional[str] = None


@dataclass(frozen=True)
class CompareRecord:
    mc: dict
    data: Optional[Hist1DRecord] = None


@dataclass(frozen=True)
class RootHistogram:
    """A ROOT TH1/TH2 and its flow-inclusive source arrays."""

    histogram: object
    values_with_flow: object
    variances_with_flow: object
    has_stored_variances: bool
    classname: str
    axis_titles: tuple


def hist1d_record(hist):
    """Return the explicit persistence record for a Hist1D."""
    return Hist1DRecord(
        hist.counts, hist.edges, hist.errors, hist.unit,
        hist.overflow, hist.underflow, hist.is_density,
    )


def hist2d_record(hist):
    """Return the explicit persistence record for a Hist2D."""
    return Hist2DRecord(
        hist.counts, hist.x_edges, hist.y_edges, hist.errors,
        hist.x_unit, hist.y_unit,
    )


def _profile_edges(centers):
    centers = np.asarray(centers, dtype=np.float64)
    if not len(centers):
        raise ValueError("cannot persist a profile without bin centers")
    half_widths = np.diff(centers) / 2 if len(centers) > 1 else np.array([0.5])
    edges = np.empty(len(centers) + 1)
    edges[0] = centers[0] - half_widths[0]
    edges[-1] = centers[-1] + half_widths[-1]
    if len(centers) > 1:
        edges[1:-1] = centers[:-1] + half_widths
    return edges


def write_histogram_record(root_file, name, record):
    """Write one typed histogram record to an open ROOT file."""
    if isinstance(record, Hist1DRecord):
        counts = np.asarray(record.counts, dtype=np.float64)
        edges = np.asarray(record.edges, dtype=np.float64)
        root_file[name] = counts, edges
        if record.errors is not None:
            errors = np.asarray(record.errors, dtype=np.float64)
            root_file[f"{name}__sumw2"] = errors**2, edges
        _write_sidecar_presence(root_file, name, "sumw2", record.errors is not None)
        root_file[f"{name}__flow"] = (
            np.asarray([record.underflow, record.overflow], dtype=np.float64),
            np.asarray([0.0, 1.0, 2.0]),
        )
        _write_density_metadata(root_file, name, record.is_density)
        return

    if isinstance(record, Hist2DRecord):
        counts = np.asarray(record.counts, dtype=np.float64)
        x_edges = np.asarray(record.x_edges, dtype=np.float64)
        y_edges = np.asarray(record.y_edges, dtype=np.float64)
        root_file[name] = counts, x_edges, y_edges
        if record.errors is not None:
            errors = np.asarray(record.errors, dtype=np.float64)
            root_file[f"{name}__sumw2"] = errors**2, x_edges, y_edges
        _write_sidecar_presence(root_file, name, "sumw2", record.errors is not None)
        return

    if isinstance(record, ProfileRecord):
        edges = _profile_edges(record.centers)
        root_file[name] = np.asarray(record.means, dtype=np.float64), edges
        if record.stderr is not None:
            root_file[f"{name}__stderr"] = (
                np.asarray(record.stderr, dtype=np.float64), edges,
            )
        _write_sidecar_presence(root_file, name, "sumw2", False)
        _write_sidecar_presence(root_file, name, "stderr", record.stderr is not None)
        return

    if isinstance(record, CompareRecord):
        for sample, histogram in record.mc.items():
            write_histogram_record(root_file, f"compare/{name}/mc/{sample}", histogram)
        if record.data is not None:
            write_histogram_record(root_file, f"compare/{name}/data", record.data)
        return

    raise TypeError(f"unsupported histogram record: {type(record).__name__}")


def save_histogram_record(name, record, file=None, mode="update"):
    """Persist one typed record using the same writer as histogram caches."""
    _require_uproot()
    if file is None:
        from . import config

        file = config.get_output_root()
    os.makedirs(os.path.dirname(file) or ".", exist_ok=True)
    with _rootsavebook(file, mode) as root_file:
        write_histogram_record(root_file, name, record)


def save_hist_to_root(name, hist, file=None, mode="update"):
    """Save a 1D histogram (Hist1D or (counts, edges) tuple) into a ROOT file."""
    _require_uproot()
    from . import config
    from .histogram import Hist1D

    if file is None:
        file = config.get_output_root()
    os.makedirs(os.path.dirname(file) or ".", exist_ok=True)
    if isinstance(hist, Hist1D):
        record = hist1d_record(hist)
    else:
        counts, edges = hist
        record = Hist1DRecord(counts, edges)
    with _rootsavebook(file, mode) as f:
        write_histogram_record(f, name, record)


def save_hist2d_to_root(name, hist, file=None, mode="update"):
    """Save a 2D histogram (Hist2D or (counts, x_edges, y_edges)) into a ROOT file."""
    _require_uproot()
    from . import config
    from .histogram import Hist2D

    if file is None:
        file = config.get_output_root()
    os.makedirs(os.path.dirname(file) or ".", exist_ok=True)
    if isinstance(hist, Hist2D):
        record = hist2d_record(hist)
    else:
        counts, xe, ye = hist
        record = Hist2DRecord(counts, xe, ye)
    with _rootsavebook(file, mode) as f:
        write_histogram_record(f, name, record)


def load_hist_from_root(file, name):
    """Load a 1D or 2D histogram from a ROOT file into Hist1D / Hist2D."""
    _require_uproot()
    from .histogram import Hist1D, Hist2D

    with uproot.open(file) as f:
        obj = f[name]
        arrays = obj.to_numpy()
        is_density = _read_density_metadata(f, name)
        sumw2 = (
            f[f"{name}__sumw2"].to_numpy()[0]
            if _sidecar_is_current(f, name, "sumw2") else None
        )
        flow = f[f"{name}__flow"].to_numpy()[0] if f"{name}__flow" in f else None
    if len(arrays) == 2:
        counts, edges = arrays
        errors = np.sqrt(np.maximum(sumw2, 0)) if sumw2 is not None else None
        underflow, overflow = flow if flow is not None else (0.0, 0.0)
        return Hist1D(
            counts=counts, edges=edges, errors=errors,
            underflow=underflow, overflow=overflow,
            is_weighted=sumw2 is not None, is_density=is_density,
        )
    counts, x_edges, y_edges = arrays
    if counts.shape != (len(x_edges) - 1, len(y_edges) - 1):
        counts = counts.T
    if sumw2 is not None and sumw2.shape != counts.shape:
        sumw2 = sumw2.T
    errors = np.sqrt(np.maximum(sumw2, 0)) if sumw2 is not None else None
    return Hist2D(
        counts=counts, x_edges=x_edges, y_edges=y_edges, errors=errors,
    )


def load_root_histogram(file, name, *, require_variances=True):
    """Load an external ROOT TH1/TH2 without discarding flow bins or sumw2.

    The returned :class:`RootHistogram` contains an analyb histogram for the
    displayed bins and exact flow-inclusive source arrays.  ``fSumw2`` must be
    present by default so weighted uncertainties cannot silently degrade to
    ``sqrt(counts)``.  Set ``require_variances=False`` only for a known
    unweighted input.
    """
    _require_uproot()
    from .histogram import Hist1D, Hist2D

    with uproot.open(file) as root_file:
        obj = root_file[name]
        classname = obj.classname
        if not (classname.startswith("TH1") or classname.startswith("TH2")):
            raise TypeError(
                f"{name!r} is {classname}, expected a ROOT TH1 or TH2"
            )

        values = np.asarray(obj.values(flow=True), dtype=np.float64)
        stored_sumw2 = np.asarray(obj.member("fSumw2"), dtype=np.float64)
        has_stored_variances = stored_sumw2.size == values.size
        if require_variances and not has_stored_variances:
            raise ValueError(f"{name!r} has no stored variances (fSumw2)")

        variances = obj.variances(flow=True)
        if variances is None:
            raise ValueError(f"{name!r} has no recoverable variances")
        variances = np.asarray(variances, dtype=np.float64)
        if variances.shape != values.shape:
            raise ValueError(
                f"{name!r} variance shape {variances.shape} does not match "
                f"value shape {values.shape}"
            )
        if np.any(~np.isfinite(variances)) or np.any(variances < 0):
            raise ValueError(f"{name!r} has invalid variances")

        axes = tuple(obj.axes)
        edges = tuple(np.asarray(axis.edges(), dtype=np.float64) for axis in axes)
        axis_titles = tuple(str(axis.member("fTitle")) for axis in axes)

    errors = np.sqrt(variances)
    if classname.startswith("TH2"):
        if values.ndim != 2 or len(edges) != 2:
            raise ValueError(f"{name!r} has invalid TH2 dimensionality")
        expected = (len(edges[0]) + 1, len(edges[1]) + 1)
        if values.shape != expected:
            raise ValueError(
                f"{name!r} flow-inclusive shape {values.shape} does not match "
                f"axis shape {expected}"
            )
        histogram = Hist2D(
            counts=values[1:-1, 1:-1],
            x_edges=edges[0],
            y_edges=edges[1],
            errors=errors[1:-1, 1:-1],
            x_label=axis_titles[0] or None,
            y_label=axis_titles[1] or None,
        )
    else:
        if values.ndim != 1 or len(edges) != 1:
            raise ValueError(f"{name!r} has invalid TH1 dimensionality")
        expected = len(edges[0]) + 1
        if values.size != expected:
            raise ValueError(
                f"{name!r} flow-inclusive length {values.size} does not match "
                f"axis length {expected}"
            )
        histogram = Hist1D(
            counts=values[1:-1],
            edges=edges[0],
            errors=errors[1:-1],
            underflow=values[0],
            overflow=values[-1],
            is_weighted=has_stored_variances,
            label=axis_titles[0] or None,
        )

    return RootHistogram(
        histogram=histogram,
        values_with_flow=values,
        variances_with_flow=variances,
        has_stored_variances=has_stored_variances,
        classname=classname,
        axis_titles=axis_titles,
    )


# --------------------------------------------------------------------------
# Histogram cache: save / load all batch results for replotting
# --------------------------------------------------------------------------


def get_cache_path(file=None):
    """Return the cache ROOT file path derived from the output root path."""
    if file is not None:
        return file
    from . import config
    out_root = config.get_output_root()
    return out_root.replace("-output.root", "-cache.root")


def save_histogram_cache(results, file=None) -> str:
    """Save all batch results (TaskResult list) to a ROOT cache file.

    The cache file is the per-script cache ROOT file
    (``{save_path}/{script_id}-cache.root``).  All histogram types (1D, 2D,
    profiles, compare) are stored under the same keys used by the output
    ROOT file.
    """
    _require_uproot()
    cache_path = os.fspath(get_cache_path(file))
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    logger = logging.getLogger(__name__)
    started = time.monotonic()
    # /tmp is host-local on the supported analysis hosts; do not inherit a
    # potentially remote TMPDIR when the destination is an SSHFS/Ceph mount.
    local_fd, local_path = tempfile.mkstemp(prefix="analyb-cache-", suffix=".root", dir="/tmp")
    os.close(local_fd)
    destination_path = None

    try:
        with _rootsavebook(local_path, "recreate") as f:
            for r in results:
                if not r.success or r.root_data is None:
                    continue
                write_histogram_record(f, r.root_name, r.root_data)

        write_seconds = time.monotonic() - started
        destination_fd, destination_path = tempfile.mkstemp(
            prefix=".analyb-cache-", suffix=".root", dir=os.path.dirname(cache_path) or ".",
        )
        os.close(destination_fd)
        copy_started = time.monotonic()
        shutil.copyfile(local_path, destination_path)
        # The staging file is in the destination directory, so replacement is
        # atomic on filesystems that support os.replace (including local POSIX).
        os.replace(destination_path, cache_path)
        destination_path = None
        copy_seconds = time.monotonic() - copy_started
    finally:
        for path in (local_path, destination_path):
            if path is not None:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    logger.info("Saved histogram cache: %s (local write %.2fs, copy %.2fs, total %.2fs)",
                cache_path, write_seconds, copy_seconds, time.monotonic() - started)
    return cache_path


def load_histogram_cache(file=None):
    """Load all histograms from a cache ROOT file.

    Returns a dict with keys:
    - ``"hist1d"``: ``{var_name: Hist1D, ...}``
    - ``"hist2d"``: ``{key: Hist2D, ...}``
    - ``"profiles"``: ``{key: (centers, means, stderr), ...}``
    - ``"compare"``: ``{var: {"mc": {name: Hist1D, ...}, "data": Hist1D or None}, ...}``
    """
    _require_uproot()
    from .histogram import Hist1D, Hist2D

    cache_path = get_cache_path(file)

    result = {
        "hist1d": {},
        "hist2d": {},
        "profiles": {},
        "compare": {},
    }

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    with uproot.open(cache_path) as f:
        for raw_key in f:
            key = raw_key.rsplit(";", 1)[0]
            if key.endswith((
                "__stderr", "__sumw2", "__flow", "__analyb_density",
                "__analyb_sumw2_present", "__analyb_stderr_present",
            )):
                continue
            obj = f[raw_key]
            if not hasattr(obj, "to_numpy"):
                continue
            arrays = obj.to_numpy()

            # Check if this is a compare key
            if key.startswith("compare/"):
                parts = key.split("/")
                if len(parts) >= 3:
                    var = parts[1]
                    if var not in result["compare"]:
                        result["compare"][var] = {"mc": {}, "data": None}
                    if len(arrays) != 2:
                        continue
                    counts, edges = arrays
                    sumw2_key = f"{key}__sumw2"
                    errors = None
                    is_weighted = False
                    if _sidecar_is_current(f, key, "sumw2"):
                        sumw2, _ = f[sumw2_key].to_numpy()
                        errors = np.sqrt(np.maximum(sumw2, 0))
                        is_weighted = True
                    underflow = overflow = 0.0
                    flow_key = f"{key}__flow"
                    if flow_key in f:
                        flow, _ = f[flow_key].to_numpy()
                        underflow, overflow = flow
                    h = Hist1D(
                        counts=counts,
                        edges=edges,
                        errors=errors,
                        underflow=underflow,
                        overflow=overflow,
                        is_weighted=is_weighted,
                        is_density=_read_density_metadata(f, key),
                    )
                    if len(parts) >= 4 and parts[2] == "mc":
                        sample_name = parts[3]
                        result["compare"][var]["mc"][sample_name] = h
                    elif parts[2] == "data":
                        result["compare"][var]["data"] = h
                continue

            # Check if profile key
            if key.startswith("prof_") or "_prof_" in key:
                if len(arrays) != 2:
                    continue
                counts, edges = arrays
                centers = (edges[:-1] + edges[1:]) / 2
                stderr = None
                stderr_key = f"{key}__stderr"
                if _sidecar_is_current(f, key, "stderr"):
                    stderr, _ = f[stderr_key].to_numpy()
                result["profiles"][key] = (centers, counts, stderr)
                continue

            # Check if 2D key
            if key.startswith("2d_"):
                if len(arrays) == 3:
                    counts, x_edges, y_edges = arrays
                    if counts.shape != (len(x_edges) - 1, len(y_edges) - 1):
                        counts = counts.T
                    errors = None
                    sumw2_key = f"{key}__sumw2"
                    if _sidecar_is_current(f, key, "sumw2"):
                        sumw2, _, _ = f[sumw2_key].to_numpy()
                        if sumw2.shape != counts.shape:
                            sumw2 = sumw2.T
                        errors = np.sqrt(np.maximum(sumw2, 0))
                    h2 = Hist2D(
                        counts=counts, x_edges=x_edges, y_edges=y_edges,
                        errors=errors,
                    )
                    result["hist2d"][key] = h2
                continue

            # Default: 1D histogram
            if len(arrays) == 2:
                counts, edges = arrays
                sumw2_key = f"{key}__sumw2"
                errors = None
                is_weighted = False
                if _sidecar_is_current(f, key, "sumw2"):
                    sumw2, _ = f[sumw2_key].to_numpy()
                    errors = np.sqrt(np.maximum(sumw2, 0))
                    is_weighted = True
                underflow = overflow = 0.0
                flow_key = f"{key}__flow"
                if flow_key in f:
                    flow, _ = f[flow_key].to_numpy()
                    underflow, overflow = flow
                result["hist1d"][key] = Hist1D(
                    counts=counts,
                    edges=edges,
                    errors=errors,
                    underflow=underflow,
                    overflow=overflow,
                    is_weighted=is_weighted,
                    is_density=_read_density_metadata(f, key),
                )

    logger = logging.getLogger(__name__)
    logger.info("Loaded histogram cache: %s", cache_path)
    return result


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
