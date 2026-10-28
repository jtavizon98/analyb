"""Test bounded-memory primitives: iterate_tree, 2D sum, chunked profile, workers."""

import os
import tempfile

import numpy as np
import pytest

import analyb as ab
import analyb.histogram as _hist
import analyb.io as _io
import analyb.parallel as _par


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_root_file(n_entries=500):
    """Create a temporary ROOT file with test data. Returns (path, arrays_dict)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, n_entries)
    w = rng.uniform(0.5, 1.5, n_entries)
    y = 2 * x + rng.normal(0, 0.5, n_entries)

    uproot = pytest.importorskip("uproot")
    tmp = tempfile.NamedTemporaryFile(suffix=".root", delete=False)
    tmp.close()
    with uproot.recreate(tmp.name) as f:
        f["tree"] = {"x": x, "w": w, "y": y}
    return tmp.name, {"x": x, "w": w, "y": y}


# ---------------------------------------------------------------------------
# iterate_tree
# ---------------------------------------------------------------------------

def test_iterate_tree_basic():
    """iterate_tree yields all entries across chunks."""
    pytest.importorskip("uproot")
    pytest.importorskip("polars")
    path, ref = _make_tiny_root_file(500)
    try:
        chunks = list(ab.iterate_tree(
            path, "tree",
            branches=["x", "y"],
            weights_branch="w",
            step_size=50,
        ))
        total = sum(c.height for c in chunks)
        assert total == 500
        for c in chunks:
            assert "w" in c.columns
            assert "x" in c.columns
            assert "y" in c.columns
        x_cat = np.concatenate([c["x"].to_numpy() for c in chunks])
        assert np.allclose(x_cat, ref["x"])
    finally:
        os.unlink(path)


def test_iterate_tree_no_weight_auto():
    """iterate_tree does NOT auto-append a weight when none is configured."""
    pytest.importorskip("uproot")
    pytest.importorskip("polars")
    path, ref = _make_tiny_root_file(200)
    try:
        chunks = list(ab.iterate_tree(
            path, "tree",
            branches=["x", "y"],
            step_size=50,
        ))
        # weight column should not appear
        for c in chunks:
            assert "x" in c.columns
            assert "y" in c.columns
            assert "w" not in c.columns
    finally:
        os.unlink(path)


def test_iterate_tree_entry_stop():
    """iterate_tree honours entry_stop (global exclusive index) across chunk boundaries."""
    pytest.importorskip("uproot")
    pytest.importorskip("polars")
    path, ref = _make_tiny_root_file(500)
    try:
        chunks = list(ab.iterate_tree(
            path, "tree",
            branches=["x", "y"],
            entry_stop=175,
            step_size=50,
        ))
        total = sum(c.height for c in chunks)
        assert total == 175
    finally:
        os.unlink(path)


def test_iterate_tree_entry_start():
    """iterate_tree honours entry_start (count via defensive guard, RNTuple path)."""
    pytest.importorskip("uproot")
    pytest.importorskip("polars")
    path, ref = _make_tiny_root_file(500)
    try:
        chunks = list(ab.iterate_tree(
            path, "tree",
            branches=["x", "y"],
            entry_start=50,
            entry_stop=200,
            step_size=50,
        ))
        total = sum(c.height for c in chunks)
        # RNTuples may ignore entry_start in uproot.iterate, but the
        # defensive guard still limits total entries to 150.
        assert total == 150
    finally:
        os.unlink(path)


def test_iterate_tree_entry_start_on_ttree(tmp_path):
    """entry_start works on real TTrees (not just RNTuples)."""
    uproot = pytest.importorskip("uproot")
    path = tmp_path / "tree.root"
    with uproot.recreate(path) as root:
        root.mktree("NOSYS/tree", {"x": "float64", "w": "float64"})
        root["NOSYS/tree"].extend({
            "x": np.arange(50, dtype=float),
            "w": np.ones(50),
        })

    # Read entries [10, 35) — should get 25 entries (indices 10..34).
    chunks = list(ab.iterate_tree(
        path,
        "NOSYS/tree",
        branches=["x"],
        weights_branch="w",
        entry_start=10,
        entry_stop=35,
        step_size=12,
    ))
    total = sum(chunk.height for chunk in chunks)
    assert total == 25
    x_cat = np.concatenate([chunk["x"] for chunk in chunks])
    assert np.array_equal(x_cat, np.arange(10, 35, dtype=float))


def test_iterate_tree_with_cuts():
    """iterate_tree applies cuts per chunk."""
    pytest.importorskip("uproot")
    pytest.importorskip("polars")
    import polars as pl

    path, ref = _make_tiny_root_file(500)
    try:
        chunks = list(ab.iterate_tree(
            path, "tree",
            branches=["x", "y"],
            cuts=pl.col("x") > 1.0,
            step_size=50,
        ))
        total = sum(c.height for c in chunks)
        # Only entries with x > 1 should pass
        assert total < 500
        for c in chunks:
            assert (c["x"] > 1.0).all()
    finally:
        os.unlink(path)


def test_iterate_tree_entry_stop_exact():
    """entry_stop at chunk boundary yields exactly that many entries starting from entry 0."""
    pytest.importorskip("uproot")
    pytest.importorskip("polars")
    path, ref = _make_tiny_root_file(200)
    try:
        chunks = list(ab.iterate_tree(
            path, "tree",
            branches=["x", "y"],
            entry_stop=100,
            step_size=50,
        ))
        total = sum(c.height for c in chunks)
        assert total == 100
    finally:
        os.unlink(path)


def test_iterate_tree_entry_stop_on_ttree(tmp_path):
    """Real TTrees return branch dictionaries rather than RNTuple records."""
    uproot = pytest.importorskip("uproot")
    path = tmp_path / "tree.root"
    with uproot.recreate(path) as root:
        root.mktree("NOSYS/tree", {"x": "float64", "w": "float64"})
        root["NOSYS/tree"].extend({
            "x": np.arange(20, dtype=float),
            "w": np.ones(20),
        })

    chunks = list(ab.iterate_tree(
        path,
        "NOSYS/tree",
        branches=["x"],
        weights_branch="w",
        entry_stop=7,
        step_size=5,
    ))

    assert sum(chunk.height for chunk in chunks) == 7
    assert np.allclose(np.concatenate([chunk["x"] for chunk in chunks]), np.arange(7))


# ---------------------------------------------------------------------------
# sum_histograms2d
# ---------------------------------------------------------------------------

def test_sum_histograms2d():
    """sum_histograms2d merges two Hist2D objects with matching edges."""
    rng = np.random.default_rng(42)
    x_edges = np.linspace(0, 10, 11)
    y_edges = np.linspace(0, 10, 11)

    h1 = ab.hist2d(
        rng.uniform(0, 10, 500), rng.uniform(0, 10, 500),
        bins=(x_edges, y_edges),
    )
    h2 = ab.hist2d(
        rng.uniform(0, 10, 800), rng.uniform(0, 10, 800),
        bins=(x_edges, y_edges),
    )
    h_sum = ab.sum_histograms2d([h1, h2])

    assert np.allclose(h_sum.counts, h1.counts + h2.counts)
    assert np.allclose(h_sum.errors, np.sqrt(h1.errors ** 2 + h2.errors ** 2))
    assert np.allclose(h_sum.x_edges, x_edges)
    assert np.allclose(h_sum.y_edges, y_edges)


def test_sum_histograms2d_labels_preserved():
    """sum_histograms2d preserves labels from first histogram."""
    h1 = ab.hist2d([0, 1], [0, 1], bins=([0, 1, 2], [0, 1, 2]),
                     x_label="X", y_label="Y", x_unit=r"\GeV", y_unit=r"\rad")
    h2 = ab.hist2d([0, 1], [0, 1], bins=([0, 1, 2], [0, 1, 2]))
    h_sum = ab.sum_histograms2d([h1, h2])
    assert h_sum.x_label == "X"
    assert h_sum.y_label == "Y"
    assert h_sum.x_unit == r"\GeV"
    assert h_sum.y_unit == r"\rad"


def test_sum_histograms2d_explicit_labels():
    """sum_histograms2d accepts explicit label/unit overrides."""
    h1 = ab.hist2d([0, 1], [0, 1], bins=([0, 1, 2], [0, 1, 2]),
                     x_label="old_x")
    h2 = ab.hist2d([0, 1], [0, 1], bins=([0, 1, 2], [0, 1, 2]))
    h_sum = ab.sum_histograms2d([h1, h2], x_label="new_x", y_label="new_y")
    assert h_sum.x_label == "new_x"
    assert h_sum.y_label == "new_y"


def test_sum_histograms2d_mismatch_raises():
    """sum_histograms2d raises on mismatched edges."""
    h1 = ab.hist2d([0, 1], [0, 1], bins=([0, 1, 2], [0, 1, 2]))
    h2 = ab.hist2d([0, 1], [0, 1], bins=([0, 1, 3], [0, 1, 2]))
    with pytest.raises(ValueError, match="identical x_edges"):
        ab.sum_histograms2d([h1, h2])

    h3 = ab.hist2d([0, 1], [0, 1], bins=([0, 1, 2], [0, 1, 3]))
    with pytest.raises(ValueError, match="identical y_edges"):
        ab.sum_histograms2d([h1, h3])


def test_sum_histograms2d_empty_raises():
    """sum_histograms2d raises on empty list."""
    with pytest.raises(ValueError, match="need at least one"):
        ab.sum_histograms2d([])


# ---------------------------------------------------------------------------
# Weighted 2D chunked merge
# ---------------------------------------------------------------------------

def test_weighted_2d_chunk_merge():
    """Chunk → merge via sum_histograms2d matches one-shot weighted 2D."""
    rng = np.random.default_rng(42)
    n = 600
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    w = rng.uniform(0.5, 1.5, n)

    x_edges = np.linspace(0, 10, 11)
    y_edges = np.linspace(0, 10, 11)

    # One-shot
    h_one = ab.hist2d(x, y, bins=(x_edges, y_edges), weights=w)

    # Chunked
    chunks = []
    for i in range(0, n, 150):
        sl = slice(i, i + 150)
        hc = ab.hist2d(x[sl], y[sl], bins=(x_edges, y_edges), weights=w[sl])
        chunks.append(hc)

    h_merged = ab.sum_histograms2d(chunks)
    assert np.allclose(h_merged.counts, h_one.counts)
    assert np.allclose(h_merged.errors, h_one.errors)


# ---------------------------------------------------------------------------
# profile_sufficient_statistics / profile_from_sufficient_statistics
# ---------------------------------------------------------------------------

def test_profile_sufficient_statistics_roundtrip_unweighted():
    """Two-step SS roundtrip is internally consistent (unweighted)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    y = 2 * x + rng.normal(0, 0.5, 1000)

    # One-step: compute SS and then from SS
    n, sw, sw2, swy, swy2, edges = ab.profile_sufficient_statistics(
        x, y, bins=20, range=(-5, 5),
    )
    cen2, mu2, err2 = ab.profile_from_sufficient_statistics(
        n, sw, sw2, swy, swy2, edges,
    )

    # Compare against weighted profile (same internal logic as SS)
    cen1, mu1, err1 = ab.profile(x, y, bins=20, range=(-5, 5),
                                  weights=np.ones(len(x)))

    assert np.allclose(cen1, cen2)
    assert np.allclose(mu1, mu2, equal_nan=True)
    assert np.allclose(err1, err2, equal_nan=True)


def test_profile_sufficient_statistics_roundtrip_weighted():
    """Roundtrip via sufficient stats matches direct profile (weighted)."""
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 0.5, n)
    w = rng.uniform(0.5, 1.5, n)

    cen1, mu1, err1 = ab.profile(x, y, bins=20, range=(-5, 5), weights=w)

    npts, sw, sw2, swy, swy2, edges = ab.profile_sufficient_statistics(
        x, y, bins=20, range=(-5, 5), weights=w,
    )
    cen2, mu2, err2 = ab.profile_from_sufficient_statistics(
        npts, sw, sw2, swy, swy2, edges,
    )

    assert np.allclose(cen1, cen2)
    assert np.allclose(mu1, mu2, equal_nan=True)
    assert np.allclose(err1, err2, equal_nan=True)


def test_weighted_profile_chunk_merge():
    """Chunked sufficient-statistics merge matches one-shot weighted profile."""
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 0.5, n)
    w = rng.uniform(0.5, 1.5, n)

    edges = np.linspace(-5, 5, 21)

    # One-shot
    cen1, mu1, err1 = ab.profile(x, y, bins=edges, weights=w)

    # Chunked accumulate
    n_acc = np.zeros(20)
    sw_acc = np.zeros(20)
    sw2_acc = np.zeros(20)
    swy_acc = np.zeros(20)
    swy2_acc = np.zeros(20)

    for i in range(0, n, 250):
        sl = slice(i, i + 250)
        npts, sw, sw2, swy, swy2, e = ab.profile_sufficient_statistics(
            x[sl], y[sl], bins=edges, weights=w[sl],
        )
        n_acc += npts
        sw_acc += sw
        sw2_acc += sw2
        swy_acc += swy
        swy2_acc += swy2

    cen2, mu2, err2 = ab.profile_from_sufficient_statistics(
        n_acc, sw_acc, sw2_acc, swy_acc, swy2_acc, edges,
    )

    assert np.allclose(cen1, cen2)
    assert np.allclose(mu1, mu2, equal_nan=True, atol=1e-14)
    assert np.allclose(err1, err2, equal_nan=True, atol=1e-14)


def test_unweighted_profile_chunk_merge():
    """Chunked sufficient-statistics merge is self-consistent (unweighted)."""
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 0.5, n)

    edges = np.linspace(-5, 5, 21)

    # One-shot via SS (same internal logic as weighted profile)
    n_full, sw_full, sw2_full, swy_full, swy2_full, edges_full = \
        ab.profile_sufficient_statistics(x, y, bins=edges)
    cen1, mu1, err1 = ab.profile_from_sufficient_statistics(
        n_full, sw_full, sw2_full, swy_full, swy2_full, edges_full,
    )

    # Chunked accumulate
    n_acc = np.zeros(20)
    sw_acc = np.zeros(20)
    sw2_acc = np.zeros(20)
    swy_acc = np.zeros(20)
    swy2_acc = np.zeros(20)

    for i in range(0, n, 250):
        sl = slice(i, i + 250)
        npts, sw, sw2, swy, swy2, e = ab.profile_sufficient_statistics(
            x[sl], y[sl], bins=edges,
        )
        n_acc += npts
        sw_acc += sw
        sw2_acc += sw2
        swy_acc += swy
        swy2_acc += swy2

    cen2, mu2, err2 = ab.profile_from_sufficient_statistics(
        n_acc, sw_acc, sw2_acc, swy_acc, swy2_acc, edges, weighted=False,
    )

    cen1, mu1, err1 = ab.profile(x, y, bins=edges)

    assert np.allclose(cen1, cen2)
    assert np.allclose(mu1, mu2, equal_nan=True, atol=1e-14)
    assert np.allclose(err1, err2, equal_nan=True, atol=1e-14)


def test_unweighted_profile_last_edge_is_inclusive():
    """Match scipy: an unweighted profile includes the final right edge."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.999, 5.0, 6.0])
    y = np.ones_like(x)
    edges = np.array([0.0, 2.0, 5.0])

    scipy_profile = ab.profile(x, y, bins=edges)
    stats = ab.profile_sufficient_statistics(x, y, bins=edges)
    streamed_profile = ab.profile_from_sufficient_statistics(
        *stats[:-1], stats[-1], weighted=False,
    )

    assert np.allclose(streamed_profile[1], scipy_profile[1], equal_nan=True)
    assert np.allclose(streamed_profile[2], scipy_profile[2], equal_nan=True)


# ---------------------------------------------------------------------------
# Chunked weighted 1D merge
# ---------------------------------------------------------------------------

def test_weighted_1d_chunk_merge():
    """Chunk → sum_histograms matches one-shot weighted 1D."""
    rng = np.random.default_rng(42)
    n = 600
    x = rng.normal(0, 1, n)
    w = rng.uniform(0.5, 1.5, n)
    edges = np.linspace(-5, 5, 21)

    h_one = ab.hist1d(x, bins=edges, weights=w)

    chunks = []
    for i in range(0, n, 150):
        sl = slice(i, i + 150)
        hc = ab.hist1d(x[sl], bins=edges, weights=w[sl])
        chunks.append(hc)

    h_merged = ab.sum_histograms(chunks)
    assert np.allclose(h_merged.counts, h_one.counts)
    assert np.allclose(h_merged.errors, h_one.errors)


# ---------------------------------------------------------------------------
# default_max_workers
# ---------------------------------------------------------------------------

def test_default_max_workers_bounds():
    """default_max_workers is int between 1 and 4 (inclusive)."""
    workers = _par.default_max_workers()
    assert isinstance(workers, int)
    assert 1 <= workers <= 4


def test_get_default_max_workers_sane():
    """_get_default_max_workers returns a capped, positive value."""
    w = _par._get_default_max_workers()
    assert isinstance(w, int)
    assert 1 <= w <= 4


def test_default_max_workers_resolved_in_run_batch():
    """When max_workers=None, _run_batch resolves to default_max_workers."""
    assert _par.default_max_workers() == _par._get_default_max_workers()


def test_all_batch_callers_supply_meaningful_progress_descriptions(monkeypatch):
    """Every public batch entry point labels its tqdm/fallback counter."""
    pl = pytest.importorskip("polars")
    descriptions = []
    monkeypatch.setattr(_par, "_run_batch", lambda *args, **kwargs: descriptions.append(kwargs["desc"]) or [])
    df = pl.DataFrame({"x": [1.0], "y": [2.0], "eventWeight": [1.0]})
    hist = ab.Hist1D(np.array([1.0]), np.array([0.0, 1.0]), errors=np.array([1.0]))

    _par.hist1d_batch(df, [])
    _par.hist2d_batch(df, [])
    _par.profile_batch(df, [])
    _par.compare_batch({"mc": df}, variables=[])
    _par.replot_1d_batch({}, [])
    _par.replot_2d_batch({}, [])
    _par.replot_profile_batch({}, [])
    _par.replot_compare_batch({"x": {"mc": {"mc": hist}}}, ["x"], plot_type="stacked", section="Stacked")

    assert descriptions == [
        "Fill 1D plots", "Fill 2D plots", "Plot profiles", "Data/MC overlay plots",
        "Replot 1D histograms", "Replot 2D histograms", "Replot profiles",
        "Replot Stacked (stacked)",
    ]


def test_sectioned_profile_cache_roundtrip(tmp_path):
    key = "Profiles__ttbar_prof_mtlep_vs_met"
    centers = np.array([0.5, 1.5])
    means = np.array([2.0, 3.0])
    stderr = np.array([0.2, 0.3])
    result = _par.TaskResult(
        name=key,
        success=True,
        root_name=key,
        root_data=_io.ProfileRecord(centers, means, stderr),
    )
    cache_path = tmp_path / "cache.root"

    ab.save_histogram_cache([result], file=cache_path)
    loaded = ab.load_histogram_cache(file=cache_path)["profiles"][key]

    assert np.allclose(loaded[0], centers)
    assert np.allclose(loaded[1], means)
    assert np.allclose(loaded[2], stderr)


def test_histogram_cache_preserves_density_metadata(tmp_path):
    result = _par.TaskResult(
        name="density", success=True, root_name="density",
        root_data=_io.Hist1DRecord(
            np.array([2.0, 3.0]), np.array([0.0, 1.0, 3.0]),
            np.array([1.0, 1.5]), r"\GeV", 4.0, 5.0, True,
        ),
    )
    cache_path = tmp_path / "cache.root"

    ab.save_histogram_cache([result], file=cache_path)
    loaded = ab.load_histogram_cache(file=cache_path)["hist1d"]["density"]

    assert loaded.is_density
    assert loaded.bin_width is None
    np.testing.assert_allclose(loaded.counts, [2.0, 3.0])


def test_root_histogram_roundtrip_preserves_density_metadata(tmp_path):
    density = ab.divide_by_bin_width(ab.Hist1D(
        [2.0, 6.0], [0.0, 1.0, 3.0], errors=[1.0, 2.0], unit=r"\GeV",
        underflow=-2.5, overflow=4.5, is_weighted=True,
    ))
    path = tmp_path / "hist.root"

    ab.save_hist_to_root("density", density, file=path, mode="recreate")
    loaded = ab.load_hist_from_root(path, "density")

    assert loaded.is_density
    assert loaded.unit is None  # ROOT TH1 persistence has no unit metadata.
    np.testing.assert_allclose(loaded.counts, density.counts)
    np.testing.assert_allclose(loaded.edges, density.edges)
    np.testing.assert_allclose(loaded.errors, density.errors)
    assert loaded.underflow == density.underflow
    assert loaded.overflow == density.overflow


def test_root_hist2d_roundtrip_preserves_weighted_errors(tmp_path):
    path = tmp_path / "hist2d.root"
    hist = ab.Hist2D(
        [[3.0, 4.0]], [0.0, 1.0], [0.0, 1.0, 2.0],
        errors=[[1.5, 2.5]],
    )

    ab.save_hist2d_to_root("weighted", hist, file=path, mode="recreate")
    loaded = ab.load_hist_from_root(path, "weighted")

    assert np.array_equal(loaded.counts, hist.counts)
    assert np.array_equal(loaded.errors, hist.errors)


def test_root_update_does_not_recreate_an_invalid_existing_file(tmp_path):
    path = tmp_path / "invalid.root"
    original = b"not a ROOT file"
    path.write_bytes(original)

    with pytest.raises((OSError, ValueError)):
        ab.save_hist_to_root(
            "hist", ab.Hist1D([1.0], [0.0, 1.0]), file=path, mode="update",
        )

    assert path.read_bytes() == original


def test_root_update_ignores_stale_optional_sidecars_across_cycles(tmp_path):
    path = tmp_path / "updated.root"
    weighted = ab.Hist1D(
        [4.0, 9.0], [0.0, 1.0, 2.0], errors=[3.0, 5.0], is_weighted=True,
    )
    ab.save_hist_to_root("hist", weighted, file=path, mode="recreate")
    ab.save_hist_to_root("hist", weighted, file=path, mode="update")
    ab.save_hist_to_root(
        "hist", (np.array([16.0]), np.array([0.0, 2.0])),
        file=path, mode="update",
    )

    loaded = ab.load_hist_from_root(path, "hist")
    assert not loaded.is_weighted
    np.testing.assert_allclose(loaded.counts, [16.0])
    np.testing.assert_allclose(loaded.errors, [4.0])


def test_profile_update_clears_stale_sumw2_presence_marker(tmp_path):
    uproot = pytest.importorskip("uproot")
    path = tmp_path / "profile-update.root"
    weighted = _io.Hist1DRecord(
        np.array([4.0, 9.0]), np.array([0.0, 1.0, 2.0]),
        errors=np.array([3.0, 5.0]),
    )
    profile = _io.ProfileRecord(
        np.array([0.5, 1.5]), np.array([2.0, 3.0]), np.array([0.2, 0.3]),
    )

    _io.save_histogram_record("shared", weighted, file=path, mode="recreate")
    _io.save_histogram_record("shared", profile, file=path, mode="update")

    with uproot.open(path) as root_file:
        assert not _io._sidecar_is_current(root_file, "shared", "sumw2")
        assert _io._sidecar_is_current(root_file, "shared", "stderr")


def test_histogram_cache_rejects_untyped_records_atomically(tmp_path):
    destination = tmp_path / "cache.root"
    result = _par.TaskResult(
        name="legacy", success=True, root_name="legacy",
        root_data=(np.array([1.0]), np.array([0.0, 1.0])),
    )

    with pytest.raises(TypeError, match="unsupported histogram record"):
        ab.save_histogram_cache([result], file=destination)

    assert not destination.exists()


def test_output_writer_rejects_untyped_records(tmp_path):
    ab.set_script_id("invalid-record")
    ab.set_data_path(str(tmp_path))
    result = _par.TaskResult(
        name="legacy", success=True, root_name="legacy",
        root_data=(np.array([1.0]), np.array([0.0, 1.0])),
    )

    with pytest.raises(TypeError, match="unsupported histogram record"):
        _par._save_results_to_root([result])


def test_compare_cache_ignores_root_directories_and_cycles(tmp_path):
    hist_data = _io.Hist1DRecord(
        np.array([3.0]),
        np.array([0.0, 1.0]),
        np.array([2.0]),
        None,
        5.0,
        4.0,
    )
    result = _par.TaskResult(
        name="met",
        success=True,
        root_name="met",
        root_data=_io.CompareRecord({"ttbar": hist_data}, hist_data),
    )
    cache_path = tmp_path / "cache.root"

    ab.save_histogram_cache([result], file=cache_path)
    loaded = ab.load_histogram_cache(file=cache_path)["compare"]["met"]

    assert set(loaded["mc"]) == {"ttbar"}
    assert np.allclose(loaded["mc"]["ttbar"].counts, [3.0])
    assert np.allclose(loaded["mc"]["ttbar"].errors, [2.0])
    assert loaded["mc"]["ttbar"].overflow == 5.0
    assert loaded["mc"]["ttbar"].underflow == 4.0
    assert np.allclose(loaded["data"].counts, [3.0])


def test_nonsquare_hist2d_cache_roundtrip(tmp_path):
    counts = np.arange(6.0).reshape(2, 3)
    errors = np.arange(1.0, 7.0).reshape(2, 3)
    x_edges = np.array([0.0, 1.0, 2.0])
    y_edges = np.array([0.0, 1.0, 2.0, 3.0])
    result = _par.TaskResult(
        name="2d_njets_ht",
        success=True,
        root_name="2d_njets_ht",
        root_data=_io.Hist2DRecord(counts, x_edges, y_edges, errors),
    )
    cache_path = tmp_path / "cache.root"

    ab.save_histogram_cache([result], file=cache_path)
    loaded = ab.load_histogram_cache(file=cache_path)["hist2d"]["2d_njets_ht"]

    assert loaded.counts.shape == (2, 3)
    assert np.array_equal(loaded.counts, counts)
    assert np.array_equal(loaded.x_edges, x_edges)
    assert np.array_equal(loaded.y_edges, y_edges)
    assert np.array_equal(loaded.errors, errors)


def test_square_hist2d_cache_preserves_asymmetric_orientation(tmp_path):
    counts = np.array([[1.0, 2.0], [30.0, 40.0]])
    errors = np.array([[0.5, 1.5], [2.5, 3.5]])
    edges = np.array([0.0, 1.0, 2.0])
    result = _par.TaskResult(
        name="2d_square", success=True, root_name="2d_square",
        root_data=_io.Hist2DRecord(counts, edges, edges, errors),
    )
    cache_path = tmp_path / "cache.root"

    ab.save_histogram_cache([result], file=cache_path)
    loaded = ab.load_histogram_cache(file=cache_path)["hist2d"]["2d_square"]

    np.testing.assert_array_equal(loaded.counts, counts)
    np.testing.assert_array_equal(loaded.errors, errors)


def test_histogram_cache_stages_local_write_then_atomic_destination(tmp_path, monkeypatch, caplog):
    """The completed local ROOT file is copied, then atomically installed."""
    from analyb import io

    result = _par.TaskResult(
        name="x", success=True, root_name="x",
        root_data=_io.Hist1DRecord(
            np.array([2.0]), np.array([0.0, 1.0]), np.array([1.0]),
        ),
    )
    copied, replaced = [], []
    real_copy, real_replace = io.shutil.copyfile, io.os.replace
    monkeypatch.setattr(io.shutil, "copyfile", lambda src, dst: (copied.append((src, dst)), real_copy(src, dst))[1])
    monkeypatch.setattr(io.os, "replace", lambda src, dst: (replaced.append((src, dst)), real_replace(src, dst))[1])

    cache_path = tmp_path / "cache.root"
    with caplog.at_level("INFO"):
        io.save_histogram_cache([result], file=cache_path)

    assert copied and copied[0][0] != str(cache_path)
    assert replaced == [(copied[0][1], str(cache_path))]
    assert cache_path.exists()
    assert "local write" in caplog.text and "copy" in caplog.text


def test_histogram_cache_removes_destination_staging_file_on_copy_failure(tmp_path, monkeypatch):
    from analyb import io

    monkeypatch.setattr(io.shutil, "copyfile", lambda src, dst: (_ for _ in ()).throw(OSError("copy failed")))
    with np.testing.assert_raises_regex(OSError, "copy failed"):
        io.save_histogram_cache([], file=tmp_path / "cache.root")
    assert not list(tmp_path.glob(".analyb-cache-*.root"))
