"""Test that replot batch functions register the same LaTeX entries as non-replot.

Mock worker execution is used — we test only the registration logic (the parent
side), not actual multiprocessing.  We verify that:
1. replot_1d_batch registers the same entries as hist1d_batch
2. replot_2d_batch registers the same entries as hist2d_batch
3. replot_profile_batch registers the same entries as profile_batch
4. replot_compare_batch registers the same entries as compare_batch
5. Only entries present in the supplied catalog are registered (no stale entries)
"""

import os
import tempfile

import numpy as np
import polars as pl
import pytest

import analyb as ab
from analyb.io import CompareRecord, Hist1DRecord, Hist2DRecord, ProfileRecord
from analyb.makelatex import _registry_var
from analyb.parallel import (
    _worker_fill_and_plot_1d,
    _worker_fill_and_plot_2d,
    _worker_plot_profiles,
    _worker_compare,
    _worker_replot_1d,
    _worker_replot_2d,
    _worker_replot_profile,
    _worker_replot_compare,
)

# ---------------------------------------------------------------------------
# Helper: override _run_batch to execute tasks sequentially in-process
# ---------------------------------------------------------------------------

def _run_sequential(worker_func, task_args, total, **kwargs):
    """Run tasks sequentially in-process instead of via ProcessPoolExecutor."""
    results = []
    for args in task_args:
        result = worker_func(args)
        results.append(result)
    return results


def _run_batch_then_register(batch_func, *args, **kwargs):
    """Call a batch func, forcing sequential execution, and capture LaTeX entries.

    Returns (results, latex_entries) where latex_entries is a list of
    (script_id, section, filename, caption) tuples from the registry
    for the script_id used.
    """
    import analyb.parallel as _parallel

    _registry_var.set({})
    # Monkey-patch _run_batch to run sequentially (avoids fork issues in tests)
    orig_run_batch = _parallel._run_batch
    _parallel._run_batch = _run_sequential
    try:
        results = batch_func(*args, **kwargs)
    finally:
        _parallel._run_batch = orig_run_batch

    reg = _registry_var.get() or {}
    script_id = ab.get_script_id()
    entries = reg.get(script_id, [])
    return results, entries


# ---------------------------------------------------------------------------
# Fixture: minimal DataFrame + tempdir
# ---------------------------------------------------------------------------

def _make_df():
    rng = np.random.default_rng(42)
    n = 100
    return pl.DataFrame({
        "met": rng.normal(50, 20, n),
        "ptl1": rng.normal(40, 15, n),
        "mtlep": rng.normal(60, 25, n),
        "eventWeight": np.ones(n),
    })


def _hist1d_from_record(record):
    assert isinstance(record, Hist1DRecord)
    return ab.Hist1D(
        counts=record.counts,
        edges=record.edges,
        errors=record.errors,
        unit=record.unit,
        overflow=record.overflow,
        underflow=record.underflow,
        is_density=record.is_density,
    )


def test_flatten_1d_plan_preserves_sections_styles_and_options():
    from analyb.parallel import _flatten_1d_plan

    assert _flatten_1d_plan({
        "Section": ["met", ("ptl1", "atlas"), ("mtlep", "step", {"ylog": True})],
    }) == [
        ("Section", "met", None, {}),
        ("Section", "ptl1", "atlas", {}),
        ("Section", "mtlep", "step", {"ylog": True}),
    ]
    with pytest.raises(ValueError, match="invalid 1D plot-plan entry"):
        _flatten_1d_plan([("met",)])


def test_replot_1d_registers_same_entries():
    """replot_1d_batch registers the same LaTeX entries as hist1d_batch."""
    df = _make_df()
    wgt = df["eventWeight"].to_numpy()

    section_plan = {
        "Test Section": [
            ("met", "atlas", {}),
            ("ptl1", "atlas", {}),
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ab.set_script_id("test_replot_1d")
        ab.set_save_path(tmpdir)
        ab.set_weight_column("eventWeight")

        # First: run hist1d_batch, capture registrations
        _, fig_entries = _run_batch_then_register(
            ab.parallel.hist1d_batch,
            df, section_plan, weights=wgt, progress=False, max_workers=1,
        )
        # Build hists dict from results (need to run actual workers)
        _registry_var.set({})
        import analyb.parallel as _parallel
        orig = _parallel._run_batch
        _parallel._run_batch = _run_sequential
        results = ab.parallel.hist1d_batch(df, section_plan, weights=wgt, progress=False, max_workers=1)
        _parallel._run_batch = orig

        # Extract Hist1D objects from ROOT data
        hists = {}
        for r in results:
            if r.success and r.root_data is not None:
                hists[r.name] = _hist1d_from_record(r.root_data)

        # Now run replot and capture registrations
        _, replot_entries = _run_batch_then_register(
            ab.parallel.replot_1d_batch,
            hists, section_plan, progress=False, max_workers=1,
        )

        # Compare: same number of entries
        assert len(fig_entries) == len(replot_entries)
        # Same sections, filenames, captions
        for fe, re in zip(sorted(fig_entries, key=lambda e: e["filename"]),
                           sorted(replot_entries, key=lambda e: e["filename"])):
            assert fe["section"] == re["section"]
            assert fe["filename"] == re["filename"]
            assert fe["caption"] == re["caption"]


def test_replot_1d_only_registers_present():
    """replot_1d_batch skips variables not in hists catalog."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ab.set_script_id("test_replot_partial")
        ab.set_save_path(tmpdir)

        # Supply a catalog with only "met", but request "met" and "ptl1"
        from analyb.histogram import Hist1D
        h_met = Hist1D(counts=np.array([10.0, 20.0]),
                        edges=np.array([0.0, 1.0, 2.0]),
                        errors=None)

        section_plan = {
            "Section": [
                ("met", "atlas", {}),
                ("ptl1", "atlas", {}),  # not in catalog
            ]
        }

        _, entries = _run_batch_then_register(
            ab.parallel.replot_1d_batch,
            {"met": h_met}, section_plan, progress=False, max_workers=1,
        )

        # Only "met" is registered; "ptl1" was skipped
        filenames = [e["filename"] for e in entries]
        assert any("met" in fn for fn in filenames)
        assert not any("ptl1" in fn for fn in filenames)


def test_replot_2d_registers_same_entries():
    """replot_2d_batch registers the same LaTeX entries as hist2d_batch."""
    df = _make_df()
    wgt = df["eventWeight"].to_numpy()

    section_plan = {
        "2D Section": [
            ("met", "ptl1"),
        ]
    }
    ranges = {"met": (0, 200), "ptl1": (0, 200)}

    with tempfile.TemporaryDirectory() as tmpdir:
        ab.set_script_id("test_replot_2d")
        ab.set_save_path(tmpdir)

        # hist2d_batch entries
        _, fig_entries = _run_batch_then_register(
            ab.parallel.hist2d_batch,
            df, section_plan, weights=wgt, ranges=ranges,
            progress=False, max_workers=1,
        )

        # Build hists for replot
        _registry_var.set({})
        import analyb.parallel as _parallel
        orig = _parallel._run_batch
        _parallel._run_batch = _run_sequential
        results = ab.parallel.hist2d_batch(df, section_plan, weights=wgt,
                                            ranges=ranges,
                                            progress=False, max_workers=1)
        _parallel._run_batch = orig

        hists = {}
        for r in results:
            if r.success and r.root_data is not None:
                data = r.root_data
                assert isinstance(data, Hist2DRecord)
                hists[r.root_name] = ab.Hist2D(
                    counts=data.counts, x_edges=data.x_edges,
                    y_edges=data.y_edges, errors=data.errors,
                    x_unit=data.x_unit, y_unit=data.y_unit,
                )

        _, replot_entries = _run_batch_then_register(
            ab.parallel.replot_2d_batch,
            hists, section_plan, progress=False, max_workers=1,
        )

        assert len(fig_entries) == len(replot_entries)
        for fe, re in zip(sorted(fig_entries, key=lambda e: e["filename"]),
                           sorted(replot_entries, key=lambda e: e["filename"])):
            assert fe["section"] == re["section"]
            assert fe["filename"] == re["filename"]


def test_replot_profile_registers_same_entries():
    """replot_profile_batch registers the same LaTeX entries as profile_batch."""
    df = _make_df()

    section_plan = {
        "Profile Section": [
            ("met", "mtlep"),
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ab.set_script_id("test_replot_prof")
        ab.set_save_path(tmpdir)

        _, fig_entries = _run_batch_then_register(
            ab.parallel.profile_batch,
            df, section_plan, progress=False, max_workers=1,
        )

        # Build profiles for replot
        _registry_var.set({})
        import analyb.parallel as _parallel
        orig = _parallel._run_batch
        _parallel._run_batch = _run_sequential
        results = ab.parallel.profile_batch(df, section_plan, progress=False, max_workers=1)
        _parallel._run_batch = orig

        profiles = {}
        for r in results:
            if r.success and r.root_data is not None:
                data = r.root_data
                assert isinstance(data, ProfileRecord)
                profiles[r.root_name] = (data.centers, data.means, data.stderr)

        _, replot_entries = _run_batch_then_register(
            ab.parallel.replot_profile_batch,
            profiles, section_plan, progress=False, max_workers=1,
        )

        assert len(fig_entries) == len(replot_entries)
        for fe, re in zip(sorted(fig_entries, key=lambda e: e["filename"]),
                           sorted(replot_entries, key=lambda e: e["filename"])):
            assert fe["section"] == re["section"]
            assert fe["filename"] == re["filename"]


def test_replot_compare_registers_same_entries():
    """replot_compare_batch registers the same LaTeX entries as compare_batch."""
    df = _make_df()

    variables = ["met", "ptl1"]

    with tempfile.TemporaryDirectory() as tmpdir:
        ab.set_script_id("test_replot_cmp")
        ab.set_save_path(tmpdir)
        ab.set_weight_column("eventWeight")

        dfs_mc = {"ttbar": df, "Wt": df}

        _, fig_entries = _run_batch_then_register(
            ab.parallel.compare_batch,
            dfs_mc, None, variables, edges={},
            plot_type="overlay", save_prefix="cmp_",
            section="Compare Section",
            caption_fmt="{var} distribution",
            progress=False, max_workers=1,
        )

        # Build compare_data for replot
        _registry_var.set({})
        import analyb.parallel as _parallel
        orig = _parallel._run_batch
        _parallel._run_batch = _run_sequential
        results = ab.parallel.compare_batch(
            dfs_mc, None, variables, edges={},
            plot_type="overlay", save_prefix="cmp_",
            section="Compare Section",
            caption_fmt="{var} distribution",
            progress=False, max_workers=1,
        )
        _parallel._run_batch = orig

        compare_data = {}
        for r in results:
            if r.success and r.root_data is not None:
                data = r.root_data
                assert isinstance(data, CompareRecord)
                entry = {
                    "mc": {
                        name: _hist1d_from_record(record)
                        for name, record in data.mc.items()
                    }
                }
                if data.data is not None:
                    entry["data"] = _hist1d_from_record(data.data)
                compare_data[r.name] = entry

        _, replot_entries = _run_batch_then_register(
            ab.parallel.replot_compare_batch,
            compare_data, variables,
            plot_type="overlay", save_prefix="cmp_",
            section="Compare Section",
            caption_fmt="{var} distribution",
            progress=False, max_workers=1,
        )

        assert len(fig_entries) == len(replot_entries)
        for fe, re in zip(sorted(fig_entries, key=lambda e: e["filename"]),
                           sorted(replot_entries, key=lambda e: e["filename"])):
            assert fe["section"] == re["section"]
            assert fe["filename"] == re["filename"]


def test_replot_compare_caption_resolves_samples():
    ab.set_script_id("test_replot_samples")
    hist = ab.Hist1D([1.0], [0.0, 1.0])
    compare_data = {"met": {"mc": {"ttbar": hist, "Wt": hist}, "data": None}}

    _, entries = _run_batch_then_register(
        ab.parallel.replot_compare_batch,
        compare_data,
        ["met"],
        section="Compare",
        caption_fmt="{var}: {samples}",
        progress=False,
        max_workers=1,
    )

    assert "ttbar + Wt" in entries[0]["caption"]
    assert "{samples}" not in entries[0]["caption"]


def test_compare_and_replot_forward_stacked_color_options(monkeypatch, tmp_path):
    """Stack-only color options survive the parent/worker global handoff."""
    import analyb.parallel as parallel

    captured = []
    monkeypatch.setattr(parallel, "_hist1d_stacked", lambda *args, **kwargs: captured.append(kwargs))
    monkeypatch.setattr(parallel, "_save_results_to_root", lambda results: None)
    monkeypatch.setattr(parallel, "_run_batch", _run_sequential)
    ab.set_script_id("test_stack_colors")
    ab.set_save_path(str(tmp_path))
    ab.set_weight_column("eventWeight")
    colors = {"ttbar": "navy"}
    df = pl.DataFrame({"met": [10.0, 20.0], "eventWeight": [1.0, 1.0]})

    parallel.compare_batch(
        {"ttbar": df}, variables=["met"], plot_type="stacked",
        stack_colors=colors, stack_data_color="magenta", progress=False,
    )
    hist = ab.Hist1D(np.array([1.0]), np.array([0.0, 1.0]), errors=np.array([1.0]))
    parallel.replot_compare_batch(
        {"met": {"mc": {"ttbar": hist}}}, ["met"], plot_type="stacked",
        stack_colors=colors, stack_data_color="magenta", progress=False,
    )

    assert len(captured) == 2
    assert all(item["colors"] == colors for item in captured)
    assert all(item["data_color"] == "magenta" for item in captured)


def test_compare_and_replot_forward_overlay_atlas_options(monkeypatch, tmp_path):
    import analyb.parallel as parallel

    captured = []
    monkeypatch.setattr(
        parallel, "_hist1d_comparison",
        lambda *args, **kwargs: captured.append(kwargs),
    )
    monkeypatch.setattr(parallel, "_save_results_to_root", lambda results: None)
    monkeypatch.setattr(parallel, "_run_batch", _run_sequential)
    ab.set_script_id("test_overlay_atlas")
    ab.set_save_path(str(tmp_path))
    ab.set_weight_column("eventWeight")
    label_kwargs = {"status": "Internal", "lumi": False}
    df = pl.DataFrame({"met": [10.0, 20.0], "eventWeight": [1.0, 1.0]})

    parallel.compare_batch(
        {"ttbar": df}, variables=["met"], plot_type="overlay",
        atlas_label_kwargs=label_kwargs, progress=False,
    )
    hist = ab.Hist1D(
        np.array([1.0]), np.array([0.0, 1.0]), errors=np.array([1.0]),
    )
    parallel.replot_compare_batch(
        {"met": {"mc": {"ttbar": hist}}}, ["met"], plot_type="overlay",
        atlas_label_kwargs=label_kwargs, progress=False,
    )

    assert len(captured) == 2
    assert all(item["atlas"] is True for item in captured)
    assert all(item["atlas_label_kwargs"] == label_kwargs for item in captured)


def test_compare_and_replot_share_ratio_renderer(monkeypatch, tmp_path):
    import analyb.parallel as parallel

    captured = []
    monkeypatch.setattr(
        parallel, "_hist1d_with_ratio",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )
    monkeypatch.setattr(parallel, "_save_results_to_root", lambda results: None)
    monkeypatch.setattr(parallel, "_run_batch", _run_sequential)
    ab.set_script_id("test_ratio_renderer")
    ab.set_save_path(str(tmp_path))
    ab.set_weight_column("eventWeight")
    frame = pl.DataFrame({"met": [10.0, 20.0], "eventWeight": [1.0, 1.0]})

    parallel.compare_batch(
        {"ttbar": frame}, frame, ["met"], plot_type="ratio", progress=False,
    )
    hist = ab.Hist1D([1.0], [0.0, 1.0], errors=[1.0])
    parallel.replot_compare_batch(
        {"met": {"mc": {"ttbar": hist}, "data": hist}},
        ["met"], plot_type="ratio", progress=False,
    )

    assert len(captured) == 2
