"""Basic import tests — no ROOT data needed, runs in local venv."""

import analyb as ab


def test_import():
    """Package is importable."""
    assert ab is not None


def test_submodules_importable():
    """All submodules can be imported."""
    import analyb.binning
    import analyb.colors
    import analyb.config
    import analyb.cutflow
    import analyb.histogram
    import analyb.io
    import analyb.makelatex
    import analyb.parallel
    import analyb.plot
    import analyb.stats
    import analyb.units
    import analyb.variables
    import analyb.yieldtable


def test_version():
    """Version is accessible."""
    assert isinstance(ab.__version__, str)
    assert ab.__version__
