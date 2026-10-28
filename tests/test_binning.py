"""Test binning rules with dummy data — no ROOT data needed."""

import numpy as np
import analyb.binning as binning


def test_sturges():
    """Sturges rule produces reasonable bin count."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    nbins = binning.sturges_nbins(data)
    assert nbins > 1


def test_scott():
    """Scott rule produces reasonable bin width."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    width = binning.scott_width(data)
    assert width > 0


def test_freedman_diaconis():
    """Freedman-Diaconis rule produces reasonable bin width."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    width = binning.freedman_diaconis_width(data)
    assert width > 0


def test_resolve_bins():
    """resolve_bins returns edges for a rule string."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    edges = binning.resolve_bins("scott", data, range=(-5, 5))
    assert len(edges) > 2
    assert edges[0] <= -5
    assert edges[-1] >= 5


def test_resolve_bins_empty_finite_data_raises():
    """Automatic ranges require at least one finite value."""
    data = np.array([np.nan, np.inf, -np.inf])
    try:
        binning.resolve_bins("scott", data)
    except ValueError as exc:
        assert "empty finite data" in str(exc)
    else:
        raise AssertionError("resolve_bins should reject all-nonfinite data")


def test_resolve_bins_empty_data_with_explicit_range():
    """Explicit ranges allow empty histograms with known bin edges."""
    edges = binning.resolve_bins("scott", [], range=(0, 1))
    assert np.allclose(edges, [0.0, 1.0])
