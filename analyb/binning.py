"""Histogram binning rules.

Implements the rules of thumb from Sec. 4.6 of the data analytics course
(Sturges, Scott, Freedman-Diaconis) plus convenience helpers.
"""

import numpy as np

SCOTT = "scott"
STURGES = "sturges"
FREEDMAN_DIACONIS = "fd"
RULES = (SCOTT, STURGES, FREEDMAN_DIACONIS)


def _data_range(data, range=None, pct_range=None):
    """Return (lo, hi) for binning.

    - If `range` given, use it directly.
    - Elif `pct_range` given as (lo_pct, hi_pct) in [0,100], use those percentiles.
    - Else use data min/max.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if range is not None:
        lo, hi = float(range[0]), float(range[1])
    elif pct_range is not None:
        if data.size == 0:
            raise ValueError("cannot determine bin range from empty finite data")
        lo, hi = np.percentile(data, pct_range)
    else:
        if data.size == 0:
            raise ValueError("cannot determine bin range from empty finite data")
        lo, hi = float(np.min(data)), float(np.max(data))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return lo, hi


def _filter_to_range(data, range=None, pct_range=None):
    """Return data filtered to the explicit range or percentile range."""
    if range is not None:
        lo, hi = float(range[0]), float(range[1])
        return data[(data >= lo) & (data <= hi)]
    if pct_range is not None:
        lo, hi = np.percentile(data, pct_range)
        return data[(data >= lo) & (data <= hi)]
    return data


def scott_width(data, range=None, pct_range=None):
    """Scott's rule: h = 3.49 * s / n^(1/3).

    When *range* or *pct_range* is given, only data within that window is
    used to compute the standard deviation, preventing sentinel/filler values
    outside the physics region from inflating the bin width.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = _filter_to_range(data, range=range, pct_range=pct_range)
    n = data.size
    if n == 0:
        return 1.0
    s = np.std(data, ddof=1) if n > 1 else 0.0
    if s == 0:
        s = 1.0
    return 3.49 * s / np.cbrt(n)


def sturges_nbins(data, range=None, pct_range=None):
    """Sturges' rule: m = 1 + log2(n)."""
    data = np.asarray(data, dtype=float)
    n = data.size
    if n == 0:
        return 1
    return int(np.ceil(1 + np.log2(n)))


def freedman_diaconis_width(data, range=None, pct_range=None):
    """Freedman-Diaconis: h = 2 * (Q75 - Q25) / n^(1/3).

    When *range* or *pct_range* is given, only data within that window is
    used to compute the IQR, preventing sentinel/filler values outside the
    physics region from inflating the bin width.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = _filter_to_range(data, range=range, pct_range=pct_range)
    n = data.size
    if n == 0:
        return 1.0
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    if iqr == 0:
        iqr = np.std(data, ddof=1) if n > 1 else 1.0
    if iqr == 0:
        iqr = 1.0
    return 2.0 * iqr / np.cbrt(n)


def rule_to_width(rule, data, range=None, pct_range=None):
    if rule == SCOTT:
        return scott_width(data, range=range, pct_range=pct_range)
    if rule == FREEDMAN_DIACONIS:
        return freedman_diaconis_width(data, range=range, pct_range=pct_range)
    if rule == STURGES:
        lo, hi = _data_range(data, range=range, pct_range=pct_range)
        return (hi - lo) / sturges_nbins(data)
    raise ValueError(f"Unknown rule {rule!r}; choose from {RULES}")


def rule_to_nbins(rule, data, range=None, pct_range=None):
    if rule == STURGES:
        return sturges_nbins(data)
    h = rule_to_width(rule, data, range=range, pct_range=pct_range)
    lo, hi = _data_range(data, range=range, pct_range=pct_range)
    return max(1, int(np.ceil((hi - lo) / h)))


def resolve_bins(arg, data, range=None, pct_range=None):
    """Resolve a flexible `bins` argument to a numpy-compatible value.

    Accepted forms:
      - "scott" | "sturges" | "fd": apply the rule.
      - int: equal-width bins over the data range.
      - array-like: returned as-is (explicit edges).
      - float: treated as a bin width.
      - tuple (int, (lo, hi)): fixed number of bins over given range.
    """
    if isinstance(arg, str):
        lo, hi = _data_range(data, range=range, pct_range=pct_range)
        n = rule_to_nbins(arg, data, range=(lo, hi), pct_range=pct_range)
        return np.linspace(lo, hi, n + 1)
    if np.isscalar(arg):
        if isinstance(arg, (int, np.integer)):
            lo, hi = _data_range(data, range=range, pct_range=pct_range)
            return np.linspace(lo, hi, int(arg) + 1)
        if isinstance(arg, (float, np.floating)):
            lo, hi = _data_range(data, range=range, pct_range=pct_range)
            n = max(1, int(np.ceil((hi - lo) / arg)))
            return np.linspace(lo, lo + n * arg, n + 1)
    if isinstance(arg, (tuple, list)) and len(arg) == 2 and np.isscalar(arg[0]):
        n, (lo, hi) = int(arg[0]), arg[1]
        return np.linspace(lo, hi, n + 1)
    arr = np.asarray(arg, dtype=float)
    if arr.ndim == 1 and arr.size >= 2:
        return arr
    raise ValueError(f"Cannot resolve bins from {arg!r}")
