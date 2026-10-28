"""Statistical utilities: chi-squared, Poisson likelihood, binomial intervals, Gaussian models."""

import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None


def chi_squared(func, xdata, ydata, params, sigma):
    """Chi-squared for given function and values."""
    return np.sum(((func(xdata, *params) - ydata) / sigma) ** 2)


def poisson_likelihood_objective(params, xdata, ydata, func):
    """Negative Poisson log-likelihood for minimisation."""
    expected = np.maximum(func(xdata, *params), 1e-10)
    return -np.sum(ydata * np.log(expected) - expected)


def poisson_likelihood(params, xdata, ydata, func):
    """Poisson likelihood (product of PMFs) for given parameters."""
    ymodel = np.maximum(func(xdata, *params), 1e-10)
    log_likelihood = np.sum(stats.poisson.logpmf(ydata, ymodel))
    return np.exp(log_likelihood)


def clopper_pearson_interval(passed, total, *, cl=0.6827):
    """Clopper-Pearson bounds ``(lower, upper)`` for genuine binomial counts.

    ``passed`` and ``total`` are unweighted integer event counts with
    ``0 <= passed <= total`` (scalars or arrays of one shape). The exact
    interval covers the binomial probability with at least ``cl`` for every
    ``total`` and every true probability; no approximation is substituted at
    small or large ``total``. ``lower = 0`` when ``passed = 0`` and ``upper = 1``
    when ``passed = total``; both are NaN when ``total = 0``.

    These are interval bounds, not a variance. Weighted, signed or
    non-integer inputs are rejected; weighted ratios use the delta-method
    variance of :func:`analyb.cutflow.nested_efficiency` and carry no coverage
    guarantee. Integer-valued weighted sums cannot be detected here and remain
    the caller's responsibility. Counts above 2**53 are rejected because they
    are not exactly representable in double precision.
    """
    k, n = np.asarray(passed), np.asarray(total)
    if k.shape != n.shape:
        raise ValueError("passed and total must have one common shape")
    if not 0 < cl < 1:
        raise ValueError("cl must lie strictly between 0 and 1")
    # Validate in the native dtype: float conversion above 2**53 could merge
    # distinct counts and hide passed > total.
    for counts in (k, n):
        if counts.dtype.kind not in "iuf":
            raise ValueError("Clopper-Pearson intervals require numeric counts")
        if counts.dtype.kind == "f" and (np.any(~np.isfinite(counts))
                                         or np.any(counts != np.floor(counts))):
            raise ValueError("Clopper-Pearson intervals require unweighted integer counts")
        if np.any(counts < 0):
            raise ValueError("Clopper-Pearson intervals require nonnegative counts")
        if np.any(counts > 2 ** 53):
            raise ValueError("counts above 2**53 are not exactly representable")
    k, n = k.astype(np.float64), n.astype(np.float64)  # exact below 2**53
    if np.any(k > n):
        raise ValueError("passed counts cannot exceed total counts")
    alpha = 1 - cl
    with np.errstate(invalid="ignore"):
        lower = np.where(k > 0, stats.beta.ppf(alpha / 2, k, n - k + 1), 0.0)
        # isf keeps the upper tail when 1 - alpha/2 would round to 1.
        upper = np.where(k < n, stats.beta.isf(alpha / 2, k + 1, n - k), 1.0)
    empty = n == 0
    lower, upper = np.where(empty, np.nan, lower), np.where(empty, np.nan, upper)
    return lower[()], upper[()]


def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def gaussian_with_constant_background(x, a, mu, sigma, c):
    return gaussian(x, a, mu, sigma) + c
