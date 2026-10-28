"""Use a noninteractive backend for deterministic plot rendering tests."""

import matplotlib

matplotlib.use("Agg", force=True)
