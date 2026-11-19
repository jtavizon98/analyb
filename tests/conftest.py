"""Use a noninteractive backend for deterministic plot rendering tests."""

import matplotlib

matplotlib.use("Agg", force=True)

from hypothesis import settings

# Default for property tests; --hypothesis-profile on the command line wins.
settings.register_profile("analyb", max_examples=200, deadline=None)
settings.register_profile("quick", max_examples=10, deadline=None)
settings.load_profile("analyb")
