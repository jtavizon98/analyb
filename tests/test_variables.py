"""Caller-owned variable metadata, with no built-in analysis catalogue."""

from contextvars import Context

import pytest

import analyb as ab
from analyb import variables


def test_default_registry_is_empty():
    def check():
        assert variables._get_registry() == {}
        assert ab.lookup("observable") == ("observable", None)
        with pytest.raises(KeyError, match="not registered"):
            ab.lookup("observable", strict=True)
    Context().run(check)


def test_register_and_register_many_are_context_local():
    def check():
        ab.register("observable", r"$x$", r"\GeV")
        ab.register_many({"angle": (r"$\phi$", r"\radian")})
        assert ab.lookup("observable", strict=True) == (r"$x$", r"\GeV")
        assert ab.lookup("angle") == (r"$\phi$", r"\radian")
        assert Context().run(ab.lookup, "observable") == ("observable", None)
    Context().run(check)


def test_local_overrides_restore_after_exception():
    def check():
        ab.register("observable", "Original")
        with pytest.raises(RuntimeError):
            with ab.local({"observable": ("Temporary", None)}):
                assert ab.lookup("observable") == ("Temporary", None)
                raise RuntimeError("synthetic failure")
        assert ab.lookup("observable") == ("Original", None)
    Context().run(check)


def test_plot_uses_explicit_registry_metadata():
    import matplotlib.pyplot as plt

    hist = ab.Hist1D([1.0], [0.0, 1.0])
    with ab.local({"observable": (r"Observable $x$", r"\GeV")}):
        fig, ax = ab.hist1d_plot(hist, variable="observable", atlas=False)
        assert "Observable" in ax.get_xlabel()
        assert "GeV" in ax.get_xlabel()
        plt.close(fig)
