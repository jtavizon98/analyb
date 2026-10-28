"""Repository-owned Matplotlib defaults for portable analysis plots."""

from pathlib import Path

from matplotlib import style


def apply():
    """Apply analyb's MathText-first plotting style."""
    style.use(Path(__file__).with_name("styles") / "analyb.mplstyle")


def plotbook_context():
    """Readable type for 7-inch source figures placed two across an A4 page."""
    from matplotlib import rc_context
    return rc_context({
        "font.size": 16, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })
