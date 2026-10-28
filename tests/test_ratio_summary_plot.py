"""ratio_summary_plot: categorical ratio summary with components stacked from one."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import analyb as ab


def _presentation(values, errors, components=None):
    colors = None if components is None else {k: c for k, c in zip(components, ("#E69F00", "#009E73"))}
    return ab.RatioPresentation(values=values, errors=errors, valid=np.isfinite(values),
                                components=components, component_colors=colors)


def test_ratio_summary_plot_places_series_in_sub_slots_with_stacks():
    dev = _presentation([0.9, 1.05, np.nan], [0.02, 0.01, 0.0],
                        {"W": [-0.15, 0.05, 0.0], "G": [0.05, 0.0, 0.0]})
    val = _presentation([0.92, 1.02, 0.99], [0.02, 0.01, 0.01])
    fig, ax = ab.ratio_summary_plot(["a", "b", "c"], {"dev": dev, "val": val},
                                    ylabel="q", xlabel="Sub-step", ratio_ylim=(0.8, 1.2),
                                    colors={"val": "blue"}, markers={"val": "s"}, separators=(2,),
                                    separator_ymax=0.8)
    points = [c.lines[0] for c in ax.containers if hasattr(c, "lines")]
    np.testing.assert_allclose(points[0].get_xdata(), [-0.25, 0.75, 1.75])
    np.testing.assert_allclose(points[1].get_xdata(), [0.25, 1.25, 2.25])
    assert points[1].get_marker() == "s"
    bars = [p for p in ax.patches if p.get_height() != 0]
    # W at slot 0 goes down by 0.15 from 1 in the first half slot; G stacks up from 1.
    w0 = [p for p in bars if np.isclose(p.get_x(), -0.5) and np.isclose(p.get_height(), -0.15)]
    assert len(w0) == 1 and np.isclose(w0[0].get_width(), 0.5) and np.isclose(w0[0].get_y(), 1.0)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["dev", "val", "W", "G"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    assert ax.get_ylim() == (0.8, 1.2) and ax.get_xlabel() == "Sub-step"
    sep = [l for l in ax.lines if len(l.get_xdata()) == 2 and np.allclose(l.get_xdata(), 1.5)]
    assert len(sep) == 1 and np.allclose(sep[0].get_ydata(), [0.0, 0.8])   # stops below the legend
    plt.close(fig)


def test_ratio_summary_plot_marks_values_beyond_the_window():
    dev = _presentation([0.5, 1.0], [0.01, 0.01])
    fig, ax = ab.ratio_summary_plot(["a", "b"], {"dev": dev}, ylabel="q", xlabel="x",
                                    ratio_ylim=(0.9, 1.1))
    below = [l for l in ax.lines if l.get_marker() == "v" and len(l.get_xdata())]
    assert len(below) == 1 and np.isclose(below[0].get_xdata()[0], 0.0)
    plt.close(fig)


def test_ratio_summary_plot_rejects_bad_input():
    good = _presentation([1.0], [0.0])
    with pytest.raises(ValueError, match="one value per category"):
        ab.ratio_summary_plot(["a", "b"], {"s": good}, ylabel="q", xlabel="x")
    with pytest.raises(ValueError, match="x and a y label"):
        ab.ratio_summary_plot(["a"], {"s": good}, ylabel="q", xlabel="")
    a = ab.RatioPresentation(values=[1.0], errors=[0.0], valid=[True], components={"W": [0.1]},
                             component_colors={"W": "red"})
    b = ab.RatioPresentation(values=[1.0], errors=[0.0], valid=[True], components={"W": [0.1]},
                             component_colors={"W": "blue"})
    with pytest.raises(ValueError, match="two colours"):
        ab.ratio_summary_plot(["a"], {"a": a, "b": b}, ylabel="q", xlabel="x")


def test_ratio_summary_plot_draws_into_a_given_axes():
    fig, (top, bottom) = plt.subplots(2, 1)
    dev = _presentation([0.98, 1.01], [0.01, 0.01])
    before = set(plt.get_fignums())
    out_fig, out_ax = ab.ratio_summary_plot(["a", "b"], {"dev": dev}, ylabel="q", xlabel="x", ax=bottom)
    assert out_fig is fig and out_ax is bottom and not top.lines
    assert set(plt.get_fignums()) == before and bottom.get_xlabel() == "x"
    plt.close(fig)
