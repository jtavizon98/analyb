"""Regression coverage for efficiency_plot, which once raised on every call."""

import matplotlib.pyplot as plt
import numpy as np

import analyb as ab


def _curve():
    centers = np.array([0.5, 1.5, 2.5])
    eff = np.array([0.2, 0.5, 0.8])
    return centers, eff, np.full(3, 0.05), np.full(3, 0.04)


def test_efficiency_plot_draws_asymmetric_errors(tmp_path):
    centers, eff, low, high = _curve()
    fig, ax = ab.efficiency_plot(centers, eff, low, high, xlabel="x",
                                 save_name="eff", save_path=tmp_path, script_id="t")
    (bars,) = ax.containers
    np.testing.assert_allclose(bars.lines[0].get_ydata(), eff)
    assert (tmp_path / "t-eff.pdf").is_file()
    plt.close(fig)


def test_efficiency_plot_with_reference_ratio():
    centers, eff, low, high = _curve()
    reference = (centers, np.array([0.4, 0.5, 0.0]), low, high)
    fig, (main, ratio) = ab.efficiency_plot(centers, eff, low, high, reference=reference)
    ratio_points = ratio.containers[0].lines[0].get_ydata()
    np.testing.assert_allclose(ratio_points[:2], [0.5, 1.0])
    assert np.isnan(ratio_points[2])
    plt.close(fig)


def test_efficiency_plot_labels_reference_errors_ratio_errors_and_ranges():
    centers, eff, low, high = _curve()
    reference = (centers, np.array([0.4, 0.5, 0.5]), low, high)
    fig, (main, ratio) = ab.efficiency_plot(
        centers, eff, low, high, reference=reference, label="BM", reference_label="BH",
        reference_color="#0072B2", reference_errors=True, ratio_label="BM / BH",
        ratio_errors=[0.1, 0.2, 0.3], ylim=(0.0, 1.3), ratio_ylim=(0.5, 2.0))
    assert [t.get_text() for t in main.get_legend().get_texts()] == ["BM", "BH"]
    ref_bars = main.containers[1]
    np.testing.assert_allclose(ref_bars.lines[0].get_ydata(), [0.4, 0.5, 0.5])
    segments = ref_bars.lines[2][0].get_segments()
    np.testing.assert_allclose(segments[0][:, 1], [0.35, 0.44])
    ratio_segments = ratio.containers[0].lines[2][0].get_segments()
    np.testing.assert_allclose(ratio_segments[1][:, 1], [0.8, 1.2])
    assert main.get_ylim() == (0.0, 1.3) and ratio.get_ylim() == (0.5, 2.0)
    assert ratio.get_ylabel() == "BM / BH"
    plt.close(fig)


def test_efficiency_plot_defaults_unchanged():
    centers, eff, low, high = _curve()
    fig, (main, ratio) = ab.efficiency_plot(centers, eff, low, high,
                                            reference=(centers, eff, low, high))
    assert {t.get_text() for t in main.get_legend().get_texts()} == {"Efficiency", "Reference"}
    assert main.get_ylim() == (0, 1.05) and ratio.get_ylabel() == "Eff / Ref"
    assert len(main.containers) == 1  # reference is a dashed line
    plt.close(fig)


def test_efficiency_plot_more_series_share_the_reference():
    centers, eff, low, high = _curve()
    reference = (centers, np.array([0.4, 0.5, 0.0]), low, high)
    more = {"PLIT": dict(centers=centers + 0.1, eff=np.array([0.2, 0.25, 0.4]), err_low=low,
                         err_high=high, color="#E69F00", ratio_errors=[0.1, 0.1, 0.1], marker="s")}
    fig, (main, ratio) = ab.efficiency_plot(centers, eff, low, high, reference=reference,
                                            reference_errors=True, label="A", reference_label="R",
                                            more_series=more)
    assert [t.get_text() for t in main.get_legend().get_texts()] == ["A", "R", "PLIT"]
    extra = ratio.containers[1].lines[0]
    np.testing.assert_allclose(extra.get_xdata(), centers + 0.1)
    np.testing.assert_allclose(extra.get_ydata()[:2], [0.5, 0.5])
    assert np.isnan(extra.get_ydata()[2]) and extra.get_marker() == "s"
    np.testing.assert_allclose(ratio.containers[1].lines[2][0].get_segments()[0][:, 1], [0.4, 0.6])
    plt.close(fig)


def test_efficiency_plot_more_series_need_a_reference():
    import pytest
    centers, eff, low, high = _curve()
    with pytest.raises(ValueError, match="needs a reference"):
        ab.efficiency_plot(centers, eff, low, high, more_series={"x": dict(
            centers=centers, eff=eff, err_low=low, err_high=high, color="red")})
