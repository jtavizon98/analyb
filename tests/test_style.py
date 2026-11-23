"""analyb's style does not depend on the host's matplotlibrc."""

import json
import subprocess
import sys


def _rc_after_import(tmp_path, rc_text):
    (tmp_path / "matplotlibrc").write_text(rc_text)
    code = ("import json, matplotlib; import analyb; r = matplotlib.rcParams; "
            "print(json.dumps([r['lines.markersize'], r['font.size'], r['legend.handlelength'], r['axes.titlesize']]))")
    out = subprocess.run([sys.executable, "-c", code], env={"MATPLOTLIBRC": str(tmp_path), "MPLBACKEND": "Agg"},
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_host_matplotlibrc_does_not_reach_analyb_figures(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = _rc_after_import(tmp_path / "a", "")
    personal = _rc_after_import(tmp_path / "b", "lines.markersize: 20\nfont.size: 30\nlegend.handlelength: 9\n"
                                                "axes.titlesize: xx-large\n")
    assert plain == personal == [4.0, 12.0, 1.5, "medium"]


def test_default_data_markers_are_filled():
    import matplotlib.pyplot as plt
    import analyb as ab
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3], "o")
    assert not [f for f in ab.check_figure(fig) if f.check == "marker-hollow"]
    plt.close(fig)
