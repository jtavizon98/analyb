"""Test YieldTable.to_latex escaping and column spec."""

import numpy as np
import analyb as ab
from analyb.yieldtable import YieldTable, YieldEntry, yield_table


def test_to_latex_column_spec():
    """Column spec uses all-l format (no broken S-column syntax)."""
    rows = [YieldEntry("ttbar", "SR", 100.0, 10.0)]
    yt = YieldTable(rows, samples=["ttbar"], regions=["SR"], row_axis="sample")
    tex = yt.to_latex()
    # spec should be "ll" (one label + one data column)
    assert r"\begin{tabular}{ll}" in tex or r"\begin{tabular}{l l}" in tex
    # Should not contain broken "S " remnants
    assert "S " not in tex.split(r"\begin{tabular")[1].split(r"\end{tabular")[0].split("}")[0]


def test_to_latex_pm_in_math_mode():
    """Uncertainties use $\\pm$ (math mode)."""
    rows = [YieldEntry("ttbar", "SR", 100.0, 10.0)]
    yt = YieldTable(rows, samples=["ttbar"], regions=["SR"], row_axis="sample")
    tex = yt.to_latex()
    # \pm should appear inside math-mode $...$ delimiters
    import re
    # Find the data cell: something like $100 \pm 10$
    assert re.search(r"\$\d+.*\\pm.*\d+\$", tex), f"No math-mode \\pm found in: {tex}"


def test_to_latex_underscore_escaped():
    """Underscores in sample/region labels are escaped."""
    rows = [YieldEntry("ttbar_enhanced", "signal_region", 100.0, 10.0)]
    yt = YieldTable(rows, samples=["ttbar_enhanced"], regions=["signal_region"],
                    row_axis="sample")
    tex = yt.to_latex()
    assert r"ttbar\_enhanced" in tex
    assert r"signal\_region" in tex


def test_to_latex_special_chars_escaped():
    """Ampersand, percent, and dollar signs are escaped in labels."""
    rows = [YieldEntry("tt & bar", "SR%", 100.0, 10.0)]
    yt = YieldTable(rows, samples=["tt & bar"], regions=["SR%"],
                    row_axis="sample")
    tex = yt.to_latex()
    assert r"\&" in tex
    assert r"\%" in tex


def test_to_latex_backslash_escape_has_no_literal_braces():
    assert YieldTable._escape_tex(r"a\b") == r"a\textbackslash{}b"


def test_to_latex_multi_row_multi_col():
    """Multiple rows and columns produce valid LaTeX."""
    rows = [
        YieldEntry("ttbar", "SR", 100.0, 5.0),
        YieldEntry("ttbar", "CR", 50.0, 3.0),
        YieldEntry("Wt", "SR", 30.0, 2.0),
        YieldEntry("Wt", "CR", 15.0, 1.0),
    ]
    yt = YieldTable(rows, samples=["ttbar", "Wt"], regions=["SR", "CR"],
                    row_axis="sample")
    tex = yt.to_latex()
    assert r"\toprule" in tex
    assert r"\midrule" in tex
    assert r"\bottomrule" in tex
    assert r"\end{tabular}" in tex
    # Check spec has 3 columns (label + 2 data)
    assert r"\begin{tabular}{lll}" in tex or r"\begin{tabular}{l l l}" in tex


def test_to_latex_missing_entry_dash():
    """Missing entries render as '--'."""
    rows = [YieldEntry("ttbar", "SR", 100.0, 10.0)]
    yt = YieldTable(rows, samples=["ttbar", "Wt"], regions=["SR"],
                    row_axis="sample")
    tex = yt.to_latex()
    assert "--" in tex


def test_to_latex_region_axis():
    """Column spec is correct when row_axis is 'region'."""
    rows = [
        YieldEntry("ttbar", "SR", 100.0, 10.0),
        YieldEntry("Wt", "SR", 50.0, 5.0),
    ]
    yt = YieldTable(rows, samples=["ttbar", "Wt"], regions=["SR"],
                    row_axis="region")
    tex = yt.to_latex()
    # 1 label + 2 data columns = 3 cols
    assert r"\begin{tabular}{lll}" in tex or r"\begin{tabular}{l l l}" in tex


def test_save_yield_table():
    """save() writes .txt and .tex files."""
    import tempfile, os
    rows = [YieldEntry("tt", "SR", 100.0, 10.0)]
    yt = YieldTable(rows, samples=["tt"], regions=["SR"], row_axis="sample")
    with tempfile.TemporaryDirectory() as tmpdir:
        txt, tex = yt.save("test_yields", save_path=tmpdir)
        assert os.path.exists(txt)
        assert os.path.exists(tex)
        with open(txt) as f:
            assert "tt" in f.read()
        with open(tex) as f:
            assert r"\toprule" in f.read()
