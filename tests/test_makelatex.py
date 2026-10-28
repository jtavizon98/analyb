"""Test makelatex registration (figures, tables, inputs), and escaping."""

import os
import tempfile
import numpy as np
import analyb as ab
from analyb.makelatex import (
    _registry_var,
    register,
    register_table,
    register_input,
    _build_sections,
    _escape_tex,
    PREAMBLE_LINES,
)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _clear_registry():
    _registry_var.set({})


def _get_entries(script_id):
    reg = _registry_var.get()
    if reg is None:
        return []
    return reg.get(script_id, [])


# ---------------------------------------------------------------------------
# Figure registration
# ---------------------------------------------------------------------------

def test_register_figure_entry():
    _clear_registry()
    register("test_script", "My Section", "test_script-met.pdf",
             caption="MET distribution", width=0.45, label="fig:met")
    entries = _get_entries("test_script")
    assert len(entries) == 1
    e = entries[0]
    assert e["__type__"] == "figure"
    assert e["section"] == "My Section"
    assert e["filename"] == "test_script-met.pdf"
    assert e["caption"] == "MET distribution"
    assert e["width"] == 0.45
    assert e["label"] == "fig:met"


# ---------------------------------------------------------------------------
# Table registration
# ---------------------------------------------------------------------------

def test_register_table_entry():
    _clear_registry()
    register_table("test_script", "Yield tables", "test_script-yields.tex",
                   caption="Event yields", label="tab:yields")
    entries = _get_entries("test_script")
    assert len(entries) == 1
    e = entries[0]
    assert e["__type__"] == "table"
    assert e["section"] == "Yield tables"
    assert e["filename"] == "test_script-yields.tex"
    assert e["caption"] == "Event yields"
    assert e["label"] == "tab:yields"
    assert "width" not in e  # tables don't have width


def test_register_input_entry():
    _clear_registry()
    register_input("test_script", "Reference", "test_script-formulas.tex")
    entries = _get_entries("test_script")
    assert entries == [{
        "__type__": "input",
        "section": "Reference",
        "filename": "test_script-formulas.tex",
    }]


# ---------------------------------------------------------------------------
# Mixed figure + table rendering (_build_sections)
# ---------------------------------------------------------------------------

def test_build_sections_mixed_figure_and_table():
    """Figures are two-per-row; tables are solo. They never pair together."""
    entries = [
        {"__type__": "figure", "section": "Test", "filename": "fig_a.pdf",
         "caption": "Fig A", "width": 0.45, "label": "fig:a"},
        {"__type__": "table", "section": "Test", "filename": "tab_x.tex",
         "caption": "Tab X", "label": "tab:x"},
        {"__type__": "figure", "section": "Test", "filename": "fig_b.pdf",
         "caption": "Fig B", "width": 0.45, "label": "fig:b"},
    ]
    tex = _build_sections(entries)

    # Check figure environment exists
    assert r"\begin{figure}[H]" in tex
    assert r"\end{figure}" in tex
    # Check table environment exists
    assert r"\begin{table}[H]" in tex
    assert r"\end{table}" in tex
    # Check figures rendered with includegraphics
    assert r"\includegraphics" in tex
    # Check table uses \input
    assert r"\input{" in tex
    assert "tab_x.tex" in tex
    # Tables never paired with figures (no \hfill in table block)
    table_block = tex.split(r"\begin{table}")[1].split(r"\end{table}")[0]
    assert r"\hfill" not in table_block


def test_build_sections_two_figures_one_row():
    """Two figures in the same section appear side-by-side."""
    entries = [
        {"__type__": "figure", "section": "Test", "filename": "fig_a.pdf",
         "caption": "Fig A", "width": 0.45, "label": "fig:a"},
        {"__type__": "figure", "section": "Test", "filename": "fig_b.pdf",
         "caption": "Fig B", "width": 0.45, "label": "fig:b"},
    ]
    tex = _build_sections(entries)
    # Exactly one figure environment (two figs share it)
    assert tex.count(r"\begin{figure}[H]") == 1
    # Both includegraphics present
    assert tex.count(r"\includegraphics") == 2
    # \hfill between them
    assert r"\hfill" in tex


def test_build_sections_table_alone_no_figure_env():
    """A lone table gets its own table[H] float, never a figure env."""
    entries = [
        {"__type__": "table", "section": "Test", "filename": "tab.tex",
         "caption": "Tab", "label": "tab:one"},
    ]
    tex = _build_sections(entries)
    assert r"\begin{table}[H]" in tex
    assert r"\begin{figure}[H]" not in tex
    assert r"\input{tab.tex}" in tex


def test_build_sections_input_is_inline_and_uses_asset_directory():
    entries = [
        {"__type__": "input", "section": "Reference", "filename": "formulas.tex"},
    ]
    tex = _build_sections(entries, asset_dir="test_script")
    assert r"\subsection{Reference}" in tex
    assert r"\input{test_script/formulas.tex}" in tex
    assert r"\begin{figure}[H]" not in tex
    assert r"\begin{table}[H]" not in tex


def test_input_page_breaks_wrap_the_section_heading_and_content():
    entries = [{
        "__type__": "input", "section": "Reference", "filename": "formulas.tex",
        "page_break_before": True, "page_break_after": True,
    }]

    tex = _build_sections(entries)

    assert tex.index(r"\clearpage") < tex.index(r"\subsection{Reference}")
    assert tex.rindex(r"\clearpage") > tex.index(r"\input{formulas.tex}")


# ---------------------------------------------------------------------------
# Table packages in preamble
# ---------------------------------------------------------------------------

def test_booktabs_in_preamble():
    """PREAMBLE_LINES includes \\usepackage{booktabs}."""
    preamble = "\n".join(PREAMBLE_LINES)
    assert r"\usepackage{booktabs}" in preamble
    assert r"\usepackage{longtable}" in preamble


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------

def test_escape_tex_underscore():
    assert _escape_tex("hello_world") == r"hello\_world"


def test_escape_tex_preserves_physics_latex():
    label = r"$E_T^{\mathrm{miss}}$"
    assert _escape_tex(label) == label


def test_escape_tex_escapes_plain_text_specials_around_math():
    caption = r"A&B_1: $E_T^{\mathrm{miss}}$ (50%)"
    assert _escape_tex(caption) == (
        r"A\&B\_1: $E_T^{\mathrm{miss}}$ (50\%)"
    )


# ---------------------------------------------------------------------------
# write_tex integration (full output check)
# ---------------------------------------------------------------------------

def test_write_tex_with_mixed_entries():
    """write_tex produces valid structure for mixed figure/table entries."""
    _clear_registry()
    register("test_mixed", "Kinematics", "test_mixed-met.pdf",
             caption="MET", width=0.45)
    register_table("test_mixed", "Yield tables", "test_mixed-yields.tex",
                   caption="Yields", label="tab:yields")

    with tempfile.TemporaryDirectory() as tmpdir:
        ab.set_save_path(tmpdir)
        ab.set_script_id("test_mixed")
        tex_path = ab.makelatex.write_tex("test_mixed", save_path=tmpdir)
        assert tex_path is not None
        assert os.path.exists(tex_path)

        content = open(tex_path, encoding="utf-8").read()

        # Preamble guard
        assert r"\ifdefined\analybmain" in content
        # Figures and tables present
        assert r"\begin{figure}[H]" in content
        assert r"\begin{table}[H]" in content
        assert r"\input{test_mixed/test_mixed-yields.tex}" in content
        assert r"\includegraphics" in content
        # Captions
        assert "MET" in content
        assert "Yields" in content


def test_write_master_with_table_entries():
    """write_master includes booktabs and handles table registrations."""
    _clear_registry()
    register("ms_tab", "Section", "ms_tab-fig.pdf")
    register_table("ms_tab", "Tables", "ms_tab-tab.tex",
                   caption="Yield table")

    with tempfile.TemporaryDirectory() as tmpdir:
        ab.set_save_path(tmpdir)
        ab.set_script_id("ms_tab")
        # Write the per-script tex first
        ab.makelatex.write_tex("ms_tab", save_path=tmpdir)
        # Then write master
        master_path = ab.makelatex.write_master(
            ["ms_tab"], save_path=tmpdir, title="Test Master"
        )
        assert master_path is not None
        content = open(master_path, encoding="utf-8").read()
        assert r"\usepackage{booktabs}" in content
        assert r"\def\analybmain{}" in content
