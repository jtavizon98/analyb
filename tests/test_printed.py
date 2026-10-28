"""Printed-value manifest: records, matching rule, and the yield-table hook."""

import json
import random
import shutil
import subprocess

import pytest

from analyb import printed
from analyb.yieldtable import YieldEntry, YieldTable


def _table():
    rows = [
        YieldEntry("ttbar", "SR", 12345.0, 111.0),
        YieldEntry("ttbar", "CR", 0.5, 0.25),
        YieldEntry("single_top", "SR", -3.2, 1.0),
        YieldEntry("single_top", "CR", 12.0, 3.5),
    ]
    return YieldTable(rows, samples=["ttbar", "single_top"], regions=["SR", "CR"])


def test_printed_formats_without_a_manifest():
    assert printed.active_manifest() is None
    assert printed.printed("x", 0.123456, ".3f") == "0.123"


def test_manifest_records_sources_and_writes_schema(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"ratio": 1.0012344}')
    out = tmp_path / "values.json"
    with printed.value_manifest(out, "generate.py", sources={"evidence": evidence}):
        text = printed.printed("ratio/ptl1", 1.0012344, ".6f", context="ptl1",
                               source="evidence", section="Comparison", page=3)
        printed.record_value("chi2/ptl1", float("nan"), "---", context="ptl1")
    assert text == "1.001234"
    record = json.loads(out.read_text())
    assert record["schema"] == printed.SCHEMA
    assert record["schema_version"] == 4
    assert record["producer"] == "generate.py"
    assert record["match_rule"] == printed.MATCH_RULE
    assert len(record["sources"]["evidence"]["sha256"]) == 64
    assert record["values"][0] == {
        "key": "ratio/ptl1", "value": 1.0012344, "text": "1.001234", "source": "evidence",
        "context": "ptl1", "section": "Comparison", "page": 3,
    }
    assert record["values"][1]["value"] is None
    assert printed.active_manifest() is None


def test_manifest_rejects_conflicts_and_unknown_sources(tmp_path):
    manifest = printed.ValueManifest("p")
    manifest.record("a", 1.0, "1.0")
    manifest.record("a", 1.0, "1.0")  # identical re-record is a no-op
    with pytest.raises(ValueError, match="twice"):
        manifest.record("a", 1.0, "1.00")
    with pytest.raises(ValueError, match="source"):
        manifest.record("b", 1.0, "1.0", source="evidence")
    with pytest.raises(ValueError):
        manifest.record("c", 1.0, "   ")
    with pytest.raises(ValueError):
        manifest.record("d", True, "1")


def test_manifest_is_not_written_when_the_block_fails(tmp_path):
    out = tmp_path / "values.json"
    with pytest.raises(RuntimeError):
        with printed.value_manifest(out, "p"):
            printed.printed("x", 1.0, ".1f")
            raise RuntimeError("generator failed")
    assert not out.exists()
    assert printed.active_manifest() is None


POSITIVE = [
    ("ptl1  1.001234  0.98", "1.001234"),
    ("value 1.25. Next", "1.25"),
    ("yield \u22123.2", "-3.2"),            # typeset minus
    ("row - 3.2", "-3.2"),                  # math spacing after a unary minus
    ("row \u2212 3.2", "-3.2"),
    ("row 1.23e + 04", "1.23e+04"),
    ("row 1.23e\u221204", "1.23e-04"),
    ("row 1,234.5", "1,234.5"),
    ("row 1, 234.5", "1,234.5"),            # one token despite the space
    ("row 12.5%", "12.5%"),
    ("row 12.5 %", "12.5%"),
    ("row 12.5 GeV", "12.5 GeV"),
    ("row 12.5GeV", "12.5 GeV"),
    ("row 12.", "12"),
    ("row 12    34%", "34%"),
    ("row \ufe633.2", "-3.2"),
    ("row \uff0d3.2", "-3.2"),
    ("range 0\u20135", "0-5"),             # not one token: literal rule
    ("row –       3.2 ± 1", "3.2"),         # placeholder stays a dash
    ("row 0 – 5", "5"),                    # dash range endpoint
    ("row 0-5", "0-5"),
    ("row --       3.2 ± 1", "3.2"),
    ("row 12    - 3.2", "-3.2"),            # separate cell with a unary sign
    ("row 12 -3.2", "-3.2"),
    ("ptl1 - 3.2", "-3.2"),                 # digits in a row label are not operands
    ("row -.5", "-.5"),
    ("row − .5", "-.5"),
    ("row +.5", "+.5"),
    ("row .5e + 04", ".5e+04"),
    ("row -.5e − 04 GeV", "-.5e-04 GeV"),
    ("row −١٢", "-12"),
    ("row −12", "-١٢"),
    ("row ١٢٣٤", "١٢٣٤"),
    ("row 12.٣٤", "12.34"),
    ("row ١ , ٢٣٤.٥e + ٠٤ GeV", "1,234.5e+04 GeV"),
    ("row 1 , 234.5", "1,234.5"),
    ("row 1 , 234 , 567.5", "1,234,567.5"),
    ("row 12.5\u2009%", "12.5%"),
    ("row 12.5\tGeV", "12.5 GeV"),
    ("row 12.5   GeV", "12.5 GeV"),
    ("row 12 ± 3 GeV", "12 ± 3 GeV"),        # numeric-bearing literal
    ("row 12 radial bins", "12"),
    ("row 12 radial bins", "12 radial bins"),
    ("row 12\tnsamples", "12"),
    ("row 12   mradial bins", "12"),
]

# Every input from both phase-2 reviews where a wrong number was accepted.
NEGATIVE = [
    ("row -3.2", "3.2"),
    ("row \u22123.2", "3.2"),
    ("row +3.2", "3.2"),
    ("row \u2212 3.2", "3.2"),             # sign separated by whitespace
    ("row - 3.2", "3.2"),
    ("row 1.25e+04", "1.25"),
    ("row 1.23e+04", "04"),
    ("row 1.23e + 04", "04"),
    ("row 1.23e + 04", "1.23"),
    ("row 1,234.5", "234.5"),
    ("row 1, 234.5", "234.5"),              # comma group with a space
    ("row 1,234.5GeV", "1"),                # no backtracking to a prefix
    ("row 1,234.5GeV", "1,234.5"),
    ("row -12.5%", "12.5%"),
    ("row -12.5 GeV", "12.5 GeV"),
    ("row 1.23e+04%", "04%"),
    ("row 12    34%", "1234%"),
    ("row 12    34%", "34"),
    ("row 12.5%", "12.5"),                  # suffix is part of the token
    ("row \ufe633.2", "3.2"),
    ("row \uff0d3.2", "3.2"),
    ("ptl1  11.001234", "1.001234"),
    ("ptl1  1.0012345", "1.001234"),
    ("value 1.25.5", "1.25"),
    ("row b77WP", "77"),
    ("row \u00e977", "77"),                 # Unicode letter before the digits
    ("row 77\u00e9", "77"),
    ("row -0.1% GeV", "0.1% GeV"),          # numbers inside literal text are tokens too
    ("row 10-5", "0-5"),
    ("row –       3.2 ± 1", "-3.2"),
    ("row 0 – 5", "-5"),
    ("row 0 − 5", "-5"),
    ("row 0 - 5", "-5"),
    ("row 0 - 5", "5"),                    # ambiguous binary/unary operand
    ("row -       3.2", "-3.2"),
    ("row -       3.2", "3.2"),             # no unsigned fallback
    ("row -\t3.2", "-3.2"),
    ("row -\t3.2", "3.2"),
    ("row -12kg", "12kg"),
    ("row 1.23e+04kg", "04kg"),
    ("row 12    34kg", "1234kg"),
    ("row 12kg", "12kg"),                   # unsupported even when identical
    ("row 12kg", "12"),
    ("row 1.23e+04kg 04", "1.23e+04kg"),
    ("row 12kg 12", "12kg"),                # unrelated valid token cannot rescue it
    ("row -.5", ".5"),
    ("row +.5", ".5"),
    ("row .5e+04", "04"),
    ("row .5e + 04", "+04"),
    ("row -.5e+04%", ".5e+04%"),
    ("row −١٢", "١٢"),
    ("row ١٢٣٤", "٣٤"),
    ("row 12.٣٤", "12"),
    ("row ١٢.٣٤", "34"),
    ("row 12.³⁴", "12"),                    # nondecimal numerals fail closed
    ("row ¹²", "¹²"),
    ("row 1 , 234.5", "234.5"),
    ("row 1 , 234.5", "1"),
    ("row 1 , 23.5", "23.5"),               # malformed group consumed whole
    ("row 1 , 23.5", "1"),
    ("row 1 , 23.5", "1 , 23.5"),
    ("row 1.2.3", "3"),
    ("row 1e+", "1"),
    ("row 1 e + foo12", "1"),
    ("row 1 e + foo12", "12"),
    ("row 1e2 , 345", "1e2"),
    ("row 1e2 , 345", "345"),
    ("row 1e2e + 04", "+04"),
    ("row 12kge + 04", "+04"),
    ("row 1,,234", "1"),
    ("row 1,,234", "234"),
    ("row 1 , -234", "-234"),
    ("row 1e+ +04", "+04"),
    ("row + +.5", "+.5"),
    ("row + +.5", ".5"),
    ("row 1 , - -234", "-234"),
    ("row 12.5\u2009%", "12.5"),
    ("row 12.5\tGeV", "12.5"),
    ("row 12.5%kg", "12.5"),
    ("row 12.5%kg", "12.5%kg"),
    ("row 12.5%3", "12.5"),
    ("row 12.5%3", "3"),
    ("row 12.5GeV3", "3"),
    ("row 12.5 fb-1kg", "12.5fb"),
    ("row 12.5 fb-1kg", "1kg"),
    ("row 1.2..3", "1.2"),
    ("row 1.2..3", "3"),
    ("row 1..2", "1"),
    ("row 1..2", "2"),
    ("row 1...2", "1"),
]


@pytest.mark.parametrize("line, text", POSITIVE)
def test_match_rule_accepts(line, text):
    assert printed.find_printed([line], text, "row" if line.startswith("row") else None) == [0]


@pytest.mark.parametrize("line, text", NEGATIVE)
def test_match_rule_rejects(line, text):
    assert printed.find_printed([line], text) == []


def test_tokenizer_splits_cells_and_keeps_signs_and_suffixes():
    assert printed.numeric_tokens("row \u2212 3.2   12    34%  1, 234.5  1.23e + 04 GeV") == [
        "-3.2", "12", "34%", "1,234.5", "1.23e+04GeV",
    ]
    assert printed.numeric_text("  1.23e + 04 ") == "1.23e+04"
    assert printed.numeric_text("7 \u00b1 12") is None
    assert printed.numeric_text("---") is None
    assert printed.numeric_text("− .٥e + ٠٤\tGeV") == "-.5e+04GeV"
    assert printed.numeric_tokens("row 12kg .5e+04 1 , 234.5 12.³⁴") == [
        ".5e+04", "1,234.5",
    ]


@pytest.mark.parametrize("dash", ["\u2012", "\u2013", "\u2010", "\u2011", "\u2014"])
def test_dash_identity_is_preserved_until_literal_comparison(dash):
    assert printed.find_printed([f"row {dash}       3.2 ± 1"], "-3.2", "row") == []
    assert printed.find_printed([f"row {dash}       3.2 ± 1"], "3.2", "row") == [0]
    assert printed.find_printed([f"row 0 {dash} 5"], "-5", "row") == []
    assert printed.find_printed([f"row 0 {dash} 5"], "0-5", "row") == [0]


@pytest.mark.parametrize("space", ["", " ", "\u00a0", "\u2009", "\t", "   "])
def test_grouping_and_suffix_typesetting_whitespace(space):
    line = f"row −.5e + 04{space}GeV  1{space},{space}234.5{space}%"
    assert printed.numeric_tokens(line) == ["-.5e+04GeV", "1,234.5%"]
    assert printed.find_printed([line], "-.5e+04 GeV", "row") == [0]
    assert printed.find_printed([line], "1,234.5%", "row") == [0]
    assert printed.find_printed([line], "234.5%", "row") == []


_FORMATS = [".3g", ".6f", ".2f", ",.1f", ".2e", ".1%", ",d"]


def _typeset(text):
    """Variants pdftotext may produce for a TeX-typeset number."""
    yield text
    yield text.replace("-", "\u2212")
    yield text.replace("e+", "e + ").replace("e-", "e \u2212 ")
    yield text.replace(",", ", ")
    yield text.replace(",", " , ")


def test_property_distinct_formatted_numbers_never_match_each_other():
    rng = random.Random(20261005)
    checked = 0
    for _ in range(3000):
        fmt = rng.choice(_FORMATS)
        scale = 10 ** rng.randint(-4, 7)
        a, b = (rng.uniform(-1, 1) * scale for _ in range(2))
        if fmt == ",d":
            a, b = int(a), int(b)
        text_a, text_b = format(a, fmt), format(b, fmt)
        if text_a == text_b:
            continue
        for suffix in ("", " GeV"):
            for line_b in _typeset(text_b + suffix):
                assert printed.find_printed(["row " + line_b], text_a + suffix, "row") == [], (
                    text_a, line_b)
            for line_a in _typeset(text_a + suffix):
                assert printed.find_printed(["row " + line_a], text_a + suffix, "row") == [0], (
                    text_a, line_a)
        checked += 1
    assert checked > 2500


def test_context_must_share_the_line_as_whole_tokens():
    lines = ["ttbar 12 3", "single top 7 12", "ttbar_extra 5", "single_top 9"]
    assert printed.find_printed(lines, "12", context="ttbar") == [0]
    assert printed.find_printed(lines, "3", context="single_top") == []
    assert printed.find_printed(lines, "7", context="single_top") == [1]  # '_' extracted as space
    assert printed.find_printed(lines, "9", context="single top") == [3]
    assert printed.find_printed(lines, "5", context="ttbar") == []
    assert printed.find_printed(lines, "5", context="ttbar_extra") == [2]
    assert printed.find_printed(lines, "12", context="tt") == []
    assert printed.find_printed(["ttbar\u00e9 12", "\u00e9ttbar 12"], "12", context="ttbar") == []


def test_cell_assignment_within_a_row_is_not_verified():
    # Documented limit: value and error swapped in the row still both match.
    assert printed.find_printed(["ttbar 7 ± 12"], "12", context="ttbar") == [0]
    assert printed.find_printed(["ttbar 7 ± 12"], "7", context="ttbar") == [0]


def test_yield_table_records_each_printed_number(tmp_path):
    table = _table()
    plain = table.to_latex(key="yields")
    with printed.value_manifest(tmp_path / "values.json", "test") as manifest:
        recorded = table.to_latex(key="yields")
    assert recorded == plain  # recording never changes the output
    by_key = {r["key"]: r for r in manifest.records}
    assert len(by_key) == 8
    assert by_key["yields/ttbar/SR/value"]["text"] == "1.23e+04"
    assert by_key["yields/ttbar/SR/value"]["value"] == 12345.0
    assert by_key["yields/single_top/SR/value"]["text"] == "-3.2"
    assert by_key["yields/single_top/CR/error"]["context"] == "single_top"
    for record in manifest.records:
        assert f"${record['text']}" in recorded or f"\\pm {record['text']}$" in recorded


def test_yield_table_save_uses_base_name_as_key(tmp_path):
    with printed.value_manifest(tmp_path / "values.json", "test"):
        _table().save("yields-sr", save_path=str(tmp_path))
    keys = [r["key"] for r in json.loads((tmp_path / "values.json").read_text())["values"]]
    assert keys[0] == "yields-sr/ttbar/SR/value"


@pytest.mark.skipif(not (shutil.which("pdflatex") and shutil.which("pdftotext")),
                    reason="needs pdflatex and pdftotext")
def test_recorded_yield_text_is_found_in_the_typeset_pdf(tmp_path):
    with printed.value_manifest(tmp_path / "values.json", "test") as manifest:
        body = _table().to_latex(key="yields")
    lines = _pdf_lines(tmp_path, body)

    for record in manifest.records:
        assert printed.find_printed(lines, record["text"], record["context"]), record
    assert printed.find_printed(lines, "-3.3", "single_top") == []


def _pdf_lines(tmp_path, body):
    (tmp_path / "doc.tex").write_text(
        "\\documentclass{article}\\usepackage{booktabs}\\begin{document}\n"
        + body + "\n\\end{document}\n")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "doc.tex"], cwd=tmp_path,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["pdftotext", "-layout", "doc.pdf", "doc.txt"], cwd=tmp_path, check=True)
    return (tmp_path / "doc.txt").read_text().splitlines()


@pytest.mark.skipif(not (shutil.which("pdflatex") and shutil.which("pdftotext")),
                    reason="needs pdflatex and pdftotext")
def test_review_cases_in_extracted_synthetic_pdf(tmp_path):
    table = YieldTable([YieldEntry("row", "present", 3.2, 1.0)],
                       samples=["row"], regions=["missing", "present"])
    with printed.value_manifest(tmp_path / "values.json", "test") as manifest:
        body = table.to_latex()
    lines = _pdf_lines(tmp_path, body + r"""
\par\noindent range $0 - 5$
\par\noindent sign $-12\mathrm{kg}$
\par\noindent exponent $1.23e+04\mathrm{kg}$
\par\noindent cells $12\qquad 34\mathrm{kg}$
\par\noindent decimal $-.5$
\par\noindent decimalexp $.5e+04$
\par\noindent grouped $1\, ,\, 234.5$
\par\noindent supported $-12\mathrm{GeV}$
\par\noindent percent $12.5\,\%$
\par\noindent unit $12.5\quad\mathrm{GeV}$
\par\noindent invalidsuffix $12.5\%\mathrm{kg}$
\par\noindent invalidfraction 1.2..3
\par\noindent unitprefix 12 radial bins
""")
    # Ensure the intended PDF extraction defects, rather than another spelling,
    # were exercised. In math mode pdftotext emits minus, not the range dash.
    assert any("–" in line and "3.2" in line for line in lines)
    assert any("0 − 5" in line for line in lines)
    assert any("1 , 234.5" in line for line in lines)
    for context, text in [
        ("row", "-3.2"), ("range", "-5"), ("sign", "12kg"),
        ("exponent", "04kg"), ("cells", "1234kg"), ("decimal", ".5"),
        ("decimalexp", "04"), ("decimalexp", "+04"),
        ("grouped", "234.5"), ("grouped", "1"),
        ("invalidsuffix", "12.5"), ("invalidsuffix", "12.5%kg"),
        ("invalidfraction", "1.2"), ("invalidfraction", "3"),
    ]:
        assert printed.find_printed(lines, text, context) == [], (context, text, lines)
    for record in manifest.records:
        assert printed.find_printed(lines, record["text"], record["context"]), record
    for context, text in [
        ("decimal", "-.5"), ("decimalexp", ".5e+04"),
        ("grouped", "1,234.5"), ("supported", "-12 GeV"),
        ("percent", "12.5%"), ("unit", "12.5 GeV"),
        ("unitprefix", "12"), ("unitprefix", "12 radial bins"),
    ]:
        assert printed.find_printed(lines, text, context), (context, text, lines)
