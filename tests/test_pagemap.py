"""Compact plotbook page map matches the book's own pagination."""

import json
import shutil
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import pytest

import analyb as ab


def _sections():
    return {
        "Particle": [(f"figs/p{i}.pdf", f"Particle observable {i}") for i in range(7)],
        "Empty": [],
        "Native": [(f"figs/n{i}.pdf", f"Native stage chart {i}") for i in range(3)],
    }


def test_page_map_lists_every_slot_in_order(tmp_path):
    book = tmp_path / "book.tex"
    tex_without = ab.write_compact_plotbook(book, _sections(), rows_per_page=3,
                                            section_rows={"Native": 2})
    plain = book.read_text()
    ab.write_compact_plotbook(book, _sections(), rows_per_page=3,
                              section_rows={"Native": 2}, page_map=True)
    assert book.read_text() == plain  # the map never changes the TeX
    assert tex_without == str(book)

    record = json.loads((tmp_path / "book.pagemap.json").read_text())
    assert record["schema"] == "analyb.plotbook_page_map"
    assert record["schema_version"] == 1
    assert (record["tex"], record["pdf"]) == ("book.tex", "book.pdf")
    assert record["page_count"] == 3 == plain.count("\\clearpage") + 1
    pages = record["pages"]
    assert [(p["page"], p["section"], p["rows"], len(p["slots"])) for p in pages] == [
        (1, "Particle", 3, 6), (2, "Particle", 3, 1), (3, "Native", 2, 3),
    ]
    assert pages[0]["slots"][:3] == [
        {"figure": "figs/p0.pdf", "caption": "Particle observable 0", "row": 1, "column": 1},
        {"figure": "figs/p1.pdf", "caption": "Particle observable 1", "row": 1, "column": 2},
        {"figure": "figs/p2.pdf", "caption": "Particle observable 2", "row": 2, "column": 1},
    ]
    assert pages[2]["slots"][2] == {
        "figure": "figs/n2.pdf", "caption": "Native stage chart 2", "row": 2, "column": 1,
    }
    figures = [slot["figure"] for page in pages for slot in page["slots"]]
    assert figures == [f for entries in _sections().values() for f, _ in entries]


def test_no_page_map_by_default(tmp_path):
    ab.write_compact_plotbook(tmp_path / "book.tex", _sections())
    assert not (tmp_path / "book.pagemap.json").exists()


@pytest.mark.skipif(not (shutil.which("pdflatex") and shutil.which("pdftotext")),
                    reason="needs pdflatex and pdftotext")
def test_page_map_matches_compiled_pages(tmp_path):
    (tmp_path / "figs").mkdir()
    with matplotlib.rc_context():
        for entries in _sections().values():
            for figure, _ in entries:
                fig, ax = plt.subplots(figsize=(4.5, 5.4))
                ax.plot([0, 1], [0, 1])
                fig.savefig(tmp_path / figure)
                plt.close(fig)
    ab.write_compact_plotbook(tmp_path / "book.tex", _sections(), rows_per_page=3,
                              section_rows={"Native": 2}, page_map=True)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "book.tex"], cwd=tmp_path,
                   check=True, stdout=subprocess.DEVNULL)
    record = json.loads((tmp_path / "book.pagemap.json").read_text())
    for page in record["pages"]:
        text = subprocess.run(
            ["pdftotext", "-f", str(page["page"]), "-l", str(page["page"]), "book.pdf", "-"],
            cwd=tmp_path, check=True, capture_output=True, text=True).stdout
        assert page["section"] in text
        for slot in page["slots"]:
            assert slot["caption"] in text
    beyond = subprocess.run(
        ["pdftotext", "-f", str(record["page_count"] + 1), "book.pdf", "-"],
        cwd=tmp_path, capture_output=True, text=True)
    assert beyond.returncode != 0 or not beyond.stdout.strip()  # no extra page
