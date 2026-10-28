"""Yield tables: integral per (sample, region) exported to text/LaTeX."""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

from .histogram import Hist1D


class YieldEntry:
    def __init__(self, sample, region, value, error):
        self.sample = sample
        self.region = region
        self.value = float(value)
        self.error = float(error)

    def __repr__(self):
        return f"YieldEntry({self.sample!r}, {self.region!r}, {self.value:.3g} ± {self.error:.3g})"


def _value_error(hist):
    if isinstance(hist, Hist1D):
        return hist.integral(), hist.integral_error()
    if isinstance(hist, tuple) and len(hist) == 2:
        counts, err = hist
        sum_counts = float(np.sum(counts))
        if hasattr(err, "__len__"):
            sum_err = float(np.sqrt(np.sum(np.asarray(err) ** 2)))
        else:
            sum_err = float(err)
        return sum_counts, sum_err
    val = float(hist)
    return val, 0.0


def yield_table(
    entries,
    samples=None,
    regions=None,
    column="sample",
):
    """Build a YieldTable from a mapping {(sample, region): Hist1D | (value, error) | number}.

    Parameters
    ----------
    samples, regions : list[str] or None
        Ordering; auto-detected from entries if None.
    column : "sample" | "region"
        Rows of the printed table; the other axis becomes columns.
    """
    if samples is None:
        samples = sorted({k[0] for k in entries})
    if regions is None:
        regions = sorted({k[1] for k in entries})

    rows = []
    for (s, r), hist in entries.items():
        v, e = _value_error(hist)
        rows.append(YieldEntry(s, r, v, e))

    return YieldTable(rows, samples=samples, regions=regions, row_axis=column)


class YieldTable:
    def __init__(self, rows, samples, regions, row_axis="sample"):
        self.rows = rows
        self.samples = list(samples)
        self.regions = list(regions)
        self.row_axis = row_axis

    @staticmethod
    def _escape_tex(s):
        """Escape LaTeX special characters in a string."""
        if not isinstance(s, str):
            return s
        replacements = {
            "\\": r"\textbackslash{}",
            "{": r"\{",
            "}": r"\}",
            "_": r"\_",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(char, char) for char in s)

    def _grid(self):
        """Return dict {(row, col): YieldEntry}."""
        grid = {}
        for entry in self.rows:
            if self.row_axis == "sample":
                key = (entry.sample, entry.region)
            else:
                key = (entry.region, entry.sample)
            grid[key] = entry
        return grid

    def to_text(self):
        grid = self._grid()
        if self.row_axis == "sample":
            row_labels = self.samples
            col_labels = self.regions
        else:
            row_labels = self.regions
            col_labels = self.samples
        lines = []
        header = [""] + col_labels
        lines.append("\t".join(header))
        for r in row_labels:
            cells = [r]
            for c in col_labels:
                e = grid.get((r, c))
                if e is None:
                    cells.append("--")
                else:
                    cells.append(f"{e.value:.3g} ± {e.error:.3g}")
            lines.append("\t".join(cells))
        return "\n".join(lines)

    def to_latex(self, *, key="yield", source=None):
        """LaTeX tabular of ``value ± error`` cells.

        When a :func:`analyb.printed.value_manifest` is active, every printed
        value and error is recorded as ``{key}/{row}/{column}/value|error``
        with the row label as ``context``; ``source`` names a manifest source
        role. The output is identical with or without a manifest.
        """
        from .printed import record_value

        grid = self._grid()
        if self.row_axis == "sample":
            row_labels = self.samples
            col_labels = self.regions
        else:
            row_labels = self.regions
            col_labels = self.samples
        ncol = len(col_labels) + 1
        spec = "l" + "l" * len(col_labels)
        lines = []
        lines.append(r"\begin{tabular}{" + spec + "}")
        lines.append(r"\toprule")
        escaped_cols = [self._escape_tex(c) for c in col_labels]
        lines.append(" & ".join([""] + escaped_cols) + r" \\")
        lines.append(r"\midrule")
        for r in row_labels:
            cells = [self._escape_tex(r)]
            for c in col_labels:
                e = grid.get((r, c))
                if e is None:
                    cells.append("--")
                else:
                    value = f"{e.value:.3g}"
                    error = f"{e.error:.3g}"
                    for part, number, text in (("value", e.value, value), ("error", e.error, error)):
                        record_value(f"{key}/{r}/{c}/{part}", number, text,
                                     context=r, source=source)
                    cells.append(rf"${value} \pm {error}$")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    def save(self, base_name, save_path=None, *, source=None):
        from . import config

        if save_path is None:
            save_path = config.get_save_path() or os.path.join(os.getcwd(), "plots")
        os.makedirs(save_path, exist_ok=True)
        txt = os.path.join(save_path, f"{base_name}.txt")
        tex = os.path.join(save_path, f"{base_name}.tex")
        with open(txt, "w", encoding="utf-8") as fp:
            fp.write(self.to_text())
        with open(tex, "w", encoding="utf-8") as fp:
            fp.write(self.to_latex(key=base_name, source=source))
        logger.info("Saved yield table: %s / %s", txt, tex)
        return txt, tex

    def __repr__(self):
        return f"YieldTable(samples={len(self.samples)}, regions={len(self.regions)})"
