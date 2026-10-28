"""Render-time figure lint: mechanical geometry checks on a laid-out figure.

``check_figure`` lays the figure out once with
``Figure.draw_without_rendering()`` and measures visible text, legends and
plotted data afterwards. Each :class:`Finding` stands alone: there is no
combined score, and an empty result never certifies that a figure is readable.
Run it after the final limits, scales and overlays, like
``reserve_annotation_headroom``.

``Finding.bbox`` is ``(x0, y0, x1, y1)`` in figure inches measured from the
lower-left corner of the figure canvas (the saved page without
``bbox_inches="tight"``). Lengths in values and messages are typographic
points.
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.collections import LineCollection, PathCollection, QuadMesh
from matplotlib.colors import to_rgba, to_rgba_array
from matplotlib.figure import SubFigure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

logger = logging.getLogger(__name__)

TOLERANCE_PT = 0.5
"""Geometric slack for canvas and overlap checks, in points."""

_MATHDEFAULT = re.compile(r"\\mathdefault\{([^{}]*)\}")

MIN_FONT_PT = 5.0
"""Blocking font floor at final placement (``font-below-floor``)."""
FLAG_FONT_PT = 7.0
"""Advisory font size at final placement (``font-below-advisory``)."""

SIDECAR_SCHEMA = "analyb.figcheck"
SIDECAR_SCHEMA_VERSION = 1
CHECKS = ("text-outside-canvas", "text-overlap", "text-over-data", "axis-unlabelled",
          "font-below-floor", "font-below-advisory")
"""Every check :func:`check_figure` can run; the sidecar lists those it ran."""


@dataclass(frozen=True)
class Finding:
    """One mechanical defect found by :func:`check_figure`.

    ``severity`` is ``"fail"`` (a mechanical defect) or ``"flag"`` (advisory).
    ``bbox`` is in figure inches from the canvas lower-left corner; ``value``
    is the measured number described by ``message`` (or ``None``).
    """

    check: str
    severity: str
    where: str
    bbox: tuple
    value: float
    message: str

    def __post_init__(self):
        if self.bbox is not None:
            object.__setattr__(self, "bbox", tuple(float(v) for v in self.bbox))
        if self.value is not None:
            object.__setattr__(self, "value", float(self.value))

    def to_dict(self):
        data = asdict(self)
        data["bbox"] = None if self.bbox is None else list(self.bbox)
        return data


class FigureCheckError(RuntimeError):
    """Raised by the ``"raise"`` save mode after saving when checks fail."""

    def __init__(self, path, findings):
        self.path = path
        self.findings = list(findings)
        failures = [f for f in self.findings if f.severity == "fail"]
        detail = "; ".join(f"{f.check} at {f.where}: {f.message}" for f in failures)
        super().__init__(f"{len(failures)} figure check failure(s) in {path}: {detail}")


@dataclass
class _Item:
    where: str
    extent: Bbox
    text: Text = None  # None for a legend box
    legend: Legend = None  # owning legend for legend texts
    overlay: bool = False  # candidate for text-over-data


# --------------------------------------------------------------------------
# Collection of visible text and legend geometry
# --------------------------------------------------------------------------

def _snippet(text, limit=40):
    text = " ".join(_MATHDEFAULT.sub(r"\1", text).split())
    if text.startswith("$") and text.endswith("$") and "\\" not in text:
        text = text.strip("$")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _all_axes(fig):
    axes = list(fig.axes)
    for ax in fig.findobj(lambda a: isinstance(a, Axes)):
        if ax not in axes:
            axes.append(ax)
    return axes


def _finite(bbox):
    return bbox is not None and np.all(np.isfinite(bbox.extents))


def _text_item(text, where, **kwargs):
    if not text.get_visible() or not text.get_text().strip():
        return None
    extent = text.get_window_extent()
    if not _finite(extent) or extent.width <= 0 or extent.height <= 0:
        return None
    return _Item(f"{where} '{_snippet(text.get_text())}'", extent, text=text, **kwargs)


def _axis_items(axis, where):
    ax = axis.axes
    name = axis.axis_name
    items = [
        _text_item(axis.label, f"{where} {name}label"),
        _text_item(axis.get_offset_text(), f"{where} {name} offset text"),
    ]
    tol = 0.5  # px: ticks drawn at the axes limits stay included
    for locs, get_ticks in (
        (axis.get_majorticklocs(), axis.get_major_ticks),
        (axis.get_minorticklocs(), axis.get_minor_ticks),
    ):
        locs = np.asarray(locs, dtype=float)
        if not locs.size:
            continue
        if name == "x":
            pos = ax.get_xaxis_transform().transform(np.column_stack([locs, np.zeros_like(locs)]))[:, 0]
            lo, hi = ax.bbox.x0, ax.bbox.x1
        elif name == "y":
            pos = ax.get_yaxis_transform().transform(np.column_stack([np.zeros_like(locs), locs]))[:, 1]
            lo, hi = ax.bbox.y0, ax.bbox.y1
        else:
            continue
        for tick, p in zip(get_ticks(len(locs)), pos):
            if not (np.isfinite(p) and lo - tol <= p <= hi + tol) or not tick.get_visible():
                continue
            for label in (tick.label1, tick.label2):
                items.append(_text_item(label, f"{where} {name}tick label"))
    return items


def _legend_box(legend):
    """Legend content box; the frame when drawn, else without border padding."""
    box = legend.get_window_extent()
    if legend.get_frame_on() and legend.get_frame().get_visible():
        return box
    pad = legend.borderpad * legend.prop.get_size_in_points() * legend.figure.dpi / 72
    x0, y0, x1, y1 = box.extents
    if x1 - x0 > 2 * pad and y1 - y0 > 2 * pad:
        return Bbox.from_extents(x0 + pad, y0 + pad, x1 - pad, y1 - pad)
    return box


def _legend_items(legend, where):
    if not legend.get_visible():
        return []
    box = _legend_box(legend)
    if not _finite(box):
        return []
    items = [_Item(where, box, legend=legend, overlay=True)]
    for text in [*legend.get_texts(), legend.get_title()]:
        items.append(_text_item(text, f"{where} text", legend=legend))
    return items


def _container_items(children, where):
    items = []
    for child in children:
        if isinstance(child, (Axes, Axis)):
            continue
        if isinstance(child, Legend):
            items += _legend_items(child, f"{where} legend")
        elif isinstance(child, SubFigure):
            if child.get_visible():
                items += _container_items(child.get_children(), "subfigure")
        else:
            for text in child.findobj(lambda a: isinstance(a, Text)):
                items.append(_text_item(text, f"{where} text", overlay=True))
    return items


def _collect_items(fig, axes):
    items = []
    for index, ax in enumerate(axes):
        if not ax.get_visible():
            continue
        where = f"axes[{index}]"
        if ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                if axis.get_visible():
                    items += _axis_items(axis, where)
        items += _container_items(ax.get_children(), where)
    items += _container_items(fig.get_children(), "figure")
    return [item for item in items if item is not None]


def _rotated(item):
    return item.text is not None and item.text.get_rotation() % 90 != 0


def _corners(item):
    """Display-space corners of an item's box, rotated for rotated text.

    The axis-aligned extent of a rotated rectangle is centred on it, so the
    unrotated size (measured by briefly resetting the rotation) and that
    centre fix the rectangle.
    """
    e = item.extent
    if not _rotated(item):
        return np.array([[e.x0, e.y0], [e.x1, e.y0], [e.x1, e.y1], [e.x0, e.y1]])
    text = item.text
    angle = text.get_rotation()
    text.set_rotation(0)
    try:
        flat = text.get_window_extent()
    finally:
        text.set_rotation(angle)
    half = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [flat.width / 2, flat.height / 2]
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return half @ np.array([[c, s], [-s, c]]) + [(e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2]


def _penetration(p, q):
    """Smallest overlap of two convex polygons over their edge normals (0 if apart)."""
    depth = np.inf
    for poly in (p, q):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            norm = np.hypot(*edge)
            if norm == 0:
                continue
            axis = np.array([-edge[1], edge[0]]) / norm
            pa, qa = p @ axis, q @ axis
            overlap = min(pa.max(), qa.max()) - max(pa.min(), qa.min())
            if overlap <= 0:
                return 0.0
            depth = min(depth, overlap)
    return float(depth)


# --------------------------------------------------------------------------
# Plotted data in display coordinates
# --------------------------------------------------------------------------

def _clip_region(artist, box):
    if artist.get_clip_on() and artist.get_clip_box() is not None:
        return Bbox.intersection(box, artist.get_clip_box())
    return box


def _inside(points, box):
    return int(np.count_nonzero(
        (points[:, 0] >= box.x0) & (points[:, 0] <= box.x1)
        & (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1)
    ))


def _crossing_segments(points, box):
    """Count segments between consecutive finite points that cross ``box``
    with neither endpoint inside it (Liang-Barsky clipping)."""
    if len(points) < 2:
        return 0
    p0, p1 = points[:-1], points[1:]
    finite = np.all(np.isfinite(p0), axis=1) & np.all(np.isfinite(p1), axis=1)
    p0, p1 = p0[finite], p1[finite]
    d = p1 - p0
    t0 = np.zeros(len(p0))
    t1 = np.ones(len(p0))
    ok = np.ones(len(p0), dtype=bool)
    for p, q in (
        (-d[:, 0], p0[:, 0] - box.x0), (d[:, 0], box.x1 - p0[:, 0]),
        (-d[:, 1], p0[:, 1] - box.y0), (d[:, 1], box.y1 - p0[:, 1]),
    ):
        parallel = p == 0
        ok &= ~(parallel & (q < 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = q / p
        t0 = np.where(~parallel & (p < 0), np.maximum(t0, r), t0)
        t1 = np.where(~parallel & (p > 0), np.minimum(t1, r), t1)
    ok &= t0 <= t1

    def contained(pts):
        return ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
                & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1))

    return int(np.count_nonzero(ok & ~contained(p0) & ~contained(p1)))


def _visible_alpha(colors):
    try:
        rgba = to_rgba_array(colors)
    except (TypeError, ValueError):
        return False
    return rgba.size > 0 and bool(np.any(rgba[:, 3] > 0))


def _label(artist):
    label = artist.get_label()
    return None if not label or label.startswith("_") else label


def _container_labels(ax):
    """Map artists of labelled containers (errorbar, bar) to the container label."""
    labels = {}
    for container in ax.containers:
        label = _label(container)
        if label:
            for child in container.get_children():
                labels[id(child)] = label
    return labels


def _painted(artist):
    """Visible with some nonzero artist alpha (per-element alpha arrays allowed;
    collections filter their elements by colour alpha afterwards)."""
    alpha = artist.get_alpha()
    return artist.get_visible() and (alpha is None or bool(np.any(np.asarray(alpha) > 0)))


def _line_color_painted(color, alpha):
    """Line2D colours: an explicit artist alpha replaces the colour's alpha,
    except for "none", exactly as ``matplotlib.colors.to_rgba(color, alpha)``."""
    try:
        return to_rgba(color, alpha)[3] > 0
    except (TypeError, ValueError):
        return False


def _cycled(values, n):
    """Matplotlib cycles per-element properties over the elements."""
    values = np.asarray(values)
    return np.resize(values, n) if values.size else np.zeros(n, dtype=values.dtype)


def _point_mask(coll, n):
    """Scatter points that paint something: nonzero size and a visible face or edge."""
    sizes = np.asarray(coll.get_sizes(), dtype=float)
    sized = _cycled(sizes > 0, n) if sizes.size else np.ones(n, dtype=bool)

    def alpha(colors):
        rgba = to_rgba_array(colors)
        return _cycled(rgba[:, 3] > 0, n) if len(rgba) else np.zeros(n, dtype=bool)

    stroked = alpha(coll.get_edgecolor()) & _cycled(np.asarray(coll.get_linewidths()) > 0, n)
    return sized & (alpha(coll.get_facecolor()) | stroked)


def _data_shapes(ax):
    """Yield (artist, kind, display arrays) for plotted data that paints pixels.

    ``kind`` is ``"points"`` (marker centres), ``"lines"`` (polylines) or
    ``"area"`` (filled display paths). Artists that are invisible, fully
    transparent, zero-width, or zero-size are skipped.
    """
    for line in ax.lines:
        if not _painted(line):
            continue
        alpha = line.get_alpha()
        has_line = (line.get_linestyle() not in ("None", "none", " ", "")
                    and line.get_linewidth() > 0 and _line_color_painted(line.get_color(), alpha))
        has_marker = (line.get_marker() not in (None, "None", "none", " ", "")
                      and line.get_markersize() > 0
                      and (_line_color_painted(line.get_markerfacecolor(), alpha)
                           or (line.get_markeredgewidth() > 0
                               and _line_color_painted(line.get_markeredgecolor(), alpha))))
        if not (has_line or has_marker):
            continue
        points = line.get_transform().transform(line.get_path().vertices)
        yield line, ("lines" if has_line else "points"), [points]
    for coll in ax.collections:
        if not _painted(coll) or isinstance(coll, QuadMesh):
            continue
        if isinstance(coll, PathCollection):
            offsets = np.asarray(coll.get_offsets(), dtype=float)
            if offsets.size:
                offsets = offsets[_point_mask(coll, len(offsets))]
            if offsets.size:
                yield coll, "points", [coll.get_offset_transform().transform(offsets)]
            continue
        transform = coll.get_transform()
        stroked = (_visible_alpha(coll.get_edgecolor())
                   and bool(np.any(np.asarray(coll.get_linewidths()) > 0)))
        if not isinstance(coll, LineCollection) and _visible_alpha(coll.get_facecolor()):
            yield coll, "area", [transform.transform_path(p) for p in coll.get_paths()]
        elif stroked:
            yield coll, "lines", [transform.transform(p.vertices) for p in coll.get_paths()]
    for patch in ax.patches:
        if not _painted(patch):
            continue
        path = patch.get_transform().transform_path(patch.get_path())
        if patch.get_fill() and _visible_alpha(patch.get_facecolor()):
            yield patch, "area", [path]
        elif patch.get_linewidth() > 0 and _visible_alpha(patch.get_edgecolor()):
            yield patch, "lines", path.to_polygons(closed_only=False)


def _covered(item, axes):
    """Covered line/marker vertices, crossing segments, and filled areas."""
    vertices = crossings = area_vertices = 0
    line_labels, area_labels = [], []
    for ax in axes:
        if not ax.get_visible():
            continue
        names = _container_labels(ax)
        for artist, kind, shapes in _data_shapes(ax):
            name = _label(artist) or names.get(id(artist)) or type(artist).__name__
            region = _clip_region(artist, item.extent)
            if region is None or region.width <= 0 or region.height <= 0:
                continue
            if kind == "area":
                hit = [p for p in shapes if p.intersects_bbox(region, filled=True)]
                if hit:
                    area_vertices += sum(
                        _inside(poly, region) for p in hit for poly in p.to_polygons(closed_only=False)
                    )
                    area_labels.append(name)
                continue
            n_in = sum(_inside(p, region) for p in shapes)
            n_cross = sum(_crossing_segments(p, region) for p in shapes) if kind == "lines" else 0
            if n_in or n_cross:
                vertices += n_in
                crossings += n_cross
                line_labels.append(name)
    return vertices, crossings, line_labels, area_vertices, area_labels


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def _inches(bbox, dpi):
    return tuple(round(float(v) / dpi, 4) for v in bbox.extents)


def check_figure(fig, *, final_width_cm=None, min_font_pt=MIN_FONT_PT, flag_font_pt=FLAG_FONT_PT):
    """Return mechanical layout findings for ``fig`` after one layout pass.

    Checks visible text, legends and plotted data; empty strings are ignored.

    - ``text-outside-canvas`` (fail): a text or legend box extends beyond the
      figure canvas, so it is clipped in the saved file.
    - ``text-overlap`` (fail): two distinct texts (legends count as one box)
      intersect.
    - ``text-over-data`` (fail): a legend or in-axes/figure text covers line
      or marker vertices or is crossed by a line segment; ``value`` counts the
      covered vertices.  Covering a filled area (bars, bands, filled steps) is
      a ``flag``.
    - ``axis-unlabelled`` (fail): a visible x or y axis shows tick labels but
      has no axis label. An axis whose tick labels are hidden (the upper panel
      of a shared x axis) needs none.
    - ``font-below-floor`` (fail): only with ``final_width_cm``; the text's
      base font size scaled to the final width is below ``min_font_pt``.
      ``font-below-advisory`` (flag): at least ``min_font_pt`` but below
      ``flag_font_pt``.
      MathText sub/superscripts are not measured separately.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    final_width_cm : float, optional
        Width of the figure as placed in the final document. Without it the
        font check is skipped.
    min_font_pt : float
        Blocking font floor at final placement (default 5 pt).
    flag_font_pt : float
        Advisory font size at final placement (default 7 pt).

    Returns
    -------
    list of Finding
    """
    if final_width_cm is not None and not (np.isfinite(final_width_cm) and final_width_cm > 0):
        raise ValueError("final_width_cm must be a positive finite number")
    if not (np.isfinite(min_font_pt) and min_font_pt > 0):
        raise ValueError("min_font_pt must be a positive finite number")
    if not (np.isfinite(flag_font_pt) and flag_font_pt >= min_font_pt):
        raise ValueError("flag_font_pt must be finite and at least min_font_pt")

    fig.draw_without_rendering()
    dpi = fig.dpi
    px_per_pt = dpi / 72
    tol = TOLERANCE_PT * px_per_pt
    axes = _all_axes(fig)
    items = _collect_items(fig, axes)
    findings = []

    canvas = fig.bbox
    for item in items:
        e = item.extent
        excess = {
            "left": canvas.x0 - e.x0, "right": e.x1 - canvas.x1,
            "bottom": canvas.y0 - e.y0, "top": e.y1 - canvas.y1,
        }
        sides = [side for side, value in excess.items() if value > tol]
        if sides:
            over = max(excess[side] for side in sides) / px_per_pt
            findings.append(Finding(
                "text-outside-canvas", "fail", item.where, _inches(e, dpi), round(over, 2),
                f"extends {over:.1f} pt beyond the {'/'.join(sides)} canvas edge; "
                "clipped in the saved file",
            ))

    # Legends take part as one box against other text; their own entries are
    # compared only with sibling entries, never with the enclosing box.
    boxes = [item for item in items if item.legend is None or item.text is None]
    pairs = [(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]]
    entries = [item for item in items if item.legend is not None and item.text is not None]
    pairs += [(a, b) for i, a in enumerate(entries) for b in entries[i + 1:]
              if a.legend is b.legend]
    for a, b in pairs:
        overlap = Bbox.intersection(a.extent, b.extent)
        if overlap is None or overlap.width <= tol or overlap.height <= tol:
            continue
        w, h = overlap.width / px_per_pt, overlap.height / px_per_pt
        if _rotated(a) or _rotated(b):
            # Axis-aligned boxes of rotated labels overlap long before the
            # labels do; compare the rotated rectangles instead.
            depth = _penetration(_corners(a), _corners(b)) / px_per_pt
            if depth <= TOLERANCE_PT:
                continue
            value, message = depth, f"rotated text boxes overlap by {depth:.1f} pt"
        else:
            value, message = min(w, h), f"overlap of {w:.1f} x {h:.1f} pt"
        findings.append(Finding(
            "text-overlap", "fail", f"{a.where} x {b.where}", _inches(overlap, dpi),
            round(value, 2), message,
        ))

    for item in items:
        if not item.overlay:
            continue
        vertices, crossings, line_labels, area_vertices, area_labels = _covered(item, axes)
        if vertices or crossings:
            findings.append(Finding(
                "text-over-data", "fail", item.where, _inches(item.extent, dpi), vertices,
                f"covers {vertices} line/marker vertices and is crossed by {crossings} "
                f"further segments of {', '.join(line_labels)}",
            ))
        if area_labels:
            findings.append(Finding(
                "text-over-data", "flag", item.where, _inches(item.extent, dpi), area_vertices,
                f"covers filled area of {', '.join(area_labels)} "
                f"({area_vertices} outline vertices inside)",
            ))

    for index, ax in enumerate(axes):
        if not (ax.get_visible() and ax.axison):
            continue
        for axis in (ax.xaxis, ax.yaxis):
            if not axis.get_visible():
                continue
            ticks = [item for item in _axis_items(axis, "")
                     if item is not None and "tick label" in item.where]
            label = axis.label
            if ticks and not (label.get_visible() and label.get_text().strip()):
                findings.append(Finding(
                    "axis-unlabelled", "fail", f"axes[{index}] {axis.axis_name} axis",
                    _inches(ax.get_window_extent(), dpi), None,
                    f"{axis.axis_name} axis shows {len(ticks)} tick labels but has no axis label",
                ))

    if final_width_cm is not None:
        scale = final_width_cm / (fig.get_figwidth() * 2.54)
        for item in items:
            if item.text is None:
                continue
            size = item.text.get_fontsize()
            effective = size * scale
            if effective < min_font_pt - 1e-9:
                check, severity, limit = "font-below-floor", "fail", f"floor {min_font_pt:g} pt"
            elif effective < flag_font_pt - 1e-9:
                check, severity, limit = "font-below-advisory", "flag", f"advisory {flag_font_pt:g} pt"
            else:
                continue
            findings.append(Finding(
                check, severity, item.where, _inches(item.extent, dpi), round(effective, 2),
                f"{effective:.2f} pt at {final_width_cm:g} cm final width "
                f"({size:g} pt in source); {limit}",
            ))
    return findings


# --------------------------------------------------------------------------
# Save hook
# --------------------------------------------------------------------------

def sidecar_path(pdf_path):
    """Return ``<pdf stem>.figcheck.json`` beside ``pdf_path``."""
    return os.path.splitext(pdf_path)[0] + ".figcheck.json"


def write_sidecar(
    fig, pdf_path, findings, *, final_width_cm=None,
    min_font_pt=MIN_FONT_PT, flag_font_pt=FLAG_FONT_PT,
):
    """Log ``findings`` and write ``<pdf stem>.figcheck.json`` beside ``pdf_path``.

    ``findings`` must come from :func:`check_figure` on ``fig`` immediately
    before it was saved to ``pdf_path``; the sidecar records the PDF hash so a
    collector can tell whether it still describes that file.
    """
    for f in findings:
        logger.warning("figcheck %s %s [%s] %s: %s", os.path.basename(pdf_path),
                       f.check, f.severity, f.where, f.message)
    with open(pdf_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    width, height = (float(v) for v in fig.get_size_inches())
    record = {
        "schema": SIDECAR_SCHEMA,
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "pdf": os.path.basename(pdf_path),
        "pdf_sha256": digest,
        "figure_size_in": [width, height],
        "figure_width_cm": round(width * 2.54, 4),
        "final_width_cm": final_width_cm,
        "min_font_pt": min_font_pt,
        "flag_font_pt": flag_font_pt,
        "bbox_units": "figure inches from the canvas lower-left corner",
        "checks_skipped": (
            [] if final_width_cm is not None else ["font-below-floor", "font-below-advisory"]
        ),
        "checks": [c for c in CHECKS
                   if final_width_cm is not None or not c.startswith("font-below")],
        "findings": [f.to_dict() for f in findings],
    }
    path = sidecar_path(pdf_path)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)
    return path
