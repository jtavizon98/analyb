"""SI prefix scaling and unit formatting for y-axis labels.

Provides tools to automatically scale bin widths to a readable range
by adjusting the SI prefix (e.g. 0.0003\\,GeV -> 0.3\\,MeV) so that
the displayed number always falls in the range [0.1, 1000).

SI prefix names and powers follow the BIPM SI Brochure, 9th edition,
section 3, Table 7: https://www.bipm.org/en/publications/si-brochure
The tables retain the yocto-through-yotta subset supported by this API.
Backslash-prefixed unit spellings are compatible with siunitx's documented
interface: https://ctan.org/pkg/siunitx (Joseph Wright, siunitx manual).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Standard SI prefix names and powers (yocto through yotta).
# ---------------------------------------------------------------------------

SI_PREFIXES = {
    r"\yotta": 24,
    r"\zetta": 21,
    r"\exa": 18,
    r"\peta": 15,
    r"\tera": 12,
    r"\giga": 9,
    r"\mega": 6,
    r"\kilo": 3,
    r"\hecto": 2,
    r"\deca": 1,
    "": 0,
    r"\deci": -1,
    r"\centi": -2,
    r"\milli": -3,
    r"\micro": -6,
    r"\nano": -9,
    r"\pico": -12,
    r"\femto": -15,
    r"\atto": -18,
    r"\zepto": -21,
    r"\yocto": -24,
}

# ---------------------------------------------------------------------------
# HEP unit families  — groups of macros whose SI prefix can be swapped
# ---------------------------------------------------------------------------

UNIT_FAMILIES = {
    "energy": [
        r"\eV", r"\keV", r"\MeV", r"\GeV", r"\TeV", r"\PeV",
    ],
    "length": [
        r"\pm", r"\nm", r"\um", r"\mm", r"\cm", r"\m", r"\km",
    ],
    "time": [
        r"\fs", r"\ps", r"\ns", r"\us", r"\ms", r"\s",
    ],
    "barn": [
        r"\ab", r"\fb", r"\pb", r"\nb", r"\ub", r"\mb", r"\b",
    ],
}

# ---------------------------------------------------------------------------
# Scale-factor lookup  (macro -> float)
# ---------------------------------------------------------------------------

# Each family follows a 10^3 spacing between consecutive members,
# starting at 10^0 for the first (smallest) member.


def _build_scales(family):
    scales = {}
    for i, macro in enumerate(family):
        scales[macro] = 1000 ** i
    return scales


UNIT_SCALES = {}
for _members in UNIT_FAMILIES.values():
    UNIT_SCALES.update(_build_scales(_members))

# Reverse lookup: macro -> family list
_MACRO_TO_FAMILY = {}
for _name, _members in UNIT_FAMILIES.items():
    for _macro in _members:
        _MACRO_TO_FAMILY[_macro] = _members


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scale_value(value, unit):
    """Scale *value* to a readable range by adjusting the SI prefix.

    Parameters
    ----------
    value : float
        Numerical value expressed in the current unit.
    unit : str or None
        siunitx macro for the current unit (e.g. ``r"\\GeV"``),
        or ``None`` for dimensionless.

    Returns
    -------
    scaled : float
        Value expressed in the best-matching unit.
    new_unit : str or None
        Chosen unit macro (may differ from the input *unit*).
    """
    family = _MACRO_TO_FAMILY.get(unit)
    if family is None:
        return value, unit

    current_scale = UNIT_SCALES.get(unit, 1)
    if current_scale == 0 or not np.isfinite(current_scale):
        return value, unit

    value_base = value * current_scale

    # Prefer the current unit if it already gives a readable number.
    if 0.1 <= abs(value) < 1000:
        return value, unit

    # Search from smallest to largest macro; the first member whose
    # scaled value lands in [0.1, 1000) is the coarsest that still
    # gives at least 0.1 — conventional for HEP plots.
    for member in family:
        member_scale = UNIT_SCALES[member]
        scaled = value_base / member_scale
        if 0.1 <= abs(scaled) < 1000:
            return scaled, member

    # Extreme value — pick the member closest to 1.
    best = min(
        family,
        key=lambda m: abs(abs(value_base / UNIT_SCALES[m]) - 1),
    )
    return value_base / UNIT_SCALES[best], best


def format_bin_width(bw, unit):
    """Format a bin width for y-axis labels, scaling the SI prefix if needed.

    Parameters
    ----------
    bw : float or None
        Bin width (may be ``None``, 0, or NaN for degenerate histograms).
    unit : str or None
        siunitx unit macro (e.g. ``r"\\GeV"``) or ``None``.

    Returns
    -------
    (value_str, unit_str) or None
        Suitable for building ``\\SI{<value_str>}{<unit_str>}``.
        Returns ``None`` when *bw* is degenerate, signalling the caller
        to omit the per-bin-width suffix entirely.
    """
    if bw is None or not np.isfinite(bw) or bw == 0:
        return None

    scaled, new_unit = scale_value(bw, unit)

    # Adaptive precision for all cases.
    # For recognised units, scale_value already brings the value into a
    # readable range; for dimensionless and unknown units we fall through.
    abs_scaled = abs(scaled)
    if abs_scaled >= 0.1:
        legacy = f"{scaled:.1f}"
    elif abs_scaled >= 0.01:
        legacy = f"{scaled:.2f}"
    elif abs_scaled >= 0.001:
        legacy = f"{scaled:.3f}"
    else:
        legacy = f"{scaled:.4f}"

    # Keep the established labels when their rounding is faithful, but do not
    # make a width look materially different from the one used in the plot.
    rounded = float(legacy)
    if np.isclose(rounded, scaled, rtol=1e-10, atol=1e-12):
        return legacy, new_unit
    return f"{scaled:.3g}", new_unit
