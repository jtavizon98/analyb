import pytest

from analyb.units import format_bin_width


@pytest.mark.parametrize(
    "width, expected",
    [(0.13, "0.13"), (0.065, "0.065"), (50.0, "50.0")],
)
def test_format_bin_width_preserves_meaningful_fractional_width(width, expected):
    assert format_bin_width(width, None) == (expected, None)
