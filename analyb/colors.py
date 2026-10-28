"""Catppuccin palette and custom colormaps, hex-only (no external package)."""

import matplotlib.colors as mcolors


def create_smooth_colormap(hex_colors, name="catppuccin_custom"):
    rgb_colors = [mcolors.hex2color(c) for c in hex_colors]
    return mcolors.LinearSegmentedColormap.from_list(name, rgb_colors)


# Catppuccin Latte palette, Copyright (c) 2021 Catppuccin (MIT).
# Source and full notice: ../THIRD_PARTY_NOTICES.md.
catppuccin_colors = {
    "rosewater": mcolors.hex2color("#dc8a78"),
    "flamingo": mcolors.hex2color("#dd7871"),
    "pink": mcolors.hex2color("#ea76cb"),
    "mauve": mcolors.hex2color("#8839ef"),
    "red": mcolors.hex2color("#d20f39"),
    "maroon": mcolors.hex2color("#e64553"),
    "peach": mcolors.hex2color("#fe640b"),
    "yellow": mcolors.hex2color("#df8e1d"),
    "green": mcolors.hex2color("#40a02b"),
    "teal": mcolors.hex2color("#179299"),
    "sky": mcolors.hex2color("#04a5e5"),
    "sapphire": mcolors.hex2color("#2096d3"),
    "blue": mcolors.hex2color("#1e66f5"),
    "lavender": mcolors.hex2color("#7287fd"),
    "text": mcolors.hex2color("#4c4f69"),
    "subtext1": mcolors.hex2color("#5c5f77"),
    "subtext0": mcolors.hex2color("#6c6f85"),
    "overlay2": mcolors.hex2color("#7d7f99"),
    "overlay1": mcolors.hex2color("#8e8f99"),
    "overlay0": mcolors.hex2color("#9ca0b0"),
    "surface2": mcolors.hex2color("#ccd0da"),
    "surface1": mcolors.hex2color("#bcc0cc"),
    "surface0": mcolors.hex2color("#acb0be"),
    "base": mcolors.hex2color("#eff1f5"),
    "mantle": mcolors.hex2color("#e6e9ef"),
    "crust": mcolors.hex2color("#dce0e8"),
}

color_names = list(catppuccin_colors.keys())

# Categorical palettes the plot helpers draw together, as catppuccin names in
# draw order. Shared by the plotting helpers.
PALETTES = {
    "stack": ("blue", "green", "peach", "mauve", "teal", "yellow", "lavender"),
    "comparison": ("red", "blue", "green"),
    "comparison_many": ("red", "blue", "green", "mauve", "teal", "peach"),
    "closure": ("blue", "green", "peach", "mauve", "teal", "yellow", "lavender", "maroon"),
    "systematics": (
        "blue", "sky", "green", "yellow", "peach", "mauve",
        "red", "teal", "lavender", "pink", "maroon", "sapphire",
    ),
}


# Colormap definitions.
light_coolwarm_hex_colors = ["#1e66f5", "#dbc3fa", "#d20f39"]
infaded_coolwarm_hex_colors = ["#78a3f9", "#d20f39"]
outfaded_coolwarm_hex_colors = ["#1e66f5", "#df5774"]
coolwarm_hex_colors = ["#1e66f5", "#ca9ee6", "#d20f39"]
nova_hex_colors = ["#0951dc", "#822fef", "#f04269", "#ef7680", "#f6a2e1"]
superheat_hex_colors = ["#0848c4", "#7111ee", "#f04269", "#ef7680", "#f6a2e1"]

catppuccin_colorbars = {
    "light_coolwarm": create_smooth_colormap(
        light_coolwarm_hex_colors, name="light_coolwarm"
    ),
    "infaded_coolwarm": create_smooth_colormap(
        infaded_coolwarm_hex_colors, name="infaded_coolwarm"
    ),
    "outfaded_coolwarm": create_smooth_colormap(
        outfaded_coolwarm_hex_colors, name="outfaded_coolwarm"
    ),
    "coolwarm": create_smooth_colormap(coolwarm_hex_colors, name="coolwarm"),
    "nova": create_smooth_colormap(nova_hex_colors, name="nova"),
    "superheat": create_smooth_colormap(superheat_hex_colors, name="superheat"),
}

current_colorbar = catppuccin_colorbars["superheat"]


def _register_colormaps_with_matplotlib():
    """Register analyb colormaps and make `superheat` the matplotlib default.

    matplotlib's rc file is parsed at startup before any user imports, so
    `image.cmap: superheat` in matplotlibrc cannot resolve to our custom cmap.
    We register the maps here so they are available by name, and we also set
    `image.cmap` to `current_colorbar` so any code that relies on the rc
    (e.g. imshow without an explicit cmap) uses superheat by default.
    """
    import matplotlib as mpl
    for name, cmap in catppuccin_colorbars.items():
        if name not in mpl.colormaps:
            mpl.colormaps.register(cmap, name=name)
    try:
        import matplotlib.pyplot as plt
        plt.rcParams["image.cmap"] = current_colorbar.name
    except ImportError:
        pass


_register_colormaps_with_matplotlib()
