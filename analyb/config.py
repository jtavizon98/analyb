"""Global configuration: figure sizes, script id, output root file."""

import os
import time
import contextvars

fig_size_square = (4.5, 4)
fig_size_2d = (5.2, 4)
fig_size_wide = (7, 3)

# Script id: identifies every plot/file produced by a script.
_script_id_var = contextvars.ContextVar("analyb_script_id", default=None)


def set_script_id(script_id):
    _script_id_var.set(script_id)


def get_script_id():
    sid = _script_id_var.get()
    if sid is None:
        sid = time.strftime("%Y%m%d-%H%M%S")
    return sid


# Save path for figures (default: ./plots relative to cwd).
_save_path_var = contextvars.ContextVar("analyb_save_path", default=None)


def set_save_path(path):
    _save_path_var.set(path)


def get_save_path():
    return _save_path_var.get()


# Data path for heavy files (cache ROOT, output ROOT).
# When set, derived paths (output-root, cache) use this instead of save_path.
_data_path_var = contextvars.ContextVar("analyb_data_path", default=None)


def set_data_path(path):
    _data_path_var.set(path)


def get_data_path():
    return _data_path_var.get()


# Default weight-column name (used by load_tree when weights_branch is left None).
_weight_column_var = contextvars.ContextVar("analyb_weight_column", default=None)


def set_weight_column(name):
    _weight_column_var.set(name)


def get_weight_column():
    return _weight_column_var.get()


# PNG output: automatically saved alongside PDFs (default on, 150 DPI).
_save_png_var = contextvars.ContextVar("analyb_save_png", default=True)


def set_save_png(enabled):
    _save_png_var.set(enabled)


def get_save_png():
    return _save_png_var.get()


_save_png_dpi_var = contextvars.ContextVar("analyb_save_png_dpi", default=150)


def set_save_png_dpi(dpi):
    _save_png_dpi_var.set(dpi)


def get_save_png_dpi():
    return _save_png_dpi_var.get()


# Figure lint at save time (analyb.figcheck): None (off), "warn" or "raise".
_FIGURE_CHECK_MODES = (None, "warn", "raise")
_figure_check_var = contextvars.ContextVar("analyb_figure_check", default=None)


def set_figure_check(mode):
    if mode not in _FIGURE_CHECK_MODES:
        raise ValueError(f"figure check mode must be one of {_FIGURE_CHECK_MODES}, got {mode!r}")
    _figure_check_var.set(mode)


def get_figure_check():
    return _figure_check_var.get()


# Final placed width of saved figures in cm (for the figure-check font floor).
_figure_final_width_cm_var = contextvars.ContextVar("analyb_figure_final_width_cm", default=None)


def set_figure_final_width_cm(width_cm):
    if width_cm is not None:
        width_cm = float(width_cm)
        if not 0 < width_cm < float("inf"):
            raise ValueError(f"final width must be a positive finite number of cm, got {width_cm!r}")
    _figure_final_width_cm_var.set(width_cm)


def get_figure_final_width_cm():
    return _figure_final_width_cm_var.get()


# Single output root file per script: {save_path}/{script_id}-output.root
_output_root_var = contextvars.ContextVar("analyb_output_root", default=None)


def set_output_root(path):
    _output_root_var.set(path)


def get_output_root():
    explicit = _output_root_var.get()
    if explicit is not None:
        return explicit
    sid = get_script_id()
    dp = get_data_path()
    if dp is not None:
        return os.path.join(dp, f"{sid}-output.root")
    sp = get_save_path()
    base = sp if sp is not None else os.path.join(os.getcwd(), "plots")
    return os.path.join(base, f"{sid}-output.root")
