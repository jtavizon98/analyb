"""Caller-owned branch-name -> (LaTeX label, unit) registry.

The registry starts empty. Local overrides use contextvars so callers can
scope labels without changing another context's registrations.
"""

import contextvars

_default_registry = {}
_registry_var = contextvars.ContextVar("analyb_variables_registry", default=None)


def _get_registry():
    reg = _registry_var.get()
    if reg is None:
        reg = dict(_default_registry)
    return reg


def register(name, label, unit=None):
    """Register or overwrite a single variable in the current registry."""
    reg = dict(_get_registry())
    reg[name] = (label, unit)
    _registry_var.set(reg)


def register_many(entries):
    """Register/overwrite many variables at once. `entries` maps name -> (label, unit)."""
    reg = dict(_get_registry())
    reg.update(entries)
    _registry_var.set(reg)


def lookup(name, strict=False):
    """Return (label, unit) for `name`.

    If not found:
      - strict=True  -> raise KeyError
      - strict=False -> return (name, None)
    """
    reg = _get_registry()
    if name in reg:
        return reg[name]
    if strict:
        raise KeyError(f"Variable {name!r} not registered")
    return (name, None)


class local:
    r"""Context manager / decorator for temporarily overriding the registry.

    Example
    -------
    >>> with variables.local({"momentum": (r"$p_T$", r"\GeV")}):
    ...     label, unit = variables.lookup("momentum")
    """

    def __init__(self, overrides):
        self._overrides = dict(overrides)
        self._token = None

    def __enter__(self):
        reg = dict(_get_registry())
        reg.update(self._overrides)
        self._token = _registry_var.set(reg)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _registry_var.reset(self._token)
        return False
