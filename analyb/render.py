"""Render independent figure tasks serially or in a process pool.

``render_many(tasks, workers=...)`` runs picklable zero-argument callables
(``functools.partial`` of a module-level function) and returns their results
in task order. Each task's LaTeX registrations and printed-value records are
captured where it runs and replayed into the caller's registry and manifest in
task order, so a serial run and a pool run produce the same registry,
manifest and results. Workers are forked: they inherit analyb's context state
(script id, save path, figure-check mode, layout rules) and the caller's data.
"""

from __future__ import annotations

import os
import traceback
from concurrent.futures import ProcessPoolExecutor

from . import makelatex, printed


class RenderError(RuntimeError):
    """One or more tasks failed; ``failures`` holds ``(index, task, traceback)``."""

    def __init__(self, failures, total):
        self.failures = failures
        lines = [f"{len(failures)} of {total} render tasks failed:"]
        for index, task, trace in failures:
            lines.append(f"--- task {index}: {_describe(task)}\n{trace.rstrip()}")
        super().__init__("\n".join(lines))


def default_workers():
    """Usable CPUs (affinity-aware), capped at 4."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError, OSError):
        n = os.cpu_count() or 1
    return max(1, min(n, 4))


def render_many(tasks, *, workers=None):
    """Run every task and return their results in task order.

    ``workers=1`` runs in this process; ``None`` uses :func:`default_workers`.
    Every task runs even when some fail; then :class:`RenderError` lists each
    failure with its traceback. Registrations and printed records of failed
    tasks are dropped.
    """
    tasks = list(tasks)
    workers = default_workers() if workers is None else int(workers)
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    manifest = printed.active_manifest()
    manifest_spec = None if manifest is None else (manifest.producer, dict(manifest.sources))
    jobs = [(task, manifest_spec) for task in tasks]
    if workers == 1 or len(tasks) <= 1:
        outcomes = [_run_captured(job) for job in jobs]
    else:
        import multiprocessing

        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=context,
                                 initializer=_configure_worker) as pool:
            outcomes = list(pool.map(_run_captured, jobs))

    results, failures = [], []
    for index, (task, (ok, value, registry, records)) in enumerate(zip(tasks, outcomes)):
        if not ok:
            failures.append((index, task, value))
            results.append(None)
            continue
        makelatex._extend_registry(registry)
        for record in records:
            manifest.record(**record)
        results.append(value)
    if failures:
        raise RenderError(failures, len(tasks))
    return results


def _run_captured(job):
    """Run one task with an empty registry and manifest; return what it added."""
    task, manifest_spec = job
    registry_token = makelatex._registry_var.set({})
    manifest = None if manifest_spec is None else printed.ValueManifest(
        manifest_spec[0], sources=manifest_spec[1])
    manifest_token = printed._active.set(manifest)
    try:
        value = task()
        records = [] if manifest is None else [
            {k: v for k, v in record.items()} for record in manifest.records]
        return True, value, makelatex._registry_var.get(), records
    except Exception:
        return False, traceback.format_exc(), None, None
    finally:
        printed._active.reset(manifest_token)
        makelatex._registry_var.reset(registry_token)


def _configure_worker():
    for name in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    import matplotlib

    matplotlib.use("Agg")


def _describe(task):
    func = getattr(task, "func", task)
    name = getattr(func, "__qualname__", repr(func))
    args = getattr(task, "args", ())
    keywords = getattr(task, "keywords", {})
    shown = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in keywords.items()]
    text = f"{name}({', '.join(shown)})"
    return text if len(text) <= 200 else text[:197] + "..."
