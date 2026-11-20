"""render_many: same results, registry and printed records for any worker count."""

import functools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import analyb as ab  # noqa: E402
from analyb import makelatex, printed  # noqa: E402


def draw(index, out):
    """One figure: saved, registered and with one printed value."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [index, index + 1])
    ab.save_figure(fig, f"fig{index}", save_path=out, verbose=False)
    makelatex.register(ab.get_script_id(), "Section", f"fig{index}.pdf", caption=f"Figure {index}")
    printed.printed(f"slope/{index}", 1.0 + index, ".2f", section="Section")
    return {"index": index, "script": ab.get_script_id()}


def fail(index):
    makelatex.register(ab.get_script_id(), "Section", f"broken{index}.pdf")
    raise RuntimeError(f"task {index} broke")


def run(tmp_path, workers, name):
    out = tmp_path / name
    ab.set_script_id("render-test")
    ab.set_save_png(False)
    token = makelatex._registry_var.set({})
    try:
        with printed.value_manifest(tmp_path / f"{name}.json", producer="test") as manifest:
            results = ab.render_many([functools.partial(draw, i, str(out)) for i in range(6)],
                                     workers=workers)
            records = manifest.records
        return results, makelatex._registry_var.get(), records, sorted(os.listdir(out))
    finally:
        makelatex._registry_var.reset(token)


def test_a_pool_run_gives_what_a_serial_run_gives(tmp_path):
    serial = run(tmp_path, 1, "serial")
    pooled = run(tmp_path, 3, "pooled")
    assert pooled == serial
    results, registry, records, files = serial
    assert [r["index"] for r in results] == list(range(6))
    assert {r["script"] for r in results} == {"render-test"}  # workers inherit context state
    assert [e["filename"] for e in registry["render-test"]] == [f"fig{i}.pdf" for i in range(6)]
    assert [r["key"] for r in records] == [f"slope/{i}" for i in range(6)]
    assert files == [f"render-test-fig{i}.pdf" for i in range(6)]


@pytest.mark.parametrize("workers", [1, 3])
def test_every_task_runs_and_each_failure_is_reported(tmp_path, workers):
    ab.set_script_id("render-test")
    ab.set_save_png(False)
    out = str(tmp_path)
    tasks = [functools.partial(draw, 0, out), functools.partial(fail, 1),
             functools.partial(draw, 2, out), functools.partial(fail, 3)]
    token = makelatex._registry_var.set({})
    try:
        with pytest.raises(ab.RenderError) as caught:
            ab.render_many(tasks, workers=workers)
        registry = makelatex._registry_var.get()
    finally:
        makelatex._registry_var.reset(token)
    assert [index for index, _, _ in caught.value.failures] == [1, 3]
    assert "task 3 broke" in str(caught.value) and "fail(1)" in str(caught.value)
    assert sorted(os.listdir(out)) == ["render-test-fig0.pdf", "render-test-fig2.pdf"]
    assert [e["filename"] for e in registry["render-test"]] == ["fig0.pdf", "fig2.pdf"]


def test_worker_count_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        ab.render_many([], workers=0)
