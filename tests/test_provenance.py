"""Run-provenance sidecars: clean-source refusal, identity, verification."""

import json
import os
import shutil
import subprocess
import types

import pytest

import analyb
from analyb import provenance

BINARY = bytes(range(256)) * 4


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch, tmp_path):
    # Isolate the user's Git configuration from both the fixture commands and
    # the code under test.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for key, value in {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }.items():
        monkeypatch.setenv(key, value)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    (path / ".gitignore").write_text("ignored.log\n")
    (path / "module.py").write_text("VALUE = 1\n")
    (path / "data.bin").write_bytes(BINARY)
    git(path, "add", ".")
    git(path, "commit", "-q", "-m", "initial")
    return path


@pytest.fixture
def run(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "result.json").write_text('{"x": 1}\n')
    data = tmp_path / "input.root"
    data.write_bytes(b"root-bytes")
    return out, data


def test_clean_sidecar_records_identities(repo, run):
    out, data = run
    document = provenance.write_sidecar(
        out, repos={"analysis": repo}, inputs={"benchmark": data},
        configs={"mapping": repo / "module.py"}, parameters={"force": False},
        seeds={"bootstrap": 7}, outputs=[out / "result.json"],
    )
    stored = json.loads((out / "provenance.json").read_text())
    assert stored == document
    assert stored["repos"]["analysis"] == {
        "path": os.path.realpath(repo), "head": git(repo, "rev-parse", "HEAD"),
        "branch": "main", "head_in_remote_ref": False,
    }
    assert stored["inputs"]["benchmark"]["size"] == len(b"root-bytes")
    assert stored["inputs"]["benchmark"]["sha256"] == provenance.sha256_file(data)
    assert stored["outputs"]["result.json"]["sha256"] == provenance.sha256_file(out / "result.json")
    assert stored["parameters"] == {"force": False} and stored["seeds"] == {"bootstrap": 7}
    assert "argv" not in stored
    location = stored["environment"]["packages"]["analyb"]["location"]
    assert location == os.path.dirname(os.path.realpath(analyb.__file__))
    provenance.verify_sidecar(out)


def test_head_in_remote_ref_is_recorded_not_required(repo, tmp_path):
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(repo), str(clone))
    assert provenance.source_state(clone)["head_in_remote_ref"] is True
    (clone / "local.py").write_text("LOCAL = 1\n")
    git(clone, "add", "local.py")
    git(clone, "commit", "-q", "-m", "local only")
    state = provenance.source_state(clone)  # an unpushed commit is accepted
    assert state["head_in_remote_ref"] is False and state["head"] == git(clone, "rev-parse", "HEAD")


@pytest.mark.parametrize("change", ["tracked", "staged-reverted", "untracked", "mode", "nested-repo"])
def test_any_uncommitted_state_is_refused(repo, run, change):
    out, data = run
    if change == "tracked":
        (repo / "module.py").write_text("VALUE = 2\n")
    elif change == "staged-reverted":
        (repo / "module.py").write_text("VALUE = 2\n")
        git(repo, "add", "module.py")
        (repo / "module.py").write_text("VALUE = 1\n")
    elif change == "untracked":
        (repo / "new.py").write_text("NEW = 1\n")
    elif change == "mode":
        git(repo, "config", "core.fileMode", "false")
        (repo / "module.py").chmod(0o755)
    else:
        (repo / "nested").mkdir()
        git(repo / "nested", "init", "-q")
    with pytest.raises(provenance.ProvenanceError, match="uncommitted or untracked"):
        provenance.write_sidecar(out, repos={"analysis": repo}, inputs={"in": data})
    assert sorted(os.listdir(out)) == ["result.json"]


def test_ignored_files_do_not_make_a_checkout_dirty(repo):
    (repo / "ignored.log").write_text("scratch\n")
    provenance.source_state(repo)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_hidden_index_flags_are_refused(repo, flag):
    git(repo, "update-index", flag, "module.py")
    with pytest.raises(provenance.ProvenanceError, match="assume-unchanged or skip-worktree"):
        provenance.source_state(repo)  # clean as far as git status can tell


def test_submodules_are_refused(repo, tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    git(child, "init", "-q", "-b", "main")
    (child / "c.txt").write_text("child\n")
    git(child, "add", ".")
    git(child, "commit", "-q", "-m", "child")
    git(repo, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(child), "sub")
    git(repo, "commit", "-q", "-m", "add submodule")
    with pytest.raises(provenance.ProvenanceError, match="submodule 'sub'"):
        provenance.source_state(repo)


@pytest.mark.parametrize("target, outside", [
    ("module.py", False),
    ("../outside.py", True),
    ("ABSOLUTE", True),
    (".git/config", True),
    ("ignored.log", True),
    ("sub", True),
])
def test_committed_symlinks_must_reach_tracked_files(repo, tmp_path, target, outside):
    (tmp_path / "outside.py").write_text("X = 1\n")
    (repo / "ignored.log").write_text("ignored\n")
    (repo / "sub").mkdir()
    (repo / "sub" / "x.py").write_text("X = 2\n")
    git(repo, "add", "sub/x.py")
    if target == "ABSOLUTE":
        target = str(repo / "module.py")  # inside, but not relocatable
    os.symlink(target, repo / "link.py")
    git(repo, "add", "link.py")
    git(repo, "commit", "-q", "-m", "link")
    if outside:
        with pytest.raises(provenance.ProvenanceError, match="symlink 'link.py'"):
            provenance.source_state(repo)
    else:
        provenance.source_state(repo)


def test_existing_sidecar_is_not_overwritten_silently(repo, run):
    out, data = run
    first = provenance.write_sidecar(out, repos={"a": repo}, inputs={"in": data})
    with pytest.raises(provenance.ProvenanceError, match="refusing to overwrite"):
        provenance.write_sidecar(out, repos={"a": repo}, inputs={"in": data})
    second = provenance.write_sidecar(out, repos={"a": repo}, inputs={"in": data}, replace=True)
    assert second["repos"] == first["repos"]
    assert sorted(os.listdir(out)) == ["provenance.json", "result.json"]


def test_failed_replacement_keeps_previous_sidecar(repo, run, monkeypatch):
    out, data = run
    first = provenance.write_sidecar(out, repos={"a": repo}, inputs={"in": data})
    before = (out / "provenance.json").read_bytes()
    original_replace = provenance.os.replace

    def failing_replace(source, destination):
        if os.path.basename(destination) == provenance.SIDECAR_NAME:
            raise OSError("synthetic publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(provenance.os, "replace", failing_replace)
    with pytest.raises(OSError, match="synthetic"):
        provenance.write_sidecar(out, repos={"a": repo}, inputs={}, replace=True)
    monkeypatch.undo()
    assert (out / "provenance.json").read_bytes() == before
    assert provenance.verify_sidecar(out)["repos"] == first["repos"]
    assert sorted(os.listdir(out)) == ["provenance.json", "result.json"]


def test_concurrent_writer_is_refused(repo, run):
    out, _ = run
    (out / provenance.LOCK_NAME).write_text("4242\n")
    with pytest.raises(provenance.ProvenanceError, match="another provenance writer"):
        provenance.write_sidecar(out, repos={"a": repo}, inputs={}, replace=True)
    assert not (out / "provenance.json").exists()


def test_parameters_must_be_an_explicit_serializable_dict(repo, run):
    out, _ = run
    with pytest.raises(provenance.ProvenanceError, match="dictionary"):
        provenance.write_sidecar(out, repos={"a": repo}, inputs={}, parameters=["--x"])
    with pytest.raises(provenance.ProvenanceError, match="JSON-serializable"):
        provenance.write_sidecar(out, repos={"a": repo}, inputs={}, parameters={"x": object()})


def test_verify_detects_altered_or_missing_outputs(repo, run):
    out, data = run
    (out / "plot.pdf").write_bytes(b"pdf")
    provenance.write_sidecar(out, repos={"a": repo}, inputs={"in": data},
                             outputs=[out / "result.json", out / "plot.pdf"])
    (out / "result.json").write_text('{"x": 2}\n')
    (out / "plot.pdf").unlink()
    with pytest.raises(provenance.ProvenanceError) as error:
        provenance.verify_sidecar(out / "provenance.json")
    assert "output 'result.json' does not match" in str(error.value)
    assert "output 'plot.pdf' is missing" in str(error.value)


def test_verify_source_detects_drift(repo):
    state = provenance.source_state(repo)
    provenance.verify_source(state)
    (repo / "late.py").write_text("LATE = 1\n")
    with pytest.raises(provenance.ProvenanceError, match="uncommitted"):
        provenance.verify_source(state)
    git(repo, "add", "late.py")
    git(repo, "commit", "-q", "-m", "late")
    with pytest.raises(provenance.ProvenanceError, match="changed since"):
        provenance.verify_source(state)


def test_verify_source_rejects_wrong_imported_checkout(repo, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(repo, other)
    state = provenance.source_state(repo)
    provenance.verify_source(state, module=types.SimpleNamespace(
        __name__="module", __file__=str(repo / "module.py")))
    with pytest.raises(provenance.ProvenanceError, match="not from the recorded checkout"):
        provenance.verify_source(state, module=types.SimpleNamespace(
            __name__="module", __file__=str(other / "module.py")))
    with pytest.raises(provenance.ProvenanceError, match="not from 'pkg.py'"):
        provenance.verify_source(state, module=types.SimpleNamespace(
            __name__="pkg", __file__=str(repo / "module.py")))


def test_package_checkout_refuses_venv_inside_another_checkout(repo):
    # analyb installed in an ignored virtual environment below the analysis
    # checkout must not be attributed to the analysis HEAD.
    (repo / ".gitignore").write_text("ignored.log\n.venv/\n")
    git(repo, "commit", "-q", "-am", "ignore venv")
    installed = repo / ".venv" / "lib" / "site-packages" / "analyb"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("")
    module = types.SimpleNamespace(__name__="analyb", __file__=str(installed / "__init__.py"))
    state = provenance.source_state(repo)
    with pytest.raises(provenance.ProvenanceError, match="not from 'analyb/__init__.py'"):
        provenance.package_checkout(module)
    with pytest.raises(provenance.ProvenanceError, match="not from 'analyb/__init__.py'"):
        provenance.verify_source(state, module=module)
    # A tracked package at the fixture root is accepted. Do not require the
    # checkout running this test to have a committed HEAD (e.g. a source export).
    tracked = repo / "analyb"
    tracked.mkdir()
    (tracked / "__init__.py").write_text("")
    git(repo, "add", "analyb/__init__.py")
    git(repo, "commit", "-q", "-m", "tracked package fixture")
    module = types.SimpleNamespace(__name__="analyb", __file__=str(tracked / "__init__.py"))
    assert provenance.package_checkout(module) == str(repo.resolve())


def test_untracked_package_file_is_refused(repo):
    (repo / "extra.py").write_text("X = 1\n")
    module = types.SimpleNamespace(__name__="extra", __file__=str(repo / "extra.py"))
    with pytest.raises(provenance.ProvenanceError, match="not tracked at HEAD"):
        provenance.package_checkout(module)


def test_accepted_identity_mismatch_is_reported(repo, run):
    out, data = run
    provenance.write_sidecar(out, repos={"analyb": repo}, inputs={"benchmark": data})
    head = git(repo, "rev-parse", "HEAD")
    accepted = {
        "repos": {"analyb": {"head": head}},
        "inputs": {"benchmark": {"sha256": provenance.sha256_file(data)}},
    }
    provenance.verify_sidecar(out, expected=accepted)
    accepted["repos"]["analyb"]["head"] = "0" * 40
    accepted["inputs"]["benchmark"]["sha256"] = "f" * 64
    with pytest.raises(provenance.ProvenanceError) as error:
        provenance.verify_sidecar(out, expected=accepted)
    assert "repos.analyb.head" in str(error.value)
    assert "inputs.benchmark.sha256" in str(error.value)
    with pytest.raises(provenance.ProvenanceError, match="configs.mapping is not recorded"):
        provenance.verify_sidecar(out, expected={"configs": {"mapping": {"sha256": "0"}}})


def test_no_remote_url_is_recorded(repo, run):
    out, data = run
    git(repo, "remote", "add", "origin", "https://user:tok3n-SECRET@git.example.invalid/x.git")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    document = provenance.write_sidecar(out, repos={"a": repo}, inputs={"in": data})
    assert document["repos"]["a"]["head_in_remote_ref"] is True
    content = (out / "provenance.json").read_bytes()
    assert b"tok3n-SECRET" not in content and b"git.example.invalid" not in content
