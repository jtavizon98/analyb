"""Run provenance sidecars: source revisions, inputs, environment and outputs.

``write_sidecar`` records which clean, committed source checkouts, inputs,
recipe parameters and environment produced an output directory in
``provenance.json``. Every recorded checkout must be clean: no index or
worktree change, no untracked non-ignored file, no submodule, no
assume-unchanged or skip-worktree entry, and every committed symlink must
resolve inside the checkout to a tracked regular file. Explore by committing
locally; an unpushed HEAD is recorded (with whether any remote-tracking ref
contains it), never refused.

A recorded revision establishes provenance, not acceptance. Accepted
identities (pinned revisions, artifact hashes) belong to the caller, which
compares them with ``verify_sidecar(..., expected=...)``. ``source_state``,
``package_checkout`` and ``verify_source`` let a caller fail early, locate the
checkout an imported package really comes from, and confirm before
publication that it did not change. Raw ``argv`` and Git remotes are never
recorded; callers pass an explicit dictionary of safe recipe parameters.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile

SCHEMA = "analyb-provenance"
SCHEMA_VERSION = 1
SIDECAR_NAME = "provenance.json"
LOCK_NAME = ".provenance.lock"
PACKAGES = ("analyb", "numpy", "scipy", "uproot", "matplotlib")
_ROLE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# Options that would otherwise let user or repository configuration hide
# changes from the cleanliness check.
_GIT_PINS = ("-c", "core.fileMode=true")


class ProvenanceError(RuntimeError):
    """Provenance cannot be recorded safely, or a sidecar does not verify."""


def sha256_file(path):
    """Return the streaming SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_within(path, root):
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _git(repo, *args):
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", repo, *args],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        command = next((a for a in args if not a.startswith("-") and "=" not in a), "")
        raise ProvenanceError(
            f"git {command} failed in {repo!r}: "
            f"{detail.decode(errors='replace').strip() or exc}"
        ) from exc


def _index(top):
    """Return tracked regular files and symlinks; refuse hidden flags and submodules."""
    for entry in _git(top, "ls-files", "-v", "-z").split(b"\0"):
        if entry and (entry[:1].islower() or entry[:1] == b"S"):
            raise ProvenanceError(
                f"{os.fsdecode(entry[2:])!r} in {top!r} is marked assume-unchanged or "
                "skip-worktree; Git would not report its changes"
            )
    files, links = set(), set()
    for entry in _git(top, "ls-files", "-s", "-z").split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        name = os.fsdecode(path)
        if meta.startswith(b"160000"):
            raise ProvenanceError(
                f"{top!r} contains submodule {name!r}; submodule source cannot be recorded"
            )
        (links if meta.startswith(b"120000") else files).add(name)
    return files, links


def _check_link(top, name, tracked_files):
    """A committed symlink must be relative and reach a tracked regular file."""
    full = os.path.join(top, name)
    target = os.readlink(full)
    real = os.path.realpath(full)
    relative = os.path.relpath(real, top).replace(os.sep, "/") if _is_within(real, top) else None
    if (
        os.path.isabs(target) or relative is None or relative.split("/")[0] == ".git"
        or not os.path.isfile(real) or relative not in tracked_files
    ):
        raise ProvenanceError(
            f"symlink {name!r} in {top!r} is absolute or does not resolve inside the "
            "checkout to a tracked regular file"
        )


def _dirty_paths(top):
    raw = _git(
        top, *_GIT_PINS, "status", "--porcelain=v2", "-z",
        "--ignore-submodules=none", "--untracked-files=all",
    )
    paths = []
    tokens = raw.split(b"\0")
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        fields = {b"1": 8, b"2": 9, b"u": 10, b"?": 1, b"!": 1}.get(token[:1])
        if fields is None:
            raise ProvenanceError(f"unexpected git status entry in {top!r}: {token[:2]!r}")
        paths.append(os.fsdecode(token.split(b" ", fields)[fields]))
        if token[:1] == b"2":
            index += 1  # rename/copy source path
    return paths


def source_state(path):
    """Return the identity of the clean checkout containing ``path``.

    The record holds ``path`` (checkout top level), full ``head``, ``branch``
    (None when detached) and ``head_in_remote_ref`` (whether any
    remote-tracking ref contains HEAD). Raises ``ProvenanceError`` for any
    uncommitted change, untracked non-ignored file, submodule, hidden index
    flag or unsafe committed symlink. Writes nothing.
    """
    top = os.path.realpath(
        _git(os.fspath(path), "rev-parse", "--show-toplevel").decode().strip()
    )
    head = _git(top, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    tracked_files, tracked_links = _index(top)
    dirty = _dirty_paths(top)
    if dirty:
        listed = ", ".join(dirty[:5]) + (" ..." if len(dirty) > 5 else "")
        raise ProvenanceError(
            f"checkout {top!r} has {len(dirty)} uncommitted or untracked paths "
            f"({listed}); commit them (locally is enough) and rerun"
        )
    for name in sorted(tracked_links):
        _check_link(top, name, tracked_files)
    branch = subprocess.run(
        ["git", "-C", top, "symbolic-ref", "--quiet", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or None
    containing = _git(
        top, "for-each-ref", "--contains", head, "--format=%(refname)", "refs/remotes"
    )
    return {
        "path": top,
        "head": head,
        "branch": branch,
        "head_in_remote_ref": bool(containing.strip()),
    }


def package_checkout(module, relative_path=None):
    """Return the Git checkout an imported module's source file belongs to.

    The module file must be tracked at HEAD at ``relative_path`` (default: the
    module's dotted name, e.g. ``analyb/__init__.py``) from the checkout's own
    top level. A package installed in an ignored directory of another
    repository (a virtual environment, say) is therefore refused.
    """
    location = os.path.realpath(module.__file__)
    if relative_path is None:
        suffix = "/__init__.py" if os.path.basename(location) == "__init__.py" else ".py"
        relative_path = module.__name__.replace(".", "/") + suffix
    top = os.path.realpath(
        _git(os.path.dirname(location), "rev-parse", "--show-toplevel").decode().strip()
    )
    actual = os.path.relpath(location, top).replace(os.sep, "/")
    if actual != relative_path:
        raise ProvenanceError(
            f"{module.__name__} is imported from {location!r}, not from "
            f"{relative_path!r} of a checkout (found {actual!r} in {top!r})"
        )
    try:
        _git(top, "cat-file", "-e", f"HEAD:{relative_path}")
    except ProvenanceError as exc:
        raise ProvenanceError(
            f"{module.__name__} file {relative_path!r} is not tracked at HEAD of {top!r}"
        ) from exc
    return top


def _file_record(path):
    if not os.path.isfile(path):
        raise ProvenanceError(f"{path!r} is not a regular file")
    return {"size": os.path.getsize(path), "sha256": sha256_file(path)}


def _role_files(mapping, kind):
    records = {}
    for role, path in (mapping or {}).items():
        if not isinstance(role, str) or not role:
            raise ProvenanceError(f"{kind} role must be a nonempty string: {role!r}")
        absolute = os.path.abspath(os.fspath(path))
        records[role] = {"path": absolute, **_file_record(absolute)}
    return records


def _packages():
    packages = {}
    for name in PACKAGES:
        spec = importlib.util.find_spec(name)
        if spec is None:
            continue
        module = sys.modules.get(name)
        version = getattr(module, "__version__", None)
        if version is None:
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version = None
        if spec.origin:
            location = os.path.dirname(os.path.realpath(spec.origin))
        else:  # namespace package
            location = next(iter(spec.submodule_search_locations or ()), None)
        packages[name] = {"version": version, "location": location}
    return packages


def _acquire_lock(out_dir):
    path = os.path.join(out_dir, LOCK_NAME)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise ProvenanceError(
            f"another provenance writer holds {path!r}; remove it only if no "
            "writer is running"
        ) from None
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"{os.getpid()}\n")
    return path


def write_sidecar(
    out_dir, *, repos, inputs, configs=None, parameters=None, seeds=None,
    outputs=None, replace=False,
):
    """Write ``out_dir/provenance.json`` and return the recorded document.

    ``repos``, ``inputs`` and ``configs`` map caller-chosen roles to paths;
    every repository must pass ``source_state``. ``parameters`` is an explicit
    JSON-serializable dictionary of safe recipe parameters (raw argv is never
    recorded). ``outputs`` lists produced files, recorded relative to
    ``out_dir`` when inside it. Publication is exclusive (``.provenance.lock``)
    and atomic; an existing sidecar is replaced only when ``replace`` is true.
    """
    out_dir = os.path.abspath(os.fspath(out_dir))
    if not os.path.isdir(out_dir):
        raise ProvenanceError(f"output directory does not exist: {out_dir!r}")
    if parameters is not None and not isinstance(parameters, dict):
        raise ProvenanceError("parameters must be a dictionary of safe recipe parameters")
    for role in repos:
        if not isinstance(role, str) or not _ROLE.fullmatch(role):
            raise ProvenanceError(f"repository role must be a simple name: {role!r}")
    sidecar = os.path.join(out_dir, SIDECAR_NAME)
    if os.path.lexists(sidecar) and not replace:
        raise ProvenanceError(f"refusing to overwrite {sidecar!r}; pass replace=True")
    output_records = {}
    for path in outputs or ():
        absolute = os.path.abspath(os.fspath(path))
        key = os.path.relpath(absolute, out_dir) if _is_within(absolute, out_dir) else absolute
        output_records[key] = _file_record(absolute)
    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "parameters": parameters or {},
        "repos": {role: source_state(path) for role, path in repos.items()},
        "inputs": _role_files(inputs, "input"),
        "configs": _role_files(configs, "config"),
        "seeds": seeds,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": _packages(),
        },
        "outputs": output_records,
    }
    try:
        encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"sidecar is not JSON-serializable: {exc}") from exc

    lock = _acquire_lock(out_dir)
    try:
        if os.path.lexists(sidecar) and not replace:
            raise ProvenanceError(f"refusing to overwrite {sidecar!r}; pass replace=True")
        staging = tempfile.mkdtemp(dir=out_dir, prefix=".provenance-staging-")
        try:
            staged = os.path.join(staging, SIDECAR_NAME)
            with open(staged, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staged, sidecar)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    finally:
        os.unlink(lock)
    return document


def _check_file(root, relative, size, digest, label, problems):
    root = os.path.realpath(root)
    path = os.path.realpath(os.path.join(root, relative))
    if not _is_within(path, root):
        problems.append(f"{label} path {relative!r} escapes {root!r}")
    elif not os.path.isfile(path):
        problems.append(f"{label} {relative!r} is missing")
    elif os.path.getsize(path) != size or sha256_file(path) != digest:
        problems.append(f"{label} {relative!r} does not match its recorded hash")


def _match(expected, actual, where, problems):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            problems.append(f"{where or 'sidecar'} is not an object")
            return
        for key, value in expected.items():
            if key not in actual:
                problems.append(f"{where}.{key} is not recorded")
            else:
                _match(value, actual[key], f"{where}.{key}" if where else key, problems)
    elif actual != expected:
        problems.append(f"{where} is {actual!r}, expected {expected!r}")


def verify_sidecar(path, *, expected=None):
    """Re-hash a sidecar's outputs and compare accepted identities.

    ``path`` is the sidecar or its directory. ``expected`` is a caller-owned
    nested subset of the document, for example
    ``{"repos": {"analyb": {"head": pin}}, "inputs": {"benchmark": {"sha256":
    digest}}}``; every listed value must be recorded exactly. Raises
    ``ProvenanceError`` listing all problems and returns the document otherwise.
    """
    path = os.path.abspath(os.fspath(path))
    if os.path.isdir(path):
        path = os.path.join(path, SIDECAR_NAME)
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if (document.get("schema"), document.get("schema_version")) != (SCHEMA, SCHEMA_VERSION):
        raise ProvenanceError(f"{path!r} is not an {SCHEMA} v{SCHEMA_VERSION} sidecar")
    root = os.path.realpath(os.path.dirname(path))
    problems = []
    for role, record in document["repos"].items():
        if not _ROLE.fullmatch(role) or not isinstance(record, dict) or not re.fullmatch(
            r"[0-9a-f]{40}", str(record.get("head"))
        ):
            problems.append(f"repository record {role!r} is malformed")
    for name, entry in document["outputs"].items():
        # Outputs outside the sidecar directory are recorded by absolute path.
        base = os.path.dirname(name) if os.path.isabs(name) else root
        _check_file(base, name, entry["size"], entry["sha256"], "output", problems)
    if expected:
        _match(expected, document, "", problems)
    if problems:
        raise ProvenanceError(f"{path!r} does not verify: " + "; ".join(problems))
    return document


def verify_source(record, *, module=None, relative_path=None):
    """Fail unless a checkout is still clean at the recorded HEAD and provides ``module``.

    ``record`` is a ``source_state`` result or a sidecar repository record.
    When ``module`` is given, ``package_checkout(module, relative_path)`` must
    be the recorded checkout.
    """
    current = source_state(record["path"])
    if (current["path"], current["head"]) != (record["path"], record["head"]):
        raise ProvenanceError(
            f"checkout {record['path']!r} changed since it was recorded "
            f"(HEAD {record['head']} -> {current['head']})"
        )
    if module is not None:
        checkout = package_checkout(module, relative_path)
        if checkout != os.path.realpath(record["path"]):
            raise ProvenanceError(
                f"{module.__name__} comes from {checkout!r}, not from the "
                f"recorded checkout {record['path']!r}"
            )
