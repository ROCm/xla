#!/usr/bin/env python3
"""Resolve XLA targets and build reusable, hash-verified runner bundles."""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from file_util import sha256_file


UPSTREAM_URL = "https://github.com/openxla/xla.git"
ROCM_CI_BAZELRC_RELATIVE_PATH = Path(
    "build_tools/rocm/rocm_xla_ci.bazelrc"
)
CONTAINER_ROCM_BAZELRC = Path("/usertools/rocm.bazelrc")
RUNNER_TARGET = "//xla/tools/multihost_hlo_runner:hlo_runner_main"
RUNNER_RELATIVE_PATH = Path(
    "xla/tools/multihost_hlo_runner/hlo_runner_main"
)
TARGETS_SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 2
BUNDLE_KIND = "hlo_stability_runner_bundle"
MIN_CANDIDATES = 1
MAX_CANDIDATES = 3
ACTIVE_PROCESS: subprocess.Popen[str] | None = None
FINALIZATION_ACTIVE = False
DEFERRED_FINALIZATION_SIGNAL: int | None = None


def bundle_finalization_active() -> bool:
    return FINALIZATION_ACTIVE


def defer_finalization_signal(signum: int) -> None:
    global DEFERRED_FINALIZATION_SIGNAL
    DEFERRED_FINALIZATION_SIGNAL = signum


def consume_deferred_finalization_signal() -> int | None:
    global DEFERRED_FINALIZATION_SIGNAL
    signum = DEFERRED_FINALIZATION_SIGNAL
    DEFERRED_FINALIZATION_SIGNAL = None
    return signum


def handled_signals() -> set[int]:
    return {
        signal.SIGINT,
        signal.SIGTERM,
        getattr(signal, "SIGHUP", signal.SIGTERM),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def signal_process_group(process: subprocess.Popen[str], signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def signal_active_process(signum: int) -> None:
    if ACTIVE_PROCESS is not None:
        signal_process_group(ACTIVE_PROCESS, signum)


def _block_handled_signals() -> Any:
    if hasattr(signal, "pthread_sigmask"):
        return signal.pthread_sigmask(
            signal.SIG_BLOCK, handled_signals()
        )
    return None


def _spawn_process(
    command: list[str],
    **kwargs: Any,
) -> tuple[subprocess.Popen[str], Any]:
    global ACTIVE_PROCESS
    previous_mask = _block_handled_signals()
    try:
        if previous_mask is not None:
            kwargs["preexec_fn"] = lambda: signal.pthread_sigmask(
                signal.SIG_SETMASK, previous_mask
            )
        process = subprocess.Popen(
            command,
            start_new_session=True,
            **kwargs,
        )
        ACTIVE_PROCESS = process
        return process, previous_mask
    except BaseException:
        if previous_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        raise


def _restore_signal_mask(previous_mask: Any) -> None:
    if previous_mask is not None:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _terminate_capture_process(
    process: subprocess.Popen[str],
    signum: int,
) -> None:
    previous_mask = _block_handled_signals()
    try:
        signal_process_group(process, signum)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            signal_process_group(process, signal.SIGKILL)
            process.communicate()
    finally:
        _restore_signal_mask(previous_mask)


def _terminate_logged_process(process: subprocess.Popen[str]) -> None:
    previous_mask = _block_handled_signals()
    try:
        signal_process_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            signal_process_group(process, signal.SIGKILL)
            process.wait()
    finally:
        _restore_signal_mask(previous_mask)


def run_capture_result(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: int | None = None,
) -> tuple[str, int]:
    global ACTIVE_PROCESS
    process, previous_mask = _spawn_process(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        mask_to_restore = previous_mask
        previous_mask = None
        _restore_signal_mask(mask_to_restore)
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_capture_process(process, signal.SIGKILL)
        raise
    except BaseException:
        _terminate_capture_process(process, signal.SIGTERM)
        raise
    finally:
        _restore_signal_mask(previous_mask)
        ACTIVE_PROCESS = None
    if check and process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {command_text(command)}\n"
            f"{stdout.strip()}"
        )
    return stdout.strip(), int(process.returncode)


def run_capture(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: int | None = None,
) -> str:
    stdout, _ = run_capture_result(
        command,
        cwd=cwd,
        check=check,
        timeout=timeout,
    )
    return stdout


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    progress_label: str | None = None,
    heartbeat_sec: int = 30,
) -> int:
    global ACTIVE_PROCESS
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {command_text(command)}\n")
        log.flush()
        process, previous_mask = _spawn_process(
            command,
            cwd=cwd,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            mask_to_restore = previous_mask
            previous_mask = None
            _restore_signal_mask(mask_to_restore)
            started = time.monotonic()
            while True:
                try:
                    return_code = process.wait(timeout=heartbeat_sec)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = int(time.monotonic() - started)
                    log_bytes = (
                        log_path.stat().st_size
                        if log_path.exists()
                        else 0
                    )
                    print(
                        f"[{utc_now()}] "
                        f"{progress_label or 'subprocess'} still running "
                        f"({elapsed}s); log={log_path}; "
                        f"log_bytes={log_bytes}",
                        flush=True,
                    )
        except BaseException:
            _terminate_logged_process(process)
            raise
        finally:
            _restore_signal_mask(previous_mask)
            ACTIVE_PROCESS = None
        log.write(f"\n[exit_code={return_code}]\n")
    return return_code


def git(repo: Path, *args: str, check: bool = True) -> str:
    return run_capture(["git", *args], cwd=repo, check=check)


def validate_git_root(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(
            f"{label} does not exist or is not a directory: {resolved}"
        )
    root = Path(git(resolved, "rev-parse", "--show-toplevel")).resolve()
    if root != resolved:
        raise ValueError(
            f"{label} must be the Git repository root: "
            f"{resolved} (root={root})"
        )
    return resolved


def load_build_target_specs(path: Path) -> list[dict[str, Any]]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(
                    f"target file contains duplicate key: {key}"
                )
            result[key] = item
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise ValueError(f"target file must contain a JSON object: {path}")
    required = {"schema_version", "targets"}
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if extra:
            details.append(f"unsupported={','.join(extra)}")
        raise ValueError(
            "target file must contain exactly schema_version and targets "
            f"({'; '.join(details)}): {path}"
        )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != TARGETS_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported target schema: {value['schema_version']!r}; "
            f"expected {TARGETS_SCHEMA_VERSION}"
        )
    raw_targets = value["targets"]
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError(
            f"target file must contain a non-empty targets list: {path}"
        )
    targets: list[dict[str, Any]] = []
    seen_revisions: set[str] = set()
    seen_labels: set[str] = set()
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, dict):
            raise ValueError(
                f"{path}: target {index} must be a JSON object"
            )
        allowed = {"revision", "commit", "label"}
        missing_fields = {"revision"} - set(raw_target)
        extra_fields = set(raw_target) - allowed
        if missing_fields or extra_fields:
            raise ValueError(
                f"{path}: target {index} must contain revision and optional "
                f"commit/label; missing={sorted(missing_fields)}, "
                f"unsupported={sorted(extra_fields)}"
            )
        revision = raw_target["revision"]
        if (
            not isinstance(revision, str)
            or not revision
            or revision != revision.strip()
            or any(character.isspace() for character in revision)
            or revision.startswith("-")
        ):
            raise ValueError(f"{path}: target {index} has an invalid revision")
        if revision in seen_revisions:
            raise ValueError(f"{path}: duplicate target revision: {revision}")
        target: dict[str, Any] = {"revision": revision}
        if "commit" in raw_target:
            configured_commit = raw_target["commit"]
            if configured_commit is not None and (
                not isinstance(configured_commit, str)
                or not re.fullmatch(r"[0-9a-fA-F]{40}", configured_commit)
            ):
                raise ValueError(
                    f"{path}: target {index} commit must be null or a full "
                    "40-character SHA"
                )
            target["commit"] = (
                configured_commit.lower()
                if isinstance(configured_commit, str)
                else None
            )
        if "label" in raw_target:
            label = raw_target["label"]
            if (
                not isinstance(label, str)
                or not label
                or label != label.strip()
                or len(label) > 128
                or any(ord(character) < 32 for character in label)
            ):
                raise ValueError(
                    f"{path}: target {index} has an invalid label"
                )
            if label in seen_labels:
                raise ValueError(f"{path}: duplicate target label: {label}")
            target["label"] = label
            seen_labels.add(label)
        targets.append(target)
        seen_revisions.add(revision)
    return targets


def load_stability_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError(
            f"unsupported benchmark profile schema: "
            f"{profile.get('schema_version') if isinstance(profile, dict) else None}"
        )
    reference = profile.get("reference")
    runner = profile.get("runner")
    if not isinstance(reference, dict) or any(
        not isinstance(reference.get(key), str) or not reference[key]
        for key in ("id", "source", "xla_ref", "xla_commit", "gpu", "container")
    ):
        raise ValueError(f"benchmark profile has no valid reference: {path}")
    if (
        not isinstance(runner, dict)
        or runner.get("num_repeats") != 2
        or runner.get("arg_mode") != "uninitialized"
        or runner.get("cmd_buffer") != "off"
        or runner.get("order") != "size"
        or runner.get("settle_sec") != 2
    ):
        raise ValueError(
            "stability requires the reference-aligned runner policy "
            "(repeats=2, uninitialized arguments, command buffers off, "
            f"size order, settle=2): {path}"
        )
    if reference["source"] != "checked_in":
        raise ValueError(
            f"unsupported reference source {reference['source']!r}: {path}"
        )
    if not re.fullmatch(r"[0-9a-fA-F]{40}", reference["xla_commit"]):
        raise ValueError(f"profile reference commit is invalid: {path}")
    return profile


def ensure_and_fetch_remotes(
    repo: Path,
    refs: list[str],
    skip_fetch: bool,
    *,
    allow_local_refs: bool = False,
) -> None:
    remotes = set(git(repo, "remote").splitlines())
    needed = {ref.split("/", 1)[0] for ref in refs if "/" in ref}
    if allow_local_refs:
        needed = {
            remote
            for remote in needed
            if remote in remotes or remote == "upstream"
        }
    if "upstream" in needed and "upstream" not in remotes:
        if skip_fetch:
            raise ValueError(
                "upstream remote is required but missing; add it with: "
                f"git remote add upstream {UPSTREAM_URL}"
            )
        print(
            f"Adding upstream remote: {UPSTREAM_URL}",
            file=sys.stderr,
            flush=True,
        )
        git(repo, "remote", "add", "upstream", UPSTREAM_URL)
        remotes.add("upstream")
    missing = sorted(needed - remotes)
    if missing:
        raise ValueError(
            f"refs use unknown Git remote(s): {', '.join(missing)}"
        )
    if not skip_fetch:
        for remote in sorted(needed):
            print(f"Fetching {remote}...", file=sys.stderr, flush=True)
            git(repo, "fetch", remote, "--prune")


def canonical_revision(repo: Path, revision: str) -> str:
    if revision.startswith("refs/") or "/" not in revision:
        return revision
    remote, _ = revision.split("/", 1)
    remotes = set(git(repo, "remote").splitlines())
    if remote in remotes:
        return f"refs/remotes/{revision}"
    return revision


def resolve_revision(repo: Path, revision: str) -> str:
    canonical = canonical_revision(repo, revision)
    output = git(
        repo,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{canonical}^{{commit}}",
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    sha = lines[-1] if lines else ""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise RuntimeError(
            f"revision {revision!r} did not resolve to one commit SHA: "
            f"{output!r}"
        )
    return sha.lower()


def resolve_target_specs(
    repo: Path,
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for spec in specs:
        ref = spec["revision"]
        configured_commit = spec.get("commit")
        sha = resolve_revision(
            repo,
            configured_commit
            if isinstance(configured_commit, str)
            else ref,
        )
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ref).strip("._-") or "xla"
        target: dict[str, Any] = {
            "id": f"candidate:{ref}",
            "role": "candidate",
            "ref": ref,
            "source_ref": ref,
            "revision": ref,
            "commit": sha,
            "slug": f"{slug}_{sha[:12]}",
        }
        if "label" in spec:
            target["label"] = spec["label"]
        if "commit" in spec:
            target["configured_commit"] = configured_commit
        resolved.append(target)
        commit_source = (
            f"configured commit: {configured_commit}"
            if isinstance(configured_commit, str)
            else "configured commit: HEAD"
        )
        print(
            f"[{spec.get('label', ref)}] requested revision: {ref}; "
            f"{commit_source}; resolved commit: {sha}",
            file=sys.stderr,
            flush=True,
        )
    return resolved


def resolve_live_control(
    repo: Path,
    profile: dict[str, Any],
) -> dict[str, str]:
    reference = profile["reference"]
    reference_ref = reference["xla_ref"]
    reference_commit = reference["xla_commit"].lower()
    commit = resolve_revision(repo, reference_commit)
    if commit != reference_commit:
        raise ValueError(
            f"reference commit resolved to {commit}, "
            f"expected {reference_commit}"
        )
    return {
        "id": f"live-control:{commit}",
        "role": "live_control",
        "ref": f"live-control/{reference_ref}",
        "source_ref": reference_ref,
        "label": str(
            reference.get("label", "pinned live control")
        ),
        "revision": commit,
        "commit": commit,
        "slug": f"live_control_{commit[:12]}",
    }


def validate_target_path_uniqueness(
    targets: list[dict[str, Any]],
) -> None:
    for field in ("id", "slug"):
        values = [str(target.get(field, "")) for target in targets]
        if any(not value for value in values):
            raise ValueError(f"runner targets contain an empty {field}")
        duplicates = sorted(
            value for value in set(values) if values.count(value) > 1
        )
        if duplicates:
            raise ValueError(
                f"runner targets contain duplicate {field}: "
                + ", ".join(duplicates)
            )


def choose_bazel(requested: str | None) -> str:
    command = requested or "bazel"
    executable = shutil.which(command)
    if executable is None:
        raise ValueError(f"Bazel executable not found: {command}")
    return executable


def bazel_version(executable: str, cwd: Path) -> str:
    output = run_capture([executable, "--version"], cwd=cwd)
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", output)
    return match.group(1) if match else output


def rocm_bazel_configuration(
    bazel: str,
    source_repo: Path,
    target_ref: str,
) -> tuple[list[str], dict[str, Any]]:
    branch_bazelrc = source_repo / ROCM_CI_BAZELRC_RELATIVE_PATH
    metadata: dict[str, Any]
    if branch_bazelrc.is_file():
        metadata = {
            "mode": "branch_ci",
            "branch_bazelrc": {
                "path": str(branch_bazelrc),
                "sha256": sha256_file(branch_bazelrc),
            },
        }
        invocation = [bazel, f"--bazelrc={branch_bazelrc}"]
    elif CONTAINER_ROCM_BAZELRC.is_file():
        metadata = {
            "mode": "container_ci_fallback",
            "branch_bazelrc": None,
        }
        invocation = [bazel, f"--bazelrc={CONTAINER_ROCM_BAZELRC}"]
    else:
        raise RuntimeError(
            f"{target_ref} has neither branch ROCm CI Bazel configuration "
            f"{branch_bazelrc} nor container fallback "
            f"{CONTAINER_ROCM_BAZELRC}"
        )
    if CONTAINER_ROCM_BAZELRC.is_file():
        metadata["container_bazelrc"] = {
            "path": str(CONTAINER_ROCM_BAZELRC),
            "sha256": sha256_file(CONTAINER_ROCM_BAZELRC),
        }
    else:
        metadata["container_bazelrc"] = None
    return invocation, metadata


def bazel_config_closure(text: str, root: str) -> list[str]:
    dependencies: dict[str, list[str]] = {}
    for line in text.splitlines():
        match = re.match(
            r"^\s*(?:common|build):(\S+)\s+(.*?)\s*$", line
        )
        if match is None:
            continue
        config, options = match.groups()
        dependencies.setdefault(config, []).extend(
            re.findall(r"--config(?:=|\s+)([^\s#]+)", options)
        )
    closure: list[str] = []
    queue = [root]
    while queue:
        config = queue.pop(0)
        if config in closure:
            continue
        closure.append(config)
        queue.extend(dependencies.get(config, []))
    return closure


def rocm_host_toolchain_metadata(
    source_repo: Path,
    target_ref: str,
) -> dict[str, Any]:
    bazelrc = source_repo / "tensorflow.bazelrc"
    text = bazelrc.read_text(encoding="utf-8")
    selected_match = re.search(
        r"(?m)^\s*common:rocm\s+--config=(\S+)\s*$", text
    )
    selected_config = selected_match.group(1) if selected_match else None
    config_chain = bazel_config_closure(text, "rocm")
    metadata: dict[str, Any] = {
        "selected_config": selected_config,
        "config_chain": config_chain,
        "host_compiler": None,
    }
    configured_paths: list[str] = []
    for config in config_chain:
        for line in text.splitlines():
            if re.match(
                rf"^\s*(?:common|build):{re.escape(config)}\s+", line
            ) is None:
                continue
            compiler_match = re.search(
                r'--action_env=CLANG_COMPILER_PATH=(?:"([^"]+)"|(\S+))',
                line,
            )
            if compiler_match:
                configured_paths.append(
                    compiler_match.group(1) or compiler_match.group(2)
                )
    configured_paths = list(dict.fromkeys(configured_paths))
    if not configured_paths:
        return metadata
    if len(configured_paths) != 1:
        raise RuntimeError(
            f"{target_ref} selects conflicting CLANG_COMPILER_PATH values: "
            f"{', '.join(configured_paths)}"
        )
    configured = configured_paths[0]
    compiler = (
        Path(configured)
        if Path(configured).is_absolute()
        else Path(shutil.which(configured) or configured)
    )
    required_tools = {
        "clang": compiler,
        "clang++": compiler.with_name("clang++"),
        "ld.lld": compiler.with_name("ld.lld"),
    }
    missing = [
        f"{name} ({path})"
        for name, path in required_tools.items()
        if not path.is_file() or not os.access(path, os.X_OK)
    ]
    if missing:
        raise RuntimeError(
            f"{target_ref} selects legacy ROCm config {selected_config}, "
            "which requires its configured LLVM toolchain. Missing: "
            f"{', '.join(missing)}. Install the branch-compatible LLVM "
            "packages in the container; do not generate "
            "xla_configure.bazelrc or replace the branch's ROCm config."
        )
    metadata["host_compiler"] = {
        "configured_path": configured,
        "resolved_path": str(compiler.resolve()),
        "version": run_capture(
            [str(compiler), "--version"], timeout=15
        ).splitlines()[0],
        "tools": {
            name: str(path.resolve()) for name, path in required_tools.items()
        },
    }
    return metadata


def source_checkout_state(repo: Path) -> dict[str, str | None]:
    return {
        "branch": git(
            repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False
        )
        or None,
        "commit": git(repo, "rev-parse", "HEAD"),
        "status": git(repo, "status", "--porcelain", "--untracked-files=all"),
    }


def restored_source_checkout_metadata(
    repo: Path,
    expected: dict[str, str | None] | None = None,
) -> dict[str, str | None]:
    state = source_checkout_state(repo)
    if expected is not None and any(
        state.get(field) != expected.get(field)
        for field in ("branch", "commit", "status")
    ):
        raise RuntimeError(
            "restored XLA checkout does not match its original state: "
            f"expected={expected}, actual={state}"
        )
    return {
        "status": "restored",
        "restored_at": utc_now(),
        "branch": state["branch"],
        "commit": state["commit"],
        "working_tree_status": state["status"],
    }


def acquire_source_lock(repo: Path, output_dir: Path) -> int:
    if os.name != "posix":
        raise RuntimeError(
            "runner preparation locking requires a POSIX environment"
        )
    fcntl = __import__("fcntl")
    git_dir = Path(git(repo, "rev-parse", "--absolute-git-dir"))
    # Coordinate with any tool that checks out this dedicated source repo.
    lock_path = git_dir / "hlo-eval-campaign.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.lseek(descriptor, 0, os.SEEK_SET)
        owner = os.read(descriptor, 4096).decode(
            "utf-8", errors="replace"
        ).strip()
        os.close(descriptor)
        details = f"\nLock owner:\n{owner}" if owner else ""
        raise ValueError(
            f"another HLO evaluation is using {repo}.{details}"
        ) from error
    try:
        owner = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "output_dir": str(output_dir.expanduser().resolve()),
            "acquired_at": utc_now(),
        }
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            (json.dumps(owner, indent=2) + "\n").encode("utf-8"),
        )
        os.fsync(descriptor)
    except BaseException:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        raise
    return descriptor


def release_source_lock(descriptor: int) -> None:
    fcntl = __import__("fcntl")
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def require_clean_source_repo(
    repo: Path,
) -> dict[str, str | None]:
    state = source_checkout_state(repo)
    if state["status"]:
        raise ValueError(
            f"XLA source repo is not clean: {repo}\n"
            "Runner preparation checks out multiple commits and will not "
            "stash, reset, or delete local changes.\n"
            f"Inspect it with: git -C {shlex.quote(str(repo))} status --short\n"
            "Commit, stash, or remove the changes before retrying:\n"
            f"{state['status']}"
        )
    return state


def restore_source_checkout(
    repo: Path,
    state: dict[str, str | None],
) -> None:
    require_clean_source_repo(repo)
    branch = state["branch"]
    if branch:
        git(repo, "checkout", "--no-overwrite-ignore", branch)
    else:
        commit = state["commit"]
        if not commit:
            raise RuntimeError(
                "original detached checkout has no recorded commit"
            )
        git(
            repo,
            "checkout",
            "--no-overwrite-ignore",
            "--detach",
            commit,
        )


def collect_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "captured_at": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "visible_devices": {
            name: os.environ.get(name)
            for name in (
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            )
        },
    }
    for name, command in {
        "rocm_smi": [
            "rocm-smi",
            "--showproductname",
            "--showdriverversion",
        ],
        "hipcc": ["hipcc", "--version"],
    }.items():
        if shutil.which(command[0]):
            try:
                environment[name] = run_capture(
                    command, check=False, timeout=15
                )
            except subprocess.TimeoutExpired:
                environment[name] = "timed out"
    return environment


def build_runner(
    *,
    target: dict[str, Any],
    source_repo: Path,
    bundle_dir: Path,
    bazel: str,
    reuse: bool,
    base_metadata: dict[str, Any] | None = None,
    reuse_metadata: dict[str, Any] | None = None,
    completion_status: str = "completed",
) -> dict[str, Any]:
    target_dir = bundle_dir / target["slug"]
    runner_copy = target_dir / "runner" / "hlo_runner_main"
    build_log = target_dir / "build.log"
    metadata_path = target_dir / "metadata.json"
    previous: dict[str, Any] = {}
    if reuse and metadata_path.is_file():
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            previous = loaded
    metadata: dict[str, Any] = {
        **previous,
        **(base_metadata or {}),
        **target,
        "started_at": utc_now(),
        "status": "preparing_runner",
        "paths": {
            **(
                base_metadata.get("paths", {})
                if isinstance(base_metadata, dict)
                and isinstance(base_metadata.get("paths"), dict)
                else {}
            ),
            "build_log": str(build_log),
            "runner": str(runner_copy),
        },
    }
    metadata.pop("error", None)
    metadata.pop("finished_at", None)
    metadata.pop("runner_reused", None)
    metadata.pop("runner_reuse_rejected", None)
    write_json(metadata_path, metadata)
    try:
        reuse_provenance = (
            reuse_metadata
            if isinstance(reuse_metadata, dict)
            else previous
        )
        expected_hash = reuse_provenance.get("runner_sha256")
        runner_reusable = (
            reuse
            and runner_copy.is_file()
            and os.access(runner_copy, os.X_OK)
            and reuse_provenance.get("commit") == target["commit"]
            and isinstance(expected_hash, str)
            and sha256_file(runner_copy) == expected_hash
        )
        if runner_reusable:
            metadata["runner_reused"] = True
            metadata["runner_sha256"] = sha256_file(runner_copy)
            metadata["status"] = completion_status
            metadata["finished_at"] = utc_now()
            write_json(metadata_path, metadata)
            print(
                f"[{target.get('label', target['ref'])}] reusing runner "
                f"{runner_copy}",
                flush=True,
            )
            return metadata
        if reuse and (runner_copy.exists() or previous):
            metadata["runner_reuse_rejected"] = (
                "missing or mismatched commit/checksum provenance"
            )
            write_json(metadata_path, metadata)

        require_clean_source_repo(source_repo)
        metadata["status"] = "checking_out"
        write_json(metadata_path, metadata)
        print(
            f"[{target.get('label', target['ref'])}] checking out "
            f"{target['commit']}",
            flush=True,
        )
        git(
            source_repo,
            "checkout",
            "--no-overwrite-ignore",
            "--detach",
            target["commit"],
        )
        checked_out_commit = git(source_repo, "rev-parse", "HEAD")
        if checked_out_commit != target["commit"]:
            raise RuntimeError(
                f"checked out {checked_out_commit}, "
                f"expected {target['commit']}"
            )
        metadata["source_head"] = checked_out_commit
        expected_bazel = (source_repo / ".bazelversion").read_text(
            encoding="utf-8"
        ).strip()
        actual_bazel = bazel_version(bazel, source_repo)
        bazel_invocation, bazelrc_metadata = rocm_bazel_configuration(
            bazel, source_repo, target["ref"]
        )
        metadata["bazel"] = {
            "command": bazel,
            "expected_version": expected_bazel,
            "actual_version": actual_bazel,
            **bazelrc_metadata,
        }
        metadata["rocm_toolchain"] = rocm_host_toolchain_metadata(
            source_repo, target["ref"]
        )
        write_json(metadata_path, metadata)
        if expected_bazel != actual_bazel:
            raise RuntimeError(
                f"{target['ref']} requires Bazel {expected_bazel}, "
                f"but {bazel} reports {actual_bazel}; ensure bazel invokes "
                "Bazelisk or pass --bazel-command with a compatible launcher"
            )
        build_options = ["-c", "opt", "--config=rocm"]
        build_command = [
            *bazel_invocation,
            "build",
            *build_options,
            RUNNER_TARGET,
        ]
        metadata["build_command"] = build_command
        metadata["status"] = "building"
        write_json(metadata_path, metadata)
        label = str(target.get("label", target["ref"]))
        print(
            f"[{label}] building {RUNNER_TARGET}; log={build_log}",
            flush=True,
        )
        build_rc = run_logged(
            build_command,
            cwd=source_repo,
            log_path=build_log,
            progress_label=f"build {label}",
        )
        metadata["build_exit_code"] = build_rc
        if build_rc != 0:
            metadata["status"] = "build_failed"
            metadata["finished_at"] = utc_now()
            write_json(metadata_path, metadata)
            print(
                f"[{label}] build failed with exit code {build_rc}; "
                f"log={build_log}",
                file=sys.stderr,
                flush=True,
            )
            return metadata
        bazel_bin = Path(
            run_capture(
                [
                    *bazel_invocation,
                    "info",
                    *build_options,
                    "bazel-bin",
                ],
                cwd=source_repo,
            ).splitlines()[-1]
        )
        built_runner = bazel_bin / RUNNER_RELATIVE_PATH
        if not built_runner.is_file():
            raise RuntimeError(f"built runner not found: {built_runner}")
        runner_copy.parent.mkdir(parents=True, exist_ok=True)
        temporary_runner = runner_copy.with_suffix(".tmp")
        shutil.copy2(built_runner, temporary_runner)
        temporary_runner.chmod(temporary_runner.stat().st_mode | 0o111)
        temporary_runner.replace(runner_copy)
        metadata["runner_sha256"] = sha256_file(runner_copy)
        metadata["status"] = completion_status
        metadata["finished_at"] = utc_now()
        write_json(metadata_path, metadata)
        print(
            f"[{label}] runner ready: {runner_copy}; "
            f"sha256={metadata['runner_sha256']}",
            flush=True,
        )
        return metadata
    except KeyboardInterrupt as error:
        metadata["status"] = "interrupted"
        metadata["error"] = str(error)
        metadata["finished_at"] = utc_now()
        write_json(metadata_path, metadata)
        raise
    except Exception as error:
        metadata["status"] = "error"
        metadata["error"] = str(error)
        metadata["finished_at"] = utc_now()
        write_json(metadata_path, metadata)
        return metadata


def acquire_and_resolve_runner_targets(
    *,
    source_repo: Path,
    output_dir: Path,
    target_specs: list[dict[str, Any]],
    profile: dict[str, Any],
    bazel_command: str | None,
    skip_fetch: bool,
) -> tuple[
    int,
    dict[str, str | None],
    list[dict[str, Any]],
    str,
]:
    source_lock = acquire_source_lock(source_repo, output_dir)
    try:
        original_state = require_clean_source_repo(source_repo)
        generated_bazelrc = source_repo / "xla_configure.bazelrc"
        if generated_bazelrc.exists():
            raise ValueError(
                "generated Bazel configuration must be absent: "
                f"{generated_bazelrc}. Runner preparation uses each branch's "
                "ROCm CI configuration."
            )
        reference_ref = profile["reference"]["xla_ref"]
        ensure_and_fetch_remotes(
            source_repo,
            [reference_ref, *(item["revision"] for item in target_specs)],
            skip_fetch,
            allow_local_refs=True,
        )
        live_control = resolve_live_control(source_repo, profile)
        candidates = resolve_target_specs(source_repo, target_specs)
        targets = [live_control, *candidates]
        validate_target_path_uniqueness(targets)
        bazel = choose_bazel(bazel_command)
        return source_lock, original_state, targets, bazel
    except BaseException:
        release_source_lock(source_lock)
        raise


def prepare_runner_bundle(
    *,
    source_repo: Path,
    bundle_dir: Path,
    targets_file: Path,
    profile_file: Path,
    bazel_command: str | None,
    skip_fetch: bool,
) -> tuple[Path, dict[str, Any]]:
    global FINALIZATION_ACTIVE
    FINALIZATION_ACTIVE = False
    consume_deferred_finalization_signal()
    source_repo = validate_git_root(source_repo, "XLA source repo")
    bundle_dir = bundle_dir.resolve()
    targets_file = targets_file.resolve()
    profile_file = profile_file.resolve()
    if bundle_dir.is_relative_to(source_repo):
        raise ValueError(
            f"runner bundle must be outside the XLA source repo: {bundle_dir}"
        )
    for required in (targets_file, profile_file):
        if not required.is_file():
            raise ValueError(f"required file not found: {required}")
    targets_file_sha256 = sha256_file(targets_file)
    profile_file_sha256 = sha256_file(profile_file)
    target_specs = load_build_target_specs(targets_file)
    if not MIN_CANDIDATES <= len(target_specs) <= MAX_CANDIDATES:
        raise ValueError(
            "stability requires one to three candidate targets; "
            f"found {len(target_specs)} in {targets_file}"
        )
    profile = load_stability_profile(profile_file)
    if (
        sha256_file(targets_file) != targets_file_sha256
        or sha256_file(profile_file) != profile_file_sha256
    ):
        raise ValueError(
            "stability target/profile configuration changed while loading"
        )
    (
        source_lock,
        original_state,
        targets,
        bazel,
    ) = acquire_and_resolve_runner_targets(
        source_repo=source_repo,
        output_dir=bundle_dir,
        target_specs=target_specs,
        profile=profile,
        bazel_command=bazel_command,
        skip_fetch=skip_fetch,
    )
    live_control = targets[0]
    manifest_path = bundle_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "created_at": utc_now(),
        "status": "building",
        "inputs": {
            "targets_file": {
                "path": targets_file.name,
                "sha256": targets_file_sha256,
            },
            "profile_file": {
                "path": profile_file.name,
                "sha256": profile_file_sha256,
            },
            "runner_bundle_script": {
                "path": Path(__file__).name,
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "xla_source_repo": {
                "directory_name": source_repo.name,
            },
        },
        "profile": profile,
        "benchmark": {
            "profile_name": profile.get("name", "unnamed"),
            "reference_aligned": True,
            "effective": dict(profile["runner"]),
            "overrides": {},
        },
        "environment": collect_environment(),
        "targets": targets,
        "target_specs": target_specs,
        "live_control_id": live_control["id"],
        "active_target_ids": [target["id"] for target in targets],
        "source_original_state": original_state,
        "results": [],
    }
    try:
        bundle_dir.mkdir(parents=True, exist_ok=False)
        write_json(manifest_path, manifest)
    except BaseException:
        release_source_lock(source_lock)
        raise
    restore_needed = True
    try:
        for index, target in enumerate(targets, start=1):
            label = target.get("label", target["ref"])
            print(
                f"[runner {index}/{len(targets)}] {label}; "
                f"ref={target['source_ref']}; commit={target['commit']}",
                flush=True,
            )
            result = build_runner(
                target=target,
                source_repo=source_repo,
                bundle_dir=bundle_dir,
                bazel=bazel,
                reuse=False,
            )
            manifest["results"].append(result)
            write_json(manifest_path, manifest)
            print(
                f"[runner {index}/{len(targets)}] {label}: "
                f"{result.get('status')}",
                flush=True,
            )
        failed = [
            result
            for result in manifest["results"]
            if result.get("status") != "completed"
        ]
        if (
            sha256_file(targets_file) != targets_file_sha256
            or sha256_file(profile_file) != profile_file_sha256
        ):
            raise ValueError(
                "stability target/profile configuration changed during "
                "runner preparation"
            )
        manifest["status"] = (
            "completed_pending_restore"
            if not failed
            else "completed_with_failures_pending_restore"
        )
        manifest["summary"] = {
            "total": len(targets),
            "completed": len(targets) - len(failed),
            "failed": len(failed),
        }
        write_json(manifest_path, manifest)
        print(
            f"Runner bundle build phase: {manifest['status']}; "
            f"manifest={manifest_path}",
            flush=True,
        )
        FINALIZATION_ACTIVE = True
        if failed:
            failed_refs = ", ".join(str(item.get("ref")) for item in failed)
            raise RuntimeError(
                f"runner preparation failed for: {failed_refs}"
            )
        return manifest_path, manifest
    except BaseException as error:
        FINALIZATION_ACTIVE = True
        if isinstance(error, KeyboardInterrupt):
            manifest["status"] = "interrupted_pending_restore"
        elif (
            manifest.get("status")
            != "completed_with_failures_pending_restore"
        ):
            manifest["status"] = "error_pending_restore"
        manifest["failed_at"] = utc_now()
        manifest["error"] = str(error)
        write_json(manifest_path, manifest)
        raise
    finally:
        original_exception_active = sys.exc_info()[0] is not None
        restore_error: BaseException | None = None
        release_error: BaseException | None = None
        handled_signals = {
            signal.SIGINT,
            signal.SIGTERM,
            getattr(signal, "SIGHUP", signal.SIGTERM),
        }
        previous_mask = None
        if hasattr(signal, "pthread_sigmask"):
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK, handled_signals
            )
        try:
            if (
                restore_needed
                and original_state is not None
                and manifest_path.is_file()
            ):
                restore_succeeded = False
                try:
                    restore_source_checkout(source_repo, original_state)
                    manifest["source_restore"] = (
                        restored_source_checkout_metadata(
                            source_repo, original_state
                        )
                    )
                    restore_succeeded = True
                except (OSError, RuntimeError, ValueError) as error:
                    restore_error = error
                    manifest["source_restore"] = {
                        "status": "failed",
                        "attempted_at": utc_now(),
                        "error": str(error),
                    }
                    manifest["status"] = "error"
                    manifest["failed_at"] = utc_now()
                    manifest["error"] = (
                        "failed to restore original XLA checkout: "
                        f"{error}"
                    )
                if restore_succeeded:
                    final_statuses = {
                        "completed_pending_restore": "completed",
                        "completed_with_failures_pending_restore": (
                            "completed_with_failures"
                        ),
                        "interrupted_pending_restore": "interrupted",
                        "error_pending_restore": "error",
                    }
                    manifest["status"] = final_statuses.get(
                        str(manifest.get("status")),
                        str(manifest.get("status")),
                    )
                    manifest["finished_at"] = utc_now()
                write_json(manifest_path, manifest)
                if restore_succeeded:
                    try:
                        print(
                            "XLA source checkout restored: "
                            f"branch={original_state.get('branch')}; "
                            f"commit={original_state.get('commit')}; "
                            f"bundle_status={manifest['status']}",
                            flush=True,
                        )
                    except BrokenPipeError:
                        pass
            if source_lock is not None:
                try:
                    release_source_lock(source_lock)
                except BaseException as error:
                    release_error = error
                    manifest["status"] = "error"
                    manifest["failed_at"] = utc_now()
                    manifest["error"] = (
                        f"failed to release XLA source lock: {error}"
                    )
                    write_json(manifest_path, manifest)
        finally:
            if previous_mask is not None:
                signal.pthread_sigmask(
                    signal.SIG_SETMASK, previous_mask
                )
        FINALIZATION_ACTIVE = False
        if restore_error is not None and not original_exception_active:
            raise RuntimeError(
                "failed to restore original XLA checkout: "
                f"{restore_error}"
            ) from restore_error
        if restore_error is not None:
            print(
                "CRITICAL: the original operation failed and the XLA source "
                f"checkout could not be restored: {restore_error}",
                file=sys.stderr,
            )
        if release_error is not None and not original_exception_active:
            raise RuntimeError(
                f"failed to release XLA source lock: {release_error}"
            ) from release_error
        if release_error is not None:
            print(
                "CRITICAL: the original operation failed and the XLA source "
                f"lock could not be released: {release_error}",
                file=sys.stderr,
            )
