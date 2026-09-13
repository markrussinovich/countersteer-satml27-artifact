#!/usr/bin/env python3
"""Validate or apply the IPI-Aware capture overlay to vLLM 0.19.0."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "patches" / "vllm-0.19.0-ipi-aware-capture"
VLLM_VERSION = "0.19.0"
VLLM_UPSTREAM_TAG = "v0.19.0"
VLLM_UPSTREAM_COMMIT = "2a69949bdadf0e8942b7a1619b229cb475beef20"

# Only files that differ from the official vLLM v0.19.0 tag are applied.
FILES = (
    "entrypoints/openai/api_server.py",
    "model_executor/models/_ipi_aware_capture.py",
    "model_executor/models/_ipi_aware_sender.py",
    "model_executor/models/gemma4.py",
    "model_executor/models/gpt_oss.py",
    "model_executor/models/qwen2.py",
    "model_executor/models/qwen3_5.py",
    "model_executor/models/qwen3_next.py",
    "v1/executor/multiproc_executor.py",
    "v1/worker/gpu_model_runner.py",
)
NEW_FILES = frozenset(
    {
        "model_executor/models/_ipi_aware_capture.py",
        "model_executor/models/_ipi_aware_sender.py",
    }
)
BASELINE_SHA256 = {
    "entrypoints/openai/api_server.py": "cab836e021b1d8a41e1708be11070294b3b87685ffa15f88a140c1bbdd1039c3",
    "model_executor/models/gemma4.py": "404bef585219f7f7bfa0110a3973d86ed43edc9193e4b2b640a025d29622cae5",
    "model_executor/models/gpt_oss.py": "6f98da1dc4251c6a3aa9195c6c2edda45d9026ea2859547650360751f562f643",
    "model_executor/models/qwen2.py": "ae3ae71fd94d32cc8a1da80eeaab5bcfdeb2418adbade798989451c681099a43",
    "model_executor/models/qwen3_5.py": "5ce2151170c38c2a2718e3a9386598c0660d30db5ab8844fd284e08633ebc904",
    "model_executor/models/qwen3_next.py": "0f7c2df8fa972a193922bad89260d6cf1b6acb97a63b9c6675bc0be362d0a1e5",
    "v1/executor/multiproc_executor.py": "2011e7d3024f1f230db98b88acc63856ca4d1e74b77f676d8a8a66c4e0dc01ad",
    "v1/worker/gpu_model_runner.py": "3afc290d3df1be3df1b89b9b35942695f5896c7729f608d6c44580899212c301",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _installed_version(target: Path) -> str | None:
    version_file = target / "_version.py"
    if not version_file.is_file():
        return None
    tree = ast.parse(version_file.read_text(), filename=str(version_file))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        if any(
            isinstance(target_node, ast.Name) and target_node.id == "__version__"
            for target_node in node.targets
        ):
            return node.value.value
    return None


def _discover_target() -> Path | None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return None
    locations = list(spec.submodule_search_locations)
    return Path(locations[0]).resolve() if len(locations) == 1 else None


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _release_git_state() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(ROOT), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the IPI-Aware overlay to an exact vLLM 0.19.0 package."
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="Path to the installed vllm package (auto-detected when omitted)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate compatibility and report state without copying files",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path for the JSON audit report",
    )
    args = parser.parse_args()

    target = args.target.resolve() if args.target else _discover_target()
    if target is None:
        parser.error("could not locate vLLM; pass --target /path/to/site-packages/vllm")
    if not target.is_dir():
        parser.error(f"target is not a directory: {target}")

    missing_patch_files = [
        relative for relative in FILES if not (PATCH_ROOT / relative).is_file()
    ]
    if missing_patch_files:
        parser.error("patch manifest is incomplete: " + ", ".join(missing_patch_files))

    installed_version = _installed_version(target)
    if installed_version != VLLM_VERSION:
        parser.error(
            f"target must be vLLM {VLLM_VERSION}, found "
            f"{installed_version or 'unknown'}"
        )

    patch_hashes = {
        relative: _sha256(PATCH_ROOT / relative) for relative in FILES
    }
    states: dict[str, str] = {}
    incompatible: list[str] = []
    for relative in FILES:
        destination = target / relative
        if not destination.is_file():
            if relative in NEW_FILES:
                states[relative] = "absent"
                continue
            incompatible.append(f"{relative}=missing")
            continue
        actual = _sha256(destination)
        if actual == patch_hashes[relative]:
            states[relative] = "patched"
        elif relative not in NEW_FILES and actual == BASELINE_SHA256[relative]:
            states[relative] = "upstream"
        else:
            incompatible.append(f"{relative}={actual}")

    if incompatible:
        parser.error(
            "target contains unknown or incompatible files; restore official vLLM "
            f"{VLLM_VERSION} before applying: " + ", ".join(incompatible)
        )

    if not args.check:
        for relative in FILES:
            source = PATCH_ROOT / relative
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.ipi-aware-tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            states[relative] = "patched"

    state_values = set(states.values())
    if state_values == {"patched"}:
        overall_state = "patched"
    elif state_values <= {"upstream", "absent"}:
        overall_state = "upstream"
    else:
        overall_state = "partial"

    release_revision, release_worktree_dirty = _release_git_state()
    report: dict[str, object] = {
        "schema": "ipi_aware.vllm_patch_report.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "mode": "check" if args.check else "apply",
        "target": str(target),
        "release_revision": release_revision,
        "release_worktree_dirty": release_worktree_dirty,
        "vllm_version": VLLM_VERSION,
        "upstream_tag": VLLM_UPSTREAM_TAG,
        "upstream_commit": VLLM_UPSTREAM_COMMIT,
        "state": overall_state,
        "files": states,
        "patch_sha256": patch_hashes,
    }
    if args.report:
        _write_report(args.report.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
