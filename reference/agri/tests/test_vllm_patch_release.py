from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "patches" / "vllm-0.19.0-ipi-aware-capture"
APPLY_SCRIPT = ROOT / "scripts" / "apply_vllm_ipi_aware_patch.py"


def _load_apply_module():
    spec = importlib.util.spec_from_file_location("apply_vllm_ipi_aware_patch", APPLY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_version(target: Path, version: str = "0.19.0") -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "_version.py").write_text(f"__version__ = version = {version!r}\n")


def test_patch_manifest_matches_every_shipped_python_file() -> None:
    module = _load_apply_module()
    shipped = {
        str(path.relative_to(PATCH_ROOT))
        for path in PATCH_ROOT.rglob("*.py")
    }
    assert shipped == set(module.FILES)
    assert set(module.BASELINE_SHA256) == set(module.FILES) - set(module.NEW_FILES)


def test_chunk_bounds_use_only_vllm_019_pre_advance_interval() -> None:
    runner_path = PATCH_ROOT / "v1/worker/gpu_model_runner.py"
    tree = ast.parse(runner_path.read_text(), filename=str(runner_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_ipi_aware_capture_chunk_bounds"
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(runner_path), "exec"), namespace)
    bounds = namespace["_ipi_aware_capture_chunk_bounds"]

    assert bounds(80, 10, 100) == (80, 90)
    assert bounds(90, 20, 100) == (90, 100)
    chunk_start, chunk_end = bounds(80, 10, 100)
    assert not chunk_start <= 79 < chunk_end
    assert chunk_start <= 89 < chunk_end


def test_apply_check_accepts_an_idempotently_patched_target(tmp_path: Path) -> None:
    module = _load_apply_module()
    target = tmp_path / "vllm"
    _write_version(target)
    for relative in module.FILES:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PATCH_ROOT / relative, destination)

    result = subprocess.run(
        [sys.executable, str(APPLY_SCRIPT), "--target", str(target), "--check"],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["vllm_version"] == "0.19.0"
    assert report["upstream_commit"] == module.VLLM_UPSTREAM_COMMIT
    assert report["state"] == "patched"


def test_apply_rejects_unknown_local_modifications(tmp_path: Path) -> None:
    module = _load_apply_module()
    target = tmp_path / "vllm"
    _write_version(target)
    for relative in module.FILES:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PATCH_ROOT / relative, destination)
    unknown = target / next(iter(module.BASELINE_SHA256))
    unknown.write_text("unknown local modification\n")

    result = subprocess.run(
        [sys.executable, str(APPLY_SCRIPT), "--target", str(target), "--check"],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "unknown or incompatible files" in result.stderr


def test_apply_from_validated_upstream_is_atomic_and_writes_report(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_apply_module()
    target = tmp_path / "vllm"
    report_path = tmp_path / "logs" / "patch.json"
    _write_version(target)
    for relative in module.BASELINE_SHA256:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"upstream fixture for {relative}\n")
        module.BASELINE_SHA256[relative] = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(APPLY_SCRIPT),
            "--target",
            str(target),
            "--report",
            str(report_path),
        ],
    )
    assert module.main() == 0
    stdout_report = json.loads(capsys.readouterr().out)

    assert stdout_report["state"] == "patched"
    assert json.loads(report_path.read_text())["patch_sha256"] == stdout_report[
        "patch_sha256"
    ]
    for relative in module.FILES:
        assert (target / relative).read_bytes() == (PATCH_ROOT / relative).read_bytes()
        assert not (target / relative).with_name(
            f".{Path(relative).name}.ipi-aware-tmp"
        ).exists()


def test_vllm_extra_is_pinned_to_the_overlay_version() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"vllm==0.19.0"' in pyproject
    assert '"vllm>=0.9"' not in pyproject
