from __future__ import annotations

import numpy as np
import pytest

from ipi_aware.probes.training.training import _binary_auroc, _wmw_auroc


def test_binary_auroc_uses_sklearn_tie_semantics() -> None:
    assert _binary_auroc([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) == pytest.approx(0.875)


def test_binary_auroc_requires_sklearn(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sklearn.metrics":
            raise ModuleNotFoundError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="sklearn"):
        _binary_auroc([0, 1], [0.1, 0.9])


def test_wmw_auroc_uses_tie_credit_and_returns_none_for_single_class() -> None:
    assert _wmw_auroc(np.array([0.1, 0.5, 0.5, 0.9]), np.array([0, 0, 1, 1])) == pytest.approx(0.875)
    assert _wmw_auroc(np.array([0.1, 0.2]), np.array([1, 1])) is None


def test_gpu_auroc_uses_sklearn_tie_semantics_and_none_for_single_class() -> None:
    torch = pytest.importorskip("torch")
    from ipi_aware.probes.training.evaluation import _gpu_auroc

    probs = torch.tensor([0.1, 0.5, 0.5, 0.9])
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

    assert _gpu_auroc(torch, probs, labels) == pytest.approx(0.875)
    assert _gpu_auroc(torch, probs[:2], torch.tensor([1, 1], dtype=torch.float32)) is None
