from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_LABEL_SET_ID = "trace_meta_heuristics_v1"


def default_label_set_id() -> str:
    return DEFAULT_LABEL_SET_ID


def load_label_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    input_path = Path(path)
    if input_path.is_dir():
        input_path = input_path / "labels.jsonl"
    rows: list[dict[str, Any]] = []
    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return tuple(rows)


def build_label_rows(
    *,
    decision_points: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    label_set_id: str | None = None,
) -> tuple[dict[str, Any], ...]:
    resolved_label_set_id = label_set_id or default_label_set_id()
    trace_by_id = {
        str(row.get("trace_id") or row.get("run_id")): row
        for row in traces
    }
    rows: list[dict[str, Any]] = []
    first_visible_by_trace: dict[str, str] = {}

    for payload in decision_points:
        trace_id = str(payload.get("trace_id") or payload.get("run_id") or "")
        trace_payload = trace_by_id.get(trace_id, {})
        assistant_message_index = int(payload.get("assistant_message_index") or 0)
        injection_round_index = trace_payload.get("injection_round_index")
        risk_visible = int(
            injection_round_index is not None
            and int(injection_round_index) < assistant_message_index
        )
        decision_point_id = str(payload.get("decision_point_id") or "")
        if risk_visible and trace_id not in first_visible_by_trace:
            first_visible_by_trace[trace_id] = decision_point_id

        rows.append(
            {
                "schema_version": "label.v1",
                "label_set_id": resolved_label_set_id,
                "decision_point_id": decision_point_id,
                "trace_id": trace_id,
                "suite_name": payload.get("suite_name"),
                "task_id": payload.get("task_id"),
                "attack_name": payload.get("attack_name"),
                "assistant_message_index": assistant_message_index,
                "injection_round_index": injection_round_index,
                "utility": _coerce_optional_int(trace_payload.get("utility")),
                "security": _coerce_optional_int(trace_payload.get("security")),
                "outcome_error": trace_payload.get("outcome_error"),
                "risk_visible": risk_visible,
                "risk_faced": 0,
            }
        )

    finalized: list[dict[str, Any]] = []
    for row in rows:
        finalized.append(
            {
                **row,
                "risk_faced": int(first_visible_by_trace.get(str(row["trace_id"])) == str(row["decision_point_id"])),
            }
        )
    return tuple(finalized)


def write_label_artifacts(
    *,
    trace_dir: str | Path,
    label_set_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    trace_root = Path(trace_dir)
    label_dir = trace_root / "labels" / label_set_id
    label_dir.mkdir(parents=True, exist_ok=True)

    labels_path = label_dir / "labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    label_manifest = {
        "label_set_id": label_set_id,
        "label_columns": {
            "risk_visible": {
                "kind": "heuristic",
                "source": "trace.injection_round_index",
                "positive_meaning": "the injected content is visible before this assistant generation",
            },
            "risk_faced": {
                "kind": "heuristic",
                "source": "trace.injection_round_index",
                "positive_meaning": "this is the first assistant generation that faces visible injected content",
            },
        },
    }
    (label_dir / "label_manifest.json").write_text(json.dumps(label_manifest, ensure_ascii=False, indent=2) + "\n")

    total = len(rows)
    stats = {
        "example_count": total,
        "trace_count": len({str(row.get("trace_id")) for row in rows}),
        "label_columns": {
            "risk_visible": _binary_label_stats(rows, "risk_visible"),
            "risk_faced": _binary_label_stats(rows, "risk_faced"),
        },
    }
    (label_dir / "label_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    return label_dir


def _binary_label_stats(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, int]:
    positives = sum(int(int(row.get(column, 0)) == 1) for row in rows)
    return {
        "positive_example_count": positives,
        "negative_example_count": len(rows) - positives,
    }


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value)) if isinstance(value, bool) else int(value)


# ---------------------------------------------------------------------------
# Protocol-based labeling (migrated from v2)
# ---------------------------------------------------------------------------

SUPPORTED_LABELING_PROTOCOLS = (
    "risk_faced",
    "risk_visible",
    "risk_actual",
)


def build_protocol_label_rows(
    decision_points: Sequence[Mapping[str, Any]],
    *,
    traces: Sequence[Mapping[str, Any]] | None = None,
    labeling_protocol: str = "risk_faced",
) -> list[dict[str, int | str]]:
    from ..utils import DEFAULT_LABELING_PROTOCOL
    if labeling_protocol not in SUPPORTED_LABELING_PROTOCOLS:
        raise ValueError(f"unsupported labeling protocol: {labeling_protocol}")
    trace_injection_rounds = {
        str(row.get("trace_id")): row.get("injection_round_index")
        for row in (traces or ())
    }
    trace_security = _trace_security_by_id(traces or ())
    visible_by_id: dict[str, int] = {}
    # (trace_id, round_index) -> first DP whose assistant_message_index is
    # strictly greater than the injection's message index. Used to mark the
    # first DP that *faces* each injection round as `risk_faced = 1`, so a
    # multi-injection trace can yield multiple positives (one per round).
    first_visible_by_round: dict[tuple[str, int], str] = {}
    iri_values_by_id: dict[str, tuple[int, ...]] = {}
    ordered = sorted(
        decision_points,
        key=lambda row: (str(row.get("trace_id")), int(row.get("decision_index") or 0)),
    )
    for row in ordered:
        decision_point_id = str(row["decision_point_id"])
        trace_id = str(row.get("trace_id") or "")
        injection_round_index = row.get("injection_round_index")
        if injection_round_index is None:
            injection_round_index = trace_injection_rounds.get(trace_id)
        iri_values = tuple(
            int(iri) for iri in (
                injection_round_index
                if isinstance(injection_round_index, list)
                else ([injection_round_index] if injection_round_index is not None else [])
            )
        )
        iri_values_by_id[decision_point_id] = iri_values
        assistant_idx = int(row.get("assistant_message_index") or 0)
        visible = int(any(iri < assistant_idx for iri in iri_values))
        visible_by_id[decision_point_id] = visible
        if visible:
            for iri in iri_values:
                if iri < assistant_idx:
                    first_visible_by_round.setdefault((trace_id, iri), decision_point_id)

    rows: list[dict[str, int | str]] = []
    for row in decision_points:
        decision_point_id = str(row["decision_point_id"])
        trace_id = str(row.get("trace_id") or "")
        risk_visible = visible_by_id.get(decision_point_id, 0)
        risk_faced = int(
            any(
                first_visible_by_round.get((trace_id, iri)) == decision_point_id
                for iri in iri_values_by_id.get(decision_point_id, ())
            )
        )
        if labeling_protocol == "risk_visible":
            label = risk_visible
        elif labeling_protocol == "risk_actual":
            if risk_visible == 1 and risk_faced == 0:
                security = trace_security.get(trace_id)
                if security is None:
                    raise ValueError(
                        f"risk_actual requires outcome.security for trace {trace_id!r}"
                    )
                label = 1 if security else 0
            else:
                label = risk_faced
        else:
            label = risk_faced
        rows.append({"decision_point_id": decision_point_id, "label": int(label)})
    return rows


def _trace_security_by_id(traces: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    security_by_id: dict[str, bool] = {}
    for row in traces:
        trace_id = str(row.get("trace_id") or "")
        security = _extract_trace_security(row)
        if security is None:
            continue
        previous = security_by_id.get(trace_id)
        if previous is not None and previous != security:
            raise ValueError(f"conflicting outcome.security values for trace {trace_id!r}")
        security_by_id[trace_id] = security
    return security_by_id


def _extract_trace_security(row: Mapping[str, Any]) -> bool | None:
    outcome = row.get("outcome") if isinstance(row.get("outcome"), Mapping) else {}
    value = outcome.get("security") if outcome else row.get("security")
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError("trace outcome.security must be boolean")


def write_protocol_labels(
    *,
    root: str | Path,
    labeling_protocol: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    from ..io import write_jsonl
    from ..utils import ProductPaths
    if labeling_protocol not in SUPPORTED_LABELING_PROTOCOLS:
        raise ValueError(f"unsupported labeling protocol: {labeling_protocol}")
    output_path = ProductPaths.from_root(root).labels(labeling_protocol)
    clean_rows: list[dict[str, int | str]] = []
    for row in rows:
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError(f"label for {row.get('decision_point_id')} must be binary 0/1")
        clean_rows.append({"decision_point_id": str(row["decision_point_id"]), "label": label})
    return write_jsonl(output_path, clean_rows)


def label_decision_points(
    *,
    root: str | Path,
    labeling_protocol: str = "risk_faced",
    grid_point_ids: Sequence[str] | None = None,
) -> Path:
    from ..utils import iter_grid_point_ids
    selected_grid_point_ids = tuple(grid_point_ids) if grid_point_ids is not None else iter_grid_point_ids(root)
    rows: list[dict[str, Any]] = []
    for grid_point_id in selected_grid_point_ids:
        decision_points = _load_decision_point_label_context(root, grid_point_id)
        traces = _load_trace_label_context(root, grid_point_id)
        rows.extend(
            build_protocol_label_rows(
                decision_points,
                traces=traces,
                labeling_protocol=labeling_protocol,
            )
        )
    return write_protocol_labels(root=root, labeling_protocol=labeling_protocol, rows=rows)


def _load_decision_point_label_context(root: str | Path, grid_point_id: str) -> list[dict[str, Any]]:
    from ..utils import ProductPaths

    path = ProductPaths.from_root(root).decision_points(grid_point_id)
    rows: list[dict[str, Any]] = []
    decode = _decision_point_context_decoder()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(decode(line))
    return rows


def _load_trace_label_context(root: str | Path, grid_point_id: str) -> list[dict[str, Any]]:
    from ..utils import ProductPaths

    path = ProductPaths.from_root(root).traces(grid_point_id)
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    decode = _trace_context_decoder()
    with handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(decode(line))
    return rows


def _decision_point_context_decoder():
    try:
        import msgspec
    except ImportError:
        import json

        def decode(line: str) -> dict[str, Any]:
            row = json.loads(line)
            return {
                "decision_point_id": row.get("decision_point_id"),
                "trace_id": row.get("trace_id"),
                "decision_index": row.get("decision_index"),
                "assistant_message_index": row.get("assistant_message_index"),
                "injection_round_index": row.get("injection_round_index"),
            }

        return decode

    class DecisionPointContext(msgspec.Struct):
        decision_point_id: str | None = None
        trace_id: str | None = None
        decision_index: int | None = None
        assistant_message_index: int | None = None
        injection_round_index: list[int] | int | None = None

    decoder = msgspec.json.Decoder(type=DecisionPointContext)

    def decode(line: str) -> dict[str, Any]:
        row = decoder.decode(line)
        return {
            "decision_point_id": row.decision_point_id,
            "trace_id": row.trace_id,
            "decision_index": row.decision_index,
            "assistant_message_index": row.assistant_message_index,
            "injection_round_index": row.injection_round_index,
        }

    return decode


def _trace_context_decoder():
    try:
        import msgspec
    except ImportError:
        import json

        def decode(line: str) -> dict[str, Any]:
            row = json.loads(line)
            return {
                "trace_id": row.get("trace_id"),
                "injection_round_index": row.get("injection_round_index"),
                "outcome": row.get("outcome"),
            }

        return decode

    class TraceContext(msgspec.Struct):
        trace_id: str | None = None
        injection_round_index: list[int] | int | None = None
        outcome: dict[str, Any] | None = None

    decoder = msgspec.json.Decoder(type=TraceContext)

    def decode(line: str) -> dict[str, Any]:
        row = decoder.decode(line)
        return {
            "trace_id": row.trace_id,
            "injection_round_index": row.injection_round_index,
            "outcome": row.outcome,
        }

    return decode
