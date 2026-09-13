from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .featurization.feature_io import examples_to_feature_payload
from .io import read_json, read_jsonl, write_feature_payload, write_grid_point_products, write_json
from .io import write_eval_groups, load_probe_checkpoint, load_partition, load_feature_payload, load_labels, load_decision_points, write_json
from .collection.labels import SUPPORTED_LABELING_PROTOCOLS, label_decision_points
from ipi_aware.model_families import SUPPORTED_MODEL_FAMILIES, position_offset_for_model_family
from .utils import DEFAULT_LABELING_PROTOCOL
from .training.partition import resolve_train_grid_points, build_partition, write_partition
from .utils import DEFAULT_FEATURE_NAME, ProductPaths, iter_grid_point_ids
from .training.training import load_labeled_dataset, train_probe_dataset
from .collection.trace_collection import collect_agentdojo_v2, default_run_id_prefix as ipi_aware_default_run_id_prefix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ipi-signal-probe",
        description="Run the IPI-Aware probing pipeline using explicit stage products.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect",
        help="Collect AgentDojo traces directly into v2 products.",
    )
    collect.add_argument("--source-run-dir", default=None, help="Optional existing run to convert instead of live collection.")
    collect.add_argument("--output-root", default=None, help="V2 output root for --source-run-dir conversion.")
    collect.add_argument("--results-dir", default="results/probe_traces/v2")
    collect.add_argument("--run-name", default=None)
    collect.add_argument("--model", default="vllm_parsed")
    collect.add_argument("--model-id", default=None)
    collect.add_argument("--benchmark-version", default="v1.2.2")
    collect.add_argument("--suite", dest="suites", action="append", default=[])
    collect.add_argument("--user-task", dest="user_tasks", action="append", default=[])
    collect.add_argument("--injection-task", dest="injection_tasks", action="append", default=[])
    collect.add_argument("--attack", dest="attacks", action="append", default=[])
    collect.add_argument("--defense", default=None)
    collect.add_argument("--tool-delimiter", default="tool")
    collect.add_argument("--system-message-name", dest="system_message_names", action="append", default=[])
    collect.add_argument("--system-message", dest="system_messages", action="append", default=[])
    collect.add_argument("--tool-output-format", choices=("yaml", "json"), default=None)
    collect.add_argument("--temperature", type=float, default=0.0)
    collect.add_argument("--max-completion-tokens", type=int, default=None)
    collect.add_argument("--max-workers", type=int, default=1)
    collect.add_argument("--num-processes", type=int, default=1)
    collect.add_argument("--epochs", type=int, default=1)
    collect.add_argument("--continue-on-error", action="store_true")
    collect.add_argument("--max-cases", type=int, default=None)
    collect.add_argument("--plan-only", action="store_true")
    collect.add_argument("--clean", action="store_true")
    collect.add_argument("--injected", action="store_true")
    collect.add_argument("--verbose", action="store_true")
    collect.add_argument("--request-timeout", type=float, default=None,
        help="Per-request HTTP client timeout in seconds (default: OpenAI client default).")
    collect.add_argument("--log-interval", type=int, default=100,
        help="Print progress every N cases (default: 100). Set to 0 to disable console progress.")
    collect.add_argument("--upstream-ports", type=str, default=None,
        help="Comma-separated list of upstream ports. Each worker process is pinned to one port "
        "(worker i -> ports[i %% len(ports)]). Improves prefix-cache locality when running "
        "multiple independent vLLM servers.")
    collect.add_argument("--num-passes", type=int, default=1,
        help="Number of collection passes to run. Each pass uses a distinct run name suffix "
        "(e.g., <run_name>__1, <run_name>__2, ...). Default: 1.")
    collect.set_defaults(func=_cmd_collect)

    label = subparsers.add_parser("label", help="Write labels/<protocol>/labels.jsonl.")
    label.add_argument("--root", required=True)
    label.add_argument(
        "--labeling-protocol",
        choices=SUPPORTED_LABELING_PROTOCOLS,
        default=DEFAULT_LABELING_PROTOCOL,
    )
    label.add_argument("--grid-point", dest="grid_points", action="append", default=[])
    label.add_argument("--verbose", action="store_true")
    label.set_defaults(func=_cmd_label)

    featurize = subparsers.add_parser(
        "featurize",
        help="Write features/<feature_name>/<grid_point_id>.pt.",
    )
    featurize.add_argument("--root", required=True)
    featurize.add_argument("--model", required=True)
    featurize.add_argument("--backend", required=True, choices=("transformers_hook", "vllm_direct"),
        help="Capture backend: transformers_hook (offline, HuggingFace) or vllm_direct (vLLM engine).")
    featurize.add_argument("--model-family", required=True, choices=SUPPORTED_MODEL_FAMILIES,
        help="Model family for position adjustments.")
    featurize.add_argument("--feature-name", default=DEFAULT_FEATURE_NAME)
    featurize.add_argument("--grid-point", dest="grid_points", action="append", default=[])
    featurize.add_argument("--selected-position", dest="selected_positions", action="append", type=int, default=[])
    featurize.add_argument("--max-points", type=int, default=None)
    featurize.add_argument("--max-prompt-tokens", type=int, default=None,
        help="Skip DPs whose prompt_token_ids exceed this length.")
    featurize.add_argument("--shard", default=None,
        help="Shard assignment as INDEX/TOTAL (e.g. 0/4). Only process grid_point_ids[i] where i %% TOTAL == INDEX.")
    featurize.add_argument("--bucket-by-prompt-length", action="store_true")
    featurize.add_argument("--longest-first", action="store_true")
    featurize.add_argument("--verbose", action="store_true")
    # transformers_hook options
    featurize.add_argument("--batch-size", type=int, default=1,
        help="Batch size for transformers_hook backend (default: 1).")
    featurize.add_argument("--device-map", default="auto", help="Device map for model loading (default: auto).")
    featurize.add_argument("--max-memory", default=None,
        help="Max memory per GPU in GiB, e.g. '0:55,1:55'. Passed to from_pretrained max_memory.")
    # vllm_direct options
    featurize.add_argument("--num-gpus", type=int, default=1,
        help="Number of GPUs for vllm_direct backend (default: 1).")
    featurize.add_argument("--gpu-memory-utilization", type=float, default=None)
    featurize.add_argument("--max-num-batched-tokens", type=int, default=None)
    featurize.add_argument("--max-num-seqs", type=int, default=None)
    featurize.add_argument("--max-model-len", type=int, default=None)
    featurize.add_argument("--enforce-eager", action="store_true")
    featurize.add_argument("--disable-chunked-prefill", action="store_true",
        help="Disable chunked prefill. Default is enabled.")
    featurize.add_argument("--layer-ids", type=int, nargs="+", default=None,
        help="Specific layer IDs to extract (vllm_direct). Defaults to all layers (1..N-1).")
    featurize.add_argument("--enable-split-saving", action="store_true",
        help="Write per-layer split files instead of bulk .pt.")
    featurize.add_argument("--use-chat-messages", action="store_true",
        help="Use LLM.chat() with replay_request messages instead of stored prompt_token_ids. "
             "For cross-model featurization where tokenizer differs.")
    featurize.add_argument(
        "--rebuild-prompt-token-ids-from-replay",
        action="store_true",
        help="Ignore stored prompt_token_ids and rebuild prompt token IDs from replay_request.messages "
             "using the target tokenizer chat template.",
    )
    featurize.add_argument(
        "--restore-chat-messages-from-prompt-token-ids",
        action="store_true",
        help="Gemma4-only cross-model mode: reconstruct chat messages from stored prompt_token_ids "
             "and feed them through the chat-message featurization path. Requires "
             "--backend vllm_direct, --use-chat-messages, and --model-family gemma4.",
    )
    featurize.add_argument(
        "--filter-last-replay-role",
        choices=("user", "tool"),
        default=None,
        help="Optionally retain only decision points whose replay_request.messages end with the given role. "
             "Currently supported only for the gemma4 model family.",
    )
    featurize.set_defaults(func=_cmd_featurize)

    featurize_external = subparsers.add_parser(
        "featurize-from-external",
        help="Normalize external trace roots or zips into v2 products, then featurize them.",
    )
    featurize_external.add_argument("--source", required=True,
        help="External trace root directory or zip archive.")
    featurize_external.add_argument("--normalized-root", default=None,
        help="Persistent normalized v2 root. Defaults to results/probe_traces/external/<source-stem>.")
    featurize_external.add_argument("--model", required=True)
    featurize_external.add_argument("--backend", required=True, choices=("transformers_hook", "vllm_direct"),
        help="Capture backend: transformers_hook (offline, HuggingFace) or vllm_direct (vLLM engine).")
    featurize_external.add_argument("--model-family", required=True, choices=SUPPORTED_MODEL_FAMILIES,
        help="Model family for position adjustments.")
    featurize_external.add_argument("--feature-name", default=DEFAULT_FEATURE_NAME)
    featurize_external.add_argument("--grid-point", dest="grid_points", action="append", default=[])
    featurize_external.add_argument("--selected-position", dest="selected_positions", action="append", type=int, default=[])
    featurize_external.add_argument("--max-points", type=int, default=None)
    featurize_external.add_argument("--max-prompt-tokens", type=int, default=None)
    featurize_external.add_argument("--shard", default=None)
    featurize_external.add_argument("--bucket-by-prompt-length", action="store_true")
    featurize_external.add_argument("--longest-first", action="store_true")
    featurize_external.add_argument("--verbose", action="store_true")
    featurize_external.add_argument("--batch-size", type=int, default=1)
    featurize_external.add_argument("--device-map", default="auto")
    featurize_external.add_argument("--max-memory", default=None)
    featurize_external.add_argument("--num-gpus", type=int, default=1)
    featurize_external.add_argument("--gpu-memory-utilization", type=float, default=None)
    featurize_external.add_argument("--max-num-batched-tokens", type=int, default=None)
    featurize_external.add_argument("--max-num-seqs", type=int, default=None)
    featurize_external.add_argument("--max-model-len", type=int, default=None)
    featurize_external.add_argument("--enforce-eager", action="store_true")
    featurize_external.add_argument("--disable-chunked-prefill", action="store_true")
    featurize_external.add_argument("--layer-ids", type=int, nargs="+", default=None)
    featurize_external.add_argument("--enable-split-saving", action="store_true")
    featurize_external.add_argument("--use-chat-messages", action="store_true")
    featurize_external.add_argument(
        "--rebuild-prompt-token-ids-from-replay",
        action="store_true",
        help="Ignore stored prompt_token_ids and rebuild prompt token IDs from replay_request.messages "
             "using the target tokenizer chat template.",
    )
    featurize_external.add_argument(
        "--restore-chat-messages-from-prompt-token-ids",
        action="store_true",
        help="Gemma4-only cross-model mode: reconstruct chat messages from stored prompt_token_ids "
             "and feed them through the chat-message featurization path.",
    )
    featurize_external.add_argument(
        "--filter-last-replay-role",
        choices=("user", "tool"),
        default=None,
        help="Optionally retain only decision points whose replay_request.messages end with the given role. "
             "Currently supported only for the gemma4 model family.",
    )
    featurize_external.add_argument("--clean-normalized-root", action="store_true",
        help="Delete and rebuild --normalized-root before normalization.")
    featurize_external.set_defaults(func=_cmd_featurize_from_external)

    cross_featurize = subparsers.add_parser(
        "cross-model-featurize",
        help="Cross-model featurization via chat-message replay, using featurizer-family position semantics.",
    )
    cross_featurize.add_argument("--root", required=True)
    cross_featurize.add_argument("--model", required=True,
        help="Featurizer model name or path, e.g. a Qwen3-8B checkpoint used to re-featurize another model's traces.")
    cross_featurize.add_argument("--source-model-family", required=True, choices=SUPPORTED_MODEL_FAMILIES,
        help="Source trace family. Used for source-specific replay handling such as Gemma4 prompt restoration.")
    cross_featurize.add_argument("--featurizer-model-family", required=True, choices=SUPPORTED_MODEL_FAMILIES,
        help="Featurizer family. Position offsets are resolved in this model family, not the source family.")
    cross_featurize.add_argument("--feature-name", required=True)
    cross_featurize.add_argument("--grid-point", dest="grid_points", action="append", default=[])
    cross_featurize.add_argument("--selected-position", dest="selected_positions", action="append", type=int, default=[])
    cross_featurize.add_argument("--max-points", type=int, default=None)
    cross_featurize.add_argument("--max-prompt-tokens", type=int, default=None,
        help="Present for CLI parity. Ignored in chat-message mode.")
    cross_featurize.add_argument("--shard", default=None,
        help="Shard assignment as INDEX/TOTAL (e.g. 0/4). Only process grid_point_ids[i] where i %% TOTAL == INDEX.")
    cross_featurize.add_argument("--bucket-by-prompt-length", action="store_true")
    cross_featurize.add_argument("--longest-first", action="store_true")
    cross_featurize.add_argument("--verbose", action="store_true")
    cross_featurize.add_argument("--batch-size", type=int, default=0,
        help="Capture batch size. Use 0 to submit a full grid point at once.")
    cross_featurize.add_argument("--num-gpus", type=int, default=1,
        help="Number of GPUs for vllm_direct backend (default: 1).")
    cross_featurize.add_argument("--gpu-memory-utilization", type=float, default=None)
    cross_featurize.add_argument("--max-num-batched-tokens", type=int, default=None)
    cross_featurize.add_argument("--max-num-seqs", type=int, default=None)
    cross_featurize.add_argument("--max-model-len", type=int, default=None)
    cross_featurize.add_argument("--enforce-eager", action="store_true")
    cross_featurize.add_argument("--disable-chunked-prefill", action="store_true",
        help="Disable chunked prefill. Default is enabled.")
    cross_featurize.add_argument("--layer-ids", type=int, nargs="+", default=None,
        help="Specific layer IDs to extract. Defaults to all layers (0..N-1).")
    cross_featurize.add_argument("--enable-split-saving", action="store_true",
        help="Write per-layer split files instead of bulk .pt.")
    cross_featurize.add_argument(
        "--restore-source-prompt-from-token-ids",
        action="store_true",
        help="Restore source prompts from stored prompt_token_ids before cross-model re-featurization. "
             "Currently supported only for Gemma4 source traces, with per-example fallback to replay_request.messages.",
    )
    cross_featurize.add_argument(
        "--filter-last-replay-role",
        choices=("user", "tool"),
        default=None,
        help="Optionally retain only decision points whose replay_request.messages end with the given role. "
             "Currently supported only for Gemma4 source traces.",
    )
    cross_featurize.set_defaults(func=_cmd_cross_model_featurize)

    partition = subparsers.add_parser(
        "partition",
        help="Write datasets/<name>/partition.json (train/val split + eval groups).",
    )
    partition.add_argument("--root", required=True)
    partition.add_argument("--dataset", required=True,
        help="Named dataset (narrow, expand_suite, expand_suite_attack, broad) or custom name "
             "used with --train-grid-point.")
    partition.add_argument("--train-grid-point", dest="train_grid_points", action="append", default=[],
        help="Train grid point. Repeat to add. If omitted, uses the named dataset defaults.")
    partition.add_argument("--val-ratio", type=float, default=0.3)
    partition.add_argument("--split-seed", type=int, default=42)
    partition.add_argument("--labeling-protocol", default=DEFAULT_LABELING_PROTOCOL)
    partition.add_argument(
        "--partition-granularity",
        choices=("decision-points", "traces"),
        default="decision-points",
        help="Split granularity: 'decision-points' (default) splits individual DPs; "
             "'traces' keeps all DPs from the same trace in the same split.",
    )
    partition.add_argument("--verbose", action="store_true")
    partition.set_defaults(func=_cmd_partition)

    train = subparsers.add_parser(
        "train",
        help="Train probe(s) from a JSON config. Grid search when multiple values per dimension.",
    )
    train.add_argument("--config", required=True, help="Path to training config JSON.")
    train.add_argument("--device", default=None, help="Override device (e.g. cuda:0).")
    train.add_argument(
        "--filter-last-replay-role",
        choices=("user", "tool"),
        default=None,
        help="Optionally retain only decision points whose replay_request.messages end with the given role.",
    )
    train.add_argument("--dry-run", action="store_true", help="Print planned runs without training.")
    train.add_argument("--verbose", action="store_true")
    train.set_defaults(func=_cmd_train)

    eval_groups = subparsers.add_parser(
        "eval-groups",
        help="Evaluate probe generalization from a YAML config. Supports batch eval.",
    )
    eval_groups.add_argument("--config", required=True, help="Path to eval-groups config YAML/JSON.")
    eval_groups.add_argument("--device", default=None, help="Override device (e.g. cuda:0).")
    eval_groups.add_argument(
        "--filter-last-replay-role",
        choices=("user", "tool"),
        default=None,
        help="Optionally evaluate only decision points whose replay_request.messages end with the given role.",
    )
    eval_groups.add_argument("--dry-run", action="store_true", help="Print discovered probes without evaluating.")
    eval_groups.add_argument("--verbose", action="store_true")
    eval_groups.set_defaults(func=_cmd_eval_groups)

    split = subparsers.add_parser(
        "split-features",
        help="Split bulk feature .pt files into per-layer directories.",
    )
    split.add_argument("--root", required=True)
    split.add_argument("--feature-name", default=DEFAULT_FEATURE_NAME)
    split.add_argument("--keep-original", action="store_true",
        help="Keep original bulk .pt files after splitting.")
    split.add_argument("--grid-point", dest="grid_points", action="append", default=[])
    split.add_argument("--verbose", action="store_true")
    split.set_defaults(func=_cmd_split_features)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False))


def _make_stage_log_fn(args: argparse.Namespace, stage: str) -> Callable[[str], None] | None:
    if not getattr(args, "verbose", False):
        return None
    started_at = time.monotonic()

    def _log(message: str) -> None:
        elapsed = _format_elapsed(time.monotonic() - started_at)
        print(f"[{stage} {elapsed}] {message}", file=sys.stderr, flush=True)

    return _log


def _emit_log(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def _cmd_collect(args: argparse.Namespace) -> dict[str, Any]:
    if args.source_run_dir is None:
        include_clean, include_injected = _resolve_modes(args.clean, args.injected)
        run_name = args.run_name or ipi_aware_default_run_id_prefix()
        num_passes = args.num_passes
        log_interval = args.log_interval
        root = Path(args.results_dir) / run_name
        root.mkdir(parents=True, exist_ok=True)
        progress_log_path = root / "collection.log"
        all_summaries: list[dict[str, Any]] = []

        print(f"[collect] run_name={run_name} root={root} num_passes={num_passes} "
              f"num_processes={args.num_processes} max_workers={args.max_workers} "
              f"epochs={args.epochs} verbose={args.verbose}", file=sys.stderr, flush=True)

        for pass_num in range(1, num_passes + 1):

            last_print_time: float | None = None
            last_write_time: float | None = None
            _printed_first = False

            def _progress_logger(event: dict[str, Any], *, _pass=pass_num) -> None:
                nonlocal last_print_time, last_write_time, _printed_first
                elapsed = event.get("elapsed_seconds", 0)
                completed = event.get("completed", 0)
                total = event.get("total", 0)
                grid_point_id = event.get("grid_point_id", "?")
                failures = event.get("failed_case_count", 0)

                if log_interval > 0:
                    should_print = (
                        not _printed_first
                        or completed % log_interval == 0
                        or (time.monotonic() - (last_print_time or 0)) >= 5.0
                    )
                    if should_print:
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total - completed) / rate if rate > 0 else 0
                        pct = 100 * completed / total if total > 0 else 0
                        pass_tag = f"[pass {_pass}/{num_passes}]" if num_passes > 1 else ""
                        print(
                            f"{pass_tag}[{_format_elapsed(elapsed)}] {completed}/{total} cases ({pct:.1f}%) | "
                            f"{rate:.1f} c/s | ETA {eta:.0f}s | failures={failures} | "
                            f"gp={grid_point_id}",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_print_time = time.monotonic()
                        _printed_first = True

                if last_write_time is None or (time.monotonic() - last_write_time) >= 1.0:
                    with progress_log_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"event": "progress", "pass": _pass, **event}, ensure_ascii=False) + "\n")
                    last_write_time = time.monotonic()

            _upstream_ports = (
                tuple(int(p.strip()) for p in args.upstream_ports.split(","))
                if args.upstream_ports else None
            )
            summary = collect_agentdojo_v2(
                results_dir=args.results_dir,
                run_name=run_name,
                model=args.model,
                model_id=args.model_id,
                benchmark_version=args.benchmark_version,
                suites=tuple(args.suites or ["workspace"]),
                user_tasks=tuple(args.user_tasks),
                injection_tasks=tuple(args.injection_tasks),
                attacks=tuple(args.attacks),
                defense=args.defense,
                tool_delimiter=args.tool_delimiter,
                system_message_names=tuple(args.system_message_names),
                system_messages=tuple(args.system_messages),
                tool_output_format=args.tool_output_format,
                temperature=args.temperature,
                max_completion_tokens=args.max_completion_tokens,
                max_workers=args.max_workers,
                num_processes=args.num_processes,
                epochs=args.epochs,
                continue_on_error=args.continue_on_error,
                max_cases=args.max_cases,
                include_clean=include_clean,
                include_injected=include_injected,
                plan_only=args.plan_only,
                case_timeout_seconds=args.request_timeout,
                progress_callback=_progress_logger,
                upstream_ports=_upstream_ports,
            )
            all_summaries.append(summary.to_dict())

        if num_passes == 1:
            return all_summaries[0]
        # Reconcile cumulative totals from disk after all passes
        total_traces = _count_all_grid_point_rows(root, "traces.jsonl")
        total_dps = _count_all_grid_point_rows(root, "decision_points.jsonl")
        total_failures = _count_all_grid_point_rows(root, "failures.jsonl")
        manifest = read_json(root / "manifest.json")
        manifest.update({
            "completed_trace_count": total_traces,
            "completed_decision_point_count": total_dps,
            "failed_case_count": total_failures,
            "num_passes": num_passes,
        })
        write_json(root / "manifest.json", manifest)
        return {
            "run_name": run_name,
            "num_passes": num_passes,
            "passes": all_summaries,
            "total_completed_traces": total_traces,
            "total_completed_decision_points": total_dps,
            "total_failed_cases": total_failures,
        }
    if args.output_root is None:
        raise ValueError("--output-root is required when --source-run-dir is used")
    source = Path(args.source_run_dir)
    output_root = Path(args.output_root)
    source_manifest = _read_optional_json(source / "manifest.json")
    summaries = []
    for source_gp_dir in sorted((source / "grid_points").iterdir()):
        if not source_gp_dir.is_dir():
            continue
        gp_manifest = _read_optional_json(source_gp_dir / "manifest.json")
        grid_point_id = source_gp_dir.name
        suite_name, system_prompt_key, attack_name = _grid_point_parts(grid_point_id, gp_manifest)
        traces = read_jsonl(source_gp_dir / "traces.jsonl") if (source_gp_dir / "traces.jsonl").exists() else []
        decision_points = (
            read_jsonl(source_gp_dir / "decision_points.jsonl")
            if (source_gp_dir / "decision_points.jsonl").exists()
            else []
        )
        write_grid_point_products(
            root=output_root,
            grid_point_id=grid_point_id,
            suite_name=suite_name,
            system_prompt_key=system_prompt_key,
            attack_name=attack_name,
            benchmark_version=str(gp_manifest.get("benchmark_version") or source_manifest.get("benchmark_version") or "v1.2.2"),
            model_id=str(args.model_id or gp_manifest.get("model_id") or source_manifest.get("model_id") or source_manifest.get("model") or "unknown-model"),
            traces=traces,
            decision_points=decision_points,
            failure_count=int(gp_manifest.get("failure_count") or 0),
        )
        summaries.append(
            {
                "grid_point_id": grid_point_id,
                "trace_count": len(traces),
                "decision_point_count": len(decision_points),
            }
        )
    return {"root": str(output_root), "grid_point_count": len(summaries), "grid_points": summaries}


def _resolve_modes(clean_only: bool, injected_only: bool) -> tuple[bool, bool]:
    if clean_only and not injected_only:
        return True, False
    if injected_only and not clean_only:
        return False, True
    return True, True


def _cmd_label(args: argparse.Namespace) -> dict[str, Any]:
    log_fn = _make_stage_log_fn(args, "label")
    selected_grid_points = tuple(args.grid_points) or iter_grid_point_ids(args.root)
    _emit_log(
        log_fn,
        f"labeling protocol={args.labeling_protocol} root={args.root} "
        f"grid_points={len(selected_grid_points)}",
    )
    path = label_decision_points(
        root=args.root,
        labeling_protocol=args.labeling_protocol,
        grid_point_ids=selected_grid_points,
    )
    label_count = len(read_jsonl(path))
    _emit_log(log_fn, f"wrote {label_count} labels to {path}")
    return {"labels_path": str(path), "labeling_protocol": args.labeling_protocol}


def _cmd_featurize(args: argparse.Namespace) -> dict[str, Any]:
    _validate_featurize_args(args)
    if args.backend == "transformers_hook":
        return _cmd_featurize_transformers(args)
    return _cmd_featurize_vllm_direct(args)


def _cmd_cross_model_featurize(args: argparse.Namespace) -> dict[str, Any]:
    delegated_args = _build_cross_model_featurize_args(args)
    return _cmd_featurize(delegated_args)


def _cmd_featurize_from_external(args: argparse.Namespace) -> dict[str, Any]:
    log_fn = _make_stage_log_fn(args, "featurize-from-external")
    normalized_root, normalization_summary = _normalize_external_trace_source(
        source=args.source,
        normalized_root=args.normalized_root,
        clean=bool(args.clean_normalized_root),
        log_fn=log_fn,
    )
    delegated_args = argparse.Namespace(**vars(args))
    delegated_args.root = str(normalized_root)
    feature_result = _cmd_featurize(delegated_args)
    return {
        "source": str(args.source),
        "normalized_root": str(normalized_root),
        "normalization": normalization_summary,
        "featurize": feature_result,
    }


def _build_cross_model_featurize_args(args: argparse.Namespace) -> argparse.Namespace:
    restore_source_prompt = bool(args.restore_source_prompt_from_token_ids)
    return argparse.Namespace(
        root=args.root,
        model=args.model,
        backend="vllm_direct",
        model_family=args.source_model_family,
        position_model_family=args.featurizer_model_family,
        feature_name=args.feature_name,
        grid_points=list(args.grid_points),
        selected_positions=list(args.selected_positions),
        max_points=args.max_points,
        max_prompt_tokens=args.max_prompt_tokens,
        shard=args.shard,
        bucket_by_prompt_length=args.bucket_by_prompt_length,
        longest_first=args.longest_first,
        verbose=args.verbose,
        batch_size=args.batch_size,
        device_map="auto",
        max_memory=None,
        num_gpus=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        disable_chunked_prefill=args.disable_chunked_prefill,
        layer_ids=args.layer_ids,
        enable_split_saving=args.enable_split_saving,
        use_chat_messages=True,
        restore_chat_messages_from_prompt_token_ids=restore_source_prompt,
        filter_last_replay_role=args.filter_last_replay_role,
    )


def _cmd_featurize_transformers(args: argparse.Namespace) -> dict[str, Any]:
    from ipi_aware.data_collection.schema import DecisionPointRecord
    from ipi_aware.probes.featurization.features import build_probe_examples_transformers_hook
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log_fn = _make_stage_log_fn(args, "featurize")
    root = Path(args.root)
    grid_point_ids = tuple(args.grid_points) or iter_grid_point_ids(root)
    if args.shard is not None:
        shard_index, shard_total = (int(x) for x in args.shard.split("/"))
        grid_point_ids = tuple(gp for i, gp in enumerate(grid_point_ids) if i % shard_total == shard_index)
    _emit_log(
        log_fn,
        f"backend=transformers_hook model={args.model} grid_points={len(grid_point_ids)} "
        f"batch_size={args.batch_size} device_map={args.device_map}"
        + (f" shard={args.shard}" if args.shard else ""),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    import torch as _torch
    _model_kwargs = dict(
        device_map=args.device_map or "auto",
        torch_dtype=_torch.bfloat16,
        trust_remote_code=True,
    )
    if args.max_memory:
        _model_kwargs["max_memory"] = {
            int(k): f"{v}GiB" for k, v in (pair.split(":") for pair in args.max_memory.split(","))
        }
    model = AutoModelForCausalLM.from_pretrained(args.model, **_model_kwargs)
    if hasattr(model, "eval"):
        model.eval()
    try:
        devices = set(str(p.device) for p in model.parameters())
        dtypes = set(str(p.dtype) for p in model.parameters())
        mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        _emit_log(log_fn, f"model loaded devices={sorted(devices)} dtypes={sorted(dtypes)} params_mem={mem_gb:.1f}GB")
    except Exception:
        _emit_log(log_fn, "model loaded")
    requested_positions = tuple(args.selected_positions) or (-1,)
    offset_family = _position_model_family_for_args(args)
    offset = position_offset_for_model_family(offset_family)
    effective_positions = tuple(p + offset for p in requested_positions)
    if offset != 0:
        _emit_log(log_fn, f"position_model_family={offset_family}, adjusted positions {requested_positions} -> {effective_positions}")
    outputs = []

    for index, grid_point_id in enumerate(grid_point_ids, start=1):
        feat_path = root / "features" / args.feature_name / f"{grid_point_id}.pt"
        paths = ProductPaths.from_root(root)
        if feat_path.exists() or paths.is_feature_split(grid_point_id, args.feature_name):
            continue
        rows = read_jsonl(root / "grid_points" / grid_point_id / "decision_points.jsonl")
        if args.max_points is not None:
            rows = rows[: args.max_points]
        if args.max_prompt_tokens is not None:
            before = len(rows)
            rows = [r for r in rows if len(r.get("prompt_token_ids") or ()) <= args.max_prompt_tokens]
            skipped = before - len(rows)
            if skipped:
                _emit_log(log_fn, f"[{index}/{len(grid_point_ids)}] {grid_point_id}: skipped {skipped}/{before} DPs exceeding max_prompt_tokens={args.max_prompt_tokens}")
        _emit_log(
            log_fn,
            f"[{index}/{len(grid_point_ids)}] featurizing grid_point={grid_point_id} "
            f"decision_points={len(rows)}",
        )
        # LOCAL PATCH (xpia-steering port, 2026-09-13; logged in tmp/agri_port/
        # DEVIATIONS.md): honor --rebuild-prompt-token-ids-from-replay on the
        # transformers_hook backend. Upstream wires the flag only into the vllm_direct
        # path (see _cmd_featurize_vllm_direct), so on this backend it was accepted and
        # silently ignored and stored (serving-render) prompt_token_ids were used.
        # Stripping them here makes build_probe_prompt take its documented fallback:
        # rebuild from replay_request.messages via the target tokenizer chat template.
        if getattr(args, "rebuild_prompt_token_ids_from_replay", False):
            rows = [{**row, "prompt_token_ids": None} for row in rows]
        decision_points = [
            DecisionPointRecord.from_dict({"replay_request_kind": "openai_chat", **row})
            for row in rows
        ]
        before_filter_count = len(decision_points)
        decision_points = _filter_decision_points_for_featurization(
            decision_points,
            model_family=args.model_family,
            filter_last_replay_role=args.filter_last_replay_role,
        )
        filtered_count = before_filter_count - len(decision_points)
        if filtered_count:
            _emit_log(
                log_fn,
                f"[{index}/{len(grid_point_ids)}] {grid_point_id}: filtered "
                f"{filtered_count}/{before_filter_count} DPs by last replay role="
                f"{args.filter_last_replay_role}",
            )
        if args.bucket_by_prompt_length or args.longest_first:
            decision_points = _reorder_decision_points_for_featurization(
                decision_points,
                bucket_by_prompt_length=args.bucket_by_prompt_length,
                longest_first=args.longest_first,
            )
            _emit_log(
                log_fn,
                f"reordered decision_points={len(decision_points)} "
                f"bucket_by_prompt_length={args.bucket_by_prompt_length} "
                f"longest_first={args.longest_first}",
            )
        def _featurize_progress(completed: int, total: int, skipped: int) -> None:
            _emit_log(
                log_fn,
                f"[{index}/{len(grid_point_ids)}] {grid_point_id}: "
                f"{completed + skipped}/{total} done={completed} skipped={skipped}",
            )

        examples = build_probe_examples_transformers_hook(
            decision_points,
            tokenizer=tokenizer,
            model=model,
            selected_positions=effective_positions,
            batch_size=args.batch_size,
            progress_callback=_featurize_progress if log_fn else None,
        )
        if not examples:
            _emit_log(log_fn, f"[{index}/{len(grid_point_ids)}] skipped grid_point={grid_point_id}: 0 examples produced from {len(decision_points)} DPs")
            continue
        payload = examples_to_feature_payload(
            grid_point_id=grid_point_id,
            feature_name=args.feature_name,
            backend="transformers_hook",
            model_id=args.model,
            examples=examples,
            requested_positions=requested_positions,
            filter_last_replay_role=args.filter_last_replay_role,
            pre_filter_decision_point_count=before_filter_count,
            post_filter_decision_point_count=len(decision_points),
        )
        if payload is None:
            _emit_log(log_fn, f"[{index}/{len(grid_point_ids)}] skipped grid_point={grid_point_id}: empty feature payload")
            continue
        if args.enable_split_saving:
            from .featurization.feature_io import write_feature_payload_split
            path = write_feature_payload_split(root=root, payload=payload)
        else:
            path = write_feature_payload(root=root, payload=payload)
        outputs.append({"grid_point_id": grid_point_id, "feature_path": str(path), "example_count": len(examples)})
        _emit_log(log_fn, f"[{index}/{len(grid_point_ids)}] wrote {len(examples)} examples to {path}")
    _emit_log(log_fn, f"completed feature_name={args.feature_name} grid_points={len(outputs)}")
    return {"root": str(root), "feature_name": args.feature_name, "grid_points": outputs}


def _cmd_featurize_vllm_direct(args: argparse.Namespace) -> dict[str, Any]:
    from ipi_aware.data_collection.schema import DecisionPointRecord
    from ipi_aware.probes.featurization.direct_capture import DirectCaptureExtractor
    from ipi_aware.probes.featurization.features import build_probe_examples_vllm_direct
    from ipi_aware.probes.utils import default_layer_ids

    log_fn = _make_stage_log_fn(args, "featurize")
    root = Path(args.root)
    grid_point_ids = tuple(args.grid_points) or iter_grid_point_ids(root)
    if args.shard is not None:
        shard_index, shard_total = (int(x) for x in args.shard.split("/"))
        grid_point_ids = tuple(gp for i, gp in enumerate(grid_point_ids) if i % shard_total == shard_index)

    offset_family = _position_model_family_for_args(args)
    offset = position_offset_for_model_family(offset_family)
    requested_positions = tuple(args.selected_positions) or (-1,)
    effective_positions = tuple(p + offset for p in requested_positions)
    if offset != 0:
        _emit_log(log_fn, f"position_model_family={offset_family}, adjusted positions {requested_positions} -> {effective_positions}")

    layer_ids = args.layer_ids or default_layer_ids(args.model)

    _emit_log(
        log_fn,
        f"backend=vllm_direct model={args.model} grid_points={len(grid_point_ids)} "
        f"num_gpus={args.num_gpus} layer_ids={len(layer_ids)} layers"
        + (f" shard={args.shard}" if args.shard else ""),
    )

    from transformers import AutoTokenizer
    if args.use_chat_messages:
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    source_tokenizer = None

    extractor_kwargs = dict(
        model_name_or_path=args.model,
        selected_positions=list(effective_positions),
        layer_ids=layer_ids,
        num_gpus=args.num_gpus,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs or 256,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    if args.gpu_memory_utilization is not None:
        extractor_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    extractor = DirectCaptureExtractor(**extractor_kwargs)
    extractor.__enter__()

    import signal
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _sigterm_handler(signum, frame):
        _emit_log(log_fn, "Received SIGTERM, shutting down extractor...")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    outputs = []
    try:
        for index, grid_point_id in enumerate(grid_point_ids, start=1):
            feat_path = root / "features" / args.feature_name / f"{grid_point_id}.pt"
            paths = ProductPaths.from_root(root)
            if feat_path.exists() or paths.is_feature_split(grid_point_id, args.feature_name):
                continue
            rows = read_jsonl(root / "grid_points" / grid_point_id / "decision_points.jsonl")
            if args.max_points is not None:
                rows = rows[: args.max_points]
            if args.max_prompt_tokens is not None and not args.use_chat_messages:
                before = len(rows)
                rows = [r for r in rows if len(r.get("prompt_token_ids") or ()) <= args.max_prompt_tokens]
                skipped = before - len(rows)
                if skipped:
                    _emit_log(log_fn, f"[{index}/{len(grid_point_ids)}] {grid_point_id}: skipped {skipped}/{before} DPs exceeding max_prompt_tokens={args.max_prompt_tokens}")
            _emit_log(
                log_fn,
                f"[{index}/{len(grid_point_ids)}] featurizing grid_point={grid_point_id} "
                f"decision_points={len(rows)}",
            )
            decision_points = [
                DecisionPointRecord.from_dict({"replay_request_kind": "openai_chat", **row})
                for row in rows
            ]
            before_filter_count = len(decision_points)
            decision_points = _filter_decision_points_for_featurization(
                decision_points,
                model_family=args.model_family,
                filter_last_replay_role=args.filter_last_replay_role,
            )
            filtered_count = before_filter_count - len(decision_points)
            if filtered_count:
                _emit_log(
                    log_fn,
                    f"[{index}/{len(grid_point_ids)}] {grid_point_id}: filtered "
                    f"{filtered_count}/{before_filter_count} DPs by last replay role="
                    f"{args.filter_last_replay_role}",
                )
            if args.bucket_by_prompt_length or args.longest_first:
                decision_points = _reorder_decision_points_for_featurization(
                    decision_points,
                    bucket_by_prompt_length=args.bucket_by_prompt_length,
                    longest_first=args.longest_first,
                )

            def _featurize_progress(completed: int, total: int, skipped: int) -> None:
                _emit_log(
                    log_fn,
                    f"[{index}/{len(grid_point_ids)}] {grid_point_id}: "
                    f"{completed + skipped}/{total} done={completed} skipped={skipped}",
                )

            examples = build_probe_examples_vllm_direct(
                decision_points,
                tokenizer=tokenizer,
                model_name_or_path=args.model,
                selected_positions=effective_positions,
                batch_size=args.batch_size,
                extractor=extractor,
                use_chat_messages=args.use_chat_messages,
                restore_chat_messages_from_prompt_token_ids=args.restore_chat_messages_from_prompt_token_ids,
                source_tokenizer=source_tokenizer,
                rebuild_prompt_token_ids_from_replay=getattr(args, "rebuild_prompt_token_ids_from_replay", False),
                progress_callback=_featurize_progress if log_fn else None,
                timing_callback=lambda m: _emit_log(
                    log_fn,
                    f"[{index}/{len(grid_point_ids)}] {grid_point_id}: timing "
                    f"batch={m['batch_size']} tokens={m['prompt_tokens_total']} "
                    f"generate={m['generate_seconds']:.2f}s "
                    f"total={m['total_seconds']:.2f}s",
                ) if log_fn else None,
            )
            if not examples:
                _emit_log(log_fn, f"[{index}/{len(grid_point_ids)}] skipped grid_point={grid_point_id}: 0 examples produced from {len(decision_points)} DPs")
                continue
            payload = examples_to_feature_payload(
                grid_point_id=grid_point_id,
                feature_name=args.feature_name,
                backend="offline_vllm_direct",
                model_id=args.model,
                examples=examples,
                requested_positions=requested_positions,
                filter_last_replay_role=args.filter_last_replay_role,
                pre_filter_decision_point_count=before_filter_count,
                post_filter_decision_point_count=len(decision_points),
            )
            if payload is None:
                _emit_log(log_fn, f"[{index}/{len(grid_point_ids)}] skipped grid_point={grid_point_id}: empty feature payload")
                continue
            if args.enable_split_saving:
                from .featurization.feature_io import write_feature_payload_split
                path = write_feature_payload_split(root=root, payload=payload)
            else:
                path = write_feature_payload(root=root, payload=payload)
            outputs.append({"grid_point_id": grid_point_id, "feature_path": str(path), "example_count": len(examples)})
            _emit_log(log_fn, f"[{index}/{len(grid_point_ids)}] wrote {len(examples)} examples to {path}")
    finally:
        signal.signal(signal.SIGTERM, original_sigterm)
        extractor.__exit__(None, None, None)
        _emit_log(log_fn, "vLLM extractor released")
    _emit_log(log_fn, f"completed feature_name={args.feature_name} grid_points={len(outputs)}")
    return {"root": str(root), "feature_name": args.feature_name, "grid_points": outputs}


def _validate_featurize_args(args: argparse.Namespace) -> None:
    if not getattr(args, "restore_chat_messages_from_prompt_token_ids", False):
        return
    if args.backend != "vllm_direct":
        raise ValueError("--restore-chat-messages-from-prompt-token-ids requires --backend vllm_direct")
    if not args.use_chat_messages:
        raise ValueError("--restore-chat-messages-from-prompt-token-ids requires --use-chat-messages")
    if args.model_family != "gemma4":
        raise ValueError("--restore-chat-messages-from-prompt-token-ids is supported only for model_family='gemma4'")


def _position_model_family_for_args(args: argparse.Namespace) -> str:
    family = getattr(args, "position_model_family", None)
    if family is None:
        family = args.model_family
    return str(family)


def _normalize_external_trace_source(
    *,
    source: str | Path,
    normalized_root: str | Path | None,
    clean: bool,
    log_fn: Callable[[str], None] | None,
) -> tuple[Path, dict[str, Any]]:
    source_path = Path(source)
    target_root = _default_external_normalized_root(source_path) if normalized_root is None else Path(normalized_root)
    if clean and target_root.exists():
        shutil.rmtree(target_root)
    target_root.parent.mkdir(parents=True, exist_ok=True)

    temp_extract_dir: Path | None = None
    try:
        if source_path.is_file() and source_path.suffix == ".zip":
            temp_extract_dir = Path(tempfile.mkdtemp(prefix="ipi-aware-external-", dir=str(target_root.parent)))
            with zipfile.ZipFile(source_path) as archive:
                archive.extractall(temp_extract_dir)
            source_root = _find_external_run_root(temp_extract_dir)
            _emit_log(log_fn, f"extracted zip to {temp_extract_dir}")
        elif source_path.is_dir():
            source_root = _find_external_run_root(source_path)
        else:
            raise FileNotFoundError(f"external source not found: {source_path}")

        summary = _normalize_external_run_root(
            source_root=source_root,
            target_root=target_root,
            log_fn=log_fn,
        )
        return target_root, summary
    finally:
        if temp_extract_dir is not None and temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)


def _default_external_normalized_root(source_path: Path) -> Path:
    source_stem = source_path.stem if source_path.suffix == ".zip" else source_path.name
    return Path("results/probe_traces/external") / source_stem


def _find_external_run_root(base: Path) -> Path:
    if (base / "manifest.json").is_file() and (base / "grid_points").is_dir():
        return base
    candidates = [
        path for path in sorted(base.rglob("manifest.json"))
        if path.parent != base and (path.parent / "grid_points").is_dir()
    ]
    if len(candidates) == 1:
        return candidates[0].parent
    if not candidates:
        raise FileNotFoundError(f"could not find external run root under {base}")
    raise ValueError(
        f"found multiple external run roots under {base}: {[str(path.parent) for path in candidates]}"
    )


def _normalize_external_run_root(
    *,
    source_root: Path,
    target_root: Path,
    log_fn: Callable[[str], None] | None,
) -> dict[str, Any]:
    root_manifest = read_json(source_root / "manifest.json")
    target_root.mkdir(parents=True, exist_ok=True)
    write_json(target_root / "manifest.json", root_manifest)

    totals = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "grid_point_count": 0,
        "trace_count": 0,
        "decision_point_count": 0,
        "dropped_decision_point_count": 0,
        "grid_points": [],
    }
    for gp_dir in sorted((source_root / "grid_points").iterdir()):
        if not gp_dir.is_dir():
            continue
        grid_point_id = gp_dir.name
        manifest = read_json(gp_dir / "manifest.json")
        traces = read_jsonl(gp_dir / "traces.jsonl")
        trace_ids = {str(row["trace_id"]) for row in traces}
        decision_points = read_jsonl(gp_dir / "decision_points.jsonl")
        cleaned_rows = []
        dropped = 0
        for row in decision_points:
            if str(row.get("trace_id") or "") not in trace_ids:
                dropped += 1
                continue
            cleaned_rows.append(_normalize_external_decision_point_row(row))
        write_grid_point_products(
            root=target_root,
            grid_point_id=grid_point_id,
            suite_name=str(manifest["suite_name"]),
            system_prompt_key=str(manifest["system_prompt_key"]),
            attack_name=str(manifest["attack_name"]),
            benchmark_version=str(manifest["benchmark_version"]),
            model_id=str(manifest["model_id"]),
            traces=traces,
            decision_points=cleaned_rows,
            failure_count=int(manifest.get("failure_count", 0)),
        )
        gp_summary = {
            "grid_point_id": grid_point_id,
            "trace_count": len(traces),
            "decision_point_count": len(cleaned_rows),
            "dropped_decision_point_count": dropped,
        }
        totals["grid_point_count"] += 1
        totals["trace_count"] += len(traces)
        totals["decision_point_count"] += len(cleaned_rows)
        totals["dropped_decision_point_count"] += dropped
        totals["grid_points"].append(gp_summary)
        _emit_log(
            log_fn,
            f"normalized {grid_point_id}: traces={len(traces)} "
            f"decision_points={len(cleaned_rows)} dropped={dropped}",
        )
    return totals


def _normalize_external_decision_point_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if normalized.get("replay_request_kind") in (None, ""):
        normalized["replay_request_kind"] = "openai_chat"
    return normalized


def _cmd_partition(args: argparse.Namespace) -> dict[str, Any]:
    log_fn = _make_stage_log_fn(args, "partition")
    train_gps = resolve_train_grid_points(
        args.train_grid_points if args.train_grid_points else args.dataset
    )
    _emit_log(
        log_fn,
        f"dataset={args.dataset} train_gps={list(train_gps)} "
        f"val_ratio={args.val_ratio} granularity={args.partition_granularity}",
    )
    partition = build_partition(
        root=args.root,
        dataset_name=args.dataset,
        train_grid_points=train_gps,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        labeling_protocol=args.labeling_protocol,
        partition_granularity=args.partition_granularity,
    )
    path = write_partition(root=args.root, partition=partition)
    train_count = sum(len(ids) for ids in partition["train"].values())
    val_count = sum(len(ids) for ids in partition["val"].values())
    _emit_log(log_fn, f"wrote partition to {path} train={train_count} val={val_count}")
    return {
        "partition_path": str(path),
        "dataset": args.dataset,
        "train_count": train_count,
        "val_count": val_count,
        "train_grid_points": list(train_gps),
    }


def _cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    from itertools import product

    log_fn = _make_stage_log_fn(args, "train")
    config = _load_config(args.config)

    root = config["root"]
    dataset_name = config["dataset"]
    output_dir = Path(config["output_dir"])
    labeling_protocol = config.get("labeling_protocol", DEFAULT_LABELING_PROTOCOL)
    feature_name = config.get("feature_name", DEFAULT_FEATURE_NAME)
    filter_last_replay_role = args.filter_last_replay_role or config.get("filter_last_replay_role")
    batch_size = config.get("batch_size", 256)
    threshold = config.get("threshold", 0.5)
    random_seed = config.get("random_seed", 42)
    device = args.device or config.get("device")

    grid = config.get("grid", {})
    layer_indices = grid.get("layer_index", [None])
    probe_architectures = grid.get("probe_architecture", ["linear"])
    feature_compositions = grid.get("feature_composition", ["single_layer"])
    concat_num_layers_values = grid.get("concat_num_layers", [3])
    hidden_dims = grid.get("hidden_dim", [None])
    dropouts = grid.get("dropout", [0.0])
    bilinear_ranks = grid.get("bilinear_rank", [None])
    learning_rates = grid.get("learning_rate", [3e-4])
    epochs_values = grid.get("epochs", [5])
    weight_decays = grid.get("weight_decay", [config.get("weight_decay", 0.0)])
    feature_normalizations = grid.get("feature_normalization", [config.get("feature_normalization", "standard")])
    pos_weights = grid.get("pos_weight", [config.get("pos_weight", None)])
    checkpoint_mode = config.get("checkpoint_mode", "best_val")

    planned = [
        {
            "layer_index": li,
            "probe_architecture": arch,
            "feature_composition": comp,
            "concat_num_layers": cnl,
            "hidden_dim": hd,
            "dropout": dr,
            "bilinear_rank": br,
            "learning_rate": lr,
            "epochs": ep,
            "weight_decay": wd,
            "feature_normalization": fn,
            "pos_weight": pw,
        }
        for li, arch, comp, cnl, hd, dr, br, lr, ep, wd, fn, pw in product(
            layer_indices, probe_architectures, feature_compositions,
            concat_num_layers_values, hidden_dims, dropouts,
            bilinear_ranks, learning_rates, epochs_values, weight_decays,
            feature_normalizations, pos_weights,
        )
    ]

    _emit_log(
        log_fn,
        f"planned {len(planned)} run(s) dataset={dataset_name}"
        + (f" filter_last_replay_role={filter_last_replay_role}" if filter_last_replay_role else ""),
    )

    if args.dry_run:
        _emit_log(log_fn, "dry run; returning planned settings without training")
        return {"run_count": len(planned), "runs": planned}

    # Read pre-computed partition and save a copy to output_dir.
    partition = load_partition(root, dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "partition.json", partition)
    train_count = sum(len(ids) for ids in partition["train"].values())
    val_count = sum(len(ids) for ids in partition["val"].values())
    _emit_log(log_fn, f"partition: train={train_count} val={val_count}")

    dataset = load_labeled_dataset(
        root=root,
        dataset_name=dataset_name,
        labeling_protocol=labeling_protocol,
        feature_name=feature_name,
        include_eval=False,
        filter_last_replay_role=filter_last_replay_role,
    )
    if log_fn is not None:
        _emit_log(log_fn, f"loaded dataset train={len(dataset.train.labels)} val={len(dataset.val.labels)}")

    # Load monitor eval group features if configured.
    monitor_config = config.get("monitor_eval_group")
    monitor_split = None
    monitor_name = None
    if monitor_config is not None:
        from .training.training import _build_split as _build_monitor_split
        monitor_name = monitor_config.get("name", "monitor")
        group_name = monitor_config["group"]
        eval_group_gp_ids = partition.get("eval_groups", {}).get(group_name)
        if not eval_group_gp_ids:
            _emit_log(log_fn, f"WARNING: eval group '{group_name}' not found in partition, skipping monitor")
        else:
            monitor_labels_map = load_labels(root, labeling_protocol)
            from .training.training import _build_split
            mon_shards = {}
            mon_metadata = {}
            for gp_id in sorted(eval_group_gp_ids):
                payload = load_feature_payload(root, gp_id, feature_name)
                if payload is not None:
                    mon_shards[gp_id] = payload
                    mon_metadata[gp_id] = {
                        str(row["decision_point_id"]): row
                        for row in load_decision_points(root, gp_id)
                    }
            if mon_shards:
                mon_id_membership = {gp_id: [str(v) for v in shard["decision_point_ids"]] for gp_id, shard in mon_shards.items()}
                monitor_split = _build_split(
                    name=monitor_name,
                    id_membership=mon_id_membership,
                    labels=monitor_labels_map,
                    shards=mon_shards,
                    metadata=mon_metadata,
                    strict=False,
                    filter_last_replay_role=filter_last_replay_role,
                )
                _emit_log(log_fn, f"loaded monitor eval_group={group_name} n={len(monitor_split.labels)}")

    outputs = []
    for index, run in enumerate(planned):
        slug = _train_run_slug(run)
        run_output_dir = output_dir / slug
        _emit_log(log_fn, f"run {index + 1}/{len(planned)} output_dir={run_output_dir}")
        trained = train_probe_dataset(
            dataset=dataset,
            output_dir=run_output_dir,
            layer_index=run["layer_index"],
            probe_architecture=run["probe_architecture"],
            feature_composition=run["feature_composition"],
            concat_num_layers=run["concat_num_layers"],
            hidden_dim=run["hidden_dim"],
            dropout=run["dropout"],
            bilinear_rank=run["bilinear_rank"],
            learning_rate=run["learning_rate"],
            epochs=run["epochs"],
            batch_size=batch_size,
            weight_decay=run["weight_decay"],
            threshold=threshold,
            random_seed=random_seed,
            device=device,
            log_fn=log_fn,
            feature_normalization=run["feature_normalization"],
            pos_weight=run["pos_weight"],
            monitor_split=monitor_split,
            monitor_name=monitor_name,
            checkpoint_mode=checkpoint_mode,
            filter_last_replay_role=filter_last_replay_role,
        )
        outputs.append({"run": run, "output_dir": str(trained)})
        _emit_log(log_fn, f"completed run {index + 1}/{len(planned)} output_dir={trained}")

    return {"run_count": len(outputs), "outputs": outputs}


def _train_run_slug(run: dict[str, Any]) -> str:
    parts = [
        f"arch_{run['probe_architecture']}",
        f"feat_{run['feature_composition']}",
        f"layer_{run['layer_index']}",
        f"concat_{run['concat_num_layers']}",
        f"lr_{run['learning_rate']:g}" if isinstance(run["learning_rate"], float) else f"lr_{run['learning_rate']}",
        f"ep_{run['epochs']}",
    ]
    if run.get("hidden_dim") is not None:
        parts.append(f"h_{run['hidden_dim']}")
    if float(run.get("dropout") or 0.0) != 0.0:
        parts.append(f"drop_{run['dropout']}")
    if run.get("bilinear_rank") is not None:
        parts.append(f"rank_{run['bilinear_rank']}")
    if float(run.get("weight_decay") or 0.0) != 0.0:
        parts.append(f"wd_{float(run['weight_decay']):g}")
    fn = run.get("feature_normalization", "standard")
    if fn != "standard":
        parts.append(f"norm_{fn}")
    pw = run.get("pos_weight")
    if pw is not None:
        if isinstance(pw, str):
            parts.append(f"pw_{pw}")
        else:
            parts.append(f"pw_{float(pw):g}")
    return "__".join(_slug_part(p) for p in parts)


def _slug_part(value: object) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "p")


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        import yaml
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config at {p} must be a mapping")
    return payload


def _cmd_eval_groups(args: argparse.Namespace) -> dict[str, Any]:
    from collections import defaultdict
    from datetime import datetime, timezone
    from .training.evaluation import (
        EVAL_GROUP_NAMES, ProbeEvalSpec, evaluate_probes_gp_first,
        _resolve_device, _import_torch,
    )
    from .utils import EVAL_GROUPS_SCHEMA

    log_fn = _make_stage_log_fn(args, "eval-groups")
    config = _load_config(args.config)

    root = Path(config["root"])
    feature_name = config.get("feature_name", "default")
    filter_last_replay_role = args.filter_last_replay_role or config.get("filter_last_replay_role")
    eval_batch_size = config.get("eval_batch_size", 4096)
    output_dir = Path(config["output_dir"])
    device_override = args.device or config.get("device")

    # Expand probe entries: explicit list + auto-discovered directories.
    probe_entries = list(config.get("probes") or [])
    for disc in config.get("discover_from") or []:
        disc_dir = Path(disc["dir"])
        disc_all_layers = disc.get("all_layers", False)
        if not disc_dir.is_dir():
            raise FileNotFoundError(f"discover_from dir not found: {disc_dir}")
        for sub in sorted(disc_dir.iterdir()):
            if sub.is_dir() and (sub / "metrics.json").exists():
                entry = {
                    "checkpoint_root": str(sub),
                    "all_layers": disc_all_layers,
                }
                if disc.get("dataset_name") is not None:
                    entry["dataset_name"] = disc["dataset_name"]
                probe_entries.append(entry)

    _emit_log(
        log_fn,
        f"discovered {len(probe_entries)} probe(s)"
        + (f" filter_last_replay_role={filter_last_replay_role}" if filter_last_replay_role else ""),
    )

    # Phase 1: Build ProbeEvalSpecs from entries.
    probe_specs: list[ProbeEvalSpec] = []
    spec_meta: list[dict[str, Any]] = []

    for entry in probe_entries:
        checkpoint_root = Path(entry["checkpoint_root"])
        all_layers = entry.get("all_layers", False)
        layer = entry.get("layer")
        dataset_name = entry.get("dataset_name")

        metrics_path = checkpoint_root / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics.json not found in {checkpoint_root}")
        metrics = read_json(metrics_path)
        dataset_name = dataset_name or metrics.get("dataset_name")
        partition = _find_partition(checkpoint_root, dataset_name=dataset_name, root=root)
        eval_groups_def = partition.get("eval_groups")
        if eval_groups_def is None:
            raise ValueError(f"partition.json has no eval_groups (checkpoint_root={checkpoint_root})")
        train_gps = list(partition.get("train", {}).keys())
        _emit_log(log_fn, f"  {checkpoint_root.name}: dataset={dataset_name} train_gps={train_gps}")

        if all_layers:
            layer_ckpts = sorted(
                p for p in (checkpoint_root / "models").glob("best_layer_*.pt")
                if not p.stem.startswith("best_layers_")
            )
            _emit_log(log_fn, f"  {len(layer_ckpts)} layer checkpoints")
            for ckpt_path in layer_ckpts:
                li = int(ckpt_path.stem.split("_")[-1])
                ckpt = load_probe_checkpoint(ckpt_path)
                probe_id = f"{checkpoint_root.name}/L{li}"
                probe_specs.append(ProbeEvalSpec(
                    probe_id=probe_id,
                    checkpoint=ckpt,
                    eval_groups_def=eval_groups_def,
                ))
                spec_meta.append({
                    "checkpoint_root": checkpoint_root,
                    "checkpoint_path": ckpt_path,
                    "layer_index": li,
                    "train_gps": train_gps,
                })
        else:
            if layer is not None:
                best_path = str(checkpoint_root / "models" / f"best_layer_{layer}.pt")
            else:
                best_path = metrics.get("best_checkpoint_path")
            if best_path is None:
                raise ValueError(f"no best checkpoint found in {checkpoint_root}")
            _emit_log(log_fn, f"  checkpoint={best_path}")
            ckpt = load_probe_checkpoint(Path(best_path))
            probe_id = f"{checkpoint_root.parent.name}/{checkpoint_root.name}"
            probe_specs.append(ProbeEvalSpec(
                probe_id=probe_id,
                checkpoint=ckpt,
                eval_groups_def=eval_groups_def,
            ))
            spec_meta.append({
                "checkpoint_root": checkpoint_root,
                "checkpoint_path": Path(best_path),
                "probe_id": probe_id,
                "layer_index": layer,
                "train_gps": train_gps,
            })

    _emit_log(log_fn, f"total probes: {len(probe_specs)}")

    if args.dry_run:
        for meta in spec_meta:
            _emit_log(log_fn, f"  {meta.get('probe_id', '?')}: {meta['checkpoint_path']}")
        return {"probe_count": len(probe_specs)}

    # Collect unique eval GPs across all specs.
    all_eval_gp_ids: list[str] = []
    seen: set[str] = set()
    for spec in probe_specs:
        for gname in EVAL_GROUP_NAMES:
            for gp_id in spec.eval_groups_def.get(gname, []):
                if gp_id not in seen:
                    all_eval_gp_ids.append(gp_id)
                    seen.add(gp_id)
    _emit_log(log_fn, f"total unique eval GPs: {len(all_eval_gp_ids)}")

    # Phase 2: GP-first batched evaluation.
    torch = _import_torch()
    device_name = _resolve_device(torch, device_override)

    eval_results = evaluate_probes_gp_first(
        root=root,
        feature_name=feature_name,
        labeling_protocol="risk_faced",
        probe_specs=probe_specs,
        eval_gp_ids=all_eval_gp_ids,
        device=device_name,
        eval_batch_size=eval_batch_size,
        log_fn=log_fn,
        filter_last_replay_role=filter_last_replay_role,
    )
    results_by_id = {r.probe_id: r for r in eval_results}

    # Phase 3: Write outputs — one JSON per checkpoint_root.
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Group specs by checkpoint_root for per-root output.
    root_indices: dict[Path, list[int]] = defaultdict(list)
    for i, meta in enumerate(spec_meta):
        root_indices[meta["checkpoint_root"]].append(i)

    for ckpt_root, indices in root_indices.items():
        if len(indices) == 1 and not any(spec_meta[i].get("all_layers") for i in indices):
            # Single probe — write eval-groups product.
            i = indices[0]
            meta = spec_meta[i]
            r = results_by_id[probe_specs[i].probe_id]
            result_groups: dict[str, dict[str, Any]] = {}
            for gname in EVAL_GROUP_NAMES:
                gp_ids = probe_specs[i].eval_groups_def.get(gname, [])
                g = r.groups.get(gname, {})
                auroc = g.get("auroc")
                n = g.get("n", 0)
                result_groups[gname] = {
                    "n_grid_points": len(gp_ids),
                    "n_examples": n,
                    "auroc": auroc,
                    "grid_point_ids": gp_ids,
                }
                _emit_log(
                    log_fn,
                    f"  {gname}: n_gps={len(gp_ids)} n_examples={n} auroc={auroc:.4f}"
                    if auroc is not None
                    else f"  {gname}: n_gps=0",
                )

            out_payload = {
                "schema": EVAL_GROUPS_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "checkpoint_root": str(meta["checkpoint_root"]),
                "best_checkpoint_path": str(meta["checkpoint_path"]),
                "train_grid_points": meta["train_gps"],
                "filter_last_replay_role": filter_last_replay_role,
                "groups": result_groups,
            }
            per_output = output_dir / f"{ckpt_root.name}.json"
            written_path = write_eval_groups(per_output, out_payload)
            _emit_log(log_fn, f"wrote eval-groups to {written_path}")
            all_results.append({
                "output_path": str(written_path),
                "train_grid_points": meta["train_gps"],
                "groups": {
                    g: {"n": result_groups[g]["n_grid_points"], "auroc": result_groups[g]["auroc"]}
                    for g in EVAL_GROUP_NAMES
                },
            })
        else:
            # Multiple layers — write per-root layer summary.
            layer_results = []
            for i in sorted(indices, key=lambda j: spec_meta[j]["layer_index"]):
                meta = spec_meta[i]
                r = results_by_id[probe_specs[i].probe_id]
                row: dict[str, Any] = {"layer_index": meta["layer_index"]}
                if filter_last_replay_role is not None:
                    row["filter_last_replay_role"] = filter_last_replay_role
                for gname in EVAL_GROUP_NAMES:
                    row[f"{gname}_auroc"] = r.groups.get(gname, {}).get("auroc")
                    row[f"{gname}_n"] = r.groups.get(gname, {}).get("n", 0)
                layer_results.append(row)
                sa = row.get("strict_auroc") or 0
                _emit_log(log_fn, f"  {ckpt_root.name} L{meta['layer_index']}: strict={sa:.4f}")

            per_output = output_dir / f"{ckpt_root.name}_all_layers.json"
            with open(per_output, "w") as f:
                json.dump(layer_results, f, indent=2)
            _emit_log(log_fn, f"wrote {len(layer_results)} layers -> {per_output}")
            best = max(layer_results, key=lambda r: r.get("strict_auroc") or 0)
            all_results.append({
                "output_path": str(per_output),
                "checkpoint_root": str(ckpt_root),
                "n_layers": len(layer_results),
                "best_layer": best["layer_index"],
                "best_strict_auroc": best.get("strict_auroc"),
            })

    if len(all_results) == 1:
        return all_results[0]
    return {"probe_results": all_results}


def _cmd_split_features(args: argparse.Namespace) -> dict[str, Any]:
    from .featurization.feature_io import write_feature_payload_split

    log_fn = _make_stage_log_fn(args, "split-features")
    root = Path(args.root)
    feature_name = args.feature_name
    paths = ProductPaths.from_root(root)
    feature_dir = paths.features_dir / feature_name

    if not feature_dir.is_dir():
        raise FileNotFoundError(f"feature directory not found: {feature_dir}")

    if args.grid_points:
        targets = [feature_dir / f"{gp}.pt" for gp in args.grid_points]
    else:
        targets = sorted(p for p in feature_dir.glob("*.pt") if p.is_file())

    _emit_log(log_fn, f"found {len(targets)} bulk feature file(s)")
    done, skipped, failed = 0, 0, 0
    t0 = time.monotonic()

    for i, pt_path in enumerate(targets, 1):
        gp_id = pt_path.stem
        if paths.is_feature_split(gp_id, feature_name):
            skipped += 1
            continue
        _emit_log(log_fn, f"[{i}/{len(targets)}] {gp_id}")
        try:
            payload = load_feature_payload(pt_path)
            write_feature_payload_split(root=root, payload=payload, keep_original=args.keep_original)
            done += 1
        except Exception as exc:
            failed += 1
            _emit_log(log_fn, f"[{i}/{len(targets)}] FAIL {gp_id}: {exc}")

    elapsed = time.monotonic() - t0
    _emit_log(log_fn, f"done in {elapsed:.0f}s: {done} split, {skipped} skipped, {failed} failed")
    return {"done": done, "skipped": skipped, "failed": failed, "elapsed_seconds": elapsed}


def _find_partition(probe_path: Path, dataset_name: str | None = None, root: Path | None = None) -> dict[str, Any]:
    """Find and return the partition.json used to train the probe.

    If *dataset_name* is given, load that specific partition directly.
    Otherwise searches two locations:
    1. ``<probe_path>/datasets/`` (probe-local)
    2. ``<probe_path>/../../datasets/`` (run-root level)

    If *root* is given, also searches ``<root>/datasets/`` as a fallback.

    Returns the first matching partition.json found.  Raises if none found.
    """
    if dataset_name is not None:
        search_dirs = [probe_path / "datasets", probe_path.parent.parent / "datasets"]
        if root is not None:
            search_dirs.append(root / "datasets")
        for datasets_dir in search_dirs:
            p = datasets_dir / dataset_name / "partition.json"
            if p.exists():
                return read_json(p)
        raise ValueError(f"Cannot find partition for dataset_name={dataset_name!r}")
    candidates = [
        probe_path / "datasets",
        probe_path.parent.parent / "datasets",
    ]
    if root is not None:
        candidates.append(root / "datasets")
    for datasets_dir in candidates:
        if not datasets_dir.is_dir():
            continue
        for d in sorted(datasets_dir.iterdir()):
            if not d.is_dir():
                continue
            partition_path = d / "partition.json"
            if not partition_path.exists():
                continue
            return read_json(partition_path)
    raise ValueError(f"Cannot find partition.json from {probe_path}")




def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except ValueError:
        return {}


def _grid_point_parts(grid_point_id: str, manifest: Mapping[str, Any]) -> tuple[str, str, str]:
    parts = grid_point_id.split("__", 2)
    suite_name = str(manifest.get("suite_name") or (parts[0] if len(parts) > 0 else "unknown"))
    system_prompt_key = str(manifest.get("system_prompt_key") or (parts[1] if len(parts) > 1 else "unknown"))
    attack_name = manifest.get("attack_name")
    if attack_name in (None, ""):
        attack_name = parts[2] if len(parts) > 2 else "clean"
    return suite_name, system_prompt_key, str(attack_name)


def _format_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _reorder_decision_points_for_featurization(
    decision_points: Sequence[Any],
    *,
    bucket_by_prompt_length: bool,
    longest_first: bool,
) -> Sequence[Any]:
    if not bucket_by_prompt_length and not longest_first:
        return decision_points
    return sorted(
        list(decision_points),
        key=lambda point: (
            -(len(point.prompt_token_ids or ())) if longest_first else len(point.prompt_token_ids or ()),
            point.decision_point_id,
        ),
    )


def _filter_decision_points_for_featurization(
    decision_points: Sequence[Any],
    *,
    model_family: str,
    filter_last_replay_role: str | None,
) -> list[Any]:
    if filter_last_replay_role is None:
        return list(decision_points)
    if model_family != "gemma4":
        raise ValueError(
            "--filter-last-replay-role is currently supported only for model_family='gemma4'"
        )

    filtered: list[Any] = []
    for point in decision_points:
        replay_request = getattr(point, "replay_request", None) or {}
        messages = replay_request.get("messages") or ()
        if not messages:
            continue
        last_role = str(messages[-1].get("role") or "")
        if last_role == filter_last_replay_role:
            filtered.append(point)
    return filtered


def _count_all_grid_point_rows(root: Path, filename: str) -> int:
    total = 0
    for path in sorted((root / "grid_points").glob(f"*/{filename}")):
        if path.exists():
            total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


if __name__ == "__main__":
    main(sys.argv[1:])
