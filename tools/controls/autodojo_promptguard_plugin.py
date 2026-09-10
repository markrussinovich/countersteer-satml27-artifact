"""Register the SoA batteries' PromptGuard-2 filter as an AutoDojo pipeline defense.

Loaded through the vendored fork's plugin seam: list this module's name in the
`AGENTDOJO_DEFENSE_PLUGINS` env var (autodojo_job.sh --arm promptguard does this), and
`agentdojo.agent_pipeline.agent_pipeline` imports it at load time, making
`--defense promptguard_soa` selectable.

The element is the EXACT instrument the SoA batteries measured (static AgentDojo profile:
security 0.205 gpt-oss / 0.228 Qwen, benign 100%, fp~0): `agentdojo_smoke.LocalPIDetector
("pi_detector_promptguard")` -- AgentDojo's own TransformersBasedPIDetector wiring
(mode="message", threshold 0.5, upstream '<Data omitted because a prompt injection was
detected>' redaction) generalised to the pinned meta-llama/Llama-Prompt-Guard-2-86M
checkpoint (rev a8ded8e6), with the 480/64-token overlapping-window scan (a single
truncated 512-token call is a measured blind spot that flatters the defense -- SoA
adversarial review 2026-09-04) and engagement counters. Reused, never reimplemented
(prereg v3.3). The gated checkpoint resolves via XPIA_MODEL_STORE
(agentdojo_smoke._pi_model_path); export it on lane boxes without hub credentials.

NOT added to PARALLEL_EVAL_SAFE_DEFENSES: plugin targets run serial evaluation anyway
(stateful GPU element), and this shared-pipeline classifier has no lock of its own here.
"""
from agentdojo.agent_pipeline.agent_pipeline import register_defense


def _factory(config):  # (config: PipelineConfig) -> BasePipelineElement
    # DEFERRED import: this module is imported by _load_defense_plugins() at the
    # BOTTOM of agent_pipeline.py's module body, while the agentdojo.agent_pipeline
    # package __init__ is still mid-execution. agentdojo_smoke does
    # `from agentdojo.agent_pipeline import AgentPipeline, ...`, which would hit the
    # partially initialized package and ImportError. By factory-call time
    # (AgentPipeline.from_config) every module is fully loaded.
    import atexit

    from agentdojo_smoke import LocalPIDetector, _pi_model_path

    el = LocalPIDetector("pi_detector_promptguard")
    path, _ = _pi_model_path("meta-llama/Llama-Prompt-Guard-2-86M")
    print(f"[promptguard_soa] detector built: checkpoint={path} "
          f"threshold={el.threshold} mode={el.mode} safe_label={el.safe_label}",
          flush=True)

    # ENGAGEMENT TELEMETRY (adversarial review BLOCKING fix 1): without this, a
    # silently no-op filter (wrong device / dropped element / dead label map) is
    # indistinguishable in the artifact from the adaptive filter-collapse result this
    # arm exists to measure. "A filter row with zero engagements is a no-op, not a
    # defense" (SoA batteries). Periodic print every 25 checks + an atexit SUMMARY.
    orig_detect = el.detect

    def _detect(tool_output):
        r = orig_detect(tool_output)
        if el.n_checked % 25 == 0:
            print(f"[promptguard_soa] checked={el.n_checked} flagged={el.n_flagged} "
                  f"chunked={el.n_chunked}", flush=True)
        return r

    el.detect = _detect
    atexit.register(lambda: print(
        f"[promptguard_soa SUMMARY] checked={el.n_checked} flagged={el.n_flagged} "
        f"chunked={el.n_chunked}", flush=True))
    return el


register_defense("promptguard_soa", _factory)
