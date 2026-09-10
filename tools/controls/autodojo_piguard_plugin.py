"""Register the SoA batteries' PIGuard filter as an AutoDojo pipeline defense.

Clone of autodojo_promptguard_plugin.py (owner-approved detector-adaptive program,
2026-09-10: PIGuard is the strongest STATIC detector on all three flagships — 41x/88x/18x
— with zero adaptive evidence; its classifier siblings collapse under AutoDojo
optimization, PromptGuard-2 0.000->0.239 / CachePrune 0.000->0.208 on Qwen). Loaded via
the fork's AGENTDOJO_DEFENSE_PLUGINS seam (autodojo_job.sh --arm piguard); the element is
the EXACT SoA instrument: agentdojo_smoke.LocalPIDetector("pi_detector_piguard") — pinned
leolee99/PIGuard rev dd78b24e, trust_remote_code via the pinned revision, 480/64-token
overlapping-window scan, upstream redaction string. Checkpoint resolves via
XPIA_MODEL_STORE (flat dir `PIGuard`), never the hub, on boxes without credentials.

Engagement telemetry as in the PromptGuard plugin: a filter row with zero engagements is
a no-op, not a defense.
"""
from agentdojo.agent_pipeline.agent_pipeline import register_defense


def _factory(config):  # (config: PipelineConfig) -> BasePipelineElement
    # DEFERRED import — see autodojo_promptguard_plugin._factory for why.
    import atexit

    from agentdojo_smoke import LocalPIDetector, _pi_model_path

    el = LocalPIDetector("pi_detector_piguard")
    path, _ = _pi_model_path("leolee99/PIGuard")
    print(f"[piguard_soa] detector built: checkpoint={path} "
          f"threshold={el.threshold} mode={el.mode} safe_label={el.safe_label}",
          flush=True)

    orig_detect = el.detect

    def _detect(tool_output):
        r = orig_detect(tool_output)
        if el.n_checked % 25 == 0:
            print(f"[piguard_soa] checked={el.n_checked} flagged={el.n_flagged} "
                  f"chunked={el.n_chunked}", flush=True)
        return r

    el.detect = _detect
    atexit.register(lambda: print(
        f"[piguard_soa SUMMARY] checked={el.n_checked} flagged={el.n_flagged} "
        f"chunked={el.n_chunked}", flush=True))
    return el


register_defense("piguard_soa", _factory)
