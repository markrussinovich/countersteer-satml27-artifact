"""xpia — activation steering against cross-prompt injection on tool output.

Layered so each module has one job and the dependency graph is acyclic:

    common      constants: repo root, roles, probe hyperparameters, the span sentinel
    model       loading, the decoder-block container, the two hook sites
    moe         MoE router discovery in a live model; router weights off a checkpoint
    templates   harmony rendering, sentinel span location
    corpora     datasets, attacker-template-disjoint splits, probe-training text
    probes      role probes, steering directions, magnitude matching
    spans       which tokens an intervention touches
    steering    the forward hook that edits the residual stream
    scoring     deterministic scoring -- no LLM anywhere in this module
    judge       the optional LLM judge (off by default; never computes correctness)
    arms        one evaluation arm: generate, then score against the unattacked reference
    cli         stage selection, sweep driver, report tables

Import direction is strictly downward in that list; there are no cycles.

`xpia_defense.py` at the repo root re-exports every name, so existing
`import xpia_defense as X` call sites in tools/ keep working unchanged.
"""
