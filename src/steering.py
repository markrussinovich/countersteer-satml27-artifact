"""The intervention itself: the forward-hook that edits the residual stream, the displacement-proportional step rules, the per-token gate, and control-direction naming."""
from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from .common import *  # noqa: F401,F403
from .model import layer_container, pick_site, rewrap, tensor_of
from .moe import block_routers, pre_mlp_norm


# ═════════════════════════════════════════════════════ the steering-mode vocabulary
# ONE definition of what `--mode` may be, read by the hook below, by run_arm's
# steer-construction guard (src/arms.py) and by the CLI's `--mode` choices.
#
# WHY IT IS CENTRALISED. Those three lists used to be written out by hand, and the
# 2026-09-01 silent-no-op incident (FINDINGS §23e) is exactly what happens when they drift:
# a DOSE-FREE mode that is missing from run_arm's guard gets `steer = None` at `--alphas 0`,
# so no hook is registered and the arm runs COMPLETELY UNDEFENDED while labelling itself a
# defense. Adding a mode to `ABLATE_MODES` now propagates to the guard, the hook and the CLI
# at once; it cannot be added to one and forgotten in another.
ADD_MODES = ("add", "ablate_add", "ablate_mp_add")
# every mode whose ablation branch fires
ABLATE_MODES = ("ablate", "ablate_add", "ablate_mp", "ablate_mp_add")
# MEAN-PRESERVING ablation: remove only the CENTERED component (see the hook).
MEAN_PRESERVING_MODES = ("ablate_mp", "ablate_mp_add")
# modes that set their OWN magnitude and therefore MUST build a Steer even at alpha 0.
# This is the tuple run_arm's guard tests; keep it as the union, never a hand-copied list.
DOSE_FREE_MODES = ABLATE_MODES
MODES = ("add",) + ABLATE_MODES


# ════════════════════════════════════════════════════════════ steering
class Steer:
    """Adds alpha/sqrt(k) * ||h|| * d_hat at given positions, at each of k layers.

    Prefill only: the payload tokens are consumed once and cached, so their influence on
    the whole generation is fixed there. `positions` is per batch ROW and must be sliced
    to match whenever the batch is split.
    """

    def __init__(self, model, layers, dirs_by_layer, alpha_total,
                 scale="sigma", sigmas=None, ablate_axes=None, mode="add",
                 gate_proj=None, gate_ramp=1.0,
                 norm_preserve=True, probes=None, tau=0.0,
                 step_rule="fixed", boundary=None, margin=0.0, step_scale=1.0,
                 delta_maps=None,
                 decode_dirs=None, decode_alpha=0.0, decode_sigmas=None,
                 decode_scale="sigma", decode_gate=None, decode_gate_ramp=1.0,
                 router_blind=None, mean_acts=None, mean_from_span=False):
        blocks = layer_container(model)
        self.mods = [pick_site(blocks[L])[0] for L in layers]
        self.dirs = dirs_by_layer
        self.a = alpha_total / (len(layers) ** 0.5)
        self.scale = scale            # "sigma" (ITI: alpha*sigma) or "norm" (alpha*||h||)
        self.sigmas = sigmas or [0.0] * len(layers)
        self.ablate = ablate_axes or [{}] * len(layers)
        if mode not in MODES:
            raise SystemExit(f"unknown steering mode {mode!r}; have {list(MODES)}")
        self.mode = mode              # see MODES above
        # MEAN-PRESERVING ABLATION (mode=ablate_mp / ablate_mp_add). `mean_acts` is one
        # FIXED mean activation vector mu per steered layer (src/probes.build_means);
        # `mean_from_span` instead uses each row's OWN mean over its edited positions.
        # Exactly one of the two, and only for a mean-preserving mode.
        #
        # NO SILENT ZERO. A missing mu would make this operator numerically IDENTICAL to
        # plain `ablate` while still reporting itself as the mean-preserving arm -- the same
        # class of bug as the missing sigma that once turned a step into `alpha*0` and got
        # reported as a defense. So it raises here, at construction, before any generation.
        self.mean_acts = mean_acts
        self.mean_from_span = bool(mean_from_span)
        self._mean_cast = [None] * len(layers)
        self._dirs32 = [None] * len(layers)     # full-precision direction, cached per layer
        if mode in MEAN_PRESERVING_MODES:
            if self.mean_from_span == (mean_acts is not None):
                raise SystemExit(
                    f"mode={mode} needs EXACTLY ONE mean source: pass mean_acts (a fixed mu "
                    f"per steered layer, from probes.build_means) or mean_from_span=True. "
                    f"Got mean_acts={'set' if mean_acts is not None else 'None'}, "
                    f"mean_from_span={self.mean_from_span}.")
            if mean_acts is not None:
                if len(mean_acts) != len(layers) or any(m is None for m in mean_acts):
                    raise SystemExit(
                        f"mode={mode}: mean_acts must hold one non-None mu per steered "
                        f"layer; got {len(mean_acts) if mean_acts is not None else 0} for "
                        f"{len(layers)} layers ({[m is None for m in mean_acts]}).")
        elif mean_acts is not None or self.mean_from_span:
            raise SystemExit(
                f"mean_acts/mean_from_span were given but mode={mode!r} never reads them -- "
                f"that reads as a mean-preserving arm and would run as {mode!r}. Use one of "
                f"{list(MEAN_PRESERVING_MODES)}.")
        # ── OPERATORS THAT SILENTLY SWALLOW EACH OTHER (adversarial review, 2026-09-02) ──
        # The hook's branches are ordered, and two of those orderings produce an arm that
        # names two interventions and runs one. Both are the FINDINGS §23e shape -- a cell
        # tag that promises a defense the hook never applied -- so they are refused here
        # rather than left to be spotted in a log.
        if mode in ABLATE_MODES and delta_maps is not None:
            raise SystemExit(
                f"mode={mode} with a learned AlphaSteer map: the delta_maps branch returns "
                f"BEFORE the ablation branch, so the ablation would never run while the arm "
                f"still labels itself `{mode}`. Run them as separate arms.")
        if mode in ABLATE_MODES and mode not in ADD_MODES and (
                step_rule != "fixed" or tau > 0 or gate_proj is not None):
            raise SystemExit(
                f"mode={mode} is ablation-ONLY, but step_rule={step_rule!r}, tau={tau} and "
                f"gate_proj are read only by the ADDITIVE branch -- they would be silently "
                f"ignored while appearing in the cell name. Use {mode}_add "
                f"(or plain `add`) if the additive step is wanted.")
        self.norm_preserve = norm_preserve
        # CONDITIONAL (per-token) steering. Steering every payload token uniformly also
        # perturbs the legitimate record content the model needs to answer -- which is why
        # correctness sat at 47-65% even where ASR hit 0. With tau > 0 each token is scaled
        # by its own tool-ness DEFICIT, max(0, tau - p_tool)/tau, so tokens the probe
        # already reads as tool-like are left alone and only the confused ones are pushed.
        self.probes = probes or [None] * len(layers)   # (W, b) per layer, tool row index
        self.tau = tau
        # DISPLACEMENT-PROPORTIONAL STEPS. `fixed` is the historical rule: every token in the
        # span gets alpha*sigma regardless of what it carries. That is the maximum-cost
        # setting on the one axis never varied here, and it is why steering an
        # INJECTION-FREE payload costs ~36% correctness: the span is a median 148 tokens of
        # which the injection is only ~33%, so two thirds of every edited token is the
        # legitimate record the model must copy parameter values out of.
        #
        # The alternatives make the step a function of how much override-ness the token
        # actually carries, so a token already on the safe side moves by exactly zero:
        #   boundary  ARGUS (arXiv:2512.05745): alpha_t = max(0, p_ov - m + margin)
        #             -- the minimum displacement that crosses the boundary, plus a margin.
        #   mirror    StMP (arXiv:2604.08169): alpha_t = 2 * max(0, p_ov - m)
        #             -- reflect across the boundary instead of landing on it.
        # p_ov is the token's projection onto dim_OVERRIDE in sigma units. Note the sign:
        # the steered direction is dim_no_override = -dim_override, so p_ov = -(h.u)/sigma.
        # Measured class means (runs/gate_separability.json, projections onto dim_override in
        # sigma units) confirm injected tokens sit HIGHER: L12 -0.83 injected vs -2.18
        # legitimate, L16 -0.72 vs -1.77, L20 -2.79 vs -3.49.
        self.step_rule = step_rule           # fixed | boundary | mirror
        self.boundary = boundary or [0.0] * len(layers)   # m, per layer, in SIGMA units
        self.margin = margin                 # tau in the ARGUS rule, in sigma units
        self.step_scale = step_scale         # global multiplier; 1.0 = exactly the rule
        # ALPHASTEER (arXiv:2506.07022): a LEARNED per-layer map replacing the fixed vector.
        #   h' = h + step_scale * (Delta @ h)
        # Delta = Delta_tilde @ P with P the projector onto the null space of BENIGN
        # activations, so Delta @ h ~ 0 on a benign token by construction rather than by a
        # tuned threshold. This is the only operator here whose clean-input cost is structural.
        # `alpha` does NOT scale it -- the magnitude is baked into the map's regression target
        # at build time (tools/controls/build_alphasteer.py --target-sigmas), so that a sweep
        # cannot silently rescale a learned map and call it the same intervention.
        self.delta_maps = delta_maps          # list per steered layer, or None
        # per-token gate on the steering direction's own projection (sigma units)
        self.gate_proj = gate_proj
        self.gate_ramp = gate_ramp
        # DECODE-TIME steering (FINDINGS §15, opt-in, 2026-08-31). Historically this class
        # was PREFILL-ONLY: the `h.shape[1] == 1` early-return made every decode step a
        # no-op, so the defense never touched the point where the model COMMITS to an
        # argument value -- which is where the decision-point fit locates the param-hijack
        # representation. With `decode_dirs` set, every generated token's residual gets
        # alpha_dec/sqrt(k) * sigma_dec[i] * d_dec[i] added (norm-preserved like the
        # prefill edit). This composes ON TOP of unchanged prefill steering: the deployed
        # cell's prefill edit is byte-identical whether or not decode mode is on.
        self.decode_dirs = decode_dirs
        # per-layer cache of the dtype-cast direction: the cast at line ~148 otherwise
        # reallocates every decode step at every steered layer (review 2026-08-31)
        self._decode_dirs_cast = [None] * len(layers) if decode_dirs else None
        self.decode_a = (decode_alpha / (len(layers) ** 0.5)) if decode_dirs else 0.0
        self.decode_sigmas = decode_sigmas or [0.0] * len(layers)
        # `sigma` = alpha_dec * sigma_dec (ITI convention). The dp sigmas are the spread of
        # UNCENTERED completion-token activations along the axis and are comparable to
        # ||h|| itself at late layers, so sigma-scaled decode steps destroyed generation at
        # alpha 4 (trunc 1.00, measured 2026-08-31). `norm` = alpha_dec * ||h||: alpha is
        # then a fraction of the activation's own magnitude, uniform across layers.
        self.decode_scale = decode_scale
        # per-token gate on the COMMIT projection (= -decode_dir), in decode-sigma units:
        # only tokens actually reading as attacker-value commitment get pushed; ordinary
        # emission is untouched. None = every decode token.
        self.decode_gate = decode_gate
        self.decode_gate_ramp = decode_gate_ramp
        self.decode_steps = 0          # decode forwards edited (counted at the first layer)
        # skip the prefill loop entirely when this instance steers ONLY at decode: with
        # alpha_total=0 the add branch would still run a step-0 edit + norm rescale over
        # the span, and a nominally-identity bf16 rescale is not worth trusting byte-for-byte
        self.prefill_off = (alpha_total == 0 and mode == "add" and step_rule == "fixed"
                            and delta_maps is None)
        # ROUTER-BLIND RESIDUAL STEERING (opt-in, default None = every existing cell is
        # untouched). When set to a RouterBlind, this hook additionally REPORTS the exact
        # per-token edit it applied, so the controller can subtract it back out of the input
        # of every DOWNSTREAM MoE router. The edit itself is unchanged; the only new work in
        # this hook is one float32 subtraction per steered layer, behind a `is not None`
        # guard, so with the flag off the tensor arithmetic is bit-for-bit what it was.
        self.router_blind = router_blind
        self.positions: list[list[int]] | None = None
        self._h = []

    def __enter__(self):
        self._h = [m.register_forward_hook(self._mk(i)) for i, m in enumerate(self.mods)]
        return self

    def __exit__(self, *a):
        for h in self._h:
            h.remove()
        self._h = []

    def _mu(self, i, h):
        """The FIXED mean activation for steered-layer index `i`, as float32 [1, D].

        Cached per layer, and re-cast if the layer moved device: under device_map=auto the
        steered layers live on different GPUs, so a mu built on cuda:0 would crash at
        layer i's device -- the same trap the direction cast at the top of the hook exists
        for."""
        m = self._mean_cast[i]
        if m is None or m.device != h.device:
            m = self.mean_acts[i].to(device=h.device, dtype=torch.float32).reshape(1, -1)
            self._mean_cast[i] = m
        return m

    def _mk(self, i):
        def hook(mod, inp, out):
            h = tensor_of(out)
            if h.shape[1] == 1:
                # DECODE STEP. Historically an unconditional no-op; with decode mode on,
                # every generated token gets a fixed step along the decode direction.
                # Applied to every row of the (sub-)batch -- there is no per-token span at
                # decode; the whole point is to touch the emission the prefill edit cannot.
                if not (self.decode_dirs and self.decode_a):
                    return out
                if not self.decode_sigmas[i] > 0:
                    raise RuntimeError(
                        f"decode steering needs a sigma for layer index {i}; got "
                        f"{self.decode_sigmas[i]} -- refit/write the decode direction's "
                        f"sigma into the probe pickle, do not fall back to another unit.")
                dd = self._decode_dirs_cast[i]
                if dd is None or dd.dtype != h.dtype:
                    dd = self.decode_dirs[i].to(device=h.device, dtype=h.dtype)
                    self._decode_dirs_cast[i] = dd
                cur = h[:, 0]
                pre = cur.norm(dim=-1, keepdim=True)
                step = (self.decode_a * pre if self.decode_scale == "norm"
                        else self.decode_a * self.decode_sigmas[i])
                if self.decode_gate is not None:
                    # commit-ness = projection onto -decode_dir (the ADDED direction is
                    # dp_no_commit = -dp_commit), in decode-sigma units, same convention
                    # as the stage-1 analysis
                    u = dd.float() / dd.float().norm().clamp_min(1e-6)
                    p_com = -(cur.float() @ u) / max(self.decode_sigmas[i], 1e-6)
                    gate = ((p_com - self.decode_gate[i])
                            / max(self.decode_gate_ramp, 1e-6)).clamp(0.0, 1.0)
                    step = step * gate.unsqueeze(-1).to(cur.dtype) if torch.is_tensor(step) \
                        else (step * gate.unsqueeze(-1)).to(cur.dtype)
                cur = cur + step * dd
                if self.norm_preserve:
                    cur = cur * (pre / cur.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                h[:, 0] = cur
                if i == 0:
                    self.decode_steps += 1
                return rewrap(out, h)
            if self.positions is None or self.prefill_off:
                return out
            # device AND dtype: under device_map=auto the layers live on different GPUs,
            # and a direction built on cuda:0 would crash (or silently sync) at layer i's
            # device -- multi-GPU support for the 80B/106B bring-ups (2026-08-31)
            d = self.dirs[i].to(device=h.device, dtype=h.dtype)
            # MEAN-PRESERVING ABLATION works in float32 and needs the FULL-PRECISION
            # direction: `d` above has already been rounded to the activation dtype, so
            # `d.float()` would be a float32 tensor carrying bf16 precision. Cached and
            # hoisted out of the per-row loop -- re-casting it per batch row was pure
            # repeated work (review, 2026-09-02).
            d32 = None
            if self.mode in MEAN_PRESERVING_MODES:
                d32 = self._dirs32[i]
                if d32 is None or d32.device != h.device:
                    d32 = self.dirs[i].to(device=h.device, dtype=torch.float32)
                    self._dirs32[i] = d32

            def commit(b, sel, cur, orig, idxs):
                """Write the edited rows back, and (only under router-blind steering) hand
                the controller the EXACT applied displacement so downstream routers can be
                shown the un-edited stream. float32 on both sides: differencing two bf16
                tensors would quantise the very quantity the correction is built from."""
                h[b, sel] = cur
                if orig is not None:
                    self.router_blind.note_edit(i, b, idxs, cur.float() - orig.float())

            for b, idxs in enumerate(self.positions):
                if b >= h.shape[0] or not idxs:
                    continue
                idxs = [j for j in idxs if j < h.shape[1]]
                if not idxs:
                    continue
                sel = torch.tensor(idxs, device=h.device)
                cur = h[b, sel]
                # advanced indexing returns a COPY, so this keeps the pre-edit rows even
                # after h[b, sel] is written. None when router-blind steering is off, which
                # makes `commit` a plain assignment.
                orig = cur if self.router_blind is not None else None

                if self.delta_maps is not None and self.delta_maps[i] is not None:
                    Dm = self.delta_maps[i]
                    pre = cur.norm(dim=-1, keepdim=True)
                    cur = cur + self.step_scale * (cur.float() @ Dm.T).to(cur.dtype)
                    if self.norm_preserve:
                        cur = cur * (pre / cur.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                    commit(b, sel, cur, orig, idxs)
                    continue

                if self.mode in ABLATE_MODES:
                    if self.mode in MEAN_PRESERVING_MODES:
                        # MEAN-PRESERVING DIRECTIONAL ABLATION.
                        #   h <- h - ((h - mu) . d) d      [= plain ablate + (mu.d) d]
                        #
                        # WHY. Plain ablation deletes the WHOLE coordinate, and the
                        # coordinate has a large non-zero MEAN, so deleting it is not a
                        # neutral removal: it applies a net push of -(mu.d) per token.
                        # Measured on Qwen3-Next-80B at the steered layers L28/32/40 that
                        # push is -1.86 sigma summed over layers from the INJECTED-SPAN
                        # capture mean (additive alpha ~ -1.1) and -2.83 sigma from the
                        # PROBE-CORPUS mean (alpha ~ -1.6) -- same measurement, two mu
                        # estimates, always quote which. Either way it points in the
                        # ATTACK-favouring direction, since attack-succeeded rows sit LOWER
                        # on d (FINDINGS §23k). The measured
                        # ablation NULL there is therefore confounded: it cannot separate
                        # "removing the axis does nothing" from "removing it would have
                        # helped and the accompanying negative shift cancelled it".
                        #
                        # Subtracting mu's own component first removes the DISCRIMINATIVE
                        # part of the coordinate while contributing zero net displacement
                        # along d -- EXACTLY when mu is the mean of the tokens being
                        # edited, and only approximately otherwise. The two sources differ
                        # in WHICH half of that they get right, and they BRACKET the
                        # operator rather than one dominating the other:
                        #
                        #   mean_acts (fixed, probes.build_means)
                        #       deletes the span-level offset along d as well as the
                        #       within-span structure, but the stored mean is a PROBE-SITE
                        #       (pre-MLP) corpus mean while this hook edits the BLOCK
                        #       OUTPUT -- so a residual push survives. Measured on
                        #       gpt-oss-20b against a block_out capture: 0.4-2.7 sigma per
                        #       layer, i.e. most of the confound removed, not all of it.
                        #       Unmeasured on the 80-106B models.
                        #   mean_from_span (per row)
                        #       mu IS the mean of these activations at THIS site, so the
                        #       net displacement is exactly 0 -- but only the WITHIN-span
                        #       component of the coordinate is deleted; the span's own
                        #       offset along d is preserved untouched. It is also
                        #       input-dependent, which is an adaptive-attack surface a
                        #       fixed mu does not have: an attacker owning most of the
                        #       span owns mu.
                        #
                        # Degenerate case, asserted in verify_ablate_mp.py: a ONE-token
                        # span under mean_from_span is an exact no-op.
                        #
                        # FLOAT32, deliberately, unlike the plain-ablate branch below, and
                        # against the FULL-PRECISION direction (`d32`, not `d.float()`).
                        # This is a small, cheap accuracy gain, not a necessity, and the
                        # measurement says so: residual error on the edited coordinate is
                        # 0.0013 sigma this way vs 0.0064 sigma all-bf16, against a mu term
                        # of ~2.26 sigma and a mu-mismatch that dominates at ~0.31 sigma. It
                        # costs ~3.5x the transient of the plain branch (23 MB vs 6.5 MB at
                        # T=400/D=4096) on a path that fires once per prefill batch per
                        # steered layer, ~18 times per arm in the n=24 smoke. The cast back
                        # to the activation dtype happens immediately, so what lands in the
                        # residual stream is the model's own dtype either way.
                        c32 = cur.float()
                        mu = (c32.mean(dim=0, keepdim=True) if self.mean_from_span
                              else self._mu(i, h))
                        proj = (c32 - mu) @ d32
                        cur = (c32 - proj.unsqueeze(-1) * d32.unsqueeze(0)).to(h.dtype)
                    else:
                        # Directional ablation (Arditi et al.): project the activation onto
                        # the orthogonal complement of the STEERING direction itself.
                        #   h <- h - (h . d) d
                        # Ablate the same difference-in-means axis we would otherwise add
                        # along. The earlier implementation projected out raw class MEANS,
                        # which were cos 0.997 with the global activation mean and cos 0.992
                        # with each other -- that removed ~45% of every token and was
                        # capability destruction, not role correction.
                        cur = cur - (cur @ d).unsqueeze(-1) * d.unsqueeze(0)

                if self.mode in ADD_MODES and self.step_rule != "fixed":
                    # per-token displacement, in sigma units, along the SAFE direction
                    if not self.sigmas[i] > 0:
                        raise RuntimeError(
                            f"step_rule={self.step_rule} needs a sigma for layer index {i}; "
                            f"got {self.sigmas[i]}. Re-run --stage probe and --stage validate.")
                    uu = d.float() / d.float().norm().clamp_min(1e-6)
                    p_ov = -(cur.float() @ uu) / self.sigmas[i]        # override-ness, sigma
                    over = (p_ov - self.boundary[i]).clamp_min(0.0)
                    delta = (over + self.margin).clamp_min(0.0) if self.step_rule == "boundary" \
                        else 2.0 * over
                    # zero for tokens already below the boundary -- `over` is 0 there, and the
                    # margin must NOT resurrect them, hence it is applied inside the clamp
                    delta = torch.where(over > 0, delta, torch.zeros_like(delta))
                    step = (self.step_scale * delta * self.sigmas[i]).unsqueeze(-1).to(cur.dtype)
                    pre = cur.norm(dim=-1, keepdim=True)
                    cur = cur + step * d
                    if self.norm_preserve:
                        cur = cur * (pre / cur.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                    commit(b, sel, cur, orig, idxs)
                    continue
                if self.mode in ADD_MODES:
                    if self.scale == "sigma":
                        # Silently falling back to alpha*||h|| here turned a step of
                        # a*0.65 into a*32.6 -- a 50x scale change that still prints a
                        # plausible table. A missing sigma is a broken pickle, not a
                        # reason to change units mid-run.
                        if not self.sigmas[i] > 0:
                            raise RuntimeError(
                                f"--scale sigma but sigma is {self.sigmas[i]} for layer "
                                f"index {i}; the probe pickle predates this direction. "
                                f"Re-run --stage probe and --stage validate.")
                        step = self.a * self.sigmas[i]        # ITI: alpha * sigma
                    else:
                        step = self.a * cur.norm(dim=-1, keepdim=True)
                    if self.gate_proj is not None:
                        # PER-TOKEN GATE ON THE DIRECTION'S OWN PROJECTION.
                        #
                        # Uniform span-wide steering perturbs every payload token,
                        # including legitimate record content -- which is why steering an
                        # INJECTION-FREE payload already costs ~25% correctness at the
                        # alpha where ASR reaches 0. That cost is a ceiling no alpha can
                        # lift.
                        #
                        # The direction we steer along is also a per-token DETECTOR: a
                        # token's OVERRIDE-ness is how much authority claim it carries. Gate
                        # on that, and only tokens actually making the claim get moved;
                        # ordinary record tokens are left alone.
                        #
                        # gate = clamp((p_ov - thresh) / ramp, 0, 1)
                        # thresh is in SIGMA units so it is comparable across layers, whose
                        # raw projections differ by ~50x (sigma 15 at L4 vs 390 at L22).
                        #
                        # SIGN, and it was WRONG here until 2026-08-04. `d` is the STEERED
                        # direction, i.e. dim_no_override = -dim_override, so a projection
                        # onto `d` is NO-override-ness and a gate that ramps UPWARD in it
                        # steers the legitimate record tokens hardest and the injected ones
                        # least -- the exact opposite of the intent. Negate, matching the
                        # step_rule branch above, which had it right and said so.
                        # Measured (runs/gate_separability.json, projections onto
                        # dim_override in sigma units, injected vs legitimate):
                        #   L12 -0.83 vs -2.18   L16 -0.72 vs -1.77   L20 -2.79 vs -3.49
                        # injected sits HIGHER on dim_override, hence LOWER on the steered
                        # direction. Under the old sign, thresh=0.5/ramp=1.0 saturated the
                        # gate at 1.0 on legitimate tokens and left injected ones at ~0.33.
                        # That is what the four archived --gate-proj runs measured
                        # (results_add-dim-no-override-random-{3598703,3622684,3622830,
                        # 3622968}): gated ASR 0.208-0.333 against 0.083 ungated with
                        # correctness NOT recovered -- precisely an inverted gate's
                        # signature. The "per-token gating is ruled out" conclusion drawn
                        # from them is void; see todo/04-review-actions-2026-08-04.md.
                        #
                        # `--gate-proj` is therefore now in dim_OVERRIDE sigma units, which
                        # is the same convention gate_separability.py reports in, so its
                        # calibration table can be read straight into this flag.
                        #
                        # PER LAYER, not one scalar. Override-ness lives at very different
                        # offsets per layer -- injected/legitimate means are -0.83/-2.18 at
                        # L12 but -2.79/-3.49 at L20 -- so any single threshold that
                        # separates the classes at L12 sits ABOVE both classes at L20 and
                        # silently switches that layer's steering off entirely.
                        u = d / d.norm().clamp_min(1e-6)
                        p_ov = -(cur.float() @ u.float()) / max(self.sigmas[i], 1e-6)
                        gate = ((p_ov - self.gate_proj[i]) / max(self.gate_ramp, 1e-6)).clamp(0.0, 1.0)
                        step = step * gate.unsqueeze(-1).to(cur.dtype)
                    elif self.tau > 0 and self.probes[i] is not None:
                        # per-token gate: scale by this token's tool-ness deficit
                        W, bvec, ti = self.probes[i]
                        W, bvec = W.to(cur.device), bvec.to(cur.device)
                        logits = cur.float() @ W.T + bvec
                        p_tool = torch.softmax(logits, dim=-1)[:, ti]
                        gate = ((self.tau - p_tool).clamp_min(0.0) / self.tau)
                        step = step * gate.unsqueeze(-1).to(cur.dtype)
                    pre = cur.norm(dim=-1, keepdim=True)
                    cur = cur + step * d
                    if self.norm_preserve:
                        # Rotate toward tool-ness without inflating magnitude. Unconstrained
                        # addition previously reached 1.2-1.5x the mean activation norm at
                        # the alpha where ASR hit 0 -- overwriting the residual stream rather
                        # than steering it, which is what destroyed correctness.
                        cur = cur * (pre / cur.norm(dim=-1, keepdim=True).clamp_min(1e-6))

                commit(b, sel, cur, orig, idxs)
            return rewrap(out, h)
        return hook


# ═══════════════════════════ GatedDeltaNet delta-rule input steering (Qwen3-Next family)
# valid `op` values for GdnValueSteer. run_arm does NOT branch on this tuple (its guard is
# `gdn is not None`, deliberately op-blind); it exists so drivers and tests enumerate the
# ops from one place instead of hand-copied lists (the DOSE_FREE_MODES lesson, §23e).
GDN_OPS = ("v_add", "beta_scale", "v_scale")


class GdnValueSteer:
    """Prefill-only steering of a GatedDeltaNet layer's DELTA-RULE INPUTS, not the residual
    stream (FINDINGS §23ae.15: the residual line on Qwen3-Next is closed by owner order).

    Site: the per-module kernel attribute `linear_attn.chunk_gated_delta_rule(query, key,
    value, g=?, beta=?, ...)` -- post-conv, post-SiLU, exactly what the delta rule
    consumes, so capture and steer coincide byte-for-byte (the §23ae.11 fa-line lesson).
    Wrapping THIS attribute makes the edit prefill-only by construction: single-token
    cached decode dispatches to `recurrent_gated_delta_rule`, a different function that is
    never touched. The wrapper edits kernel INPUTS and passes everything else through, so
    it is identical under the fallback torch kernel and the pinned fused `fla` kernel
    (§23ae.10 box-scoping still rides every behavioral number).

    Ops (GDN_OPS):
      v_add       v_t <- renorm(v_t + (alpha/sqrt(k)) * sigma_L * d_hat_L) at the span
                  positions, in flattened (num_v_heads*head_v_dim) v-space. `renorm`
                  preserves the pre-edit flattened norm (norm_preserve, project
                  convention). v_t is WHAT the delta rule writes into the recurrent state;
                  addressing (k), write strength (beta), decay (g), the output gate (z)
                  and the query path are untouched.
      beta_scale  beta_t <- gamma * beta_t at the span positions: attenuate how strongly
                  the span is WRITTEN into the recurrent state (gamma=0: the span writes
                  nothing to state at the steered layers; span tokens still read state for
                  their own outputs, and full-attention layers still see the span). A
                  dose-parametrized mechanism probe -- no fitted direction, so it must be
                  built even at alpha 0 (run_arm's guard: `gdn is not None` is the test,
                  exactly so this class cannot reproduce the §23e alpha-0 no-op incident).
      v_scale     v_t <- v_t + (gamma - 1) * (v_t . d_hat_L) d_hat_L at the span
                  positions: MULTIPLICATIVE subspace scaling (arXiv 2602.22719; DESIGN.md
                  amendment 2, 2026-09-07). gamma=0 removes the direction's component
                  entirely; gamma=2 doubles it (the causal gate's attack side). gamma=1.0
                  exactly is refused (no-op wearing a label); production defense arms use
                  gamma in [0,1). Deliberately NO norm-preserve rescale: the operator
                  shrinks/scales one component, and renormalizing would reintroduce the
                  bulk displacement the additive gate showed to be the failure mode.
                  Dose-free (alpha stays 0).

    Interface parity with Steer, which is what lets run_arm/_generate thread it through
    unchanged: `.positions` (per batch ROW, left-pad offsets applied, sliced by _generate
    on sub-batch backoff; None = strict no-op, which is also what RouterBlind.prime relies
    on for its clean pass), `__enter__`/`__exit__`, `.router_blind` (attribute only --
    pairing requires RouterBlind mode="clean", enforced in run_arm, because `accum` needs
    per-edit residual deltas this class does not produce).

    Positive firing evidence for artifacts: `prefill_calls_edited` / `edited_tokens` --
    an arm reporting 0 is an undefended arm wearing a GDN label and run_arm warns on it.
    """

    def __init__(self, model, layers, dirs=None, alpha=0.0, sigmas=None,
                 op="v_add", gamma=1.0, norm_preserve=True):
        if op not in GDN_OPS:
            raise SystemExit(f"unknown gdn op {op!r}; have {list(GDN_OPS)}")
        blocks = layer_container(model)
        self.mods = []
        for L in layers:
            la = getattr(blocks[L], "linear_attn", None)
            if la is None or not hasattr(la, "chunk_gated_delta_rule"):
                raise SystemExit(
                    f"layer {L} has no GatedDeltaNet `linear_attn` with a "
                    f"`chunk_gated_delta_rule` attribute -- either it is a full-attention "
                    f"layer (Qwen3-Next: every 4th) or the installed modeling code moved. "
                    f"Refusing to run an arm whose hook cannot exist.")
            self.mods.append(la)
        self.layers = list(layers)
        self.op = op
        if op == "v_add":
            if dirs is None or sigmas is None or not alpha:
                raise SystemExit(
                    "gdn op=v_add needs dirs, sigmas and a non-zero alpha -- an alpha-0 "
                    "v_add arm is a no-op wearing a defense label (the §23e incident "
                    "class). Use op=beta_scale for the dose-free intervention.")
            if len(dirs) != len(self.layers) or len(sigmas) != len(self.layers):
                raise SystemExit(
                    f"gdn v_add: {len(self.layers)} layers but {len(dirs)} dirs / "
                    f"{len(sigmas)} sigmas")
            if any(not s or s <= 0 for s in sigmas):
                raise SystemExit(f"gdn v_add: non-positive sigma in {sigmas} -- a zero "
                                 f"sigma silently turns the step into alpha*0")
        elif op == "v_scale":
            if dirs is None or len(dirs) != len(self.layers):
                raise SystemExit(f"gdn v_scale needs one direction per layer; got "
                                 f"{0 if dirs is None else len(dirs)} for "
                                 f"{len(self.layers)}")
            if alpha:
                raise SystemExit("gdn v_scale is dose-free (the scale is gamma); passing "
                                 "alpha reads as a v_add arm that would silently not run")
            if gamma < 0 or gamma == 1.0:
                raise SystemExit(
                    f"gdn v_scale: gamma must be >= 0 and != 1.0, got {gamma} -- "
                    f"gamma=1.0 is an exact no-op that would still count 'edited' tokens "
                    f"(§23e class); gamma>1 is the causal gate's attack side only")
        else:
            if not (0.0 <= gamma < 1.0):
                raise SystemExit(
                    f"gdn beta_scale: gamma must be in [0,1), got {gamma} -- gamma=1.0 is "
                    f"an exact no-op that would still count 'edited' tokens and pass every "
                    f"firing-evidence guard (the §23e no-op-wearing-a-label class; "
                    f"adversarial review 2026-09-07, defect 9)")
            if dirs is not None or alpha:
                raise SystemExit("gdn op=beta_scale takes no dirs/alpha -- passing them "
                                 "reads as a v_add arm that would silently not run")
        self.dirs = dirs
        self.sigmas = sigmas or [0.0] * len(self.layers)
        self.a = (alpha / (len(self.layers) ** 0.5)) if op == "v_add" else 0.0
        self.gamma = float(gamma)
        self.norm_preserve = bool(norm_preserve)
        self._cast = [None] * len(self.layers)   # unit direction, float32, per layer/device
        self.positions: list[list[int]] | None = None
        self.router_blind = None
        self.prefill_calls_edited = 0
        self.edited_tokens = 0
        self._orig = []

    def __enter__(self):
        if self._orig:
            raise RuntimeError("GdnValueSteer entered twice without exit -- the second "
                               "enter would wrap the wrapper and double-apply the edit")
        self._orig = [m.chunk_gated_delta_rule for m in self.mods]
        for i, m in enumerate(self.mods):
            m.chunk_gated_delta_rule = self._wrap(i, self._orig[i])
        return self

    def __exit__(self, *a):
        for m, o in zip(self.mods, self._orig):
            m.chunk_gated_delta_rule = o
        self._orig = []

    def _dhat(self, i, device):
        d = self._cast[i]
        if d is None or d.device != device:
            d = self.dirs[i].to(device=device, dtype=torch.float32)
            d = d / d.norm().clamp_min(1e-30)
            self._cast[i] = d
        return d

    def _wrap(self, i, orig):
        def kern(query, key, value, *args, **kw):
            # value [B, S, num_v_heads, head_v_dim]; beta [B, S, num_v_heads] (kwarg).
            # Edit only multi-token forwards (prefill; the decode path uses the OTHER
            # kernel anyway, this is belt-and-braces) and only when positions are set --
            # None must be a strict no-op (RouterBlind.prime's clean pass depends on it).
            if self.positions is None or value.shape[1] <= 1:
                return orig(query, key, value, *args, **kw)
            if kw.get("initial_state") is not None:
                # cached CHUNKED CONTINUATION: the chunk kernel resuming from a prior
                # recurrent state sees CHUNK-RELATIVE positions, so prompt-relative span
                # indices would edit the wrong tokens. No path used here hits this
                # (generate() = full prefill then S=1 decode; RouterBlind.prime uses
                # use_cache=False), but a future chunked-prefill/speculative path must
                # skip rather than silently mis-edit (adversarial review 2026-09-07,
                # defect 10).
                return orig(query, key, value, *args, **kw)
            B, S = value.shape[0], value.shape[1]
            edited = 0
            if self.op == "v_scale":
                v = None
                d = self._dhat(i, value.device)
                for b, idxs in enumerate(self.positions):
                    if b >= B or not idxs:
                        continue
                    sel = [j for j in idxs if j < S]
                    if not sel:
                        continue
                    if v is None:
                        v = value.clone()
                    st = torch.tensor(sel, device=value.device)
                    cur = v[b, st].reshape(len(sel), -1).float()
                    proj = cur @ d
                    cur = cur + (self.gamma - 1.0) * proj.unsqueeze(-1) * d
                    v[b, st] = cur.to(v.dtype).reshape(len(sel), *value.shape[2:])
                    edited += len(sel)
                if v is not None:
                    value = v
            elif self.op == "v_add":
                v = None                     # clone lazily, only if something is edited
                d = self._dhat(i, value.device)
                step = self.a * float(self.sigmas[i])
                for b, idxs in enumerate(self.positions):
                    if b >= B or not idxs:
                        continue
                    sel = [j for j in idxs if j < S]
                    if not sel:
                        continue
                    if v is None:
                        v = value.clone()
                    st = torch.tensor(sel, device=value.device)
                    cur = v[b, st].reshape(len(sel), -1).float()
                    pre = cur.norm(dim=-1, keepdim=True)
                    cur = cur + step * d
                    if self.norm_preserve:
                        cur = cur * (pre / cur.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                    v[b, st] = cur.to(v.dtype).reshape(len(sel), *value.shape[2:])
                    edited += len(sel)
                if v is not None:
                    value = v
            else:                            # beta_scale
                beta = kw.get("beta")
                if beta is None:
                    raise RuntimeError(
                        "gdn beta_scale: the kernel was called without a `beta` kwarg -- "
                        "the installed modeling code passes beta positionally now; the "
                        "wrapper must be updated, not silently skipped.")
                nb = None
                for b, idxs in enumerate(self.positions):
                    if b >= B or not idxs:
                        continue
                    sel = [j for j in idxs if j < beta.shape[1]]
                    if not sel:
                        continue
                    if nb is None:
                        nb = beta.clone()
                    st = torch.tensor(sel, device=beta.device)
                    nb[b, st] = nb[b, st] * self.gamma
                    edited += len(sel)
                if nb is not None:
                    kw = dict(kw, beta=nb)
            if edited:
                self.edited_tokens += edited
                self.prefill_calls_edited += 1
            return orig(query, key, value, *args, **kw)
        return kern


# ═══════════════════════════════════════════ router-blind residual steering
class RouterBlind:
    """Make the residual edit invisible to the downstream MoE routers AT THE STEERED PREFILL
    POSITIONS, and visible to everything else.

    THE SCOPE QUALIFIER IN THAT SENTENCE IS LOAD-BEARING (adversarial review 2026-09-01,
    defect 2). Two things are deliberately NOT blinded and must never be claimed as blinded:
      * LATER, UNSTEERED POSITIONS in the same prefill. The edit reaches them through
        attention, and their router inputs are left perturbed -- correcting them would mean
        rewriting rows the intervention was never scoped to touch.
      * DECODE-TIME ROUTERS. Steering is prefill-only, so there is no decode edit to
        subtract, but the prefill edit still reaches decode through the KV cache.

    MOTIVATION (FINDINGS section 23, finding 3). The working gpt-oss cell leaves ~90% of
    top-k expert routing intact; on Qwen3-Next-80B and GLM-4.5-Air every behaviourally
    effective dose moves 55-90% of the routing mass, and the resulting degeneration
    (rumination, repetition loops, tool-call spam) is the signature of computing whole spans
    on the wrong experts. Direction quality is NOT the discriminator -- fired-vs-not
    separation is 0.8-1.0 sigma on all four models. So: keep the edit, hide it from the
    routers, let the experts that the CLEAN stream would have selected do the computing.

    ── WHERE THE ROUTER'S INPUT COMES FROM ──────────────────────────────────────────────
    Every installed MoE family has the same block shape:

        residual      = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)   # r -> x
        hidden_states = mlp(hidden_states)                        # x -> gate, experts, shared
        hidden_states = residual + hidden_states

    and inside `mlp` the SAME post-norm tensor `x` feeds the router, the routed experts and
    (where present) the shared expert. Editing only the router's copy therefore changes
    expert SELECTION and the combination WEIGHTS while leaving the expert INPUTS steered --
    which is exactly "steer in the experts, not beneath them".

      qwen3_next  Qwen3NextSparseMoeBlock.forward: x2 = x.view(-1, H); self.gate(x2);
                  self.experts(x2, ...); self.shared_expert(x2);
                  sigmoid(self.shared_expert_gate(x2)) -- router sees a 2D (B*S, H) tensor.
      glm4_moe    Glm4MoeMoE.forward: self.gate(hidden_states) with the 3D (B, S, H) tensor
                  (the router does its own .view(-1, H) internally); experts get
                  hidden_states.view(-1, H); shared_experts get the 3D `residuals`.
      gpt_oss     GptOssMLP.forward: reshape(-1, H) then self.router(...) -- 2D.
      qwen3_moe   Qwen3MoeSparseMoeBlock.forward: view(-1, H) then self.gate(...) -- 2D.

    Both 2D (B*S, H) and 3D (B, S, H) router inputs are handled; a row is addressed by the
    flat index b*S + j either way.

    ── THE RMSNorm PROBLEM, AND HOW IT IS AVOIDED ───────────────────────────────────────
    RMSNorm is NONLINEAR: it divides by the RMS of the vector it is given. You therefore
    CANNOT subtract the pre-norm edit from the post-norm tensor. This class never tries.
    It corrects in PRE-NORM space and RE-APPLIES THE MODEL'S OWN NORM MODULE:

        router input  <-  post_attention_layernorm( r - e )        (steered positions only)

    where `r` is the actual pre-norm residual in the steered pass (captured by a forward-PRE
    hook on the norm) and `e` is the estimate of "how much this residual differs from the
    unsteered one". Because the real module is called, the normalisation is exact for
    whatever `r - e` is handed to it -- including Qwen3Next's ZERO-CENTERED formula
    `out * (1 + w)` (weights initialised to zeros; getting this wrong flips the sign on
    every negative-weight dimension and already invalidated one CPU forensics run) and
    GLM-4.5-Air's STANDARD `w * out`. NOTHING about the norm is reimplemented here.

    ── WHAT IS EXACT AND WHAT IS APPROXIMATE: the two modes ─────────────────────────────
    The approximation, if any, lives ENTIRELY in `e`. Nothing else is estimated.

      mode="accum"  (single pass, APPROXIMATE, ~free)
          e = sum of the EXACT displacements this run's Steer hook applied at every steered
          layer above, propagated down the identity path of the residual stream. Exact for
          the direct term; it IGNORES the network's RESPONSE to the edit (attention outputs
          and expert outputs at the intervening blocks). This is the same first-order model
          the FINDINGS section 23 CPU forensics used, now applied inside the live forward.

          IT IS EXACT ONLY AT THE ROUTER IMMEDIATELY BELOW A STEERED LAYER, and degrades
          with depth even when attention is an identity map, because the MLP/EXPERT response
          is not on the identity path either. Measured on the toy stack with 3 steered
          layers and NO attention mixing (adversarial review 2026-09-01, defect 1):

              router depth   resid_rel   cos(accum,true)   logit_err_rel
              L2 (depth 1)      0.000        +1.000            0.000
              L3                0.171        +0.985            0.133
              L4                0.238        +0.972            0.352
              L5                0.370        +0.931            0.304

          The deployed configuration is 3 steered layers over 17-19 downstream routers, i.e.
          ENTIRELY in the degrading regime. **No `accum` cell may be quoted without
          --router-blind-report**, whose table is exactly the quantity above.

      mode="clean"  (two passes, EXACT, ~1 extra prefill)
          A no-edit forward over the identical inputs is run first and the pre-norm residual
          at every downstream MoE layer is cached at the steered positions. The router is
          then given post_attention_layernorm(r_clean) -- literally the tensor the unsteered
          model would have routed on. `e = r - r_clean` exactly, response term included.

    `report=True` forces the clean pass in BOTH modes and measures how good `accum` is:
    per downstream router it accumulates ||e_accum - e_true||/||e_true||, their cosine, and
    the relative router-LOGIT error ||W x_accum - W x_clean|| / ||W x - W x_clean||
    (0 = the correction removed the whole router-visible perturbation, 1 = it removed none).
    An approximation whose error is not measured is not shipped as exact.

    ── SCOPE ────────────────────────────────────────────────────────────────────────────
    * PREFILL ONLY, matching Steer: the decode branch (seq len 1) is left alone, and in
      `accum` there is nothing to subtract there anyway because no decode-position edit was
      applied. Decode-time routers still see the KV-cache-mediated effect of the prefill
      edit; that is a limitation, not an oversight.
    * STEERED POSITIONS ONLY, per batch ROW, using the same `positions` convention as Steer
      (left-padding offsets already applied). Routers at unsteered positions are untouched.
    * Layers at or below the FIRST steered layer are not hooked FOR BLOCK-OUTPUT EDITS
      (`edit_site="block_output"`, the residual Steer): at layer L the MoE runs BEFORE the
      block-output edit at layer L, so no upstream edit has entered the stream. A
      TOKEN-MIXER edit (`edit_site="token_mixer"`, GdnValueSteer: the perturbation enters
      via `linear_attn` output, which is added to the residual BEFORE the same layer's
      pre-MLP norm + MoE) is already router-visible at the steered layer itself, so that
      layer's router IS hooked (adversarial review 2026-09-07, defect 1 -- the first
      steered layer's own router was the one-router blind spot).
    """

    def __init__(self, model, steer_layers, mode="accum", report=False,
                 include_shared=False, verbose=True, edit_site="block_output"):
        if mode not in ("accum", "clean"):
            raise SystemExit(f"router-blind mode must be accum|clean, got {mode!r}")
        if edit_site not in ("block_output", "token_mixer"):
            raise SystemExit(f"router-blind edit_site must be block_output|token_mixer, "
                             f"got {edit_site!r}")
        self.mode = mode
        self.report = bool(report)
        self.needs_clean = (mode == "clean") or self.report
        self.steer_layers = list(steer_layers)
        self.edit_site = edit_site
        self._first = min(self.steer_layers)
        blocks = layer_container(model)
        hidden = model.config.get_text_config().hidden_size
        self.sites = []          # (layer, norm_mod, norm_name, [(name, router_mod)])
        self._upstream = {}      # layer -> [steered-layer INDEX whose edit reaches layer L]
        for L in range(len(blocks)):
            # block_output: the edit at layer Ls first becomes router-visible at L > Ls.
            # token_mixer: it is visible at L >= Ls (the mixer output precedes L's MoE).
            if (L <= self._first) if edit_site == "block_output" else (L < self._first):
                continue
            rts = block_routers(blocks[L], hidden, include_shared=include_shared)
            if not rts:
                continue
            nmod, nname = pre_mlp_norm(blocks[L])
            if nmod is None:
                continue
            self.sites.append((L, nmod, nname, rts))
            self._upstream[L] = [i for i, Ls in enumerate(self.steer_layers)
                                 if (Ls < L if edit_site == "block_output" else Ls <= L)]
        if not self.sites:
            raise SystemExit(
                f"--router-blind found no MoE router downstream of layer {self._first}. "
                f"Either this model is dense (router-blind steering is a no-op there and "
                f"must not be reported as an arm) or its router child is named something "
                f"src/moe.py does not recognise.")
        self.positions = None
        self._edits = {}         # steered-layer index -> {row -> (idxs, delta float32)}
        self._r = {}             # layer -> pre-norm residual of the CURRENT forward
        self._clean = {}         # layer -> {row -> clean pre-norm rows, float32}
        self._priming = False    # inside the clean pass: capture, never correct
        self._recomputing = False   # inside our own norm call: do not re-capture
        self.n_corrected = 0     # router forwards actually corrected (positive evidence)
        self.per_router = {}     # (layer, id(router)) -> corrected forwards; see defect 6
        # every hooked router must fire exactly once per prefill forward, so this is the
        # per-forward denominator a partial failure would fall short of
        self.expected_per_forward = sum(len(r) for _L, _n, _nn, r in self.sites)
        self.stats = {}          # layer -> running report sums
        self._hooks = []
        if verbose:
            print(f"[router-blind] mode={mode} edit_site={edit_site} report={self.report} "
                  f"{len(self.sites)} downstream MoE layers "
                  f"{[L for L, *_ in self.sites][:8]}"
                  f"{'...' if len(self.sites) > 8 else ''} "
                  f"(norm `{self.sites[0][2]}`, routers "
                  f"{[n for n, _ in self.sites[0][3]]}) steered={self.steer_layers}",
                  flush=True)

    # ── lifecycle ────────────────────────────────────────────────────────────────────
    def __enter__(self):
        for L, nmod, _nname, rts in self.sites:
            self._hooks.append(nmod.register_forward_pre_hook(self._mk_norm(L)))
            for _nm, rmod in rts:
                self._hooks.append(rmod.register_forward_pre_hook(self._mk_router(L, nmod)))
        return self

    def __exit__(self, *a):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        # removing the hooks does not release the caches they filled; a batch's worth of
        # cached residuals staying resident until the next begin_batch() is exactly the
        # pressure the OOM backoff cannot relieve
        self.reset()

    def begin_batch(self, positions):
        """Call once per generate() -- i.e. wherever Steer.positions is set. The caches are
        per-forward: reusing a previous chunk's edits or clean rows would correct the wrong
        tokens with the wrong vectors, silently."""
        self.reset()
        self.positions = positions

    def reset(self):
        """Drop every cached tensor. MUST be called before `torch.cuda.empty_cache()` in the
        OOM backoff: `_clean` is ~445 MB on GLM-4.5-Air at batch 4 / span 800 / 17 sites,
        `_edits` ~157 MB, and `_r` pins another ~134 MB left over from prime(). Those are
        live references, so empty_cache() frees NONE of them and the halved-batch retry would
        start with ~700 MB MORE resident than the attempt that just OOM'd (adversarial
        review 2026-09-01, defect 3)."""
        self.positions = None
        self._edits.clear()
        self._r.clear()
        self._clean.clear()

    def note_edit(self, i, b, idxs, delta):
        """Called by Steer with the EXACT displacement it applied at steered-layer index
        `i`, batch row `b`, span positions `idxs`."""
        self._edits.setdefault(i, {})[b] = (list(idxs), delta)

    def prime(self, model, ids, attn, steer=None):
        """Run the UNSTEERED forward and cache each downstream router's pre-norm residual.

        No-op unless the mode (or the report) needs it. `steer` is disabled for the duration
        by clearing its positions -- Steer's prefill branch returns the block output
        untouched when `positions is None`, so no edit is applied and no `note_edit` fires.
        """
        if not self.needs_clean:
            return
        keep = getattr(steer, "positions", None) if steer is not None else None
        self._priming = True
        try:
            if steer is not None:
                steer.positions = None
            kw = dict(input_ids=ids, attention_mask=attn, use_cache=False)
            try:
                # the (B, S, vocab) logits of a full-prompt forward are several GB on these
                # vocabularies; we need none of them
                model(**kw, logits_to_keep=1)
            except TypeError:
                model(**kw)
        finally:
            self._priming = False
            if steer is not None:
                steer.positions = keep

    # ── hooks ────────────────────────────────────────────────────────────────────────
    def _mk_norm(self, L):
        def pre(mod, args):
            if self._recomputing:
                return None
            r = args[0]
            # SINGLE SLOT, deliberately. Blocks run in order, and every router of layer L
            # fires between L's norm and L+1's norm, so nothing older is ever needed --
            # while HOLDING a reference to each layer's (B, S, H) residual would pin
            # ~1.6 GB on Qwen3-Next and ~3.4 GB on GLM-4.5-Air at a 4x4096 prefill that
            # would otherwise be freed as the forward advances.
            self._r.clear()
            self._r[L] = r
            if self._priming and self.positions is not None and r.dim() == 3 \
                    and r.shape[1] > 1:
                rows = {}
                for b, idxs in enumerate(self.positions):
                    if b >= r.shape[0] or not idxs:
                        continue
                    sel = [j for j in idxs if j < r.shape[1]]
                    if sel:
                        # kept at the SOURCE dtype, not upcast. `r` is already bf16 in the
                        # model, so storing bf16 is LOSSLESS while `.float()` would double
                        # the largest allocation in this class -- ~891 MB -> ~445 MB on
                        # GLM-4.5-Air at batch 4 / span 800 / 17 sites (review defect 4).
                        rows[b] = r[b, torch.tensor(sel, device=r.device)].clone()
                self._clean[L] = rows
            return None
        return pre

    def _accum_edit(self, L, b, sel, device):
        """Sum of the applied displacements from every steered layer ABOVE L, for row b."""
        tot = None
        for i in self._upstream.get(L, []):
            ent = self._edits.get(i, {}).get(b)
            if ent is None:
                continue
            idxs_i, delta = ent
            if list(idxs_i) != list(sel):
                # Steer clips to h.shape[1] and so do we, on the same prefill, so this can
                # only mean the caches are from different forwards. Fail loudly: silently
                # subtracting a mis-aligned edit is a wrong correction that still runs.
                raise RuntimeError(
                    f"router-blind: layer {L} row {b} span has {len(sel)} positions but the "
                    f"cached edit from steered-layer index {i} has {len(idxs_i)} -- "
                    f"begin_batch() was not called for this forward.")
            d = delta.to(device=device, dtype=torch.float32)
            tot = d if tot is None else tot + d
        return tot

    def _mk_router(self, L, nmod):
        def pre(mod, args):
            if self._priming or self.positions is None or not args:
                return None
            r = self._r.get(L)
            if r is None or r.dim() != 3 or r.shape[1] == 1:
                return None                      # decode step: prefill-only, like Steer
            B, S, H = r.shape
            x = args[0]
            flat, rows, dbg = [], [], []
            for b, idxs in enumerate(self.positions):
                if b >= B or not idxs:
                    continue
                sel = [j for j in idxs if j < S]
                if not sel:
                    continue
                base = r[b, torch.tensor(sel, device=r.device)].float()
                e_acc = self._accum_edit(L, b, sel, r.device)
                clean = self._clean.get(L, {}).get(b)
                if clean is not None:
                    clean = clean.to(device=r.device, dtype=torch.float32)
                if self.mode == "clean":
                    if clean is None:
                        # Falling back to `accum` here would turn the EXACT mode into the
                        # approximate one while still printing `+rblind@clean` on the arm --
                        # a silently different intervention under the cell's own name.
                        raise RuntimeError(
                            f"router-blind mode=clean has no cached unsteered residual for "
                            f"layer {L} row {b}: prime() did not run for this forward. "
                            f"Refusing to silently degrade to mode=accum.")
                    tgt = clean
                elif e_acc is not None:
                    tgt = base - e_acc
                else:
                    continue                      # no upstream edit reached this row
                flat.extend([b * S + j for j in sel])
                rows.append(tgt)
                if self.report and clean is not None:
                    dbg.append((base, e_acc, clean))
            if not rows:
                return None
            X = torch.cat(rows, 0).to(r.dtype)
            self._recomputing = True
            try:
                Xn = nmod(X)                      # the model's OWN norm -- never reimplemented
            finally:
                self._recomputing = False
            xn = x.clone()                        # never in place: the same tensor feeds the
            # experts and the shared expert. Both router-input layouts are supported: 3D
            # (B, S, H) as GLM-4.5-Air passes it, and 2D (B*S, H) as qwen3_next / qwen3_moe /
            # gpt-oss pass it. Anything else means the modeling code moved under us -- RAISE.
            # Returning None there would silently turn a `+rblind@` arm into an ordinary
            # steering arm that still prints under the router-blind cell name.
            if xn.dim() == 3 and xn.shape[0] == B and xn.shape[1] == S:
                xf = xn.view(-1, xn.shape[-1])
            elif xn.dim() == 2 and xn.shape[0] == B * S:
                xf = xn
            else:
                raise RuntimeError(
                    f"router-blind: layer {L} router `{type(mod).__name__}` received a "
                    f"{tuple(xn.shape)} input, but the pre-norm residual is "
                    f"{(B, S, H)} -- expected (B, S, H) or (B*S, H). The installed "
                    f"transformers version routes differently than src/moe.py assumes; "
                    f"re-run tools/controls/verify_router_blind.py.")
            fi = torch.tensor(flat, device=xf.device)
            xf[fi] = Xn.to(device=xf.device, dtype=xf.dtype)
            self.n_corrected += 1
            # PER-ROUTER, not just a total: a bare counter makes a PARTIAL failure invisible
            # -- one router of nineteen silently returning None still leaves n_corrected > 0
            # and the arm still prints `+rblind@` (review defect 6). `expected_per_forward`
            # is what run_arm checks this against.
            self.per_router[(L, id(mod))] = self.per_router.get((L, id(mod)), 0) + 1
            if self.report and dbg:
                self._note_report(L, mod, nmod, dbg)
            return (xn,) + tuple(args[1:])
        return pre

    def _note_report(self, L, rmod, nmod, dbg):
        base = torch.cat([d[0] for d in dbg], 0)
        acc = torch.cat([d[1] if d[1] is not None else torch.zeros_like(d[0])
                         for d in dbg], 0)
        clean = torch.cat([d[2] for d in dbg], 0)
        e_true = base - clean
        self._recomputing = True
        try:
            # FLOAT32 here, deliberately unlike the correction path (which stays at the
            # model's dtype so it reproduces production exactly). These three norms feed a
            # RATIO OF SMALL DIFFERENCES: at bf16 the numerator ||l_acc - l_clean|| carries a
            # rounding floor of its own, so a genuinely tiny correction error would be
            # reported as bf16 noise instead of as zero (review defect 7).
            x_base = nmod(base.float()).float()
            x_acc = nmod((base - acc).float()).float()
            x_cln = nmod(clean.float()).float()
        finally:
            self._recomputing = False
        W = rmod.weight.float()
        lb, la, lc = x_base @ W.T, x_acc @ W.T, x_cln @ W.T
        s = self.stats.setdefault(L, dict(n=0, e_true=0.0, e_resid=0.0, cos=0.0,
                                          logit_base=0.0, logit_acc=0.0))
        n = base.shape[0]
        s["n"] += n
        s["e_true"] += float(e_true.norm(dim=-1).sum())
        s["e_resid"] += float((acc - e_true).norm(dim=-1).sum())
        s["cos"] += float(torch.nn.functional.cosine_similarity(
            acc, e_true, dim=-1).nan_to_num().sum())
        s["logit_base"] += float((lb - lc).norm(dim=-1).sum())
        s["logit_acc"] += float((la - lc).norm(dim=-1).sum())

    def summary(self):
        """Per-downstream-router exactness of the `accum` estimate. Only populated with
        report=True (which forces the clean pass); empty otherwise, and an empty summary
        must be reported as "not measured", never as "exact"."""
        out = {}
        for L, s in sorted(self.stats.items()):
            n = max(s["n"], 1)
            out[L] = dict(
                n=s["n"],
                # fraction of the TRUE residual perturbation left uncorrected by `accum`
                resid_rel=s["e_resid"] / max(s["e_true"], 1e-9),
                cos_accum_true=s["cos"] / n,
                # 0.0 = the correction removed the whole router-visible perturbation;
                # 1.0 = it removed none of it
                logit_err_rel=s["logit_acc"] / max(s["logit_base"], 1e-9))
        return out


def resolve_boundary(args, steer_layers, root):
    """(per-layer boundary in sigma units, label suffix) for a displacement-proportional rule.

    `auto` IS DISABLED. It is kept only to raise, because the bug it caused was paid for.

    It calibrated m from runs/gate_separability.json as legit_mean + 1.645*legit_sd -- but
    that file measures projections onto `dim_override` at the PRE-MLP capture site, while a
    sweep steers whatever `--directions` names, at the BLOCK-OUTPUT site. Different axes:

        cos(dim_override, dim_no_override_both) = -0.519 / -0.305 / -0.231 at L12/16/20

    NOT -1.0, which is what an earlier version of this docstring asserted. So the threshold
    did not gate: 93% / 71% / 100% of injected spans sat above it, mean steps 1.21 / 0.63 /
    6.76 sigma -- at L20 a LARGER step than the fixed alpha=8 rule delivers (8/sqrt(3) =
    4.62). The seed-0 random control got 0.30 / 0.00 / 2.39 sigma, a 3.04x smaller RMS
    displacement whose size is set entirely by that draw's accidental alignment with the
    mean activation vector (12-26% of seeds give exactly zero at L12/L16). The arms were not
    magnitude-matched, violating CLAUDE.md's mandatory control invariant, and the resulting
    "direction beats random 24/0, p<1e-6" was an artifact of one arm being pushed 3x harder.

    Explicit per-layer values remain supported and are the only supported path. A fair
    boundary control equalises a BUDGET across arms -- the steered-token fraction (each
    direction's own 95th legit percentile) or, better, the total edit norm
    E[max(0, p_ov - m)]*sigma -- measured on the direction actually steered, at the site
    actually steered.

    KNOWN LIMIT, measured, do not rediscover it: at a 90th-percentile-of-legitimate
    threshold only ~37% of injected tokens sit above the boundary
    (runs/gate_separability.json, AUC 0.736 / 0.701 / 0.588 at L12/16/20). A HARD gate
    therefore misses most of the injection. A proportional rule is not a hard gate -- it
    degrades smoothly rather than dropping to zero -- but if ASR collapses under it, weak
    per-token separability is the first suspect, not the rule.
    """
    if args.step_rule == "fixed":
        return None, ""
    if args.step_boundary == "auto":
        raise SystemExit(
            "--step-boundary auto is DISABLED: runs/gate_separability.json calibrates "
            "`dim_override` at the pre-MLP site, but steering applies a different direction "
            "at the block-output site (cos -0.52/-0.30/-0.23), so the threshold does not "
            "gate and the arms are not magnitude-matched. See resolve_boundary's docstring. "
            "Pass explicit per-layer values calibrated on the steered direction instead.")
    if args.step_boundary != "auto":
        vals = [float(x) for x in args.step_boundary.split(",")]
        if len(vals) != len(steer_layers):
            raise SystemExit(f"--step-boundary has {len(vals)} values for "
                             f"{len(steer_layers)} steered layers")
        tag = f"+{args.step_rule}@{args.step_boundary}"
    else:
        p = f"{root}/runs/gate_separability.json"
        if not os.path.exists(p):
            raise SystemExit(
                f"--step-boundary auto needs {p}; run "
                f"tools/controls/gate_separability.py, or pass explicit values")
        rows = json.load(open(p))["rows"]
        missing = [L for L in steer_layers if str(L) not in rows]
        if missing:
            raise SystemExit(f"gate_separability.json has no rows for layers {missing}")
        vals = [rows[str(L)]["legit_mean"] + 1.645 * rows[str(L)]["legit_sd"]
                for L in steer_layers]
        tag = f"+{args.step_rule}@auto"
    if args.step_margin:
        tag += f"m{args.step_margin}"
    if args.step_scale != 1.0:
        tag += f"x{args.step_scale}"
    print(f"[step] rule={args.step_rule} boundary(sigma)="
          f"{[round(v, 3) for v in vals]} margin={args.step_margin} "
          f"scale={args.step_scale}", flush=True)
    return vals, tag


def resolve_gate_proj(spec, steer_layers, root, direction=None):
    """(per-layer gate threshold in dim_OVERRIDE sigma units, label suffix) or (None, "").

    Accepts three forms:
      "1.5"            one value, broadcast to every steered layer
      "-1.5,-1.2,-3.1" one value per steered layer, in layer order
      "legit90"        the Nth percentile of LEGITIMATE-token override-ness at each layer,
                       read from runs/gate_separability.json -- the GSS convention (calibrate
                       the threshold on clean spans, so the gate is defined by how often it
                       may fire on legitimate text rather than by a magic number).

    WHY PER LAYER. Override-ness is not on a common offset across layers. Measured injected
    vs legitimate means (runs/gate_separability.json, dim_override, sigma units):

        L12  -0.83 / -2.18      L16  -0.72 / -1.77      L20  -2.79 / -3.49

    A single threshold of -1.5 separates the classes at L12 and L16 and sits ABOVE BOTH at
    L20, which turns that layer's steering off without saying so. The old scalar flag could
    not express a working setting for this layer set at all.

    UNITS: dim_OVERRIDE, higher = more override-ness = more likely injected, matching what
    gate_separability.py prints. Steer negates internally because the direction it ADDS is
    dim_no_override. Before 2026-08-04 it did not, and the four archived --gate-proj runs
    steered the legitimate record tokens instead of the injected ones -- see
    todo/04-review-actions-2026-08-04.md.
    """
    if spec in (None, ""):
        return None, ""
    spec = str(spec).strip()
    m = re.fullmatch(r"legit(\d+(?:\.\d+)?)", spec)
    if m:
        pct = float(m.group(1))
        p = f"{root}/runs/gate_separability.json"
        if not os.path.exists(p):
            raise SystemExit(f"--gate-proj {spec} needs {p}; run "
                             f"tools/controls/gate_separability.py, or pass explicit values")
        rows = json.load(open(p))["rows"]
        missing = [L for L in steer_layers if str(L) not in rows]
        if missing:
            raise SystemExit(f"gate_separability.json has no rows for layers {missing}")
        vals = []
        for L in steer_layers:
            r = rows[str(L)]
            q = r.get("legit_pct", {}).get(str(int(pct))) if isinstance(
                r.get("legit_pct"), dict) else None
            if q is None:
                # normal approximation from the stored mean/sd when the percentile table is
                # absent, and SAY SO -- a silently substituted statistic is how a threshold
                # ends up meaning something other than what its name claims.
                z = {50: 0.0, 75: 0.6745, 90: 1.2816, 95: 1.6449, 99: 2.3263}.get(int(pct))
                if z is None:
                    raise SystemExit(
                        f"--gate-proj {spec}: no stored percentile and no z for {pct}; "
                        f"use 50/75/90/95/99 or pass explicit per-layer values")
                q = r["legit_mean"] + z * r["legit_sd"]
                print(f"[gate] L{L}: no stored p{int(pct)} in gate_separability.json; "
                      f"NORMAL APPROXIMATION legit_mean={r['legit_mean']:+.3f} + "
                      f"{z:.4f}*sd={r['legit_sd']:.3f} -> {q:+.3f}", flush=True)
            vals.append(float(q))
        tag = f"+gate@{spec}"
    else:
        vals = [float(x) for x in spec.split(",")]
        if len(vals) == 1:
            vals = vals * len(steer_layers)
        if len(vals) != len(steer_layers):
            raise SystemExit(f"--gate-proj has {len(vals)} values for "
                             f"{len(steer_layers)} steered layers")
        tag = f"+gate@{spec}"
    print(f"[gate] per-token gate on dim_OVERRIDE-ness, thresholds(sigma)="
          f"{[round(v, 3) for v in vals]} for layers {steer_layers}"
          + (f" (direction {direction})" if direction else ""), flush=True)
    return vals, tag


# ════════════════════════════════════════════════════════════ CachePrune (arXiv:2504.21228)
from transformers.cache_utils import DynamicCache  # noqa: E402


class PrunedKVCache(DynamicCache):
    """CachePrune's inference-time intervention (arXiv:2504.21228 eq. 6): multiplicative
    zeroing of a fixed set of K/V cache COORDINATES (layer, K-or-V, kv_head, dim) at the
    context-span positions, m_i = 1 - alpha * 1{i pruned}.

    POST-RoPE BY CONSTRUCTION: in this transformers version every supported model calls
    `past_key_values.update(key_states, value_states, layer_idx)` AFTER
    `apply_rotary_pos_emb`, so masking inside update() edits exactly the cached keys the
    paper masks, and it happens "during KV cache encoding": the masked cache is what the
    rest of the prefill AND the whole decode attend to.

    PREFILL ONLY: masking fires when the query length is > 1 and the layer's cache is
    still empty; decode steps append unmasked, matching the paper -- only context/data
    tokens are pruned, never the model's own response tokens. `positions` is per batch
    ROW (left-padding offsets already applied) and must be sliced when a batch is split;
    a FRESH instance must be built per generate() call because the cache is stateful.
    Subclassing DynamicCache with `config` keeps sliding-window layers (gpt-oss
    alternates sliding/full) on their correct layer types.
    """

    def __init__(self, spec, positions):
        super().__init__(config=spec.config)
        self._kv_spec = spec
        self._kv_positions = positions

    def update(self, key_states, value_states, layer_idx, *a, **kw):
        if key_states.shape[-2] > 1 and self.get_seq_length(layer_idx) == 0:
            spec = self._kv_spec
            km = spec.mask_for(layer_idx, "k", key_states.device, key_states.dtype)
            vm = spec.mask_for(layer_idx, "v", value_states.device, value_states.dtype)
            if km is not None or vm is not None:
                key_states = key_states.clone()
                value_states = value_states.clone()
                for b, idxs in enumerate(self._kv_positions):
                    if b >= key_states.shape[0] or not idxs:
                        continue
                    sel = [j for j in idxs if j < key_states.shape[-2]]
                    if not sel:
                        continue
                    sel = torch.tensor(sel, device=key_states.device)
                    if km is not None:
                        # (kv_heads, n_sel, head_dim) * (kv_heads, 1, head_dim)
                        key_states[b, :, sel, :] = (
                            key_states[b, :, sel, :] * km.unsqueeze(1))
                    if vm is not None:
                        value_states[b, :, sel, :] = (
                            value_states[b, :, sel, :] * vm.unsqueeze(1))
        return super().update(key_states, value_states, layer_idx, *a, **kw)


class KVMaskSpec:
    """Per-layer (kv_heads, head_dim) multiplicative masks for keys and values, plus the
    model config needed to build a correctly-typed DynamicCache. See PrunedKVCache."""

    def __init__(self, config, k_masks, v_masks, meta=None):
        self.config = config
        self.k_masks = k_masks      # list per layer: float32 (kv_heads, head_dim) or None
        self.v_masks = v_masks
        self.meta = meta or {}
        self._cast = {}             # (layer, kind, device, dtype) -> tensor

    def mask_for(self, layer_idx, kind, device, dtype):
        m = (self.k_masks if kind == "k" else self.v_masks)[layer_idx]
        if m is None:
            return None
        key = (layer_idx, kind, str(device), dtype)
        got = self._cast.get(key)
        if got is None:
            got = m.to(device=device, dtype=dtype)
            self._cast[key] = got
        return got

    def make_cache(self, positions):
        return PrunedKVCache(self, positions)


def load_kv_mask(path, config):
    """runs/cacheprune_mask.json -> KVMaskSpec. The artifact stores the PRUNED coordinate
    list; masks are dense (kv_heads, head_dim) float32 with `1 - alpha` at pruned coords."""
    d = json.load(open(path))
    tc = config.get_text_config()
    n_layers = tc.num_hidden_layers
    kvh = getattr(tc, "num_key_value_heads", tc.num_attention_heads)
    hd = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
    for k, want in (("n_layers", n_layers), ("num_kv_heads", kvh), ("head_dim", hd)):
        if d.get(k) != want:
            raise SystemExit(f"{path}: mask built for {k}={d.get(k)} but the loaded model "
                             f"has {want} -- a KV mask is not transferable across shapes.")
    alpha = float(d.get("alpha", 1.0))
    k_masks = [None] * n_layers
    v_masks = [None] * n_layers
    for c in d["pruned"]:
        tgt = k_masks if c["kv"] == "k" else v_masks
        if tgt[c["layer"]] is None:
            tgt[c["layer"]] = torch.ones(kvh, hd, dtype=torch.float32)
        tgt[c["layer"]][c["head"], c["dim"]] = 1.0 - alpha
    print(f"[cacheprune] {path}: {len(d['pruned'])} pruned coords "
          f"(alpha={alpha}, layers with K mask "
          f"{sum(m is not None for m in k_masks)}, V mask "
          f"{sum(m is not None for m in v_masks)}; fit meta: "
          f"n_fit={d.get('_meta', {}).get('n_fit')} "
          f"model={d.get('_meta', {}).get('model')})", flush=True)
    return KVMaskSpec(config, k_masks, v_masks, meta=d.get("_meta", {}))


def ctrl_kind(name):
    """('random'|'shuffled', seed) for a control arm, else None.

    `random` was a single fixed draw, so "random also blocks" was n=1 in direction
    space and carried no error bar. `random2`, `random7`, ... are independent draws of
    the same control, letting the control become a distribution.
    """
    m = re.fullmatch(r"(random|shuffled)(\d*)", name or "")
    return (m.group(1), int(m.group(2) or 0)) if m else None
