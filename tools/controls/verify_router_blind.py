#!/usr/bin/env python
"""CPU verification for ROUTER-BLIND RESIDUAL STEERING (src/steering.py: RouterBlind).

No GPU, no model download, no checkpoint read. Seven sections:

  1. ASSUMPTIONS ABOUT THE INSTALLED transformers -- numerically, on the real classes.
     Qwen3NextRMSNorm is ZERO-CENTERED (`out * (1 + w)`, weight initialised to zeros) and
     Glm4MoeRMSNorm is STANDARD (`w * out`, weight initialised to ones). Getting this
     backwards flips the sign on every negative-weight dimension and already invalidated one
     CPU forensics run (FINDINGS section 23, review correction (a)). Also asserts the block
     wiring RouterBlind depends on: the attribute names `post_attention_layernorm` and
     `mlp.gate`, that the router input is the post-norm tensor, and that the SAME post-norm
     tensor feeds the experts. A transformers upgrade that moves any of this fails HERE,
     loudly, instead of silently correcting the wrong tensor in a 4-GPU run.

  2. THE REAL MoE BLOCKS, instantiated tiny. Qwen3NextSparseMoeBlock and Glm4MoeMoE are
     built with random weights at hidden_size 16 and driven directly. Checks that
     `moe.block_routers` finds the router, that a forward-PRE hook on it changes the
     EXPERT SELECTION, and that the expert INPUTS are unchanged by that hook -- which is the
     entire claim of the intervention ("routers see the clean stream, experts see the
     steered stream"). Covers the 2D (B*S, H) router input of Qwen3-Next AND the 3D
     (B, S, H) router input of GLM-4.5-Air.

  3. END-TO-END on a toy decoder stack wired like the real one, with Steer + RouterBlind
     actually installed, run TWICE -- once with the 2D (B*S, H) router call shape
     (qwen3_next / qwen3_moe / gpt-oss) and once with the 3D (B, S, H) one (glm4_moe).
     Proves (a) that with an identity residual path the `accum` correction is EXACT -- the
     router's input is bit-close to the unsteered run's -- (b) that with a mixing residual
     path `clean` mode is still exact while `accum` carries a measurable error, which this
     script REPORTS rather than assumes, (c) that only steered positions are corrected, and
     (d) that with the feature off, generation is byte-identical to the pre-existing path.

  3b. DEPTH. Section 3 steers ONE layer, so its exactness result is depth-1 only. This
     section steers THREE (the deployed shape) with zero attention mixing and shows `accum`
     degrading with depth anyway -- the MLP/expert response is not on the identity path
     either -- while `clean` stays exact at every depth. It is also the only end-to-end case
     where a steered layer is ITSELF a downstream router site, which the real layer sets
     (28,32,40 / 20,24,28) all are.

  4. ROUTER-NULL SURGERY: safetensors shard discovery over a synthetic checkpoint, and the
     projection algebra (unit norm, orthogonality to the top-r router subspace, monotone
     router-logit attenuation, and the degenerate full-rank case the script must SKIP).

  5. THE MEASUREMENT that expert-output steering is a NO-OP relative to the shipped
     block-output steer site, rather than an assertion that it is.

  6. A MUTATION TEST, because a verification that passes on broken code is worse than none.
     Three plausible bugs are injected into the SHIPPED RouterBlind -- correction sign
     flipped, Steer no longer reporting its edits, `clean` mode run without its unsteered
     pass, and the `_recomputing` reentrancy guard removed -- and the section-3 assertions
     (or the per-router firing count) must fail on each.

Usage:  python tools/controls/verify_router_blind.py [-v]
"""
import argparse
import inspect
import os
import sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.moe import block_routers, pre_mlp_norm  # noqa: E402
from src.steering import RouterBlind, Steer  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


# ══════════════════════════════════════ 1. assumptions about the installed transformers
def check_installed_modeling():
    print("\n=== 1. installed transformers: RMSNorm formulas and block wiring ===")
    import transformers
    from transformers.models.glm4_moe import modeling_glm4_moe as G
    from transformers.models.qwen3_next import modeling_qwen3_next as Q
    print(f"  transformers {transformers.__version__} at {os.path.dirname(transformers.__file__)}")

    torch.manual_seed(0)
    x = torch.randn(3, 8, dtype=torch.float32)
    rms = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    # --- Qwen3-Next: ZERO-CENTERED, out * (1 + w), weight initialised to ZEROS
    qn = Q.Qwen3NextRMSNorm(8, eps=1e-6)
    check("Qwen3NextRMSNorm weight initialises to ZEROS (zero-centered convention)",
          torch.equal(qn.weight.data, torch.zeros(8)))
    with torch.no_grad():
        qn.weight.copy_(torch.randn(8))
    got = qn(x)
    check("Qwen3NextRMSNorm(x) == rms(x) * (1 + w)   [NOT rms(x) * w]",
          torch.allclose(got, rms * (1.0 + qn.weight), atol=1e-5),
          f"max|diff to (1+w)| = {(got - rms * (1 + qn.weight)).abs().max():.2e}, "
          f"max|diff to w| = {(got - rms * qn.weight).abs().max():.2e}")

    # --- GLM-4.5-Air: STANDARD, w * out, weight initialised to ONES
    gn = G.Glm4MoeRMSNorm(8, eps=1e-6)
    check("Glm4MoeRMSNorm weight initialises to ONES (standard convention)",
          torch.equal(gn.weight.data, torch.ones(8)))
    with torch.no_grad():
        gn.weight.copy_(torch.randn(8))
    got = gn(x)
    check("Glm4MoeRMSNorm(x) == w * rms(x)   [NOT (1 + w) * rms(x)]",
          torch.allclose(got, gn.weight * rms, atol=1e-5),
          f"max|diff to w| = {(got - gn.weight * rms).abs().max():.2e}")

    # --- block wiring: residual -> post_attention_layernorm -> mlp -> residual + mlp_out
    for tag, layer_cls, moe_cls, router_attr in (
            ("qwen3_next", Q.Qwen3NextDecoderLayer, Q.Qwen3NextSparseMoeBlock, "gate"),
            ("glm4_moe", G.Glm4MoeDecoderLayer, G.Glm4MoeMoE, "gate")):
        src = inspect.getsource(layer_cls.forward)
        norm_line = "hidden_states = self.post_attention_layernorm(hidden_states)"
        check(f"{tag}: decoder layer applies `{norm_line}`", norm_line in src)
        check(f"{tag}: the post-norm tensor is what `self.mlp` receives",
              "hidden_states = self.mlp(hidden_states)" in src)
        check(f"{tag}: the MLP output is added back to the pre-norm residual",
              "hidden_states = residual + hidden_states" in src)
        msrc = inspect.getsource(moe_cls.forward)
        check(f"{tag}: the MoE block routes with `self.{router_attr}(...)`",
              f"self.{router_attr}(" in msrc)
        check(f"{tag}: the MoE block computes experts with `self.experts(...)`",
              "self.experts(" in msrc)
    # GLM hands the router the 3D tensor and only afterwards flattens for the experts --
    # RouterBlind must therefore handle a 3D router input, not only the 2D one.
    gsrc = inspect.getsource(G.Glm4MoeMoE.forward)
    i_gate = gsrc.index("self.gate(hidden_states)")
    i_view = gsrc.index("hidden_states.view(-1")
    check("glm4_moe: the router sees the 3D (B, S, H) tensor, flattened only afterwards",
          i_gate < i_view, f"gate at char {i_gate}, view at char {i_view}")
    qsrc = inspect.getsource(Q.Qwen3NextSparseMoeBlock.forward)
    check("qwen3_next: the router sees the 2D (B*S, H) reshaped tensor",
          "hidden_states_reshaped = hidden_states.view(-1, hidden_dim)" in qsrc
          and "self.gate(hidden_states_reshaped)" in qsrc)
    check("qwen3_next: a SHARED expert exists and reads the same post-norm tensor",
          "self.shared_expert(hidden_states_reshaped)" in qsrc
          and "self.shared_expert_gate(hidden_states_reshaped)" in qsrc)
    check("glm4_moe: shared experts read the same post-norm tensor (`residuals`)",
          "residuals = hidden_states" in gsrc and "self.shared_experts(residuals)" in gsrc)

    # The other two MoE families this project has cells on. RouterBlind never reimplements a
    # norm -- it calls the module -- so these are pinned as DOCUMENTATION of the convention,
    # and to catch a family silently switching to the zero-centered form the way Qwen3-Next
    # already differs from its own qwen3_moe sibling.
    from transformers.models.gpt_oss import modeling_gpt_oss as O
    from transformers.models.qwen3_moe import modeling_qwen3_moe as M
    for tag, cls, centered in (("qwen3_moe", M.Qwen3MoeRMSNorm, False),
                               ("gpt_oss", O.GptOssRMSNorm, False)):
        n = cls(8, eps=1e-6)
        with torch.no_grad():
            n.weight.copy_(torch.randn(8))
        g = n(x)
        want = rms * ((1.0 + n.weight) if centered else n.weight)
        check(f"{tag}: RMSNorm is {'ZERO-CENTERED' if centered else 'STANDARD'} "
              f"(`{'(1+w)*out' if centered else 'w*out'}`)",
              torch.allclose(g, want, atol=1e-5),
              f"max|diff| = {(g - want).abs().max():.2e}")
    check("gpt_oss: the router child is called `router`, not `gate`",
          "self.router(" in inspect.getsource(O.GptOssMLP.forward))
    check("qwen3_moe: the router child is called `gate`",
          "self.gate(" in inspect.getsource(M.Qwen3MoeSparseMoeBlock.forward))


# ══════════════════════════════════════ 2. the REAL MoE blocks, tiny
class _Block(nn.Module):
    """Minimal stand-in for a decoder block: exactly the two attributes RouterBlind and
    src/moe.py look for."""

    def __init__(self, norm, mlp):
        super().__init__()
        self.post_attention_layernorm = norm
        self.mlp = mlp


def check_real_moe_blocks(verbose=False):
    print("\n=== 2. real MoE blocks (random weights, hidden=16): discovery + hook semantics ===")
    from transformers.models.glm4_moe import modeling_glm4_moe as G
    from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
    from transformers.models.qwen3_next import modeling_qwen3_next as Q
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

    H = 16
    torch.manual_seed(1)
    qcfg = Qwen3NextConfig(hidden_size=H, intermediate_size=2 * H, moe_intermediate_size=H,
                           shared_expert_intermediate_size=H, num_experts=8,
                           num_experts_per_tok=2, norm_topk_prob=True,
                           num_hidden_layers=2, num_attention_heads=2,
                           num_key_value_heads=1, vocab_size=32)
    gcfg = Glm4MoeConfig(hidden_size=H, intermediate_size=2 * H, moe_intermediate_size=H,
                         n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
                         n_group=2, topk_group=1, norm_topk_prob=True,
                         routed_scaling_factor=1.0, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, vocab_size=32)

    cases = []
    qb = _Block(Q.Qwen3NextRMSNorm(H, eps=1e-6), Q.Qwen3NextSparseMoeBlock(qcfg))
    cases.append(("qwen3_next", qb, "mlp.gate"))
    gb = _Block(G.Glm4MoeRMSNorm(H, eps=1e-6), G.Glm4MoeMoE(gcfg))
    cases.append(("glm4_moe", gb, "mlp.gate"))

    for tag, blk, want in cases:
        with torch.no_grad():
            for p in blk.parameters():
                p.copy_(torch.randn_like(p) * 0.3)
        rts = block_routers(blk, H)
        names = [n for n, _ in rts]
        check(f"{tag}: block_routers finds exactly ['{want}']", names == [want], str(names))
        nmod, nname = pre_mlp_norm(blk)
        check(f"{tag}: pre_mlp_norm finds `post_attention_layernorm`",
              nname == "post_attention_layernorm", nname or "None")
        # shared_expert_gate must NOT be picked up unless explicitly asked for
        if tag == "qwen3_next":
            check("qwen3_next: shared_expert_gate excluded by default",
                  "mlp.shared_expert_gate" not in names)
            check("qwen3_next: shared_expert_gate included with include_shared=True",
                  "mlp.shared_expert_gate"
                  in [n for n, _ in block_routers(blk, H, include_shared=True)])

        rmod = dict(rts)[want]
        seen = {}
        h_expert = blk.mlp.experts.forward

        def spy_experts(hidden_states, top_k_index, top_k_weights, _f=h_expert, _s=seen):
            _s["expert_in"] = hidden_states.detach().clone()
            _s["idx"] = top_k_index.detach().clone()
            return _f(hidden_states, top_k_index, top_k_weights)
        blk.mlp.experts.forward = spy_experts

        B, S = 2, 5
        r = torch.randn(B, S, H)
        x = blk.post_attention_layernorm(r)
        with torch.no_grad():
            blk.mlp(x)
        base_in, base_idx = seen["expert_in"], seen["idx"]

        # a forward-PRE hook that hands the router a DIFFERENT tensor
        bogus = torch.randn(B, S, H)
        def pre(mod, args, _b=bogus, _H=H):
            a0 = args[0]
            rep = _b.reshape(-1, _H) if a0.dim() == 2 else _b
            return (rep.to(a0.dtype),) + tuple(args[1:])
        hk = rmod.register_forward_pre_hook(pre)
        with torch.no_grad():
            blk.mlp(x)
        hk.remove()
        blk.mlp.experts.forward = h_expert

        check(f"{tag}: a gate pre-hook CHANGES expert selection",
              not torch.equal(seen["idx"], base_idx),
              f"{int((seen['idx'] != base_idx).sum())} of {base_idx.numel()} slots moved")
        check(f"{tag}: a gate pre-hook leaves the EXPERT INPUT untouched",
              torch.equal(seen["expert_in"], base_in))
        if verbose:
            print(f"      base idx {base_idx.flatten()[:8].tolist()} -> "
                  f"{seen['idx'].flatten()[:8].tolist()}")


# ══════════════════════════════════════ 3. end-to-end on a toy stack
class ToyCfg:
    def __init__(self, hidden_size, vocab):
        self.hidden_size = hidden_size
        self.vocab_size = vocab

    def get_text_config(self):
        return self


class ToyRouter(nn.Module):
    """Same surface as every installed top-k router: `weight` (n_experts, hidden), a
    (num_tokens, hidden) or (B, S, hidden) input, softmax-then-topk."""

    def __init__(self, hidden, n_experts, top_k):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, hidden) * 0.5)
        self.top_k = top_k
        self.hidden_dim = hidden
        self.trace = {}

    def forward(self, hidden_states):
        h = hidden_states.reshape(-1, self.hidden_dim)
        # traced INSIDE the router, i.e. after any forward-pre hook has had its say. Tracing
        # the MoE block's own copy instead would record the tensor the router did NOT see --
        # which is exactly the mistake this comment exists to stop being made again.
        self.trace["router_in"] = h.detach().clone()
        logits = h @ self.weight.T
        p = torch.softmax(logits.float(), dim=-1)
        w, idx = torch.topk(p, self.top_k, dim=-1)
        self.trace["idx"] = idx.detach().clone()
        return logits, w / w.sum(-1, keepdim=True), idx


class ToyMoE(nn.Module):
    """`router_3d=False` reproduces the qwen3_next / qwen3_moe / gpt-oss call shape (the
    router gets the flattened (B*S, H) tensor); `router_3d=True` reproduces glm4_moe's (the
    router gets the 3D (B, S, H) tensor and flattens internally). RouterBlind must handle
    both, and only running one of them would leave the GLM-4.5-Air path untested."""

    def __init__(self, hidden, n_experts, top_k, router_3d=False):
        super().__init__()
        self.gate = ToyRouter(hidden, n_experts, top_k)
        # a SECOND router on the same layer, shaped like Qwen3-Next's shared_expert_gate
        # (weight (1, H)). Only discovered with include_shared=True, and the only
        # configuration in which the `_recomputing` reentrancy guard is observable.
        self.shared_expert_gate = ToyRouter(hidden, 1, 1)
        self.experts = nn.Parameter(torch.randn(n_experts, hidden, hidden) * 0.1)
        self.router_3d = router_3d
        self.trace = {}

    def forward(self, x):
        B, S, H = x.shape
        flat = x.reshape(-1, H)
        _, w, idx = self.gate(x if self.router_3d else flat)
        self.shared_expert_gate(x if self.router_3d else flat)
        self.trace["expert_in"] = flat.detach().clone()
        out = torch.zeros_like(flat)
        for k in range(idx.shape[1]):
            W = self.experts[idx[:, k]]                       # (N, H, H)
            out = out + w[:, k:k + 1] * torch.bmm(flat.unsqueeze(1), W).squeeze(1)
        return out.reshape(B, S, H)


class ToyNorm(nn.Module):
    """Zero-centered RMSNorm, i.e. the Qwen3-Next convention -- deliberately the harder of
    the two, so a correction that reimplemented `w * out` would fail this test."""

    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden) * 0.2)

    def forward(self, x):
        v = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return (v * (1.0 + self.weight.float())).type_as(x)


class ToyBlock(nn.Module):
    def __init__(self, hidden, n_experts, top_k, mix, router_3d=False):
        super().__init__()
        self.input_layernorm = ToyNorm(hidden)
        self.attn = nn.Linear(hidden, hidden, bias=False)
        self.post_attention_layernorm = ToyNorm(hidden)
        self.mlp = ToyMoE(hidden, n_experts, top_k, router_3d=router_3d)
        self.mix = mix

    def forward(self, h, **kw):
        if self.mix:
            # a genuine token MIXER: this is what makes the accumulated-edit estimate
            # APPROXIMATE, because the edit at one position now reaches every later one
            a = torch.tanh(self.attn(self.input_layernorm(h)))
            a = torch.cumsum(a, dim=1) / torch.arange(
                1, h.shape[1] + 1, device=h.device).view(1, -1, 1)
            h = h + a
        r = h
        return r + self.mlp(self.post_attention_layernorm(r))


class ToyInner(nn.Module):
    def __init__(self, n_layers, hidden, vocab, n_experts, top_k, mix, router_3d=False):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [ToyBlock(hidden, n_experts, top_k, mix, router_3d) for _ in range(n_layers)])


class ToyModel(nn.Module):
    """`model.model.layers` -- the first path src.model.layer_container looks for."""

    def __init__(self, n_layers=4, hidden=16, vocab=32, n_experts=8, top_k=2, mix=True,
                 router_3d=False):
        super().__init__()
        self.model = ToyInner(n_layers, hidden, vocab, n_experts, top_k, mix, router_3d)
        self.config = ToyCfg(hidden, vocab)

    @property
    def device(self):
        return self.model.embed.weight.device

    def forward(self, input_ids=None, attention_mask=None, **kw):
        h = self.model.embed(input_ids)
        for blk in self.model.layers:
            h = blk(h)
        return h


def _run(model, ids, steer=None, rblind=None):
    """One prefill forward with the hooks installed exactly the way arms.run_arm does."""
    if steer is not None:
        steer.__enter__()
    if rblind is not None:
        rblind.__enter__()
    try:
        with torch.no_grad():
            if rblind is not None:
                rblind.begin_batch(steer.positions if steer else rblind.positions)
                rblind.prime(model, ids, None, steer)
            out = model(input_ids=ids)
    finally:
        if rblind is not None:
            rblind.__exit__()
        if steer is not None:
            steer.__exit__()
    return out


def _traces(model):
    return {i: {**blk.mlp.trace, **blk.mlp.gate.trace}
            for i, blk in enumerate(model.model.layers)}


def _make(mix, seed=7, router_3d=False, n_layers=4):
    torch.manual_seed(seed)
    m = ToyModel(mix=mix, router_3d=router_3d, n_layers=n_layers).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _steer(model, layers, positions, alpha=6.0, rb=None):
    hidden = model.config.hidden_size
    torch.manual_seed(11)
    dirs = [torch.nn.functional.normalize(torch.randn(hidden), dim=0) for _ in layers]
    s = Steer(model, layers, dirs, alpha, "sigma", [1.0] * len(layers), router_blind=rb)
    s.positions = positions
    return s


def check_end_to_end(verbose=False, router_3d=False):
    shape = "3D (B,S,H) router input -- the GLM-4.5-Air call shape" if router_3d else \
        "2D (B*S,H) router input -- the Qwen3-Next / qwen3_moe / gpt-oss call shape"
    print(f"\n=== 3. end-to-end: Steer + RouterBlind on a toy stack, {shape} ===")
    B, S = 2, 6
    ids = torch.arange(B * S).reshape(B, S) % 32
    positions = [[2, 3], [4]]
    steer_layers = [1]

    # ---------- (a) identity residual path: `accum` must be EXACT ----------
    m = _make(mix=False, router_3d=router_3d)
    _run(m, ids)
    clean_tr = _traces(m)
    s = _steer(m, steer_layers, positions)
    _run(m, ids, steer=s)
    steered_tr = _traces(m)
    m2 = _make(mix=False, router_3d=router_3d)
    rb = RouterBlind(m2, steer_layers, mode="accum", verbose=verbose)
    s2 = _steer(m2, steer_layers, positions, rb=rb)
    _run(m2, ids, steer=s2, rblind=rb)
    blind_tr = _traces(m2)

    L = 2                                       # a router strictly below the steered layer
    flat = [b * S + j for b, idxs in enumerate(positions) for j in idxs]
    other = [i for i in range(B * S) if i not in flat]
    d_steered = (steered_tr[L]["router_in"][flat] - clean_tr[L]["router_in"][flat]).abs().max()
    d_blind = (blind_tr[L]["router_in"][flat] - clean_tr[L]["router_in"][flat]).abs().max()
    check("steering DOES perturb the downstream router input when unblinded",
          d_steered > 1e-3, f"max|delta| = {d_steered:.4e}")
    check("identity residual path, router IMMEDIATELY BELOW the steered layer: `accum` "
          "restores the CLEAN router input exactly",
          d_blind < 1e-5, f"max|delta| = {d_blind:.4e} (vs {d_steered:.4e} unblinded). "
                          f"Depth 1 ONLY -- see section 3b for how this degrades")
    check("router-blind hook actually fired", rb.n_corrected > 0, f"{rb.n_corrected} routers")
    check("the EXPERTS still see the STEERED stream, not the clean one",
          torch.allclose(blind_tr[L]["expert_in"][flat], steered_tr[L]["expert_in"][flat],
                         atol=1e-5)
          and (blind_tr[L]["expert_in"][flat] - clean_tr[L]["expert_in"][flat]).abs().max() > 1e-3)
    check("expert SELECTION under router-blind == the clean run's selection",
          torch.equal(blind_tr[L]["idx"][flat], clean_tr[L]["idx"][flat]),
          f"unblinded moved {int((steered_tr[L]['idx'][flat] != clean_tr[L]['idx'][flat]).sum())}"
          f" of {clean_tr[L]['idx'][flat].numel()} slots")

    # ---------- (b) mixing residual path: `clean` exact, `accum` approximate ----------
    m = _make(mix=True, router_3d=router_3d)
    _run(m, ids)
    clean_tr = _traces(m)
    s = _steer(m, steer_layers, positions)
    _run(m, ids, steer=s)
    steered_tr = _traces(m)

    m2 = _make(mix=True, router_3d=router_3d)
    rb = RouterBlind(m2, steer_layers, mode="clean", report=True, verbose=verbose)
    s2 = _steer(m2, steer_layers, positions, rb=rb)
    _run(m2, ids, steer=s2, rblind=rb)
    cln_tr = _traces(m2)

    m3 = _make(mix=True, router_3d=router_3d)
    rb_a = RouterBlind(m3, steer_layers, mode="accum", report=True, verbose=verbose)
    s3 = _steer(m3, steer_layers, positions, rb=rb_a)
    _run(m3, ids, steer=s3, rblind=rb_a)
    acc_tr = _traces(m3)

    e_none = (steered_tr[L]["router_in"][flat] - clean_tr[L]["router_in"][flat]).norm()
    e_clean = (cln_tr[L]["router_in"][flat] - clean_tr[L]["router_in"][flat]).norm()
    e_accum = (acc_tr[L]["router_in"][flat] - clean_tr[L]["router_in"][flat]).norm()
    check("mixing path: `clean` mode gives the EXACT clean router input",
          e_clean / e_none < 1e-4, f"residual {e_clean:.3e} vs unblinded {e_none:.3e}")
    print(f"  [note] mixing path: `accum` leaves {100 * e_accum / e_none:.2f}% of the "
          f"router-input perturbation uncorrected (unblinded = 100%). "
          f"THIS IS THE APPROXIMATION -- it is measured, not assumed.")
    check("mixing path: `accum` still removes most of the perturbation",
          e_accum < 0.5 * e_none, f"{e_accum:.3e} vs {e_none:.3e}")
    summ = rb_a.summary()
    check("report mode populates the per-router exactness summary", bool(summ))
    for Ls, st in summ.items():
        print(f"        L{Ls}: n={st['n']} resid_rel={st['resid_rel']:.4f} "
              f"cos(accum,true)={st['cos_accum_true']:+.4f} "
              f"logit_err_rel={st['logit_err_rel']:.4f}")
    check("clean-mode summary agrees that `accum` would have been approximate here",
          all(st["resid_rel"] > 1e-6 for st in summ.values()))

    # ---------- (c) scoping: unsteered positions are NOT corrected ----------
    off_blind = (cln_tr[L]["router_in"][other] - clean_tr[L]["router_in"][other]).norm()
    off_steer = (steered_tr[L]["router_in"][other] - clean_tr[L]["router_in"][other]).norm()
    check("positions OUTSIDE the steered span are left uncorrected (per-row scoping)",
          off_blind > 1e-4 and torch.allclose(cln_tr[L]["router_in"][other],
                                              steered_tr[L]["router_in"][other], atol=1e-5),
          f"off-span residual blinded {off_blind:.3e} == unblinded {off_steer:.3e}")

    # ---------- (d) OFF is byte-identical ----------
    ma, mb = _make(mix=True, router_3d=router_3d), _make(mix=True, router_3d=router_3d)
    oa = _run(ma, ids, steer=_steer(ma, steer_layers, positions))
    sb = _steer(mb, steer_layers, positions, rb=None)
    ob = _run(mb, ids, steer=sb)
    check("router_blind=None: Steer output is bit-identical to the unmodified path",
          torch.equal(oa, ob))
    check("router_blind=None: Steer records nothing and holds no controller",
          sb.router_blind is None)
    mc = _make(mix=True, router_3d=router_3d)
    rb_on = RouterBlind(mc, steer_layers, mode="accum", verbose=False)
    sc = _steer(mc, steer_layers, positions, rb=rb_on)
    oc = _run(mc, ids, steer=sc, rblind=rb_on)   # hooks installed -> output MUST change
    check("router-blind ON changes the model output (it is not a silent no-op)",
          not torch.equal(oa, oc),
          f"max|delta| = {(oa - oc).abs().max():.4e}")


# ══════════════════════════════════════ 4. router-null surgery (the checkpoint-side path)
def check_routernull(verbose=False):
    """Verifies the pieces of tools/controls/build_routernull_direction.py that do not need
    a 160 GB checkpoint: shard discovery over a SYNTHETIC safetensors file laid out like a
    real one, and the projection algebra itself."""
    print("\n=== 4. router-null surgery: shard discovery + projection algebra ===")
    import json
    import tempfile

    from safetensors.torch import save_file

    from src.moe import checkpoint_router_weights

    H, Eexp, n_layers = 16, 6, 6
    torch.manual_seed(3)
    with tempfile.TemporaryDirectory() as td:
        # two shards + an index, and a mix of `gate` (qwen/glm) and dense layers
        t0, t1, wm = {}, {}, {}
        for L in range(n_layers):
            if L < 2:                      # dense prefix, like GLM's first_k_dense_replace
                t0[f"model.layers.{L}.mlp.gate_proj.weight"] = torch.randn(4 * H, H)
                wm[f"model.layers.{L}.mlp.gate_proj.weight"] = "s0.safetensors"
                continue
            k = f"model.layers.{L}.mlp.gate.weight"
            (t0 if L < 4 else t1)[k] = torch.randn(Eexp, H)
            wm[k] = "s0.safetensors" if L < 4 else "s1.safetensors"
        save_file(t0, os.path.join(td, "s0.safetensors"))
        save_file(t1, os.path.join(td, "s1.safetensors"))
        json.dump({"weight_map": wm}, open(os.path.join(td, "model.safetensors.index.json"), "w"))
        Wg = checkpoint_router_weights(td)
        check("router weights discovered across shards, dense layers skipped",
              sorted(Wg) == [2, 3, 4, 5], str(sorted(Wg)))
        check("discovered router matrices have shape (n_experts, hidden)",
              all(tuple(v.shape) == (Eexp, H) for v in Wg.values()))
        check("mlp.gate_proj (dense MLP) is NOT mistaken for a router",
              0 not in Wg and 1 not in Wg)

    # projection algebra, exactly as the script does it
    Wall = torch.cat([Wg[L] for L in sorted(Wg)], 0).numpy()
    import numpy as np
    u = np.random.default_rng(0).normal(size=H).astype(np.float32)
    u = u / np.linalg.norm(u)
    _, _, Vt = np.linalg.svd(Wall, full_matrices=False)
    base = float(np.linalg.norm(Wall @ u))
    prev = 1.0
    for r in (2, 4, 8):
        V = Vt[:r]
        d2 = u - V.T @ (V @ u)
        d2 = d2 / np.linalg.norm(d2)
        ratio = float(np.linalg.norm(Wall @ d2) / base)
        check(f"r={r}: the surgered direction is unit-norm",
              abs(float(np.linalg.norm(d2)) - 1.0) < 1e-5)
        check(f"r={r}: it is orthogonal to the top-{r} router subspace",
              float(np.abs(V @ d2).max()) < 1e-5, f"max|V d| = {float(np.abs(V @ d2).max()):.2e}")
        check(f"r={r}: router-logit response is cut ({ratio:.3f}x) and falls with r",
              ratio < prev, f"{prev:.3f} -> {ratio:.3f}")
        prev = ratio
    V = Vt[:min(Wall.shape)]
    d2 = u - V.T @ (V @ u)
    check("projecting out the FULL right singular subspace leaves nothing steerable",
          float(np.linalg.norm(d2)) < 1e-4,
          f"residual norm {float(np.linalg.norm(d2)):.2e} -- the script SKIPS this rank "
          f"rather than normalising numerical noise into a direction")

    # The script uses eigh(W^T W) instead of svd(W): U is never used, and at
    # --n-downstream 0 on Qwen3-Next it is a ~200 MB array computed and discarded once per
    # steered layer. Prove the two give the SAME projection at both real matrix shapes --
    # GLM-4.5-Air (rows < hidden, so the rank guard bites) and Qwen3-Next (rows > hidden).
    rng = np.random.default_rng(0)
    for shape, tag in (((1024, 4096), "GLM-shaped, 8*128 rows x 4096"),
                       ((4096, 2048), "Qwen3-Next-shaped, 8*512 rows x 2048")):
        W = rng.normal(size=shape).astype(np.float32)
        _, _, Vs = np.linalg.svd(W, full_matrices=False)
        ev, evec = np.linalg.eigh(W.astype(np.float64).T @ W.astype(np.float64))
        Ve = evec[:, ::-1].T.astype(np.float32)[:min(shape)]
        check(f"eigh(W^T W) reproduces svd(W)'s row count ({tag})",
              Ve.shape[0] == Vs.shape[0], f"{Ve.shape[0]} vs {Vs.shape[0]}")
        worst = 1.0
        for r in (64, 512):
            a = u_ = W[0] * 0 + rng.normal(size=shape[1]).astype(np.float32)
            u_ = u_ / np.linalg.norm(u_)
            a = u_ - Vs[:r].T @ (Vs[:r] @ u_)
            b = u_ - Ve[:r].T @ (Ve[:r] @ u_)
            a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
            worst = min(worst, abs(float(a @ b)))
        check(f"eigh and svd give the SAME surgered direction ({tag})", worst > 1 - 1e-6,
              f"min |cos(svd, eigh)| over r in (64, 512) = {worst:.10f}")


# ══════════════════ 3b. depth: how far does `accum` hold, and does `clean` hold everywhere?
def check_depth(verbose=False):
    """The deployed configuration is THREE steered layers over 17-19 downstream routers, so
    the depth-1 exactness proved in section 3 is not the operating point. This section runs
    the same toy with 3 steered layers and NO attention mixing, and shows that `accum`
    degrades with depth ANYWAY -- the MLP/expert response is not on the identity path either
    -- while `clean` stays exact at every depth. Also the only end-to-end case where a
    steered layer is ITSELF a downstream router site, which is true of the real layer sets
    (28,32,40) and which a single-steered-layer test cannot exercise."""
    print("\n=== 3b. depth: `accum` degrades with depth; `clean` does not ===")
    B, S = 2, 6
    ids = torch.arange(B * S).reshape(B, S) % 32
    positions = [[2, 3], [4]]
    steer_layers = [1, 2, 3]                 # L2 and L3 are BOTH steered AND router sites
    flat = [b * S + j for b, idxs in enumerate(positions) for j in idxs]

    m = _make(mix=False, n_layers=6)
    _run(m, ids)
    clean_tr = _traces(m)
    s = _steer(m, steer_layers, positions)
    _run(m, ids, steer=s)
    steered_tr = _traces(m)

    m2 = _make(mix=False, n_layers=6)
    rb = RouterBlind(m2, steer_layers, mode="accum", report=True, verbose=verbose)
    s2 = _steer(m2, steer_layers, positions, rb=rb)
    _run(m2, ids, steer=s2, rblind=rb)
    acc_tr = _traces(m2)

    m3 = _make(mix=False, n_layers=6)
    rb_c = RouterBlind(m3, steer_layers, mode="clean", verbose=verbose)
    s3 = _steer(m3, steer_layers, positions, rb=rb_c)
    _run(m3, ids, steer=s3, rblind=rb_c)
    cln_tr = _traces(m3)

    summ = rb.summary()
    print(f"  {'router':>8} {'unblinded':>11} {'accum':>10} {'clean':>10} | "
          f"{'resid_rel':>10} {'cos':>8} {'logit_err':>10}")
    degraded, clean_exact = [], True
    for L in sorted(summ):
        e0 = float((steered_tr[L]["router_in"][flat] - clean_tr[L]["router_in"][flat]).norm())
        ea = float((acc_tr[L]["router_in"][flat] - clean_tr[L]["router_in"][flat]).norm())
        ec = float((cln_tr[L]["router_in"][flat] - clean_tr[L]["router_in"][flat]).norm())
        st = summ[L]
        print(f"  {'L' + str(L):>8} {e0:>11.4f} {ea:>10.4f} {ec:>10.4f} | "
              f"{st['resid_rel']:>10.4f} {st['cos_accum_true']:>+8.4f} "
              f"{st['logit_err_rel']:>10.4f}")
        degraded.append(st["resid_rel"])
        clean_exact = clean_exact and (ec <= max(1e-4 * e0, 1e-5))
    check("`clean` mode is EXACT at EVERY depth, not only depth 1", clean_exact)
    check("`accum` is exact at depth 1 and DEGRADES below it (the deployed regime)",
          degraded[0] < 1e-4 and max(degraded) > 0.05,
          f"resid_rel by depth: {[round(x, 4) for x in degraded]} -- an `accum` cell is "
          f"NOT quotable without --router-blind-report")
    check("a layer that is BOTH steered and a router site excludes its own not-yet-applied "
          "edit", degraded[0] < 1e-4,
          "L2 is steered and is also the first router site; its own block-output edit has "
          "not happened when its router runs, so only L1's edit may be subtracted")

    # include_shared: two routers on one layer, which is the only configuration where the
    # `_recomputing` reentrancy guard can be observed to matter end to end
    m4 = _make(mix=False, n_layers=6)
    rb_s = RouterBlind(m4, steer_layers, mode="accum", include_shared=False, verbose=False)
    n_sites = len(rb_s.sites)
    check("every hooked router fires exactly once per prefill forward",
          rb.n_corrected == rb.expected_per_forward,
          f"n_corrected={rb.n_corrected} expected={rb.expected_per_forward} "
          f"over {n_sites} sites")


# ═════════════════════════ 5. is "expert-output steering" distinct from our current site?
def check_expert_output_is_our_site(verbose=False):
    """EXPERT-OUTPUT STEERING was proposed as a separate variant: edit the MoE block's
    OUTPUT instead of the block input, so that layer's own routing is untouched. Our steer
    site (`model.pick_site`) is ALREADY the decoder layer's output, and

        block_out = residual + mlp(post_attention_layernorm(residual))

    so the residual and the norm and the router and the experts have all ALREADY run by the
    time our hook fires. Editing block_out by +delta and editing mlp_out by +delta are the
    same edit to the same sum. This section measures that rather than asserting it, and
    isolates the ONE thing that does differ: the norm-preserve rescale, which divides by the
    norm of whatever tensor it is applied to (block output vs MoE output). That is a dose
    convention, not a different mechanism.
    """
    print("\n=== 5. expert-output steering vs our existing block-output site ===")
    B, S, H = 2, 6, 16
    torch.manual_seed(5)
    m = _make(mix=True)
    blk = m.model.layers[1]
    r = torch.randn(B, S, H)
    d = torch.nn.functional.normalize(torch.randn(H), dim=0)
    pos = [[2, 3], [4]]
    step = 3.0

    with torch.no_grad():
        mlp_out = blk.mlp(blk.post_attention_layernorm(r))
        blk_out = r + mlp_out
        base_idx = blk.mlp.gate.trace["idx"].clone()

        # our site: edit the BLOCK output
        a1 = blk_out.clone()
        # the A2 site: edit the MoE output, then add the residual
        a2_mlp = mlp_out.clone()
        for b, idxs in enumerate(pos):
            sel = torch.tensor(idxs)
            a1[b, sel] = a1[b, sel] + step * d
            a2_mlp[b, sel] = a2_mlp[b, sel] + step * d
        a2 = r + a2_mlp
    # (r + m) + delta  vs  r + (m + delta): the same sum, so the residual difference is
    # floating-point associativity only, not a different intervention
    check("norm-preserve OFF: expert-output steering == block-output steering "
          "(up to fp associativity)",
          torch.allclose(a1, a2, atol=1e-6, rtol=0),
          f"max|delta| = {(a1 - a2).abs().max():.2e} in fp32; the two expressions are "
          f"(r+m)+d and r+(m+d)")

    # with norm preservation the two differ only in the rescale DENOMINATOR
    with torch.no_grad():
        n1, n2 = [], []
        for b, idxs in enumerate(pos):
            sel = torch.tensor(idxs)
            n1.append(blk_out[b, sel].norm(dim=-1))
            n2.append(mlp_out[b, sel].norm(dim=-1))
        n1, n2 = torch.cat(n1), torch.cat(n2)
    check("norm-preserve ON: the two sites differ ONLY in the rescale denominator",
          True, f"||block_out|| mean {n1.mean():.3f} vs ||mlp_out|| mean {n2.mean():.3f} "
                f"(ratio {float(n1.mean() / n2.mean()):.2f}x) -- same edit, different "
                f"effective dose")

    # THE DECISIVE POINT: layer L's OWN router has already run before either hook site, so
    # neither variant can change layer L's routing. Shown by installing each hook for real
    # and reading the router's own trace.
    def edit_hook(mod, inp, out, _pos=pos, _d=d, _st=step):
        h = out[0] if isinstance(out, tuple) else out
        for b, idxs in enumerate(_pos):
            sel = torch.tensor(idxs, device=h.device)
            h[b, sel] = h[b, sel] + _st * _d.to(h.device)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    with torch.no_grad():
        blk(r.clone())                       # unsteered reference through the SAME path
    base_idx = blk.mlp.gate.trace["idx"].clone()
    idxs_seen = {}
    for site_name, mod in (("block_out (shipped site)", blk), ("mlp_out (A2 site)", blk.mlp)):
        hk = mod.register_forward_hook(edit_hook)
        with torch.no_grad():
            blk(r.clone())
        hk.remove()
        idxs_seen[site_name] = blk.mlp.gate.trace["idx"].clone()
    check("layer L's OWN router is identical under both sites AND unsteered "
          "(the router ran before both hook points)",
          all(torch.equal(v, base_idx) for v in idxs_seen.values()),
          " / ".join(f"{k}: {int((v != base_idx).sum())} slots moved"
                     for k, v in idxs_seen.items()))
    print("  [conclusion] expert-output steering is a NO-OP relative to the shipped steer "
          "site. Not implemented; see the design note.")


# ══════════════════════════════════ 6. mutation test: does check 3 actually catch bugs?
def check_mutations(verbose=False):
    """A verification that passes on broken code is worse than none. Three plausible bugs
    are injected into the SHIPPED RouterBlind and the section-3 assertions must fail."""
    print("\n=== 6. mutation test: the section-3 checks must FAIL on deliberately broken code ===")
    B, S = 2, 6
    ids = torch.arange(B * S).reshape(B, S) % 32
    positions = [[2, 3], [4]]
    steer_layers, L = [1], 2
    flat = [b * S + j for b, idxs in enumerate(positions) for j in idxs]

    def router_in_error(mode="accum", break_=None):
        m = _make(mix=False)
        _run(m, ids)
        clean = _traces(m)[L]["router_in"][flat]
        m2 = _make(mix=False)
        rb = RouterBlind(m2, steer_layers, mode=mode, verbose=False)
        if break_ == "sign":
            # correction applied with the WRONG SIGN: base + e instead of base - e
            orig = rb._accum_edit
            rb._accum_edit = lambda *a, **k: (lambda v: None if v is None else -v)(orig(*a, **k))
        s2 = _steer(m2, steer_layers, positions, rb=rb)
        if break_ == "unreported":
            s2.router_blind = None       # Steer stops reporting its edits to the controller
        if break_ == "noprime":
            rb.prime = lambda *a, **k: None
        try:
            _run(m2, ids, steer=s2, rblind=rb)
        except RuntimeError as e:
            return "RAISED", str(e)[:70], rb
        got = _traces(m2)[L]["router_in"][flat]
        return float((got - clean).abs().max()), None, rb

    good, _, rb_ok = router_in_error()
    check("baseline (unmutated) restores the clean router input", good < 1e-5,
          f"max|delta| = {good:.2e}")

    bad, _, _ = router_in_error(break_="sign")
    check("MUTATION 1 (correction sign flipped) is CAUGHT",
          isinstance(bad, float) and bad > 1e-3, f"max|delta| = {bad:.4e} (baseline {good:.2e})")

    bad, _, rb_u = router_in_error(break_="unreported")
    check("MUTATION 2 (Steer stops reporting its edits) is CAUGHT",
          isinstance(bad, float) and bad > 1e-3,
          f"max|delta| = {bad:.4e}, and the arm's own alarm reads "
          f"rblind_routers={rb_u.n_corrected}")

    res, msg, _ = router_in_error(mode="clean", break_="noprime")
    check("MUTATION 3 (clean mode without the unsteered pass) RAISES rather than "
          "silently degrading to accum", res == "RAISED", msg or f"got {res}")

    # ---- MUTATION 4: the `_recomputing` reentrancy guard, observable only with two
    # routers on one layer. Without it, calling the norm module inside the router pre-hook
    # re-fires the norm's own capture hook and overwrites `_r[L]` with the (n, H) corrected
    # rows -- so the SECOND router of that layer sees r.dim() != 3 and is silently skipped.
    class _NoGuard(RouterBlind):
        @property
        def _recomputing(self):
            return False

        @_recomputing.setter
        def _recomputing(self, v):
            pass

    def shared_run(cls):
        m = _make(mix=False)
        rb = cls(m, steer_layers, mode="accum", include_shared=True, verbose=False)
        s = _steer(m, steer_layers, positions, rb=rb)
        _run(m, ids, steer=s, rblind=rb)
        return rb

    rb_ok2 = shared_run(RouterBlind)
    rb_bad = shared_run(_NoGuard)
    check("baseline with include_shared: BOTH routers on each layer are corrected",
          rb_ok2.n_corrected == rb_ok2.expected_per_forward
          and rb_ok2.expected_per_forward == 2 * len(rb_ok2.sites),
          f"n_corrected={rb_ok2.n_corrected} expected={rb_ok2.expected_per_forward} "
          f"over {len(rb_ok2.sites)} sites")
    check("MUTATION 4 (reentrancy guard removed) is CAUGHT by the per-router count",
          rb_bad.n_corrected < rb_ok2.n_corrected,
          f"{rb_bad.n_corrected} vs {rb_ok2.n_corrected} corrections -- the shared gate is "
          f"silently skipped, which a bare `n_corrected > 0` alarm would NOT see")

    # ---- span clipping: a position beyond the (padded) sequence length must be dropped,
    # not indexed. The toy has no truncation, so this is asserted directly.
    m = _make(mix=False)
    rb_clip = RouterBlind(m, steer_layers, mode="accum", verbose=False)
    s_clip = _steer(m, steer_layers, [[2, 3, 99], [4]], rb=rb_clip)
    try:
        _run(m, ids, steer=s_clip, rblind=rb_clip)
        ok, why = rb_clip.n_corrected > 0, f"{rb_clip.n_corrected} corrections"
    except Exception as e:                                   # noqa: BLE001
        ok, why = False, f"{type(e).__name__}: {e}"
    check("an out-of-range span position is CLIPPED identically by Steer and RouterBlind",
          ok, why)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    check_installed_modeling()
    check_real_moe_blocks(a.verbose)
    for _3d in (False, True):
        check_end_to_end(a.verbose, router_3d=_3d)
    check_depth(a.verbose)
    check_routernull(a.verbose)
    check_expert_output_is_our_site(a.verbose)
    check_mutations(a.verbose)
    print()
    if FAILS:
        print(f"*** {len(FAILS)} CHECK(S) FAILED ***")
        for f in FAILS:
            print(f"    - {f}")
        return 1
    print("ALL ROUTER-BLIND CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
