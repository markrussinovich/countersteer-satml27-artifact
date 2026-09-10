#!/usr/bin/env python
"""CPU verification for MEAN-PRESERVING ABLATION (src/steering.py, --mode ablate_mp).

No GPU, no model, no checkpoint read: a toy decoder stack of identity blocks is enough,
because the object under test is the forward hook's arithmetic, not the model.

WHY THIS OPERATOR EXISTS. Plain `--mode ablate` deletes the whole coordinate,
`h <- h - (h.d)d`. The coordinate has a large non-zero MEAN, so that deletion is not a
neutral removal -- it applies a net push of -(mu.d) per edited token. On Qwen3-Next-80B at
the steered layers L28/32/40 the stored means project at +2.26/+0.95/-0.39 sigma, i.e. a
net -2.83 sigma along d, in the ATTACK-favouring direction (attack-succeeded rows sit LOWER
on d). FINDINGS section 23k therefore records the ablation null as CONFOUNDED: it cannot
separate "removing the axis does nothing" from "removing it would have helped and the
accompanying negative shift cancelled it". Mean-preserving ablation removes the same
coordinate while CANCELLING that push:

    h  <-  h - ((h - mu) . d) d       ==      h_ablate + (mu . d) d

HOW EXACT THAT CANCELLATION IS DEPENDS ON WHERE mu CAME FROM, and section 3 measures it
rather than asserting it. With `mean_from_span` mu is the mean of exactly these
activations at exactly this site, so the net displacement is 0 to arithmetic precision.
With a FIXED stored mu it is 0 only insofar as the stored mean matches the mean of the
edited tokens -- and it does not, because the probe pickles are captured at
`post_attention_layernorm` while this hook edits the block output. The residual is exactly
n*(mu - span_mean).d, which is what section 3 checks.

SEVEN SECTIONS:

  1. THE ALGEBRA, on synthetic tensors: the centered coordinate is zeroed, the operator
     equals plain ablate plus (mu.d)d, and plain ablate zeroes the RAW coordinate instead.
  2. NET DISPLACEMENT along d summed over the edited tokens: ~0 for the new operator,
     and for plain ablate the non-zero push -sum(h.d), reported in sigma units so it can be
     read against the additive alphas the same run sweeps. Run in float32 AND in bfloat16,
     because bf16 is what the models actually run in and the cancellation is the point.
  3. EXACT vs APPROXIMATE. With mu = the mean of the edited tokens the cancellation is
     exact; with a FIXED stored mu it is exact only to the extent that mu matches this
     span's mean, and the residual is measured to be exactly n*(mu - span_mean).d.
  3b. PER-LAYER mu and the *_add composition. Every numeric check above steers ONE layer,
     which is blind to a mu indexed to the wrong layer -- adversarial review demonstrated
     that `_mu` hardwired to layer 0 passed the whole file. This section steers three
     layers with deliberately distinct mu, and gives `ablate_mp_add` its only numeric
     coverage (against an explicit two-stage reference, under both norm_preserve settings).
  4. LOCALITY: only the intended positions change; every other position, row and layer is
     BIT-identical, and the pre-existing modes are bit-identical to a literal transcription
     of the code as it stood before this operator was added.
  5. THE NO-SILENT-ZERO GUARDS: a mean-preserving mode with no mu, with both mu sources,
     with a short mu list, or a mu handed to a mode that would ignore it -- all must raise.
     A mu of zero makes this operator numerically identical to plain `ablate`, which is the
     silent-fallback class of bug this project has already paid for twice.
  6. WIRING + CLI, no model: run_arm builds a Steer for every dose-free mode at alpha 0
     (FINDINGS section 23e), and the real argparse parser accepts `--mode ablate_mp`,
     `--mu-source`, and a NEGATIVE `--alphas` (the alpha ~ -1 additive control that isolates
     the shift from the deletion).

  7. MUTATION TEST, because a verification that passes on broken code is worse than none.
     Six plausible bugs are injected into the SHIPPED implementation -- mu dropped, mu sign
     flipped, mu indexed to the wrong layer, the hook narrowed to `mode == "ablate_mp"` (so
     `ablate_mp_add` loses its mu), the mu guard disabled, and the new mode removed from the
     steer-construction guard -- and the checks above must FAIL on each.

Usage:  .venv/bin/python tools/controls/verify_ablate_mp.py [-v]      (exit 0 = pass)
"""
import argparse
import json
import os
import sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import src.arms as A  # noqa: E402
import src.steering as S  # noqa: E402

FAILED = []
VERBOSE = False
QUIET = False          # set while the mutation section is deliberately breaking things


def say(*a):
    if not QUIET:
        print(*a)


def check(name, cond, detail=""):
    ok = bool(cond)
    if not ok:
        FAILED.append(name)
    if (not ok or VERBOSE) and not QUIET:
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return ok


# ═════════════════════════════════════════════════ the toy stack the hook is installed on
class _Block(nn.Module):
    """A decoder block that is the identity. `pick_site` hooks the BLOCK, so this is all
    the hook needs; anything the block would compute is downstream of the edit."""

    def forward(self, h, **kw):
        return h


class _Inner(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n)])


class _Model(nn.Module):
    """`layer_container` finds `model.model.layers`."""

    def __init__(self, n):
        super().__init__()
        self.model = _Inner(n)

    @property
    def device(self):
        return torch.device("cpu")


def make_steer(mode, layers, dirs, *, alpha=0.0, sigmas=None, mean_acts=None,
               mean_from_span=False, norm_preserve=True, scale="sigma"):
    return S.Steer(_Model(max(layers) + 1), layers, dirs, alpha, scale,
                   sigmas if sigmas is not None else [1.0] * len(layers), None, mode,
                   norm_preserve=norm_preserve,
                   mean_acts=mean_acts, mean_from_span=mean_from_span)


def apply_once(steer, h, positions):
    """Push a CLONE of `h` through steered-layer index 0 and return the edited tensor.

    Each steered layer is exercised on its own fresh input rather than in sequence: with
    identity blocks a second ablation of an already-ablated tensor is a no-op, which is an
    artefact of the toy and not of the model, where the coordinate is regenerated by the
    intervening computation."""
    out = h.clone()
    steer.positions = positions
    with steer:
        out = steer.mods[0](out)
    return out


def proj(x, d):
    """Projection of every token onto unit `d`, in float64 so the check is not measuring
    its own arithmetic."""
    return x.double() @ d.double()


# ═════════════════════════════════════════════════════════════ 1-3. the operator itself
def synth(B=3, T=9, D=32, seed=0, dtype=torch.float32, mu_shift=2.26):
    """A batch whose coordinate along `d` has a deliberate non-zero mean, which is the
    situation the operator exists for."""
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(D, generator=g)
    d = d / d.norm()
    h = torch.randn(B, T, D, generator=g)
    # plant the offset: every token sits mu_shift above the origin along d
    h = h + mu_shift * d
    return h.to(dtype), d.to(torch.float32)


def section_operator():
    say("=== 1-3. the operator: h <- h - ((h-mu).d)d ===")
    for dtype in (torch.float32, torch.bfloat16):
        tag = str(dtype).replace("torch.", "")
        h, d = synth(dtype=dtype)
        pos = [[1, 2, 3, 4], [0, 5], [2, 3, 6, 7, 8]]
        sel = [(b, j) for b, idxs in enumerate(pos) for j in idxs]
        edited = torch.tensor([[b, j] for b, j in sel])
        rows = h[edited[:, 0], edited[:, 1]]
        span_mean = rows.double().mean(0)

        # the FIXED mu the deployed arm uses is a stored corpus mean, NOT this span's mean.
        # Model that here: mu = span mean plus a deliberate offset along d, so section 3 can
        # measure the residual the mismatch leaves.
        mismatch = 0.31            # sigma, the Qwen3-Next capture-vs-probe-corpus gap
        mu_fixed = (span_mean + mismatch * d.double()).to(torch.float32)

        st_mp = make_steer("ablate_mp", [0], [d], mean_acts=[mu_fixed])
        st_ab = make_steer("ablate", [0], [d])
        st_sp = make_steer("ablate_mp", [0], [d], mean_from_span=True)
        out_mp = apply_once(st_mp, h, pos)
        out_ab = apply_once(st_ab, h, pos)
        out_sp = apply_once(st_sp, h, pos)

        p_pre = proj(h[edited[:, 0], edited[:, 1]], d)
        p_mp = proj(out_mp[edited[:, 0], edited[:, 1]], d)
        p_ab = proj(out_ab[edited[:, 0], edited[:, 1]], d)
        p_sp = proj(out_sp[edited[:, 0], edited[:, 1]], d)
        tol = 2e-5 if dtype == torch.float32 else 6e-2

        # (i) the CENTERED coordinate is zeroed: every edited token ends on mu's coordinate
        mu_c = float(mu_fixed.double() @ d.double())
        check(f"[{tag}] ablate_mp zeroes the CENTERED coordinate (h'.d == mu.d)",
              (p_mp - mu_c).abs().max() < tol,
              f"max|h'.d - mu.d| = {(p_mp - mu_c).abs().max():.2e}")
        # and plain ablation zeroes the RAW coordinate -- the contrast the arm is paired to
        check(f"[{tag}] plain ablate zeroes the RAW coordinate (h'.d == 0)",
              p_ab.abs().max() < tol, f"max|h'.d| = {p_ab.abs().max():.2e}")
        # (i') the algebraic identity h_mp = h_ablate + (mu.d)d
        recon = out_ab.double() + 0.0
        recon[edited[:, 0], edited[:, 1]] += mu_c * d.double()
        gap = (recon[edited[:, 0], edited[:, 1]]
               - out_mp[edited[:, 0], edited[:, 1]].double()).abs().max()
        check(f"[{tag}] h_mp == h_ablate + (mu.d)d on the edited rows", gap < tol,
              f"max|diff| = {gap:.2e}")

        # (ii) NET displacement along d, summed over the edited tokens
        n = len(sel)
        net_mp = float((p_mp - p_pre).sum())
        net_ab = float((p_ab - p_pre).sum())
        net_sp = float((p_sp - p_pre).sum())
        # plain ablation's push is exactly -sum(h.d) and is NOT small
        check(f"[{tag}] plain ablate applies the NON-ZERO push -sum(h.d)",
              abs(net_ab + float(p_pre.sum())) < tol * n and abs(net_ab) > 0.5 * n,
              f"net = {net_ab:+.3f} over {n} tokens = {net_ab / n:+.3f} per token")
        # the fixed-mu operator's residual is EXACTLY n*(mu - span_mean).d -- the mismatch,
        # nothing more. It is what makes the zero-displacement claim approximate here.
        want = n * (mu_c - float(span_mean @ d.double()))
        check(f"[{tag}] ablate_mp residual == n*(mu - span_mean).d, i.e. ONLY the mismatch",
              abs(net_mp - want) < max(tol * n, 1e-3),
              f"net = {net_mp:+.4f}, predicted {want:+.4f} "
              f"({net_mp / n:+.4f} vs plain ablate {net_ab / n:+.4f} per token)")
        # (iii) with mu = the span's own mean the cancellation is EXACT
        check(f"[{tag}] mean_from_span gives ~ZERO net displacement (exact by construction)",
              abs(net_sp) < max(tol * n, 1e-3),
              f"net = {net_sp:+.2e} over {n} tokens "
              f"({abs(net_sp) / max(abs(net_ab), 1e-9):.1e}x plain ablate's)")
        check(f"[{tag}] and it is MUCH smaller than plain ablate's push",
              abs(net_sp) < 0.01 * abs(net_ab), f"|{net_sp:+.2e}| vs |{net_ab:+.3f}|")
    say()


def section_multilayer():
    """PER-LAYER mu must reach the layer it belongs to, and ablate_mp_add must ablate.

    Both of these were BLIND SPOTS in the first version of this file, found by adversarial
    review (2026-09-02) by mutation: `Steer._mu` patched to always return `mean_acts[0]`
    passed every check, because every numeric section steered exactly ONE layer. On
    Qwen3-Next the per-layer mu.d_hat are +2.26/+0.95/-0.39 sigma, so a wrong-layer mu
    applies an arbitrary multi-sigma push -- exactly the confound the operator exists to
    remove. `ablate_mp_add` had no numeric coverage at all: only its CONSTRUCTION was
    exercised, so a hook written `if self.mode == "ablate_mp"` would have passed."""
    say("=== 3b. per-layer mu, and the *_add composition ===")
    D, K = 24, 3
    g = torch.Generator().manual_seed(11)
    dirs, mus = [], []
    for k in range(K):
        d = torch.randn(D, generator=g)
        dirs.append(d / d.norm())
        # DISTINCT and far apart, so using layer 0's mu at layer 2 cannot pass by luck
        mus.append((3.0 * (k + 1)) * dirs[-1] + torch.randn(D, generator=g))
    h = torch.randn(2, 6, D, generator=g)
    pos = [[1, 2, 3], [0, 4]]
    edited = torch.tensor([[b, j] for b, idxs in enumerate(pos) for j in idxs])

    st = make_steer("ablate_mp", list(range(K)), dirs, mean_acts=mus)
    for k in range(K):
        out = h.clone()
        st.positions = pos
        with st:
            out = st.mods[k](out)                       # drive layer index k only
        rows = out[edited[:, 0], edited[:, 1]]
        want = float(mus[k].double() @ dirs[k].double())
        got = proj(rows, dirs[k])
        check(f"layer index {k} uses ITS OWN mu (h'.d_k == mu_k.d_k)",
              (got - want).abs().max() < 2e-5, f"got {got.mean():.4f} want {want:.4f}")
        for other in range(K):
            if other == k:
                continue
            wrong = float(mus[other].double() @ dirs[k].double())
            check(f"...and NOT layer {other}'s mu at layer {k}",
                  abs(want - wrong) > 1e-3 and (got - wrong).abs().min() > 1e-3,
                  f"mu_{other}.d_{k} = {wrong:+.4f}")

    # ablate_mp_add: the ablation must actually happen underneath the additive step.
    # Checked against an explicit two-stage reference, and against plain ablate_add, which
    # must land on a DIFFERENT point (it deletes the raw coordinate, not the centered one).
    d, mu = dirs[0], mus[0]
    for npres in (False, True):
        st_mp = make_steer("ablate_mp_add", [0], [d], alpha=2.0, sigmas=[1.5],
                           mean_acts=[mu], norm_preserve=npres)
        st_ab = make_steer("ablate_add", [0], [d], alpha=2.0, sigmas=[1.5],
                           norm_preserve=npres)
        out_mp = apply_once(st_mp, h, pos)
        out_ab = apply_once(st_ab, h, pos)
        want = h.clone()
        for b, idxs in enumerate(pos):
            sel = torch.tensor(idxs)
            cur = want[b, sel]
            cur = cur - ((cur - mu.reshape(1, -1)) @ d).unsqueeze(-1) * d.unsqueeze(0)
            pre = cur.norm(dim=-1, keepdim=True)
            cur = cur + (2.0 / 1.0) * 1.5 * d                  # alpha/sqrt(1) * sigma
            if npres:
                cur = cur * (pre / cur.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            want[b, sel] = cur
        check(f"ablate_mp_add (norm_preserve={npres}) = mean-preserving ablation THEN the "
              f"additive step", (out_mp - want).abs().max() < 2e-5,
              f"max|diff| = {(out_mp - want).abs().max():.2e}")
        check(f"...and is NOT the same edit as ablate_add (norm_preserve={npres})",
              (out_mp - out_ab).abs().max() > 1e-3,
              f"max|diff| = {(out_mp - out_ab).abs().max():.3f}")
    # THE ADDITIVE STEP IS A DOSE, so the *_add form is NOT zero-net-displacement. State it
    # as a measurement rather than leaving the reader to assume the mp property survives.
    st_mp = make_steer("ablate_mp_add", [0], [d], alpha=2.0, sigmas=[1.5], mean_acts=[mu])
    out = apply_once(st_mp, h, pos)
    net = float((proj(out[edited[:, 0], edited[:, 1]], d)
                 - proj(h[edited[:, 0], edited[:, 1]], d)).sum())
    check("ablate_mp_add does NOT preserve the mean -- the additive step is a dose, by "
          "design", abs(net) > 1.0, f"net = {net:+.3f} sigma-units over {len(edited)} tokens")
    say()


def section_locality():
    say("=== 4. locality and the untouched code paths ===")
    h, d = synth(seed=3)
    pos = [[1, 2], [], [4]]
    mu = [(h[0, 1].double() + 0.4 * d.double()).float()]      # a FIXED, mismatched mu
    st = make_steer("ablate_mp", [0], [d], mean_acts=mu)
    out = apply_once(st, h, pos)
    touched = {(b, j) for b, idxs in enumerate(pos) for j in idxs}
    others_same = all(torch.equal(out[b, j], h[b, j])
                      for b in range(h.shape[0]) for j in range(h.shape[1])
                      if (b, j) not in touched)
    check("every position OUTSIDE the span is BIT-identical", others_same)
    check("every position INSIDE the span changed",
          all(not torch.equal(out[b, j], h[b, j]) for b, j in touched))
    check("a row with NO positions is untouched", torch.equal(out[1], h[1]))
    # A DOCUMENTED DEGENERACY OF `--mu-source span`, not a bug: with mu taken from the
    # span's own mean, a span of ONE token has nothing to be centered against and the edit
    # is exactly a no-op. Real spans are ~148 tokens, so this never fires in deployment --
    # but it is the reason `span` is not the default, and it is asserted rather than assumed.
    out_sp = apply_once(make_steer("ablate_mp", [0], [d], mean_from_span=True), h, pos)
    check("mean_from_span on a ONE-token span is a no-op (documented degeneracy)",
          torch.equal(out_sp[2, 4], h[2, 4]))
    check("...while a multi-token span in the same batch IS edited",
          not torch.equal(out_sp[0, 1], h[0, 1]))

    # BYTE-IDENTICAL WHEN OFF. The pre-existing branches are compared against a literal
    # transcription of the code as it stood before this change; anything but bitwise
    # equality means an existing cell moved.
    def legacy_ablate(cur, dd):
        return cur - (cur @ dd).unsqueeze(-1) * dd.unsqueeze(0)

    def legacy_add(cur, dd, step, norm_preserve):
        pre = cur.norm(dim=-1, keepdim=True)
        cur = cur + step * dd
        if norm_preserve:
            cur = cur * (pre / cur.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return cur

    for dtype in (torch.float32, torch.bfloat16):
        tag = str(dtype).replace("torch.", "")
        h, d = synth(seed=5, dtype=dtype)
        pos = [[1, 2, 3], [0], [7, 8]]
        dd = d.to(dtype)
        for mode, alpha, npres in (("ablate", 0.0, True), ("add", 4.0, True),
                                   ("add", 4.0, False), ("ablate_add", 4.0, True)):
            st = make_steer(mode, [0], [d], alpha=alpha, sigmas=[1.7],
                            norm_preserve=npres)
            out = apply_once(st, h, pos)
            want = h.clone()
            for b, idxs in enumerate(pos):
                if not idxs:
                    continue
                cur = want[b, torch.tensor(idxs)]
                if mode in ("ablate", "ablate_add"):
                    cur = legacy_ablate(cur, dd)
                if mode in ("add", "ablate_add"):
                    cur = legacy_add(cur, dd, alpha / 1.0 * 1.7, npres)
                want[b, torch.tensor(idxs)] = cur
            check(f"[{tag}] mode={mode} alpha={alpha} norm_preserve={npres} is "
                  f"BIT-identical to the pre-change code", torch.equal(out, want))
    # prefill_off is gated on mode=="add": a dose-free mode must NOT be short-circuited
    check("prefill_off stays True for add@0 (clean / base-XPIA arms unchanged)",
          make_steer("add", [0], [d], alpha=0.0).prefill_off)
    for m in S.ABLATE_MODES:
        check(f"prefill_off is False for {m}@0 (the hook must run)",
              not make_steer(m, [0], [d], alpha=0.0,
                             **({"mean_from_span": True}
                                if m in S.MEAN_PRESERVING_MODES else {})).prefill_off)
    say()


def section_guards():
    say("=== 5. no silent zero: the mu guards ===")
    _, d = synth()

    def raises(fn):
        try:
            fn()
        except SystemExit:
            return True
        except Exception:
            return False
        return False

    check("ablate_mp with NO mu raises (a zero mu would silently be plain `ablate`)",
          raises(lambda: make_steer("ablate_mp", [0], [d])))
    check("ablate_mp with BOTH mu sources raises",
          raises(lambda: make_steer("ablate_mp", [0], [d], mean_acts=[d],
                                    mean_from_span=True)))
    check("ablate_mp with a mu list shorter than the steered layers raises",
          raises(lambda: make_steer("ablate_mp", [0, 1], [d, d], mean_acts=[d])))
    check("ablate_mp with a None inside the mu list raises",
          raises(lambda: make_steer("ablate_mp", [0, 1], [d, d], mean_acts=[d, None])))
    check("a mu handed to mode=add raises (it would run as plain `add` under an mp name)",
          raises(lambda: make_steer("add", [0], [d], alpha=1.0, mean_acts=[d])))
    check("an unknown mode raises", raises(lambda: make_steer("ablate_meanpreserving",
                                                              [0], [d])))
    # the loader refuses to invent a mu, and says which artifact is missing
    from src.probes import build_means
    check("build_means refuses `span` (it is a per-input quantity, not a stored vector)",
          raises(lambda: build_means(".", [0], "cpu", "span")))
    check("build_means refuses an unknown source",
          raises(lambda: build_means(".", [0], "cpu", "grand_mean")))
    check("build_means refuses a capture path that does not exist",
          raises(lambda: build_means(".", [0], "cpu", "capture:/nonexistent/x.json")))
    # a capture from ANOTHER model must be refused by NAME: Qwen3-30B-A3B-Thinking and
    # Qwen3-Next-80B share hidden_size 2048, so a shape check cannot see this
    cap = os.path.join(ROOT, "tmp", "verify_ablate_mp_capture.json")
    os.makedirs(os.path.dirname(cap), exist_ok=True)
    with open(cap, "w") as f:
        json.dump({"config": {"model": "vendor/model-A"}, "layers": [0],
                   "rows": [{"act": {"0": [1.0, 2.0]}}]}, f)
    try:
        check("build_means loads a capture and averages its rows",
              build_means(".", [0], "cpu", f"capture:{cap}")[0].tolist() == [1.0, 2.0])
        check("build_means refuses a capture taken on a DIFFERENT model",
              raises(lambda: build_means(".", [0], "cpu", f"capture:{cap}",
                                         model_id="vendor/model-B")))
        check("...and accepts it for the model it was taken on",
              not raises(lambda: build_means(".", [0], "cpu", f"capture:{cap}",
                                             model_id="vendor/model-A")))
        check("build_means refuses a capture missing a steered layer",
              raises(lambda: build_means(".", [0, 7], "cpu", f"capture:{cap}")))
    finally:
        os.remove(cap)
    say()


# ═══════════════════════════════════════════════════════ 6. wiring and the real CLI parser
class _BuiltError(Exception):
    def __init__(self, mode, kw):
        self.mode, self.kw = mode, kw


def steer_built(**kw):
    """(was a Steer constructed, the mode it got, the kwargs it got) for one run_arm call."""
    real = A.Steer

    class Stub:
        def __init__(self, model, layers, dirs, alpha, scale, sigmas, ablate_axes, mode,
                     *a, **k):
            raise _BuiltError(mode, k)

    A.Steer = Stub
    try:
        A.run_arm(object(), object(), [], **kw)
    except _BuiltError as b:
        return True, b.mode, b.kw
    except Exception:
        return False, None, {}
    finally:
        A.Steer = real
    return False, None, {}


def section_wiring():
    say("=== 6. run_arm wiring and the real argparse parser ===")
    base = dict(layers=[0, 1, 2], dirs=[None] * 3, sigmas=[1.0] * 3)
    for m in S.DOSE_FREE_MODES:
        built, mode, kw = steer_built(alpha=0.0, mode=m, **base)
        check(f"mode={m}, alpha=0 BUILDS a Steer (the FINDINGS 23e no-op bug)", built)
        check(f"...and that Steer receives mode={m!r}", mode == m, f"got {mode!r}")
    built, _, kw = steer_built(alpha=0.0, mode="ablate_mp", mean_acts=[1, 2, 3], **base)
    check("run_arm forwards mean_acts to Steer", kw.get("mean_acts") == [1, 2, 3])
    built, _, kw = steer_built(alpha=0.0, mode="ablate_mp", mean_from_span=True, **base)
    check("run_arm forwards mean_from_span to Steer", kw.get("mean_from_span") is True)
    built, _, _ = steer_built(alpha=0.0, mode="add", **base)
    check("mode=add, alpha=0 still builds NOTHING (clean / base-XPIA stay undefended)",
          not built)
    built, _, _ = steer_built(alpha=0.0, mode="ablate_mp", layers=None, dirs=None,
                              sigmas=None)
    check("no layers builds NOTHING even under a dose-free mode", not built)

    # THE REAL PARSER, no model: argparse is patched to hand back the namespace and stop
    # before src.cli.main() touches a checkpoint.
    import argparse as _ap

    import src.cli as C
    got = {}
    real_parse = _ap.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        ns = real_parse(self, args, namespace)
        got["ns"] = ns
        raise SystemExit(0)

    def parse(argv):
        got.clear()
        _ap.ArgumentParser.parse_args = spy
        old = sys.argv
        sys.argv = ["xpia_defense.py"] + argv
        try:
            C.main()
        except SystemExit:
            pass
        finally:
            sys.argv = old
            _ap.ArgumentParser.parse_args = real_parse
        return got.get("ns")

    common = ["--model", "m/x", "--stage", "sweep", "--steer-layers", "28,32,40"]
    ns = parse(common + ["--mode", "ablate_mp", "--alphas", "0"])
    check("--mode ablate_mp parses", ns is not None and ns.mode == "ablate_mp")
    from src.probes import MU_DEFAULT
    check("--mu-source is unset by default (resolved to MU_DEFAULT at run time)",
          ns is not None and ns.mu_source is None and MU_DEFAULT == "probe_grand")
    ns = parse(common + ["--mode", "ablate_mp", "--mu-source", "span", "--alphas", "0"])
    check("--mu-source span parses", ns is not None and ns.mu_source == "span")
    # THE NEGATIVE ADDITIVE CONTROL. argparse only refuses a bare negative number when the
    # parser itself defines options that look like negative numbers; this one does not, so
    # BOTH spellings must work -- and that is asserted rather than assumed.
    for argv, want, how in ((["--alphas", "-1"], [-1.0], "--alphas -1"),
                            (["--alphas=-1"], [-1.0], "--alphas=-1"),
                            (["--alphas", "-1", "-1.63"], [-1.0, -1.63],
                             "--alphas -1 -1.63")):
        ns = parse(common + ["--mode", "add"] + argv)
        check(f"a NEGATIVE additive alpha parses: `{how}`",
              ns is not None and ns.alphas == want,
              f"got {None if ns is None else ns.alphas}")
    say()


# ══════════════════════════════════════════════════════════════════ 7. mutation test
def section_mutations():
    say("=== 7. mutation test: the checks must FAIL on broken code ===")
    global FAILED
    real_mu = S.Steer._mu
    real_dose = A.DOSE_FREE_MODES

    def run_under(name, setup, teardown, sections):
        global FAILED, QUIET
        keep, FAILED = FAILED, []
        QUIET = True                # the failures below are the POINT; do not print them
        setup()
        try:
            for fn in sections:
                fn()
        except (Exception, SystemExit) as e:         # a crash is also a caught mutation --
            # and SystemExit is the NORMAL way this code refuses a broken configuration, so
            # it must be caught here (it is a BaseException, not an Exception, and an
            # earlier version of this file let it kill the run instead of scoring it)
            FAILED.append(f"raised {type(e).__name__}: {e}")
        finally:
            teardown()
            QUIET = False
        caught, FAILED = list(FAILED), keep
        check(f"MUTATION `{name}` is caught", caught,
              f"{len(caught)} check(s) failed: {caught[:3]}")

    # (a) the `- mu` term dropped: the operator silently becomes plain `ablate`
    def drop_mu():
        S.Steer._mu = lambda self, i, h: torch.zeros(1, h.shape[-1])
    run_under("mu dropped (operator degenerates to plain ablate)", drop_mu,
              lambda: setattr(S.Steer, "_mu", real_mu), [section_operator])

    # (b) mu's sign flipped -- doubles the shift instead of cancelling it
    def flip_mu():
        S.Steer._mu = lambda self, i, h, _r=real_mu: -_r(self, i, h)
    run_under("mu sign flipped", flip_mu,
              lambda: setattr(S.Steer, "_mu", real_mu), [section_operator])

    # (b2) EVERY layer served layer 0's mu. This passed the first version of this file --
    # every numeric section steered one layer -- and is why section_multilayer exists.
    def wrong_layer_mu():
        S.Steer._mu = lambda self, i, h, _r=real_mu: _r(self, 0, h)
    run_under("per-layer mu indexed wrong (always layer 0)", wrong_layer_mu,
              lambda: setattr(S.Steer, "_mu", real_mu), [section_multilayer])

    # (b3) the hook testing `mode == "ablate_mp"` instead of `in MEAN_PRESERVING_MODES`:
    # ablate_mp_add then silently runs plain ablate + add with mu ignored, while every
    # construction-time check still passes.
    def mp_only_ablate_mp():
        S._MP_SAVED2 = S.MEAN_PRESERVING_MODES
        S.MEAN_PRESERVING_MODES = ("ablate_mp",)
    run_under("hook narrowed to mode=='ablate_mp' (ablate_mp_add loses its mu)",
              mp_only_ablate_mp,
              lambda: setattr(S, "MEAN_PRESERVING_MODES", S._MP_SAVED2),
              [section_multilayer])

    # (c) the guards relaxed: a missing mu silently defaults to zero
    def relax_guard():
        S._MP_SAVED = S.MEAN_PRESERVING_MODES
        S.MEAN_PRESERVING_MODES = ()
    def restore_guard():
        S.MEAN_PRESERVING_MODES = S._MP_SAVED
    run_under("mean-preserving guard disabled (missing mu no longer raises)",
              relax_guard, restore_guard, [section_guards])

    # (d) the new mode removed from run_arm's steer-construction guard -- FINDINGS 23e
    def break_guard():
        A.DOSE_FREE_MODES = ("ablate", "ablate_add")
    run_under("ablate_mp removed from run_arm's steer-construction guard (23e)",
              break_guard, lambda: setattr(A, "DOSE_FREE_MODES", real_dose),
              [section_wiring])
    say()


def main():
    global VERBOSE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    VERBOSE = a.verbose
    torch.manual_seed(0)
    section_operator()
    section_multilayer()
    section_locality()
    section_guards()
    section_wiring()
    section_mutations()
    if FAILED:
        print(f"FAILED {len(FAILED)} check(s):")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL MEAN-PRESERVING-ABLATION CHECKS PASSED")


if __name__ == "__main__":
    main()
