#!/usr/bin/env python
"""AutoDojo target-LLM plugin: our steered (or undefended) local agent as the
optimization target of AutoDojo's adaptive attack (arXiv:2606.15057).

AutoDojo's optimizer evaluates every candidate injection by running the target agent
end-to-end inside AgentDojo's pipeline. Its `get_llm` was given a plugin seam
(vendored fork `reference/autodojo`, branch xpia-integration: upstream
github.com/xhOwenMa/AutoDojo pinned at commit
abbcbd8d59ea19115dc874eeb2cf294169ac5e0d, plus the 8-patch series shipped at
reference/autodojo-patches/, applied on top with `git am`):

    --target-model "plugin:autodojo_target?model=openai/gpt-oss-20b&probe_dir=runs/gpt-oss-20b-userabl&direction=combo_ovr8_pat1&alpha=8.06&layers=12,16,20&match_sigma_to=dim_no_override&max_new=4096&name=gpt-oss-20b-countersteer"

resolves here, and `make_llm` returns the SAME `SteeredLLM` pipeline element every one
of our recorded AgentDojo numbers runs on (tools/controls/agentdojo_bridge.py). An
empty/absent `direction` is the undefended arm; the two arms share one loaded model
(module-level cache keyed on the model id), so the defended run's reachability
pipeline (AUTODOJO_REACHABILITY_LLM, same spec with direction dropped) costs no extra
VRAM.

The element counts `n_steered_tokens`; a periodic progress line plus an atexit summary
give the smoke gate its positive evidence that the defense was live in every defended
episode (steered_tokens > 0) and OFF in the undefended arm (== 0).

Run under: PYTHONPATH=<repo>/tools/controls:<repo> plus the AutoDojo fork's
agentdojo/src, from the fork's repo root (its optimizer resolves prompts/seeds
relative to itself).
"""
import atexit
import os
import sys
from urllib.parse import parse_qsl

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _probe_eval as E  # noqa: E402
from agentdojo_bridge import SteeredLLM  # noqa: E402

X = E.X

_MODELS = {}     # model id -> (model, tok)
_ELEMENTS = []   # for the atexit summary


def _load(model_id, device):
    # AUTODOJO_TARGET_ATTN: optional attn_implementation override (e.g. flex_attention
    # for episodes beyond the gpt-oss eager-prefill memory wall). A non-empty value
    # CHANGES THE SERVING PATH (src/model.py docstring; the 23ae.10 rule) -- cells run
    # under it are LABELED and not directly comparable to eager-path cells. The label is
    # this loader's own print plus the env echoed into the lane log.
    attn = os.environ.get("AUTODOJO_TARGET_ATTN") or None
    key = (model_id, device, attn)
    if key not in _MODELS:
        print(f"[autodojo-target] loading {model_id} on {device}"
              + (f" attn_impl={attn} (LABELED PATH, non-eager)" if attn else ""),
              flush=True)
        _MODELS[key] = X.load_model_and_tok(model_id, device, attn_impl=attn)
    return _MODELS[key]


class _LoggedSteeredLLM(SteeredLLM):
    """SteeredLLM + periodic liveness/steering-evidence lines for the watchdog."""

    def query(self, *a, **kw):
        out = super().query(*a, **kw)
        if self.n_calls % 20 == 0:
            agri = (f" agri_checked={self.agri.n_checked} agri_fired={self.agri.n_fired} "
                    f"agri_prefilled={self.agri.n_prefilled}" if self.agri else "")
            print(f"[autodojo-target {self.name}] llm_calls={self.n_calls} "
                  f"steered_tokens={self.n_steered_tokens} "
                  f"truncated={self.n_truncated}{agri}", flush=True)
        return out


def _summary():
    for el in _ELEMENTS:
        agri = (f" agri_checked={el.agri.n_checked} agri_fired={el.agri.n_fired} "
                f"agri_prefilled={el.agri.n_prefilled}" if getattr(el, "agri", None) else "")
        print(f"[autodojo-target SUMMARY {el.name}] llm_calls={el.n_calls} "
              f"steered_tokens={el.n_steered_tokens} truncated={el.n_truncated}{agri}",
              flush=True)


atexit.register(_summary)


def make_llm(spec: str):
    """AutoDojo plugin entry point. `spec` = "plugin:<module>?k=v&..."."""
    # keep_blank_values: `match_sigma_to=` (empty = use the direction's OWN sigma,
    # the Qwen convention) must survive parsing — dropping it would silently fall
    # back to the gpt-oss default `dim_no_override`, a unit change, not an error.
    q = dict(parse_qsl(spec.split("?", 1)[1], keep_blank_values=True)) if "?" in spec else {}
    model_id = q.get("model")
    if not model_id:
        raise ValueError(f"plugin spec needs model=<hf id>: {spec}")
    device = q.get("device", "cuda:0")
    direction = q.get("direction") or None
    # CachePrune arm (arXiv:2504.21228): kv_mask = path to a mask JSON from
    # tools/controls/build_cacheprune_mask.py. SteeredLLM enforces mutual
    # exclusivity with `direction` (two defenses in one arm is not an arm).
    kv_mask = q.get("kv_mask") or None
    if kv_mask and not os.path.isabs(kv_mask):
        kv_mask = os.path.join(_ROOT, kv_mask)
    if kv_mask and not os.path.exists(kv_mask):
        raise ValueError(f"kv_mask not found: {kv_mask}")
    probe_dir = q.get("probe_dir")
    if probe_dir and not os.path.isabs(probe_dir):
        probe_dir = os.path.join(_ROOT, probe_dir)
    if direction and not probe_dir:
        raise ValueError(f"a steered plugin target needs probe_dir: {spec}")
    # AGRI arm (arXiv:2608.02657): agri_probe = spec JSON from
    # tools/controls/build_agri_probe.py. Needs in-process hidden states (the gate reads
    # a decoder-block output during an extra prefill pass), which is exactly why the
    # AutoDojo target stays a plugin element rather than an HTTP-served model.
    # SteeredLLM enforces mutual exclusivity with direction/kv_mask.
    agri_spec = q.get("agri_probe") or None
    agri = None
    if agri_spec:
        if not os.path.isabs(agri_spec):
            agri_spec = os.path.join(_ROOT, agri_spec)
        if not os.path.exists(agri_spec):
            raise ValueError(f"agri_probe not found: {agri_spec}")
    layers = tuple(int(x) for x in q.get("layers", "12,16,20").split(","))
    model, tok = _load(model_id, device)
    if agri_spec:
        from agri_gate import AGRIGate
        agri = AGRIGate(model, agri_spec)
    el = _LoggedSteeredLLM(
        model, tok,
        probe_dir=probe_dir,
        direction=direction,
        layers=layers,
        alpha=float(q.get("alpha", "8.0")),
        match_sigma_to=q.get("match_sigma_to", "dim_no_override"),
        max_new=int(q.get("max_new", "4096")),
        kv_mask=kv_mask,
        agri=agri,
    )
    el.name = q.get("name") or (model_id.split("/")[-1]
                                + ("-cacheprune" if kv_mask
                                   else "-agri" if agri_spec
                                   else "-countersteer" if direction
                                   else "-undefended"))
    print(f"[autodojo-target] element ready: name={el.name} model={model_id} "
          f"direction={direction or 'NONE'} kv_mask={kv_mask or 'NONE'} "
          f"agri_probe={agri_spec or 'NONE'} "
          f"layers={list(layers)} alpha={q.get('alpha', '8.0')} "
          f"max_new={q.get('max_new', '4096')}", flush=True)
    _ELEMENTS.append(el)
    return el
