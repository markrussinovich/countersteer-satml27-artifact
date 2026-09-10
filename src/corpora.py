"""Datasets and splits: the Nemotron XPIA corpus, the parameter-abuse corpus and its manifest, attacker-template-disjoint splitting, and the text corpora the role probe trains on."""
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


# ═══════════════════════════════════════════════ pinned upstream revisions
# `paper/sections/opensci.tex` promises that "benchmarks and the red-team dataset
# are referenced by public identifier and pinned revision". That promise is only
# kept if EVERY load names one, so the shas live here, in one table, and every
# `load_dataset` in this repo goes through `hf_dataset()` below.
#
# Provenance of each sha: the revision the cached copy our results were computed
# on actually carries (`$HF_HOME/hub/datasets--*/refs/main`, fetched 2026-07-31),
# re-resolved against the hub on 2026-09-02 -- every one of them was still that
# repo's `main`, so pinning changes nothing about what loads today and only fixes
# what loads tomorrow. dolma3 is the exception and was already pinned in code to
# an OLDER revision than current `main` (afa92bfb...); that explicit pin is the
# corpus the shipped probes were fit on and is preserved, full sha recorded.
HF_DATASET_REVISIONS = {
    # the XPIA corpus behind every single-turn number in the paper
    "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1":
        "d738d4f361cc38bb4d7a42b9066776dade5332f5",
    # role-probe training text
    "allenai/c4": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    "allenai/dolma3_mix-150B-1025": "3a8349c2f7946cdc56f8ccf22c555672be0b3208",
    # general-utility benchmarks (tools/controls/general_utility_bench.py)
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    # CachePrune fidelity anchor (tools/controls/cacheprune_anchor.py)
    "rajpurkar/squad": "7b6d24c440a36b6815f21b70d25016731768db1f",
    # carrier documents for the webpage corpora (build_paper_injection_dataset.py)
    "wikimedia/wikipedia": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    # The LLMail-Inject red-team dataset. UNENFORCED: the challenge data is vendored
    # under runs/llmail/ and `build_llmail_dataset.py` reads those files rather than
    # calling `hf_dataset`, so this entry documents which upstream revision the vendored
    # copy came from. It is the one row in this table with no call site.
    "microsoft/llmail-inject-challenge": "1063bdf01ec8762b812d5e06ee768a06faa5a6f7",
}


# Model revisions our results were computed on, read from the fleet's HF cache
# (`$HF_HOME/hub/models--*/refs/main`) on 2026-09-02. These are DOCUMENTED here and in
# the released manifest, but deliberately NOT yet enforced at load time: `from_pretrained`
# calls in src/model.py and tools/controls/* still resolve `main`. Enforcing them is a
# separate change, because it must first be verified that every box in the fleet (.7/.9/
# .11/.12 and the cluster workspaces) has the SAME sha cached -- pinning against an
# unverified cache turns a working job into a multi-hundred-GB re-download or a hard
# failure mid-sweep. Until that check is done, treat this table as provenance, not as a
# guarantee, and say so wherever the pinning claim is made.
MODEL_REVISIONS = {
    "openai/gpt-oss-20b": "6cee5e81ee83917806bbde320786a8fb61efebee",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
    # Gemma is the one model whose revision was in doubt, and it was RESOLVED by forensics
    # on 2026-09-02 (FINDINGS §18c-provenance): the reported test rung ran on the box that
    # holds only `842da379...`, and the .9 box's `ba74f5b6...` carries **byte-identical
    # weights** (same safetensors blob SHAs) -- the two revisions differ only in
    # `chat_template.jinja` and `tokenizer_config.json`. So the numbers are unambiguous,
    # but a reproducer resolving `ba74f5b6...` gets the same weights with a DIFFERENT chat
    # template, and our prompts are rendered against the template. Both keys are recorded
    # because the model id is spelled both ways on the hub and in our runs; the reported
    # runs used the lowercase spelling.
    "google/gemma-4-31B-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    "google/gemma-4-31b-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    "microsoft/Phi-3-medium-128k-instruct": "a088b37c71d441ab6d862bb3fcfe6165b3014702",
    "Qwen/Qwen3-Next-80B-A3B-Thinking": "e502dd4100cc68c0de57643fd4317ec93a128670",
    "zai-org/GLM-4.5-Air": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
    # baseline / anchor models
    "meta-llama/Meta-Llama-3-8B-Instruct": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
}


def hf_revision(repo: str) -> str:
    """The pinned revision for `repo`. An unpinned repo is an ERROR, not a default.

    Defaulting to `main` is exactly the failure this exists to prevent: it makes the
    Open Science claim silently false the moment upstream force-pushes, and it did so
    for the Nemotron corpus until 2026-09-02. Adding a load means adding its sha here.
    """
    try:
        return HF_DATASET_REVISIONS[repo]
    except KeyError:
        raise SystemExit(
            f"[data] {repo} has no pinned revision. Add its sha to "
            f"src/corpora.HF_DATASET_REVISIONS (resolve with "
            f"`HfApi().dataset_info('{repo}').sha`) -- see paper/sections/opensci.tex.")


def hf_dataset(repo: str, *args, **kw):
    """`load_dataset` at the pinned revision. Use this everywhere, never load_dataset."""
    from datasets import load_dataset
    kw.setdefault("revision", hf_revision(repo))
    return load_dataset(repo, *args, **kw)


def hf_file(repo: str, filename: str, **kw):
    """`hf_hub_download` at the pinned revision, from a DATASET repo.

    `repo_type` defaults to "dataset" because the revision is looked up in
    `HF_DATASET_REVISIONS`: a model repo would find no pin and raise anyway. Without
    this default, omitting `repo_type` gives a bare 404 that reads like a bad filename.
    """
    from huggingface_hub import hf_hub_download
    kw.setdefault("revision", hf_revision(repo))
    kw.setdefault("repo_type", "dataset")
    return hf_hub_download(repo, filename, **kw)


# ════════════════════════════════════════════════════════════ dataset
NEMOTRON_REPO = "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1"


def build_dataset(limit=None):
    """Nemotron agentic IPI -> single-turn prompts. Returns list of sample dicts."""
    ds = hf_dataset(NEMOTRON_REPO, split="train")
    out, skipped = [], 0
    for r in ds:
        inj = r["injection"]
        probe = inj["injection_text"].strip()[:60]
        rec, field_name = None, None
        stack = [r["environment"]]
        while stack and rec is None:
            node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(v, str) and probe and probe in v:
                        rec, field_name = dict(node), k
                        break
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        if rec is None or not r["required_tools"]:
            skipped += 1
            continue
        # NOTE: never stamp bookkeeping keys into `rec` -- an earlier version added
        # "__vector__": "<field with the injection>" and it leaked into every payload.
        clean = json.loads(json.dumps(rec))
        clean[field_name] = clean[field_name].replace(inj["injection_text"], " ")
        msgs = r["responses_create_params"]["input"]
        out.append({
            "id": f"nem-{r['id']}",
            "domain": r["domain"], "attack_category": r["attack_category"],
            "system": next((m["content"] for m in msgs if m["role"] == "system"), ""),
            "user": next((m["content"] for m in msgs if m["role"] == "user"), ""),
            "tools": [t if "function" in t else
                      {"type": "function",
                       "function": {"name": t["name"],
                                    "description": t.get("description", ""),
                                    "parameters": t.get("parameters",
                                                        {"type": "object",
                                                         "properties": {}})}}
                      for t in r["responses_create_params"]["tools"]],
            "retrieval_tool": r["required_tools"][0],
            "payload": rec, "payload_clean": clean,
            "injection_text": inj["injection_text"], "injection_field": field_name,
            "target_tool": inj["target_tool"], "target_args": inj.get("target_args", {}),
            "goal": inj.get("goal", ""),
        })
        if limit and len(out) >= limit:
            break
    print(f"[data] {len(out)} samples ({skipped} skipped: injection not locatable)")
    return out



def template_key(s):
    """Attacker-template identity used for template-disjoint splitting.

    Some payloads carry LITERAL backslash-n/-t two-character escapes rather than real
    whitespace. Without folding those first, "[priority message]\\nhi assistant" and
    "[priority message] hi assistant" look like different templates and the same attack
    family lands in both probe-train and test.
    """
    t = re.sub(r"\\[ntr]", " ", s.get("injection_text", ""))
    return re.sub(r"\s+", " ", t).strip().lower()[:60]


PARAM_MANIFEST = "runs/param_abuse_split_manifest.json"


CORPUS_MANIFEST = "runs/corpus_manifest.json"


def corpus_manifest(root=None, write=False):
    """Provenance record for the Nemotron corpus: pinned revision + content hashes.

    This is the artifact that makes the Open Science "pinned revision" claim checkable
    by someone who has only the repo: it names the upstream revision, the number of
    samples the loader keeps, a per-sample content hash and one corpus-level hash over
    all of them, and the deterministic split the fits draw from. A reproduction that
    lands a different `corpus_sha` has a different corpus, whatever its sample count.

    Rebuild with:  .venv/bin/python -c "import xpia_defense as X; X.corpus_manifest(write=True)"
    """
    samples = build_dataset()
    fields = ("id", "domain", "attack_category", "system", "user", "tools",
              "retrieval_tool", "payload", "payload_clean", "injection_text",
              "injection_field", "target_tool", "target_args", "goal")
    per = {s["id"]: hashlib.sha256(
        json.dumps({k: s[k] for k in fields}, sort_keys=True,
                   ensure_ascii=False).encode()).hexdigest() for s in samples}
    splits = build_splits(samples, verbose=False)
    man = {
        "dataset": NEMOTRON_REPO,
        "revision": hf_revision(NEMOTRON_REPO),
        "revision_resolved": "2026-09-02 (== upstream `main` on that date; the same "
                             "revision the cached copy every result was computed on "
                             "carries, fetched 2026-07-31)",
        "n_samples": len(samples),
        "corpus_sha": hashlib.sha256(
            "".join(f"{k}:{per[k]}" for k in sorted(per)).encode()).hexdigest(),
        "split_seed": 0,
        "split_ids": {k: [samples[i]["id"] for i in v] for k, v in splits.items()},
        "dataset_revisions": dict(HF_DATASET_REVISIONS),
        "model_revisions": dict(MODEL_REVISIONS),
        "model_revisions_enforced": False,   # documented, not yet pinned at load time
        "per_sample_sha": per,
    }
    # The deployed direction's own 24 fitting samples. NOT a prefix of the probe split:
    # `paired_samples` first keeps only base samples carrying BOTH action types (195 of
    # 250) and takes the first 24 of THOSE. The difference from a naive prefix is small
    # but real, and it is stated exactly because an earlier version of this comment
    # overstated it: the two agree on 23 of 24 and are identical for their first 15
    # entries -- the naive prefix carries `nem-88`, which the pairing filter drops, and
    # stops at `nem-122`, where the real selector goes on to `nem-128`. Recorded here
    # because §2.2 of the paper says an attacker replaying the recipe lands on exactly
    # these ids, and that claim should be checkable from the artifact, not taken on trust.
    man["direction_fit_selector"] = ("tools/controls/override_slope_experiment."
                                     "paired_samples(split='probe', n=24)")
    try:
        _tools = f"{root or ROOT}/tools/controls"
        if _tools not in sys.path:                # never grow sys.path on repeat calls
            sys.path.insert(0, _tools)
        from override_slope_experiment import paired_samples  # noqa: E402
        man["direction_fit_ids"] = [s["id"] for s, _ in paired_samples("probe", 24)]
    # `paired_samples` raises SystemExit (a BaseException) when the parameter-abuse
    # corpus has not been built -- which is exactly the state a third party reproducing
    # this repo is in, since that corpus is a generated artifact. Catching only
    # `Exception` here let the manifest builder abort the whole process on a fresh
    # clone; caught by the 2026-09-02 adversarial review, which reproduced it by
    # renaming the file.
    except (Exception, SystemExit) as e:
        man["direction_fit_ids"] = None
        man["direction_fit_note"] = f"unresolved ({type(e).__name__}); build the " \
                                    f"parameter-abuse corpus and rewrite this manifest"
    if write:
        p = f"{root or ROOT}/{CORPUS_MANIFEST}"
        blob = json.dumps(man, indent=1)          # build, then write, then replace
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            with open(p + ".tmp", "w") as fh:
                fh.write(blob)
            with open(p + ".tmp") as fh:           # not written until it parses
                json.load(fh)
            os.replace(p + ".tmp", p)
        except BaseException:
            if os.path.exists(p + ".tmp"):         # never leave a half-written .tmp
                os.remove(p + ".tmp")
            raise
        print(f"[corpus] wrote {p}: {man['n_samples']} samples, "
              f"corpus_sha {man['corpus_sha'][:16]}, revision {man['revision'][:12]}")
    return man


def param_corpus_path(split, tset="fit", root=None):
    """Path to a parameter-abuse corpus file. `tset` selects the ATTACKER TEMPLATE SET."""
    tag = split if tset in (None, "", "fit") else f"{split}-{tset}"
    return f"{root or ROOT}/runs/param_abuse_dataset.{tag}.json"


def param_split_manifest(root=None, split=None):
    """The CANONICAL sample ids per parameter-abuse split, shared by every template set.

    WHY THIS EXISTS. The builder drops a base sample when the hijack cannot be placed in the
    call the unattacked model actually made -- and whether it can be placed depends on the
    ATTACKER TEMPLATE. So each template set produced a slightly different survivor list:
    dev/fit has 69 samples, each held-out set has 68, and they share only 66. Comparing
    "fit templates" against "held-out templates" across different sample sets confounds the
    template effect with a 5-sample composition change, which is exactly the thing a split
    is supposed to hold constant.

    The manifest is the INTERSECTION over every template set built for that split, in a
    fixed order. Every corpus load and every scored table restricts to it, so a template
    comparison is paired by construction. Rebuild with:

        python tools/controls/build_param_abuse_dataset.py --manifest <split>
    """
    p = f"{root or ROOT}/{PARAM_MANIFEST}"
    if not os.path.exists(p):
        return {}
    man = json.load(open(p))["splits"]
    return man.get(split, []) if split else man


def build_param_manifest(split, root=None, verbose=True):
    """Intersect the sample ids of every template set present on disk for `split`."""
    import glob as _glob
    root = root or ROOT
    pat = f"{root}/runs/param_abuse_dataset.{split}*.json"
    files = [f for f in sorted(_glob.glob(pat)) if ".shard" not in os.path.basename(f)]
    if not files:
        raise SystemExit(f"no merged corpora matching {pat} -- merge the shards first")
    per = {}
    for f in files:
        tag = os.path.basename(f)[len("param_abuse_dataset."):-len(".json")]
        per[tag] = [s["id"] for s in json.load(open(f))["samples"]]
    common = set.intersection(*(set(v) for v in per.values()))
    # canonical ORDER comes from the fit set (or the first file), so `_meta.sample_ids`
    # is reproducible and two runs of different template sets align row for row
    order = per.get(split, per[sorted(per)[0]])
    ids = [i for i in order if i in common]
    if verbose:
        for tag, v in sorted(per.items()):
            print(f"  {tag:<28} n={len(v):4d}  dropped by manifest: {len(set(v) - common)}")
        print(f"  MANIFEST {split}: {len(ids)} samples common to {len(per)} template sets")
    return ids, per


def build_splits(all_samples, n_test=96, n_probe=250, n_eval=24, seed=0, verbose=True):
    """-> {"probe": idx, "dev": idx, "test": idx}, disjoint by ATTACKER TEMPLATE.

    Factored out of main() so anything needing the splits (tools/controls/*) uses the SAME
    partition rather than a copy that can silently drift. Splitting on sample INDEX does
    not separate attacker STRINGS: 1271 samples share only ~121 distinct prefixes, so an
    index split put identical attacker text in probe-train and test.

    The corpus is brutally template-skewed -- 33 templates, one covering 987/1271 (78%) --
    so draining groups in shuffled order starves later splits. Assign whole groups
    largest-first to whichever split has the biggest deficit, then subsample.
    """
    rng = np.random.default_rng(seed)
    groups = {}
    for i, s in enumerate(all_samples):
        groups.setdefault(template_key(s), []).append(i)
    # Template ASSIGNMENT uses a fixed dev reserve, never n_eval, so growing the dev sample
    # cannot slide which templates are held out for test.
    DEV_RESERVE = 96
    quota = {"test": n_test, "probe": n_probe, "dev": DEV_RESERVE}
    bins = {k: [] for k in quota}
    for k in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        tgt = max(quota, key=lambda b: (quota[b] - len(bins[b]), b))
        bins[tgt].extend(groups[k])
    need = {"test": n_test, "probe": n_probe, "dev": n_eval}
    for b in bins:
        idx = np.array(sorted(bins[b]), dtype=int)
        if len(idx) > need[b]:
            idx = idx[rng.permutation(len(idx))[: need[b]]]
        bins[b] = np.sort(idx)
        if len(bins[b]) < need[b] and verbose:
            print(f"[split] WARNING {b} has {len(bins[b])} < requested {need[b]} "
                  f"(template-disjointness binds)", flush=True)
    for a in bins:
        for c in bins:
            if a < c:
                assert not (set(bins[a].tolist()) & set(bins[c].tolist())), f"{a}/{c} overlap"
    tset = {template_key(all_samples[i]) for i in bins["test"]}
    pset = {template_key(all_samples[i]) for i in bins["probe"]}
    assert not (tset & pset), f"attacker template leak: {sorted(tset & pset)[:3]}"
    if verbose:
        print(f"[split] probe={len(bins['probe'])} dev={len(bins['dev'])} "
              f"test={len(bins['test'])} (disjoint by ATTACKER TEMPLATE, {len(groups)} "
              f"templates, seed {seed}; test templates {len(tset)}, probe templates "
              f"{len(pset)}, 0 shared)", flush=True)
    return bins



def load_tool_json(tok, n, max_tokens=1024, split="probe"):
    """Probe content that is IN DISTRIBUTION: the JSON tool payloads this probe is applied to.

    THE MISMATCH THIS FIXES. `load_corpus` is 25% C4 + 75% dolma3 -- plain PROSE -- and every
    span the probe is ever asked to score is a JSON tool payload. This project has already
    been burned by exactly that gap: the role-confusion measurement came back INVERTED, and
    the controls showed the cause was a prose-vs-JSON text-type effect rather than role
    confusion (`FINDINGS.md`, README "Why the role-confusion test inverted"). A probe whose
    reference class is out of distribution cannot be trusted on the class it is used on --
    `CLAUDE.md`'s own corollary lists "the reference class was out of distribution (JSON
    scored by a prose-trained probe)" as a way a null has already been wrong here.

    Content is `json.dumps(payload_clean)` -- the UNPOISONED record, so no injected text
    enters the probe's training data under any role header.

    SPLIT-DISJOINT: drawn from the `probe` split only, which `build_splits` makes
    attacker-template-disjoint from dev and test.
    """
    allx = build_dataset()
    bins = build_splits(allx, verbose=False)
    out = []
    for i in bins.get(split, []):
        s = allx[i]
        pc = s.get("payload_clean")
        if not pc:
            continue
        t = json.dumps(pc, ensure_ascii=False)
        ids = tok(t, add_special_tokens=False)["input_ids"]
        out.append(tok.decode(ids[:max_tokens]) if len(ids) > max_tokens else t)
        if len(out) >= n:
            break
    return out


def load_corpus(tok, n, max_tokens=1024, kind="paper"):
    """Probe-training content.

    kind=paper      25% C4 + 75% dolma3, the paper's mix (02-train-role-probes.ipynb cell 8)
    kind=tool_json  JSON tool payloads from the probe split -- IN DISTRIBUTION, see
                    load_tool_json
    kind=mixed      half each, so the probe sees both text types under every role header

    DEVIATION LOGGING is mandatory here, not decorative: `CLAUDE.md` requires that a corpus is
    never silently substituted for the one the replicated method specifies. Anything other
    than `paper` prints a deviation line at runtime and is recorded in probe_report.json.
    """
    if kind == "tool_json":
        print("[probe][DEVIATION vs paper] corpus = JSON tool payloads (probe split), NOT "
              "the paper's 25% C4 + 75% dolma3 prose mix. Rationale: the probe is applied to "
              "JSON tool payloads and a prose-trained probe already produced one inverted "
              "result here.", flush=True)
        got = load_tool_json(tok, n, max_tokens)
        if len(got) < n:
            print(f"[probe] only {len(got)} tool payloads available for n={n}; "
                  f"NOT topping up from prose (that would silently re-introduce the mismatch)",
                  flush=True)
        return got
    if kind == "mixed":
        print("[probe][DEVIATION vs paper] corpus = 50% paper prose mix + 50% JSON tool "
              "payloads (probe split).", flush=True)
        half = n // 2
        tj = load_tool_json(tok, half, max_tokens)
        return (load_corpus(tok, n - len(tj), max_tokens, kind="paper") + tj)[:n]
    # PAPER-EXACT sampling (02-train-role-probes.ipynb cell 8, `load_raw_ds`): C4 en
    # validation + dolma3 pinned at revision 3a8349c, both streamed through
    # shuffle(seed=123, buffer_size=50_000); int(n*.25) C4 texts then int(n*.75) dolma3
    # texts, NO content filters, NO whitespace collapsing. Note int truncation makes
    # n=250 yield 249 texts, exactly as the paper's cell does. Texts are round-tripped
    # through the tokenizer at max_tokens even when short, matching the paper's
    # batch_decode of every truncated encoding (cell 11, `build_sample_seqs`).
    # Before 2026-08-25 this path read an unshuffled C4 TRAIN shard with extra content
    # filters, and dolma3 (unpinned) failed with ValueError, so shipped probes were
    # C4-only -- the OPEN row in the deviations table (README).
    def take(ds, k, out):
        it = iter(ds)
        for _ in range(k):
            ex = next(it, None)
            if ex is None:
                break
            ids = tok(ex["text"], add_special_tokens=False,
                      truncation=True, max_length=max_tokens)["input_ids"]
            out.append(tok.decode(ids))

    out: list[str] = []
    take(hf_dataset("allenai/c4", "en", split="validation", streaming=True)
         .shuffle(seed=123, buffer_size=50_000), int(n * .25), out)
    try:
        take(hf_dataset("allenai/dolma3_mix-150B-1025", split="train", streaming=True)
             .shuffle(seed=123, buffer_size=50_000), int(n * .75), out)
    except Exception as e:
        print(f"[probe][DEVIATION vs paper] dolma3 unavailable ({type(e).__name__}); "
              f"topping up from C4 validation -- corpus is NOT the paper's mix", flush=True)
        take(hf_dataset("allenai/c4", "en", split="validation", streaming=True)
             .shuffle(seed=124, buffer_size=50_000), n - len(out), out)
    return out


def load_c4(tok, n, max_tokens=1024):
    p = hf_file("allenai/c4", "en/c4-train.00000-of-01024.json.gz")
    out = []
    with gzip.open(p, "rt") as fh:
        for line in fh:
            t = " ".join(json.loads(line)["text"].split())
            if len(t.split()) < 25 or any(k in t.lower()
                                          for k in ("user:", "assistant:", "system:")):
                continue
            ids = tok(t, add_special_tokens=False)["input_ids"]
            out.append(tok.decode(ids[:max_tokens]) if len(ids) > max_tokens else t)
            if len(out) >= n:
                break
    return out
