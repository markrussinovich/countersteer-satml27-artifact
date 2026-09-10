#!/usr/bin/env python
"""SecAlign-style LoRA DPO fine-tune: prefer ignoring injected instructions.

THE BASELINE THIS BUILDS. SecAlign (arXiv:2410.05451) / Meta-SecAlign (arXiv:2507.02735)
preference-optimize the model so that, on a prompt carrying an injected instruction, the
response that ignores the injection is preferred over the response that follows it. This
trains that defense on the SAME base model our activation steering runs on, from the pairs
built by tools/controls/build_secalign_pairs.py (probe split + synthetic framing grid ONLY
-- the fairness constraint lives in the pair builder, not here).

IMPLEMENTATION NOTES, each a deliberate choice:
  * DPO loss is HAND-ROLLED (~40 lines) rather than pulled from trl: the in-job stack pins
    transformers 5.14 / torch 2.7 and trl's tokenization path re-applies chat templates and
    appends EOS -- both wrong for pre-rendered harmony text whose completions must end at
    <|call|>. The loss itself is elementary: -logsigmoid(beta * ((pol_c - ref_c) -
    (pol_r - ref_r))), reference logps from the SAME model with the adapter disabled.
  * LoRA targets ATTENTION ONLY (q/k/v/o_proj). GptOss MoE expert weights are fused 3-D
    parameters (mlp.experts.gate_up_proj/down_proj), not nn.Linear, so peft cannot wrap
    them; the router IS an nn.Linear but steering the router distribution is exactly the
    kind of capability damage the guard exists to catch. Attention-only is disclosed
    wherever this baseline is reported.
  * LoRA B is zero-initialised, so policy == reference at step 0 and the initial DPO loss
    must equal ln 2 = 0.6931. That identity is ASSERTED (tolerance 0.02) -- it is a free
    end-to-end test of the logp computation, the masking, and the adapter-disable path.
  * Multi-GPU is manual data-parallel: each rank a full replica, gradients of the (tiny)
    LoRA parameters all-reduced at step boundaries. No DDP wrapper -- it fights both
    gradient checkpointing and peft's disable_adapter context, and the LoRA gradient
    volume (~35M params) makes the all-reduce trivial.

Usage (single GPU smoke):
  python tools/controls/train_secalign_dpo.py --pairs runs/secalign_pairs.train.jsonl \
      --eval-pairs runs/secalign_pairs.eval.jsonl --out outputs/secalign_smoke \
      --max-steps 100 --max-pairs 500 --gen-samples 3
Full train (8 GPU):
  torchrun --nproc_per_node 8 tools/controls/train_secalign_dpo.py --pairs ... --out ... \
      --epochs 1 --merge
"""
import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


def log(rank, *a):
    if rank == 0:
        print(f"[dpo {time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_pairs(path, tok, max_tokens, limit=0):
    """-> list of dicts with prompt_ids, chosen_ids, rejected_ids (no padding here)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(r["chosen"], add_special_tokens=False)["input_ids"]
            j = tok(r["rejected"], add_special_tokens=False)["input_ids"]
            if len(p) + max(len(c), len(j)) > max_tokens:
                continue
            rows.append({"id": r["id"], "p": p, "c": c, "r": j})
            if limit and len(rows) >= limit:
                break
    return rows


def batch_logps(model, rows, device, pad_id, no_grad=False, return_tokens=False):
    """Sum of completion-token logps for the 2*len(rows) sequences (chosen then rejected).

    Returns (logp_chosen, logp_rejected) tensors of shape [len(rows)]. Right-padding;
    prompt tokens and padding are masked out of the sum.

    MEMORY (review M3): the per-token logp is computed with F.cross_entropy over sequence
    CHUNKS, never materializing the full [B, L, 201k-vocab] log-softmax in fp32 -- that
    tensor plus its saved-for-backward copy was a ~13 GB transient and OOM'd at
    micro_pairs=2 x max length on an 80 GB card.
    """
    seqs, masks = [], []
    for r in rows:
        for comp in (r["c"], r["r"]):
            seqs.append(r["p"] + comp)
            masks.append([0] * len(r["p"]) + [1] * len(comp))
    M = max(len(s) for s in seqs)
    ids = torch.tensor([s + [pad_id] * (M - len(s)) for s in seqs], device=device)
    cm = torch.tensor([m + [0] * (M - len(m)) for m in masks], device=device)
    attn = torch.tensor([[1] * len(s) + [0] * (M - len(s)) for s in seqs], device=device)
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        # next-token prediction: logits at t score token t+1
        logits = out.logits[:, :-1]
        tgt = ids[:, 1:]
        V = logits.shape[-1]
        lps = []
        for k in range(0, logits.shape[1], 512):
            lg = logits[:, k:k + 512].float()
            ce = F.cross_entropy(lg.reshape(-1, V), tgt[:, k:k + 512].reshape(-1),
                                 reduction="none")
            lps.append(-ce.view(lg.shape[0], -1))
        tok_lp = torch.cat(lps, dim=1) * cm[:, 1:]
        per_seq = tok_lp.sum(-1)
    if return_tokens:
        return per_seq[0::2], per_seq[1::2], cm[0::2, 1:].sum(-1), cm[1::2, 1:].sum(-1)
    return per_seq[0::2], per_seq[1::2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--eval-pairs", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epoch count")
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    ap.add_argument("--micro-pairs", type=int, default=1,
                    help="preference pairs per rank per micro-step (2 sequences each)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj",
                    help="attention-only by default; GptOss fused MoE experts are not "
                         "nn.Linear and cannot take LoRA (disclosed limitation)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=10,
                    help="linear warmup steps; LR is CONSTANT after warmup (disclosed "
                         "deviation: SecAlign used a decaying schedule)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-pairs-n", type=int, default=128)
    ap.add_argument("--ckpt-every", type=int, default=100,
                    help="save the adapter every N steps (0 = off); a late NaN must not "
                         "discard the run")
    ap.add_argument("--merge", action="store_true",
                    help="merge the adapter into bf16 weights and save to OUT/merged")
    ap.add_argument("--gen-samples", type=int, default=0,
                    help="after training, greedy-generate N eval prompts and print them")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    device = f"cuda:{local}"
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    rows = load_pairs(args.pairs, tok, args.max_tokens, args.max_pairs)
    log(rank, f"{len(rows)} training pairs from {args.pairs}")
    ev = load_pairs(args.eval_pairs, tok, args.max_tokens) if args.eval_pairs else []
    if len(ev) > args.eval_pairs_n:
        # STRIDE, don't take the first N: eval rows are grouped by sid, so first-N would
        # cover ~3 sids (review m10)
        ev = ev[:: max(1, len(ev) // args.eval_pairs_n)][: args.eval_pairs_n]
    log(rank, f"{len(ev)} eval pairs")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device)
    model.config.use_cache = False
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout,
                      target_modules=args.target_modules.split(","),
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    if rank == 0:
        model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable parameters -- LoRA target modules matched nothing"
    # peft keeps adapters fp32 on a bf16 base (autocast_adapter_dtype=True, verified on
    # peft 0.20). GUARD it: with bf16 adapters, lr=5e-6 updates sit below the bf16 ulp of
    # lora_A's ~0.1-magnitude entries and silently vanish -- training would no-op.
    for p in trainable:
        if p.dtype != torch.float32:
            p.data = p.data.float()
    assert all(p.dtype == torch.float32 for p in trainable), "adapter params not fp32"
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    def dpo_loss(pol_c, pol_r, ref_c, ref_r):
        margin = args.beta * ((pol_c - ref_c) - (pol_r - ref_r))
        return -F.logsigmoid(margin).mean(), margin

    ref_cache = {}          # eval rows are fixed; their ref logps never change (review m9)

    def eval_dpo(rows_):
        """Mean DPO loss + preference accuracy over rows_, policy vs adapter-disabled ref.

        Rank-strided (each rank scores a disjoint slice, SUM-all-reduced) and ref-cached
        -- evaluating identically on all 8 ranks was an 8x waste (review m9).
        """
        model.eval()
        mine = rows_[rank::world]
        tot, acc, n = 0.0, 0.0, 0
        for k in range(0, len(mine), args.micro_pairs):
            chunk = mine[k:k + args.micro_pairs]
            key = tuple(r["id"] for r in chunk)
            if key not in ref_cache:
                with model.disable_adapter():
                    ref_cache[key] = batch_logps(model, chunk, device, pad_id, no_grad=True)
            ref_c, ref_r = ref_cache[key]
            pol_c, pol_r = batch_logps(model, chunk, device, pad_id, no_grad=True)
            loss, margin = dpo_loss(pol_c, pol_r, ref_c, ref_r)
            tot += loss.item() * len(chunk)
            acc += (margin > 0).float().sum().item()
            n += len(chunk)
        if world > 1:
            t = torch.tensor([tot, acc, float(n)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            tot, acc, n = t[0].item(), t[1].item(), int(t[2].item())
        model.train()
        return tot / max(1, n), acc / max(1, n)

    # STEP-0 IDENTITY CHECK: adapter B is zero-init, so policy == reference and the loss
    # must be ln 2. NOTE what this does and does not test (review m7): it exercises
    # determinism and the adapter-disable path, but any deterministic batch_logps --
    # however wrong its masking -- gives margin==0 at zero-init. The masking itself is
    # tested by the batched-vs-unbatched cross-check below.
    probe_rows = (ev or rows)[: 8]
    l0, a0 = eval_dpo(probe_rows)
    log(rank, f"step-0 identity: loss={l0:.4f} (must be ln2={math.log(2):.4f}) acc={a0:.3f}")
    assert abs(l0 - math.log(2)) < 0.02, \
        f"step-0 DPO loss {l0:.4f} != ln2 -- logp computation or adapter-disable is broken"
    # BATCHED-vs-UNBATCHED SELF-TEST (review m7): the padded 2-pair batch must agree with
    # each pair scored alone -- this is what actually catches a masking/padding bug.
    two = probe_rows[:2]
    if len(two) == 2:
        bc, br = batch_logps(model, two, device, pad_id, no_grad=True)
        for i, r in enumerate(two):
            sc, sr = batch_logps(model, [r], device, pad_id, no_grad=True)
            assert (abs(sc.item() - bc[i].item()) < 0.5
                    and abs(sr.item() - br[i].item()) < 0.5), \
                f"batched vs unbatched logp mismatch on pair {i}: " \
                f"{sc.item():.3f}/{bc[i].item():.3f} {sr.item():.3f}/{br[i].item():.3f}"
        log(rank, "batched-vs-unbatched logp self-test passed")
    # DEGENERATE-CONTRAST DIAGNOSTIC (review M1): a large per-token ref-logp gap between
    # chosen and rejected means the margin can be won by crushing already-improbable text.
    with model.disable_adapter():
        rc0, rr0, nc0, nr0 = batch_logps(model, probe_rows, device, pad_id,
                                         no_grad=True, return_tokens=True)
    log(rank, f"step-0 ref per-token logp: chosen={(rc0 / nc0).mean().item():.3f} "
              f"rejected={(rr0 / nr0).mean().item():.3f} "
              f"(gap >> 1 nat = degenerate contrast risk)")
    # CROSS-RANK PARAM IDENTITY (review m8): manual DP assumes identical init everywhere.
    if world > 1:
        ck = torch.stack([p.detach().float().sum() for p in trainable]).sum()
        cks = [torch.zeros_like(ck) for _ in range(world)]
        dist.all_gather(cks, ck)
        assert all(torch.allclose(c, cks[0]) for c in cks), \
            f"rank {rank}: trainable-param checksum differs across ranks: {cks}"
        log(rank, "cross-rank param checksum identical")
    # PRE-FLIGHT at the memory ceiling (review M3): one forward+backward on the LONGEST
    # pair before the loop, so an OOM costs seconds, not the run.
    longest = max(rows, key=lambda r: len(r["p"]) + max(len(r["c"]), len(r["r"])))
    pc_, pr_ = batch_logps(model, [longest] * args.micro_pairs, device, pad_id)
    (-F.logsigmoid(args.beta * (pc_ - pr_)).mean()).backward()
    opt.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    log(rank, f"pre-flight fwd+bwd at max length "
              f"({len(longest['p']) + max(len(longest['c']), len(longest['r']))} tokens, "
              f"micro_pairs={args.micro_pairs}) OK; "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    # deterministic shuffle, shared across ranks; each rank strides
    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(rows), generator=g).tolist()
    micro_per_step = world * args.micro_pairs * args.accum
    steps_per_epoch = len(order) // micro_per_step
    total_steps = args.max_steps or max(1, int(steps_per_epoch * args.epochs))
    log(rank, f"world={world} micro_pairs={args.micro_pairs} accum={args.accum} -> "
              f"{micro_per_step} pairs/step, {steps_per_epoch} steps/epoch, "
              f"training {total_steps} steps")

    os.makedirs(args.out, exist_ok=True)
    logf = open(f"{args.out}/train_log.jsonl", "a") if rank == 0 else None
    if rank == 0:
        # serialise FULLY before writing -- json.dump streaming into the handle on a
        # non-serialisable value leaves a truncated artifact (CLAUDE.md, paid for twice)
        cfg_payload = json.dumps(vars(args) | {"world": world, "n_pairs": len(rows),
                                               "steps_per_epoch": steps_per_epoch,
                                               "total_steps": total_steps},
                                 indent=2, default=str)
        with open(f"{args.out}/config.json.tmp", "w") as fh:
            fh.write(cfg_payload)
        os.replace(f"{args.out}/config.json.tmp", f"{args.out}/config.json")
        json.load(open(f"{args.out}/config.json"))

    step, cursor, t0 = 0, 0, time.time()
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        stats = torch.zeros(4, device=device)      # loss_sum, margin_sum, acc_sum, n
        for _ in range(args.accum):
            idx = [order[(cursor + i) % len(order)] for i in
                   range(rank * args.micro_pairs, (rank + 1) * args.micro_pairs)]
            cursor += world * args.micro_pairs
            chunk = [rows[i] for i in idx]
            with model.disable_adapter():
                ref_c, ref_r = batch_logps(model, chunk, device, pad_id, no_grad=True)
            pol_c, pol_r = batch_logps(model, chunk, device, pad_id)
            loss, margin = dpo_loss(pol_c, pol_r, ref_c, ref_r)
            assert torch.isfinite(loss), f"non-finite DPO loss at step {step}"
            (loss / args.accum).backward()
            stats += torch.tensor([loss.item() * len(chunk),
                                   margin.sum().item(),
                                   (margin > 0).float().sum().item(),
                                   float(len(chunk))], device=device)
        if world > 1:
            for p in trainable:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        for gparam in opt.param_groups:
            gparam["lr"] = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1
        if rank == 0:
            n = stats[3].item()
            rec = {"step": step, "loss": stats[0].item() / n,
                   "margin": stats[1].item() / n, "acc": stats[2].item() / n,
                   "lr": opt.param_groups[0]["lr"],
                   "elapsed_s": round(time.time() - t0, 1)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if step % 5 == 0 or step == 1:
                log(rank, f"step {step}/{total_steps} loss={rec['loss']:.4f} "
                          f"margin={rec['margin']:.3f} acc={rec['acc']:.3f}")
        if rank == 0 and args.ckpt_every and step % args.ckpt_every == 0:
            model.save_pretrained(f"{args.out}/adapter_ckpt")   # MBs; step-270 NaN insurance
        if ev and (step % args.eval_every == 0 or step == total_steps):
            el, ea = eval_dpo(ev)
            log(rank, f"eval @ step {step}: loss={el:.4f} acc={ea:.3f}")
            if rank == 0:
                logf.write(json.dumps({"step": step, "eval_loss": el,
                                       "eval_acc": ea}) + "\n")
                logf.flush()

    # Barrier BEFORE the rank-0 merge/save (review M2): ranks 1..N-1 idling in a barrier
    # while rank 0 writes ~42 GB to blob would trip the 10-minute NCCL watchdog and tear
    # the job down mid-write of the final artifact.
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        model.save_pretrained(f"{args.out}/adapter")
        log(rank, f"adapter saved to {args.out}/adapter")
        if args.merge:
            merged = model.merge_and_unload()
            merged.config.use_cache = True      # the eval harness must not inherit cache-off
            # save_original_format=False is LOAD-BEARING (2026-08-30): gpt-oss ships mxfp4
            # weights and transformers 5.14's default reverse-maps the checkpoint to the
            # ORIGINAL format on save -- with a dequantized-to-bf16 model that silently
            # DROPS every MoE expert weight tensor (saved 5.2 GB of a 42 GB model; only
            # the expert *biases* survived). Verified both ways on a layer-0 partial save.
            merged.save_pretrained(f"{args.out}/merged", safe_serialization=True,
                                   max_shard_size="4GB", save_original_format=False)
            tok.save_pretrained(f"{args.out}/merged")
            # POST-SAVE COMPLETENESS GATE: existence is not success (CLAUDE.md).
            idx = json.load(open(f"{args.out}/merged/model.safetensors.index.json"))
            wm = idx["weight_map"]
            n_exp = sum(k.endswith(("experts.gate_up_proj", "experts.down_proj"))
                        for k in wm)
            total = idx.get("metadata", {}).get("total_size", 0)
            assert n_exp >= 2 * model.config.num_hidden_layers and total > 35e9, \
                f"merged save incomplete: {n_exp} expert weight tensors, {total/1e9:.1f} GB"
            log(rank, f"merged bf16 model saved to {args.out}/merged "
                      f"({total/1e9:.1f} GB, {n_exp} expert weight tensors)")
            model = merged
        if args.gen_samples and ev:
            model.eval()
            model.config.use_cache = True
            for r in ev[: args.gen_samples]:
                ids = torch.tensor([r["p"]], device=device)
                with torch.no_grad():
                    o = model.generate(input_ids=ids, max_new_tokens=512, do_sample=False,
                                       pad_token_id=pad_id)
                comp = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=False)
                calls = comp.count("to=functions.")
                log(rank, f"--- sample {r['id']} (tool calls: {calls}) ---\n{comp[:900]}")
    log(rank, "done")


if __name__ == "__main__":
    main()
