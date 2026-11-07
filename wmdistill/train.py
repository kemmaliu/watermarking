"""Student training: (1 - lam) * CE + lam * L_stat, with the watermark key available.

L_stat works on the standardized per-token score z_t(w) = (s_t(w) - mu0) / sigma0 so that the
test statistic is T_std(Y) = sum_t z_t(Y_t). For each training sequence (a teacher watermarked
sample, key known) we form the student's differentiable expected per-token score

    E_seq = mean_t  sum_w p_student(w | y_<t) z_t(w)

and compare it with the realized per-token score of the teacher's tokens R_seq = mean_t z_t(Y_t):

    mode "match": L_stat = mean_seq (E_seq - R_seq)^2      (match T(Y)/n of the teacher text)
    mode "max":   L_stat = -mean_seq E_seq                 (push the statistic up)
"""
import json
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .watermarks import Watermark


def collate(samples, pad_id, train_on_prompt=False):
    seqs = [s["prompt"] + s["completion"] for s in samples]
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros(len(seqs), L, dtype=torch.long)
    loss_mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for b, (s, q) in enumerate(zip(samples, seqs)):
        ids[b, : len(q)] = torch.tensor(q)
        attn[b, : len(q)] = 1
        loss_mask[b, (0 if train_on_prompt else len(s["prompt"])): len(q)] = True
    loss_mask[:, 0] = False  # nothing predicts position 0
    prompt_lens = torch.tensor([len(s["prompt"]) for s in samples])
    return ids, attn, loss_mask, prompt_lens


def _expected_and_realized(logits, seeds, targets, wm: Watermark):
    z = wm.standardized_score_vectors(seeds)  # [n, V], constant w.r.t. parameters
    expected = (torch.softmax(logits.float(), dim=-1) * z).sum(-1)
    realized = z.gather(-1, targets[:, None]).squeeze(-1)
    return expected, realized


def stat_loss(logits, seeds, targets, seq_idx, n_seq, wm: Watermark, mode="match", chunk=256):
    """logits/seeds/targets: flattened over the scored positions; seq_idx maps rows to sequences.

    Computed in checkpointed chunks so the [n, V] score matrix is never held for the whole batch.
    """
    E, R = [], []
    for s in range(0, logits.shape[0], chunk):
        e, r = checkpoint(_expected_and_realized, logits[s: s + chunk], seeds[s: s + chunk],
                          targets[s: s + chunk], wm, use_reentrant=False)
        E.append(e)
        R.append(r.detach())
    E, R = torch.cat(E), torch.cat(R)
    counts = torch.bincount(seq_idx, minlength=n_seq).clamp(min=1).float()
    E_seq = torch.zeros(n_seq, device=E.device).index_add(0, seq_idx, E) / counts
    R_seq = torch.zeros(n_seq, device=R.device).index_add(0, seq_idx, R) / counts
    if mode == "match":
        loss = ((E_seq - R_seq) ** 2).mean()
    elif mode == "max":
        loss = -E_seq.mean()
    else:
        raise ValueError(f"unknown stat loss mode {mode!r}")
    return loss, {"student_z": float(E_seq.mean()), "teacher_z": float(R_seq.mean())}


def train_student(student, tok, samples, wm: Watermark, lam: float, tcfg: dict, device, log_path=None):
    student.train()
    if tcfg.get("gradient_checkpointing"):
        student.gradient_checkpointing_enable()
    V = wm.vocab_size
    bs, accum = min(tcfg["batch_size"], len(samples)), tcfg.get("grad_accum", 1)
    steps_per_epoch = math.ceil(len(samples) / (bs * accum))
    total = tcfg.get("max_steps") or steps_per_epoch * tcfg["epochs"]
    opt = torch.optim.AdamW(student.parameters(), lr=float(tcfg["lr"]),
                            weight_decay=tcfg.get("weight_decay", 0.0))
    from transformers import get_cosine_schedule_with_warmup

    sched = get_cosine_schedule_with_warmup(opt, int(tcfg.get("warmup_frac", 0.05) * total), total)
    amp = device.type == "cuda" and tcfg.get("amp", "bf16") == "bf16"
    g = torch.Generator().manual_seed(tcfg.get("data_seed", 0))
    log_f = open(log_path, "w") if log_path else None
    step, micro, t0 = 0, 0, time.time()
    while step < total:
        order = torch.randperm(len(samples), generator=g).tolist()
        for i in range(0, len(order) - bs + 1, bs):
            ids, attn, mask, plens = (x.to(device) for x in collate(
                [samples[j] for j in order[i: i + bs]], tok.pad_token_id, tcfg.get("train_on_prompt", False)))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = student(input_ids=ids, attention_mask=attn).logits[:, :-1, :V]
            tgt_mask = mask[:, 1:]  # logits[:, j] predicts ids[:, j + 1]
            flat_logits, targets = logits[tgt_mask], ids[:, 1:][tgt_mask]
            ce = F.cross_entropy(flat_logits.float(), targets)

            # Stat loss only on completion tokens (prompt tokens were not watermarked).
            comp_mask = tgt_mask & (torch.arange(1, ids.shape[1], device=device)[None, :] >= plens[:, None])
            seeds = wm.keys.seeds_for_sequences(ids, plens)[:, 1:][comp_mask]
            seq_idx = torch.arange(ids.shape[0], device=device)[:, None].expand_as(comp_mask)[comp_mask]
            want_log = (step % tcfg.get("log_every", 10) == 0) and micro % accum == 0
            if lam > 0:
                ls, info = stat_loss(logits[comp_mask], seeds, ids[:, 1:][comp_mask], seq_idx, ids.shape[0],
                                     wm, tcfg.get("stat_loss", "match"), tcfg.get("stat_chunk", 256))
            elif want_log:
                with torch.no_grad():
                    ls, info = stat_loss(logits[comp_mask], seeds, ids[:, 1:][comp_mask], seq_idx,
                                         ids.shape[0], wm, tcfg.get("stat_loss", "match"),
                                         tcfg.get("stat_chunk", 256))
            else:
                ls, info = torch.zeros((), device=device), {}
            loss = (1 - lam) * ce + lam * ls if lam > 0 else ce
            (loss / accum).backward()
            micro += 1
            if want_log:
                rec = {"step": step, "ce": float(ce), "stat": float(ls), "loss": float(loss),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1), **info}
                print("  " + json.dumps(rec), flush=True)
                if log_f:
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), tcfg.get("max_grad_norm", 1.0))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step >= total:
                    break
    if log_f:
        log_f.close()
    if tcfg.get("gradient_checkpointing"):
        student.gradient_checkpointing_disable()
    student.eval()
    del opt
    return student
