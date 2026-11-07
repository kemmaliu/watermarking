"""Batched generation (watermarked or plain sampling) and evaluation of samples."""
import math

import numpy as np
import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

from .watermarks import Watermark


class VocabTrim(LogitsProcessor):
    """Never emit tokens outside the key's vocab (only matters if the LM head is padded)."""

    def __init__(self, vocab_size):
        self.V = vocab_size

    def __call__(self, input_ids, scores):
        if scores.shape[-1] > self.V:
            scores[:, self.V:] = -float("inf")
        return scores


class WatermarkProcessor(LogitsProcessor):
    """Replaces the next-token distribution by a point mass on the watermark decoder's choice."""

    def __init__(self, wm: Watermark, temperature: float):
        self.wm, self.temperature, self.start = wm, temperature, None

    def __call__(self, input_ids, scores):
        if self.start is None:
            self.start = input_ids.shape[1]
        seeds = self.wm.keys.seeds_at_step(input_ids, input_ids.shape[1] - self.start)
        tok = self.wm.sample(scores[:, : self.wm.vocab_size] / self.temperature, seeds)
        out = torch.full_like(scores, -float("inf"))
        return out.scatter_(1, tok[:, None], 0.0)


@torch.no_grad()
def generate(model, tok, prompts, *, vocab_size, max_new_tokens, batch_size, temperature=1.0,
             watermark=None, desc=""):
    """prompts: list of token-id lists. Returns [{"prompt": ids, "completion": ids}].

    EOS is suppressed (min_new_tokens == max_new_tokens) so every completion has the same length
    and the test has a fixed number of tokens.
    """
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        L = max(map(len, batch))
        ids = torch.tensor([[tok.pad_token_id] * (L - len(p)) + p for p in batch], device=device)
        attn = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in batch], device=device)
        procs = [VocabTrim(vocab_size)]
        if watermark is None:
            gen_cfg = GenerationConfig(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
        else:
            gen_cfg = GenerationConfig(do_sample=False)
            procs.append(WatermarkProcessor(watermark, temperature))
        gen_cfg.update(max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        res = model.generate(input_ids=ids, attention_mask=attn, generation_config=gen_cfg,
                             logits_processor=LogitsProcessorList(procs))
        for p, c in zip(batch, res[:, L:].tolist()):
            out.append({"prompt": p, "completion": c})
        print(f"  [{desc}] generated {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return out


# ---- detection --------------------------------------------------------------------------------
@torch.no_grad()
def detect(wm: Watermark, prompt, completion, device="cpu", dedup=True):
    """Test statistic and p-value for one sample.

    With context-hashed seeds a repeated (context, token) pair re-uses the same key, which breaks
    the i.i.d. null; following standard practice only the first occurrence is scored.
    """
    ids = torch.tensor([prompt + completion], device=device)
    P = len(prompt)
    seeds = wm.keys.seeds_for_sequences(ids, torch.tensor([P], device=device))[0, P:]
    toks = ids[0, P:]
    keep = torch.ones(len(toks), dtype=torch.bool)
    if dedup and wm.keys.scheme == "context":
        seen = set()
        for j, pair in enumerate(zip(seeds.tolist(), toks.tolist())):
            keep[j] = pair not in seen
            seen.add(pair)
    keep = keep.to(device)
    scores = wm.token_scores(seeds[keep], toks[keep])
    T, n = float(scores.sum()), int(keep.sum())
    z_per_tok = (T / max(n, 1) - wm.null_mean) / wm.null_std
    return {"T": T, "n": n, "pvalue": wm.pvalue(T, n), "z_per_token": z_per_tok}


@torch.no_grad()
def teacher_nll(model, samples, batch_size, vocab_size):
    """Total NLL / token count of completions under `model` (for perplexity of generated text)."""
    device = next(model.parameters()).device
    tot, cnt = 0.0, 0
    for i in range(0, len(samples), batch_size):
        batch = samples[i: i + batch_size]
        seqs = [s["prompt"] + s["completion"] for s in batch]
        L = max(map(len, seqs))
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        attn = torch.zeros(len(seqs), L, dtype=torch.long)
        lab = torch.full((len(seqs), L), -100, dtype=torch.long)
        for b, (s, q) in enumerate(zip(batch, seqs)):
            ids[b, : len(q)] = torch.tensor(q)
            attn[b, : len(q)] = 1
            lab[b, len(s["prompt"]): len(q)] = torch.tensor(s["completion"])
        ids, attn, lab = ids.to(device), attn.to(device), lab.to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1, :vocab_size].float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), lab[:, 1:], reduction="sum")
        tot += float(nll)
        cnt += int((lab[:, 1:] != -100).sum())
    return tot, cnt


def repetition_rate(completion, n=3):
    grams = [tuple(completion[i: i + n]) for i in range(len(completion) - n + 1)]
    return 1.0 - len(set(grams)) / max(len(grams), 1)


def evaluate_samples(samples, wm: Watermark, *, alpha, ppl_model=None, ppl_batch_size=8, device="cpu"):
    """Detection power + text-quality metrics for a list of generated samples."""
    det = [detect(wm, s["prompt"], s["completion"], device=device) for s in samples]
    pv = np.array([d["pvalue"] for d in det])
    m = {
        "n_samples": len(samples),
        "detect_rate": float((pv < alpha).mean()),
        "alpha": alpha,
        "median_pvalue": float(np.median(pv)),
        "mean_log10_pvalue": float(np.mean(np.log10(np.clip(pv, 1e-300, 1)))),
        "mean_z_per_token": float(np.mean([d["z_per_token"] for d in det])),
        "mean_scored_tokens": float(np.mean([d["n"] for d in det])),
        "rep3_rate": float(np.mean([repetition_rate(s["completion"]) for s in samples])),
    }
    if ppl_model is not None:
        tot, cnt = teacher_nll(ppl_model, samples, ppl_batch_size, wm.vocab_size)
        m["teacher_ppl"] = math.exp(tot / max(cnt, 1))
    return m
