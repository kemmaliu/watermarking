"""Pipeline stages. Each stage reads/writes files under <output_dir>/<exp_name>/ and skips work
whose outputs already exist, so any stage can be re-run / resumed after preemption.

    prepare    prompts -> prompts.jsonl (+ optional model download)   [needs internet]
    teacher    watermarked training data + controls (teacher wm/plain, base student)
    student    train + evaluate one (watermark, lambda) job              [SLURM array unit]
    aggregate  results.csv + plots
"""
import json
from pathlib import Path

from .common import (free, key_vocab_size, load_lm, load_tokenizer, pick_device, read_jsonl, run_dir,
                     set_seed, student_dir, sweep_jobs, write_json, write_jsonl)
from .generation import evaluate_samples, generate
from .train import train_student
from .watermarks import make_watermark


# ---- prepare ----------------------------------------------------------------------------------
def _iter_texts(pcfg):
    src = pcfg["source"]
    if src == "file":
        path = Path(pcfg["path"])
        if path.suffix == ".jsonl":
            for row in read_jsonl(path):
                yield row[pcfg.get("text_field", "text")]
        else:
            yield from (line.strip() for line in path.read_text().splitlines() if line.strip())
    elif src == "hf":
        from datasets import load_dataset

        ds = load_dataset(pcfg["dataset"], pcfg.get("subset"), split=pcfg.get("split", "train"),
                          streaming=True, trust_remote_code=False)
        for row in ds:
            yield row[pcfg.get("text_field", "text")]
    else:
        raise ValueError(f"unknown prompt source {src!r}")


def prepare(cfg, download_models=False):
    out = run_dir(cfg) / "prompts.jsonl"
    if download_models:
        from huggingface_hub import snapshot_download

        for role in ("teacher", "student"):
            print(f"downloading {cfg[role]['model']}")
            snapshot_download(cfg[role]["model"], allow_patterns=["*.json", "*.safetensors", "*.model",
                                                                   "*.txt", "*.tiktoken"])
    if out.exists():
        print(f"{out} exists, skipping")
        return
    pcfg = cfg["prompts"]
    tok = load_tokenizer(cfg["teacher"]["model"])
    P, need = pcfg["prompt_tokens"], pcfg["n_train"] + pcfg["n_eval"]
    assert P > cfg["watermark"].get("context_width", 2), "prompt must be longer than the hash context"
    rows = []
    for text in _iter_texts(pcfg):
        ids = tok(text).input_ids
        if len(ids) >= P + pcfg.get("min_extra_tokens", 0):
            rows.append(ids[:P])
            if len(rows) == need:
                break
    if len(rows) < need and not pcfg.get("allow_reuse", False):
        raise RuntimeError(f"only {len(rows)} prompts of >= {P} tokens, need {need}")
    while len(rows) < need:  # tiny smoke-test sources: cycle
        rows.append(rows[len(rows) % max(len(rows), 1)])
    write_jsonl(out, [{"split": "train" if i < pcfg["n_train"] else "eval", "ids": ids,
                       "text": tok.decode(ids)} for i, ids in enumerate(rows)])
    print(f"wrote {len(rows)} prompts to {out}")


def _prompts(cfg, split):
    return [r["ids"] for r in read_jsonl(run_dir(cfg) / "prompts.jsonl") if r["split"] == split]


def _eval_prompts(cfg):
    k = cfg["eval"].get("samples_per_prompt", 1)
    return [p for p in _prompts(cfg, "eval") for _ in range(k)]


def _add_text(tok, samples):
    for s in samples:
        s["text"] = tok.decode(s["completion"])
    return samples


def _gen_cached(path, fn):
    if Path(path).exists():
        return read_jsonl(path)
    rows = fn()
    write_jsonl(path, rows)
    return rows


# ---- teacher ----------------------------------------------------------------------------------
def teacher(cfg):
    d, dev = run_dir(cfg), pick_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"])
    V = key_vocab_size(cfg)
    tok = load_tokenizer(cfg["teacher"]["model"])
    tmodel = load_lm(cfg["teacher"]["model"], cfg["teacher"]["dtype"], dev)
    g, e = cfg["generation"], cfg["eval"]
    gen = dict(vocab_size=V, max_new_tokens=g["max_new_tokens"], batch_size=g["batch_size"],
               temperature=g.get("temperature", 1.0))
    eval_prompts = _eval_prompts(cfg)

    plain = _gen_cached(d / "teacher_plain_eval.jsonl", lambda: _add_text(
        tok, generate(tmodel, tok, eval_prompts, desc="teacher plain eval", **gen)))
    controls = {}
    for wm_name in cfg["watermarks"]:
        wm = make_watermark(wm_name, cfg, V)
        _gen_cached(d / f"teacher_{wm_name}_train.jsonl", lambda: _add_text(tok, generate(
            tmodel, tok, _prompts(cfg, "train"), watermark=wm, desc=f"teacher {wm_name} train", **gen)))
        wm_eval = _gen_cached(d / f"teacher_{wm_name}_eval.jsonl", lambda: _add_text(tok, generate(
            tmodel, tok, eval_prompts, watermark=wm, desc=f"teacher {wm_name} eval", **gen)))
        ev = dict(alpha=e["alpha"], ppl_model=tmodel, ppl_batch_size=e["ppl_batch_size"], device=dev)
        controls[wm_name] = {"teacher_watermarked": evaluate_samples(wm_eval, wm, **ev),
                             "teacher_plain": evaluate_samples(plain, wm, **ev)}

    # Base (untrained) student, plain sampling: the lambda-independent floor.
    smodel = load_lm(cfg["student"]["model"], cfg["student"]["dtype"], dev)
    base = _gen_cached(d / "student_base_eval.jsonl", lambda: _add_text(
        tok, generate(smodel, tok, eval_prompts, desc="base student eval", **gen)))
    for wm_name in cfg["watermarks"]:
        wm = make_watermark(wm_name, cfg, V)
        controls[wm_name]["student_base"] = evaluate_samples(
            base, wm, alpha=e["alpha"], ppl_model=tmodel, ppl_batch_size=e["ppl_batch_size"], device=dev)
    write_json(d / "controls.json", controls)
    print(json.dumps(controls, indent=2))
    free(tmodel, smodel)


# ---- student ----------------------------------------------------------------------------------
def student(cfg, wm_name, lam, force=False):
    out = student_dir(cfg, wm_name, lam)
    if (out / "metrics.json").exists() and not force:
        print(f"{out}/metrics.json exists, skipping (use --force to redo)")
        return
    out.mkdir(parents=True, exist_ok=True)
    d, dev = run_dir(cfg), pick_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"])  # same init/data order for every lambda
    V = key_vocab_size(cfg)
    wm = make_watermark(wm_name, cfg, V)
    tok = load_tokenizer(cfg["teacher"]["model"])
    data = read_jsonl(d / f"teacher_{wm_name}_train.jsonl")
    print(f"== student wm={wm_name} lambda={lam:.2f}: {len(data)} teacher samples -> {out}")

    smodel = load_lm(cfg["student"]["model"], cfg["student"].get("train_dtype", "float32"), dev)
    train_student(smodel, tok, data, wm, lam, cfg["train"], dev, log_path=out / "train_log.jsonl")
    if cfg["train"].get("save_model"):
        smodel.save_pretrained(out / "model")
        tok.save_pretrained(out / "model")

    g, e = cfg["generation"], cfg["eval"]
    set_seed(cfg["seed"] + 1)
    samples = _add_text(tok, generate(smodel, tok, _eval_prompts(cfg), vocab_size=V,
                                      max_new_tokens=g["max_new_tokens"], batch_size=g["batch_size"],
                                      temperature=g.get("temperature", 1.0), desc="student eval"))
    write_jsonl(out / "samples.jsonl", samples)
    free(smodel)
    tmodel = load_lm(cfg["teacher"]["model"], cfg["teacher"]["dtype"], dev) if e.get("teacher_ppl", True) else None
    metrics = {"watermark": wm_name, "lambda": lam,
               **evaluate_samples(samples, wm, alpha=e["alpha"], ppl_model=tmodel,
                                  ppl_batch_size=e["ppl_batch_size"], device=dev)}
    log = read_jsonl(out / "train_log.jsonl")
    if log:
        metrics["final_train"] = log[-1]
    write_json(out / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


def student_job(cfg, job_index=None, wm=None, lam=None, force=False):
    jobs = sweep_jobs(cfg)
    if job_index is not None:
        jobs = [jobs[job_index]]
    elif wm is not None or lam is not None:
        jobs = [j for j in jobs if (wm is None or j[0] == wm) and (lam is None or abs(j[1] - lam) < 1e-9)]
    for wm_name, l in jobs:
        student(cfg, wm_name, l, force=force)


# ---- aggregate ----------------------------------------------------------------------------------
def aggregate(cfg):
    from .report import write_report

    write_report(cfg)
