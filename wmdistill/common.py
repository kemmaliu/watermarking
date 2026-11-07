"""Config, device, model loading and jsonl helpers."""
import copy
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


# ---- config ---------------------------------------------------------------------------------
def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, overrides=()) -> dict:
    """YAML config with `inherit: other.yaml` (relative path) and dotted CLI overrides a.b=val."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "inherit" in cfg:
        cfg = _deep_merge(load_config(path.parent / cfg.pop("inherit")), cfg)
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def run_dir(cfg) -> Path:
    d = Path(cfg["output_dir"]) / cfg["exp_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def sweep_jobs(cfg) -> list:
    """Cartesian product (watermark, lambda); a SLURM array index selects one entry."""
    return [(wm, float(lam)) for wm in cfg["watermarks"] for lam in cfg["lambdas"]]


def student_dir(cfg, wm: str, lam: float) -> Path:
    return run_dir(cfg) / "students" / wm / f"lam_{lam:.2f}"


# ---- runtime --------------------------------------------------------------------------------
def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def free(*objs):
    del objs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_lm(name: str, dtype: str, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=DTYPES[dtype]).to(device)
    model.eval()
    return model


def key_vocab_size(cfg) -> int:
    """Vocab the watermark key ranges over: shared logit dims of teacher and student.

    (Families sometimes pad the LM head differently, e.g. Qwen2.5-7B vs 0.5B; tokens beyond the
    real tokenizer are never sampled, so truncating to the common prefix is safe.)
    """
    from transformers import AutoConfig

    return min(AutoConfig.from_pretrained(cfg[r]["model"]).vocab_size for r in ("teacher", "student"))


# ---- io -------------------------------------------------------------------------------------
def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)  # atomic: a preempted job never leaves a half-written file behind


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.rename(path)
