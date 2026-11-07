"""CLI:  python -m wmdistill <stage> --config configs/<machine>.yaml [key.sub=value ...]"""
import argparse
import os

os.environ.setdefault("USE_TF", "0")  # never let transformers import TensorFlow/Flax
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # torch<2.3 MPS lacks a few tiny ops (isin)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from . import stages
from .common import load_config, sweep_jobs


def main():
    ap = argparse.ArgumentParser(prog="wmdistill")
    ap.add_argument("stage", choices=["prepare", "teacher", "student", "aggregate", "all", "jobs"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--download-models", action="store_true", help="prepare: also fetch HF weights")
    ap.add_argument("--job-index", type=int, help="student: index into the (watermark, lambda) grid")
    ap.add_argument("--wm", help="student: only this watermark")
    ap.add_argument("--lam", type=float, help="student: only this lambda")
    ap.add_argument("--force", action="store_true", help="student: redo even if metrics exist")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. train.lr=1e-5 lambdas=[0.5]")
    a = ap.parse_intermixed_args()
    cfg = load_config(a.config, a.overrides)

    if a.stage == "jobs":  # used by slurm/submit_sweep.sh to size the array
        for i, (wm, lam) in enumerate(sweep_jobs(cfg)):
            print(i, wm, lam)
        return
    if a.stage in ("prepare", "all"):
        stages.prepare(cfg, download_models=a.download_models)
    if a.stage in ("teacher", "all"):
        stages.teacher(cfg)
    if a.stage in ("student", "all"):
        stages.student_job(cfg, a.job_index, a.wm, a.lam, a.force)
    if a.stage in ("aggregate", "all"):
        stages.aggregate(cfg)


if __name__ == "__main__":
    main()
