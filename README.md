# wmdistill — Part 1: λ sweep for watermark-aware distillation

A teacher LM generates **watermarked** text with a secret key. A student is fine-tuned on those
samples with

    L = (1 − λ) · CE  +  λ · L_stat          (key available during training)

and then **samples normally (no key)**. We test whether the student's own text carries the
teacher's watermark. λ sweeps over `lambdas` in the config (0.1 … 0.9, plus 0.0 as the
plain-distillation baseline).

## Watermarks and the stat loss

| | decoder (teacher) | per-token score s_t(w) | test |
|---|---|---|---|
| `gumbel` (Aaronson) | argmax_w U_{t,w}^{1/p_w} | −log(1 − U_{t,w}) | Σ ~ Gamma(n,1) under H0 |
| `inverse` (Kuditipudi et al.) | permuted-CDF inverse at u_t | (u_t − ½)(η_t(w) − ½) | z-test |

Keys are `seed_t = H(key, previous h tokens)` (`seed_scheme: context`, default h = 2) or
`H(key, t)` (`position`). They're computed with integer hashing, bit-identical on CPU/CUDA/MPS,
so data made on one machine verifies on another.

`L_stat` uses scores standardized to mean 0 / std 1 under H0, z_t(w):
- `match` (default): (E_seq − R_seq)², where E_seq = mean_t Σ_w p_student(w) z_t(w) is the student's
  differentiable expected statistic and R_seq = mean_t z_t(Y_t) is the realized statistic T(Y)/n of
  the teacher's sample.
- `max`: −E_seq, which pushes the statistic up without a target.

Switch with `train.stat_loss=max`. Implementation: `wmdistill/train.py`.

## Running

```bash
pip install -r requirements.txt
python -m pytest -p no:cov tests                          # watermark math (13 tests, ~7s)
python -m wmdistill all --config configs/tiny.yaml        # end-to-end smoke test, ~1 min, random tiny model
python -m wmdistill all --config configs/laptop.yaml      # SmolLM2 360M -> 135M on C4, real signal
```

Stages, each resumable (existing outputs are skipped):

| stage | what | writes (under `runs/<exp_name>/`) |
|---|---|---|
| `prepare` | C4 prompts (+ `--download-models`) — **needs internet** | `prompts.jsonl` |
| `teacher` | watermarked training data; controls: teacher wm / teacher plain / base student | `teacher_*.jsonl`, `controls.json` |
| `student` | train + evaluate one (watermark, λ); `--job-index i` or `--wm/--lam` | `students/<wm>/lam_<λ>/{metrics.json,samples.jsonl,train_log.jsonl}` |
| `aggregate` | table + plot | `results.csv`, `lambda_sweep.png` |

Any config value can be overridden on the command line. Quote brackets in zsh:
`python -m wmdistill student --config configs/laptop.yaml --lam 0.5 train.lr=1e-5 'watermarks=[gumbel]'`.

## Moving to the cluster

1. Clone the repo, create the env (`pip install -r requirements.txt`), edit **`slurm/env.sh`**
   (module/conda lines, `HF_HOME` + output dir on scratch). Also uncomment the
   `--account/--partition/--constraint` lines in `slurm/*.sbatch`.
2. For gated Llama weights, accept the license on HF, then run `huggingface-cli login` on the login node.
   To pick other models, edit `teacher.model` / `student.model` in `configs/cluster.yaml`.
   The two must share a tokenizer.
3. On the login node run `bash slurm/submit_sweep.sh configs/cluster.yaml`. It runs `prepare`
   there (internet), then submits teacher → student array (20 tasks = 2 watermarks × 10 λ) →
   aggregate, chained with dependencies. Compute jobs run with `HF_HUB_OFFLINE=1`.

Re-submitting is safe: finished λ's are skipped. Rerun a single one with
`sbatch --array=7 slurm/student.sbatch configs/cluster.yaml`. List the index → (wm, λ) mapping with
`python -m wmdistill jobs --config configs/cluster.yaml`.

## Metrics (per λ, plus controls)

- `detect_rate`: fraction of student samples with p < `eval.alpha`.
- `median_pvalue`, `mean_z_per_token`: watermark strength (0 = no signal).
- `teacher_ppl`: perplexity of student samples under the teacher (quality).
- `rep3_rate`: 3-gram repetition. Watch it at high λ, since gaming the statistic can degrade text.

Repeated (context, token) pairs are scored once (standard for context-hashed keys).
`teacher_watermarked` is the ceiling. `student_base` / `teacher_plain` are the null floor and should
sit near `alpha`.
