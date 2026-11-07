# Site-specific setup, sourced by every job. EDIT THIS once you have cluster access.
# (Defaults guess at Princeton Research Computing / Della conventions.)

# module purge
# module load anaconda3/2024.6
# conda activate wmdistill

export HF_HOME="${HF_HOME:-/scratch/gpfs/$USER/hf_cache}"   # model weights: big, keep off $HOME
export WMD_OUTPUT_DIR="${WMD_OUTPUT_DIR:-/scratch/gpfs/$USER/wmdistill_runs}"
export TOKENIZERS_PARALLELISM=false
# Compute nodes typically have no internet: everything is fetched by `prepare` on the login node.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1; fi
