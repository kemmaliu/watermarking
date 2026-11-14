# Site-specific setup, sourced by every job.

module purge
module load python                        # check `module avail python`
set +u; mamba activate wmdistill; set -u  # set +u avoids "unbound variable" errors from conda

LAB_SCRATCH="/n/netscratch/sur_lab/Everyone/$USER"
export HF_HOME="${HF_HOME:-$LAB_SCRATCH/hf_cache}"            # model weights: big, keep off $HOME
export WMD_OUTPUT_DIR="${WMD_OUTPUT_DIR:-$LAB_SCRATCH/wmdistill_runs}"
export TOKENIZERS_PARALLELISM=false
# Compute nodes may have no internet: everything is fetched by `prepare` on the login node.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1; fi