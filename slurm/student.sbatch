#!/bin/bash
#SBATCH --job-name=wmd-student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
# #SBATCH --constraint=gpu80
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --partition=YOUR_PARTITION
# One array task = one (watermark, lambda); list them with `python -m wmdistill jobs --config ...`.
# Re-submitting is safe: finished jobs (metrics.json present) are skipped.
set -euo pipefail
CONFIG=${1:?usage: sbatch --array=0-N slurm/student.sbatch configs/cluster.yaml [overrides...]}; shift
source slurm/env.sh
python -m wmdistill student --config "$CONFIG" --job-index "$SLURM_ARRAY_TASK_ID" output_dir="$WMD_OUTPUT_DIR" "$@"
