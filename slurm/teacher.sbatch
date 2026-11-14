#!/bin/bash
#SBATCH --job-name=wmd-teacher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# #SBATCH --constraint=gpu80          # e.g. Della: request 80GB A100s
#SBATCH --account=sur_lab
#SBATCH --partition=gpu
set -euo pipefail
CONFIG=${1:?usage: sbatch slurm/teacher.sbatch configs/cluster.yaml [overrides...]}; shift
source slurm/env.sh
python -m wmdistill teacher --config "$CONFIG" output_dir="$WMD_OUTPUT_DIR" "$@"
