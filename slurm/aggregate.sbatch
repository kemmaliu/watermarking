#!/bin/bash
#SBATCH --job-name=wmd-aggregate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
CONFIG=${1:?usage: sbatch slurm/aggregate.sbatch configs/cluster.yaml [overrides...]}; shift
source slurm/env.sh
python -m wmdistill aggregate --config "$CONFIG" output_dir="$WMD_OUTPUT_DIR" "$@"
