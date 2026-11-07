#!/bin/bash
# Run ON THE LOGIN NODE:  bash slurm/submit_sweep.sh configs/cluster.yaml [overrides...]
#   1. prepare (here, with internet): prompts + model weights into $HF_HOME
#   2. teacher job  ->  3. student array (one task per (watermark, lambda))  ->  4. aggregate
set -euo pipefail
CONFIG=${1:?usage: bash slurm/submit_sweep.sh configs/cluster.yaml [overrides...]}; shift
source slurm/env.sh
mkdir -p slurm/logs
python -m wmdistill prepare --download-models --config "$CONFIG" output_dir="$WMD_OUTPUT_DIR" "$@"

N=$(python -m wmdistill jobs --config "$CONFIG" "$@" | wc -l | tr -d ' ')
T=$(sbatch --parsable slurm/teacher.sbatch "$CONFIG" "$@")
S=$(sbatch --parsable --dependency=afterok:$T --array=0-$((N - 1)) slurm/student.sbatch "$CONFIG" "$@")
A=$(sbatch --parsable --dependency=afterany:$S slurm/aggregate.sbatch "$CONFIG" "$@")
echo "teacher=$T  students=$S (array 0-$((N - 1)))  aggregate=$A"
echo "results -> $WMD_OUTPUT_DIR/<exp_name>/results.csv, lambda_sweep.png"
