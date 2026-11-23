#!/usr/bin/env bash
# Submit from the CUDA-Agent project root:
#   sbatch scripts/slurm_workflow_array.sh
# Run up to four batches concurrently:
#   sbatch --array=0-108%4 scripts/slurm_workflow_array.sh
# Override partition/QoS/GPU/time with sbatch options if needed.
#SBATCH --job-name=cuda-agent
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:A100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-108%1
#SBATCH --output=cuda-agent-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the CUDA-Agent project root}"
[[ -f main.py && -f pyproject.toml ]] || {
  echo 'Please run sbatch from the CUDA-Agent project root.' >&2
  exit 2
}

index="${SLURM_ARRAY_TASK_ID:?This script requires a Slurm job array}"
[[ "$index" =~ ^[0-9]+$ ]] && (( index <= 108 )) || {
  echo "Invalid array index: $index (expected 0-108)" >&2
  exit 2
}
task_start=$((3261 + index * 10))
task_end=$((task_start + 9))

# Use the devices allocated by Slurm, ignoring any local GPU override.
unset CUDA_AGENT_GPUS
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUDA_AGENT_LOG_DIR="$PWD/logs/slurm_${SLURM_ARRAY_JOB_ID}_${index}"
mkdir -p "$CUDA_AGENT_LOG_DIR"
command -v uv >/dev/null || { echo 'uv is not available in PATH.' >&2; exit 127; }

# Prepare dependencies with `uv sync` before submission. Compute jobs must
# not modify the shared environment or require dependency downloads.
echo "Started $(date -Is) job=${SLURM_JOB_ID} tasks=${task_start}-${task_end}"
srun uv run --frozen --no-sync python main.py \
  --task-start "$task_start" --task-end "$task_end"
echo "Finished $(date -Is)"
