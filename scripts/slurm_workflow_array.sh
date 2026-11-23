#!/usr/bin/env bash
# Submit from the CUDA-Agent project root:
#   sbatch scripts/slurm_workflow_array.sh
# One job runs all 109 batches sequentially (no job array).
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
#SBATCH --output=cuda-agent-%j.log

set -euo pipefail
# Load API credentials on the compute node without printing their values.
secrets_file="$HOME/.config/cuda-agent/secrets.env"
if [[ ! -r "$secrets_file" ]]; then
  echo "Cannot read API credentials: $secrets_file" >&2
  exit 2
fi
set -a
source "$secrets_file"
set +a

cd "${SLURM_SUBMIT_DIR:?Submit from the CUDA-Agent project root}"
[[ -f main.py && -f pyproject.toml ]] || {
  echo 'Please run sbatch from the CUDA-Agent project root.' >&2
  exit 2
}

# Use the devices allocated by Slurm, ignoring any local GPU override.
unset CUDA_AGENT_GPUS
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUDA_AGENT_LOG_DIR="$PWD/logs/slurm_${SLURM_JOB_ID}"
mkdir -p "$CUDA_AGENT_LOG_DIR"
command -v uv >/dev/null || { echo 'uv is not available in PATH.' >&2; exit 127; }

# Prepare dependencies with `uv sync` before submission. Compute jobs must
# not modify the shared environment or require dependency downloads.
echo "Started $(date -Is) job=${SLURM_JOB_ID} tasks=3261-4350"
failed_batches=0
for ((task_start=3261; task_start<=4350; task_start+=10)); do
  task_end=$((task_start + 9))
  echo "Running tasks ${task_start}-${task_end} at $(date -Is)"
  # Match the original semicolon-separated commands: continue after a failure.
  if srun uv run --frozen --no-sync python main.py \
      --task-start "$task_start" --task-end "$task_end"; then
    echo "Completed tasks ${task_start}-${task_end}"
  else
    status=$?
    echo "Failed tasks ${task_start}-${task_end} (exit $status); continuing." >&2
    failed_batches=$((failed_batches + 1))
  fi
done
echo "Finished $(date -Is)"
echo "Failed batches: $failed_batches"
(( failed_batches == 0 ))
