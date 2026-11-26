#!/usr/bin/env bash
#SBATCH --job-name=sass-refresh
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=sass-refresh-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
echo "Started $(date -Is) job=${SLURM_JOB_ID} host=$(hostname)"
.venv-train-ready/.venv/bin/python scripts/check_sass_training_inputs.py \
  --data data/sass_cuda_refresh_20261125_final \
  --model models/Qwen2.5-Coder-7B-Instruct
.venv-train-ready/.venv/bin/python -m unittest tests.test_sass_completion_loss -v
nvidia-smi
.venv-train-ready/.venv/bin/python -u -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/train_sass_lora.py \
  --model models/Qwen2.5-Coder-7B-Instruct \
  --data data/sass_cuda_refresh_20261125_final \
  --output runs/sass_lora_refresh_20261125_v1 \
  --epochs 2 --grad-accum 8 "$@"
echo "Finished $(date -Is)"
