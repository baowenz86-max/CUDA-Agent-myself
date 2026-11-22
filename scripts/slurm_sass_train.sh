#!/usr/bin/env bash
#SBATCH --job-name=sass-train
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=sass-train-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "Started: $(date -Is) host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi
.venv-train-ready/.venv/bin/python -m unittest tests.test_sass_completion_loss -v
.venv-train-ready/.venv/bin/python -u -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/train_sass_lora.py \
  --model models/Qwen2.5-Coder-7B-Instruct \
  --data data/sass_cuda_compact_32k_v1 \
  --output runs/sass_lora_formal_v1 \
  --grad-accum 8 --epochs 2 "$@"
echo "Finished: $(date -Is)"
