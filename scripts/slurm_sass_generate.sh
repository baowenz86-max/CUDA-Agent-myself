#!/usr/bin/env bash
#SBATCH --job-name=sass-generate
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=sass-generate-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
kind="${1:-lora}"
case "$kind" in base) adapter=();; lora) adapter=(--adapter runs/sass_lora_formal_v1/adapter);; *) echo 'Use base or lora'; exit 2;; esac
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
echo "Started $(date -Is) job=${SLURM_JOB_ID} mode=$kind"
nvidia-smi
.venv-train-ready/.venv/bin/python -u scripts/generate_sass.py \
  --model models/Qwen2.5-Coder-7B-Instruct "${adapter[@]}" \
  --dataset data/sass_cuda_compact_32k_v1/test.jsonl \
  --output "runs/${kind}_test113" --limit 113
echo "Finished $(date -Is)"
