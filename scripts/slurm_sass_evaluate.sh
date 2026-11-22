#!/usr/bin/env bash
#SBATCH --job-name=sass-evaluate
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=sass-evaluate-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
kind="${1:-lora}"
case "$kind" in base|lora) ;; *) echo 'Use base or lora'; exit 2;; esac
module load cuda/12.6
export CUDA_HOME=/public/app/cuda/12.6
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 OMP_NUM_THREADS=1
echo "Started $(date -Is) job=${SLURM_JOB_ID} mode=$kind"
nvidia-smi
.venv/bin/python -u scripts/evaluate_sass.py \
  --predictions "runs/${kind}_test113" --results results \
  --output "runs/${kind}_test113_eval_batch_v1" --python .venv/bin/python --resume
echo "Finished $(date -Is)"
