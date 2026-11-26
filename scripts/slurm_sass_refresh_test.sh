#!/usr/bin/env bash
#SBATCH --job-name=sass-refresh-test
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=sass-refresh-test-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
stage="${1:-generate}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
echo "Started $(date -Is) job=${SLURM_JOB_ID} stage=$stage"
nvidia-smi
case "$stage" in
  generate)
    .venv-train-ready/.venv/bin/python scripts/check_sass_training_inputs.py \
      --data data/sass_cuda_refresh_20261125_final --model models/Qwen2.5-Coder-7B-Instruct
    .venv-train-ready/.venv/bin/python -u scripts/generate_sass.py \
      --model models/Qwen2.5-Coder-7B-Instruct \
      --adapter runs/sass_lora_refresh_20261125_v1/adapter \
      --dataset data/sass_cuda_refresh_20261125_final/test.jsonl \
      --output runs/lora_refresh_20261125_test113 --limit 113
    ;;
  evaluate)
    module load cuda/12.6
    export CUDA_HOME=/public/app/cuda/12.6
    export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
    export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4
    .venv/bin/python -u scripts/evaluate_sass.py \
      --predictions runs/lora_refresh_20261125_test113 --results results \
      --output runs/lora_refresh_20261125_test113_eval_v1 \
      --python .venv/bin/python --resume
    ;;
  *) echo 'Use generate or evaluate'; exit 2;;
esac
echo "Finished $(date -Is)"
