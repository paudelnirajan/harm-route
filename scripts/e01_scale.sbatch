#!/bin/bash
#SBATCH --job-name=e01-scale
#SBATCH --partition=blanca
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
# Per model: base generations + judge for every language, then the activation cache (probes / directions).
# Preemptable: can be killed and requeued; every step skips files that already exist, so it resumes.
# Usage (login node, repo root):  mkdir -p logs && sbatch scripts/e01_scale.sbatch <model> [langs...]
# Override the GPU type at submit time, e.g. sbatch --gres=gpu:l40:1 scripts/e01_scale.sbatch qwen3-8b
set -euo pipefail
export HF_HOME=/projects/$USER/hf UV_CACHE_DIR=/projects/$USER/.cache/uv
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export PATH=/curc/sw/install/uv/0.8.15/.local/bin:$PATH
cd /projects/$USER/harm-route
git rev-parse --short HEAD; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MODEL="$1"; shift
LANGS=("$@"); [ ${#LANGS[@]} -eq 0 ] && LANGS=(eng_Latn npi_Deva srp_Cyrl srp_Latn)
for L in "${LANGS[@]}"; do
  echo "=== $MODEL $L: generate + judge ==="
  uv run python scripts/generate_e0.py --config configs/experiments/e01_scale.yaml --model "$MODEL" --lang "$L"
  uv run python scripts/judge.py --judge qwen3guard-4b --config configs/experiments/e01_scale.yaml --model "$MODEL" --lang "$L"
done
echo "=== $MODEL: activation cache ==="
uv run python scripts/cache_e1_acts.py --config configs/experiments/e01_acts.yaml --model "$MODEL" \
  --device cuda --dtype bfloat16 --lang "${LANGS[@]}"
