#!/bin/bash
#SBATCH --job-name=e01-gen
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Generations then judge, for each language in turn (resumable: existing files are skipped).
# Usage (login node, repo root):  mkdir -p logs && sbatch scripts/e01_generate.sbatch [config] [judge] [langs...]
set -euo pipefail
export HF_HOME=/projects/$USER/hf UV_CACHE_DIR=/projects/$USER/.cache/uv
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PATH=/curc/sw/install/uv/0.8.15/.local/bin:$PATH
cd /projects/$USER/harm-route
git rev-parse --short HEAD
CFG="${1:-configs/experiments/e01_generate.yaml}"; JUDGE="${2:-qwen3guard-4b}"; shift $(( $# > 2 ? 2 : $# ))
LANGS=("$@"); [ ${#LANGS[@]} -eq 0 ] && LANGS=(eng_Latn npi_Deva srp_Cyrl srp_Latn)
for L in "${LANGS[@]}"; do
  echo "=== $L ==="
  uv run python scripts/generate_e0.py --config "$CFG" --lang "$L"
  uv run python scripts/judge.py --judge "$JUDGE" --config "$CFG" --lang "$L"
done
