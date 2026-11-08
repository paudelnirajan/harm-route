#!/bin/bash
#SBATCH --job-name=e00-gen
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Usage (login node, repo root):  mkdir -p logs && sbatch scripts/e00_generate.sbatch [config]
set -euo pipefail
export HF_HOME=/projects/$USER/hf UV_CACHE_DIR=/projects/$USER/.cache/uv
export HF_HUB_OFFLINE=1   # weights already cached; no token on the cluster -> anonymous calls get 429
export PYTHONUNBUFFERED=1   # print progress lines immediately (stdout to a file is block-buffered otherwise)
export PATH=/curc/sw/install/uv/0.8.15/.local/bin:$PATH
cd /projects/$USER/harm-route
git rev-parse --short HEAD   # record the code version in the log
uv run python scripts/generate_e0.py --config "${1:-configs/experiments/e00_generate.yaml}"
