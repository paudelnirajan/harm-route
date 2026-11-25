#!/bin/bash
#SBATCH --job-name=e01-causal
#SBATCH --partition=blanca
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --mem=14G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
# E1 causal stage 1 (patch-h readout, forward passes only). Resumable: languages already written are skipped.
# Usage (login node, repo root):  sbatch scripts/e01_causal.sbatch <model> [langs...]
# Layer sweep:  sbatch --export=ALL,CFG=configs/experiments/e01_causal_sweep.yaml scripts/e01_causal.sbatch qwen3-8b
set -euo pipefail
export HF_HOME=/projects/$USER/hf UV_CACHE_DIR=/projects/$USER/.cache/uv
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export PATH=/curc/sw/install/uv/0.8.15/.local/bin:$PATH
cd /projects/$USER/harm-route
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MODEL="$1"; shift
CFG=${CFG:-configs/experiments/e01_causal.yaml}
uv run python scripts/causal_readout.py --config "$CFG" --model "$MODEL" ${@:+--lang "$@"}
