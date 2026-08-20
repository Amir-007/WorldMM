#!/bin/bash
# Track A - sleep-cycle consolidation.  [Objective 2]
#
# Needs only the text embedding model (Qwen3-Embedding-4B, ~8GB), not the 30B
# generator, so a MIG slice is enough and this can run alongside other work.
#
#   CONDITION=fixed sbatch script/slurm/4_consolidate.sbatch
#   CONDITION=event sbatch script/slurm/4_consolidate.sbatch
#
#SBATCH --job-name=trackA-consolidate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/parallel_scratch/ms04938/logs/consolidate_%j.out
#SBATCH --error=/parallel_scratch/ms04938/logs/consolidate_%j.err
#SBATCH --chdir=/parallel_scratch/ms04938/WorldMM

set -eo pipefail

PERSON="${PERSON:-A1_JAKE}"
CONDITION="${CONDITION:-fixed}"
MODEL="${MODEL:-qwen3vl-30b}"
SWEEP="${SWEEP:-0.78,0.82,0.86,0.90,0.94}"

cd /parallel_scratch/ms04938/WorldMM
mkdir -p /parallel_scratch/ms04938/logs

echo "host:      $(hostname)"
echo "condition: $CONDITION"
echo "started:   $(date)"

if [ -f "/parallel_scratch/ms04938/env.sh" ]; then
    source /parallel_scratch/ms04938/env.sh
fi
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:$PATH
source /parallel_scratch/ms04938/WorldMM/.venv/bin/activate
export HF_HOME="${HF_HOME:-/parallel_scratch/ms04938/hf_cache}"

python preprocess/consolidation/run_consolidation.py \
    --semantic "output/metadata/semantic_memory/${PERSON}_${CONDITION}/semantic_extraction_results_${MODEL}.json" \
    --output-dir "output/consolidation/${PERSON}_${CONDITION}" \
    --sweep "$SWEEP" \
    --materialise \
    "$@"

echo "finished: $(date)"
