#!/bin/bash
# Track A - semantic extraction over 5-minute time-based nodes.
#
# Small next to OpenIE: ~620 nodes per condition at a few seconds each, so a
# single shard per condition is enough and both conditions fit on one card in
# sequence. Checkpointed per node; resubmit to resume.
#
#   CONDITION=fixed sbatch script/slurm/3_semantic.sbatch
#   CONDITION=event sbatch script/slurm/3_semantic.sbatch
#   CONDITION=fixed sbatch script/slurm/3_semantic.sbatch --merge
#
#SBATCH --job-name=trackA-semantic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/parallel_scratch/ms04938/logs/semantic_%j.out
#SBATCH --error=/parallel_scratch/ms04938/logs/semantic_%j.err
#SBATCH --chdir=/parallel_scratch/ms04938/WorldMM

set -eo pipefail

PERSON="${PERSON:-A1_JAKE}"
CONDITION="${CONDITION:-fixed}"
MODEL="${MODEL:-qwen3vl-30b}"
PERIOD="${PERIOD:-300}"

cd /parallel_scratch/ms04938/WorldMM
mkdir -p /parallel_scratch/ms04938/logs

echo "host:      $(hostname)"
echo "condition: $CONDITION   period: ${PERIOD}s   model: $MODEL"
echo "started:   $(date)"

if [ -f "/parallel_scratch/ms04938/env.sh" ]; then
    source /parallel_scratch/ms04938/env.sh
fi
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:$PATH
source /parallel_scratch/ms04938/WorldMM/.venv/bin/activate
export HF_HOME="${HF_HOME:-/parallel_scratch/ms04938/hf_cache}"

python preprocess/semantic/run_semantic.py \
    --chunks "output/chunks/${PERSON}/${CONDITION}/chunks.json" \
    --openie "output/metadata/episodic_memory/${PERSON}_${CONDITION}/openie_results_${MODEL}.json" \
    --output-dir "output/metadata/semantic_memory/${PERSON}_${CONDITION}" \
    --model "$MODEL" \
    --period-seconds "$PERIOD" \
    "$@"

echo "finished: $(date)"
