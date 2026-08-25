#!/bin/bash
# Track A - EgoLifeQA evaluation for one chunking condition.
#
# Stages the condition into the layout eval_egolife.py expects, then runs that
# script unmodified. Both conditions use identical settings, so the comparison
# isolates chunking.
#
#   CONDITION=fixed sbatch script/slurm/7_eval.sbatch
#   CONDITION=event sbatch script/slurm/7_eval.sbatch
#
# Resumable: results are appended per question to a JSONL checkpoint and
# --resume (the default) skips completed ones, so requeue after a timeout.
#
# VRAM: a first attempt with both embedders on the GPU and visual retrieval
# enabled ran out of memory on 493 of 500 questions. Weights are only part of
# the footprint - visual retrieval feeds up to 8 frames through the 30B, and
# those activations were the 1-4GB allocations that failed at 76-79GB of 80.
#
# So the defaults here are conservative: embedders on CPU (~12GB back),
# visual retrieval off (removes the frame activations entirely), and an
# expandable allocator to limit fragmentation. Set VISUAL_K>0 to re-enable
# visual retrieval if you have headroom to spare.
#
#SBATCH --job-name=trackA-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/parallel_scratch/ms04938/logs/eval_%j.out
#SBATCH --error=/parallel_scratch/ms04938/logs/eval_%j.err
#SBATCH --chdir=/parallel_scratch/ms04938/WorldMM

set -eo pipefail

PERSON="${PERSON:-A1_JAKE}"
CONDITION="${CONDITION:-fixed}"
MODEL="${MODEL:-qwen3vl-30b}"
MAX_ROUNDS="${MAX_ROUNDS:-2}"
EMBED_DEVICE="${EMBED_DEVICE:-cpu}"
VISUAL_K="${VISUAL_K:-0}"
LIMIT="${LIMIT:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
CANDIDATES="${CANDIDATES:-30sec:5}"

cd /parallel_scratch/ms04938/WorldMM
mkdir -p /parallel_scratch/ms04938/logs

echo "host:      $(hostname)"
echo "condition: $CONDITION  rounds: $MAX_ROUNDS  embed: $EMBED_DEVICE  visual_k: $VISUAL_K  limit: ${LIMIT:-all}  max_new_tokens: $MAX_NEW_TOKENS"
echo "started:   $(date)"

if [ -f "/parallel_scratch/ms04938/env.sh" ]; then
    source /parallel_scratch/ms04938/env.sh
fi
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:$PATH
source /parallel_scratch/ms04938/WorldMM/.venv/bin/activate
export HF_HOME="${HF_HOME:-/parallel_scratch/ms04938/hf_cache}"

# Fragmentation was contributing to the OOM: allocations of 1-4GB failed while
# hundreds of MB sat reserved-but-unallocated.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# EgoLifeQA answers are a single letter. The model otherwise degenerates into
# thousands of tokens of repetition on some prompts, which at ~15 tok/s cost
# minutes per question in the first run.
export WORLDMM_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"

STAGE="output/eval/${CONDITION}"

# Cheap and idempotent; re-staging on a requeue costs seconds and guarantees
# the inputs match the current artifacts.
python eval/stage_eval_inputs.py --condition "$CONDITION" --person "$PERSON" \
    --model "$MODEL" ${LIMIT:+--limit "$LIMIT"}

# Retriever and responder are the same local model, so eval_egolife.py loads a
# single shared instance. Using qwen3vl-30b for both also avoids needing API
# access for the default gpt-5 responder.
python eval/eval_egolife.py \
    --subject "$PERSON" \
    --retriever-model "$MODEL" \
    --respond-model "$MODEL" \
    --memory-model "$MODEL" \
    --data-dir "${STAGE}/data" \
    --metadata-dir "${STAGE}/metadata" \
    --output-dir "${STAGE}/results" \
    --episodic-candidates "$CANDIDATES" \
    --visual-top-k "$VISUAL_K" \
    --max-rounds "$MAX_ROUNDS" \
    --text-embed-device "$EMBED_DEVICE" \
    --visual-embed-device "$EMBED_DEVICE" \
    --resume \
    "$@"

echo "finished: $(date)"
