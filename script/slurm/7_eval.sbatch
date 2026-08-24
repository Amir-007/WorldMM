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
# VRAM: one shared 30B instance (~60GB) plus the text (~8GB) and visual (~4GB)
# embedders is ~72GB of an 80GB A100. If it OOMs, set EMBED_DEVICE=cpu - the
# embedders are called once per query, so CPU is slower but workable.
#
#SBATCH --job-name=trackA-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/parallel_scratch/ms04938/logs/eval_%j.out
#SBATCH --error=/parallel_scratch/ms04938/logs/eval_%j.err
#SBATCH --chdir=/parallel_scratch/ms04938/WorldMM

set -eo pipefail

PERSON="${PERSON:-A1_JAKE}"
CONDITION="${CONDITION:-fixed}"
MODEL="${MODEL:-qwen3vl-30b}"
MAX_ROUNDS="${MAX_ROUNDS:-2}"
EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
CANDIDATES="${CANDIDATES:-30sec:5}"

cd /parallel_scratch/ms04938/WorldMM
mkdir -p /parallel_scratch/ms04938/logs

echo "host:      $(hostname)"
echo "condition: $CONDITION   rounds: $MAX_ROUNDS   embed: $EMBED_DEVICE"
echo "started:   $(date)"

if [ -f "/parallel_scratch/ms04938/env.sh" ]; then
    source /parallel_scratch/ms04938/env.sh
fi
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:$PATH
source /parallel_scratch/ms04938/WorldMM/.venv/bin/activate
export HF_HOME="${HF_HOME:-/parallel_scratch/ms04938/hf_cache}"

STAGE="output/eval/${CONDITION}"

# Cheap and idempotent; re-staging on a requeue costs seconds and guarantees
# the inputs match the current artifacts.
python eval/stage_eval_inputs.py --condition "$CONDITION" --person "$PERSON" --model "$MODEL"

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
    --max-rounds "$MAX_ROUNDS" \
    --text-embed-device "$EMBED_DEVICE" \
    --visual-embed-device "$EMBED_DEVICE" \
    --resume \
    "$@"

echo "finished: $(date)"
