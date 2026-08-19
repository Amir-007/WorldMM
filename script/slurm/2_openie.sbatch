#!/bin/bash
# Track A - OpenIE over one chunk set, one shard per full A100.
#
# Checkpointed per chunk, so a timeout or preemption costs at most one chunk.
# Resubmit the identical command to resume.
#
#   CONDITION=fixed SHARD=0 sbatch script/slurm/2_openie.sbatch
#   CONDITION=fixed SHARD=1 sbatch script/slurm/2_openie.sbatch
#   CONDITION=event SHARD=0 sbatch script/slurm/2_openie.sbatch
#
# Extra arguments are forwarded to the runner, e.g. to re-process only the
# chunks that came back with no triples:
#
#   CONDITION=fixed SHARD=0 sbatch script/slurm/2_openie.sbatch --retry-empty
#
# NUM_SHARDS must match across the shards of one condition, and the merge step
# reads it too. Default 2.
#
#SBATCH --job-name=trackA-openie
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/parallel_scratch/ms04938/logs/openie_%j.out
#SBATCH --error=/parallel_scratch/ms04938/logs/openie_%j.err
#SBATCH --chdir=/parallel_scratch/ms04938/WorldMM

set -eo pipefail

PERSON="${PERSON:-A1_JAKE}"
CONDITION="${CONDITION:-fixed}"
MODEL="${MODEL:-qwen3vl-30b}"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-2}"

cd /parallel_scratch/ms04938/WorldMM
mkdir -p /parallel_scratch/ms04938/logs

echo "host:      $(hostname)"
echo "condition: $CONDITION   shard: $SHARD/$NUM_SHARDS   model: $MODEL"
echo "started:   $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

if [ -f "/parallel_scratch/ms04938/env.sh" ]; then
    source /parallel_scratch/ms04938/env.sh
fi
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:$PATH
source /parallel_scratch/ms04938/WorldMM/.venv/bin/activate

# Read the model from parallel scratch rather than home NFS; the 13 shards are
# ~60GB and loading them over NFS cost 45 minutes.
export HF_HOME="${HF_HOME:-/parallel_scratch/ms04938/hf_cache}"

# Generation budgets are pinned here so both conditions provably share them.
# Changing either for one condition alone would confound the ablation.
export WORLDMM_NER_MAX_TOKENS="${WORLDMM_NER_MAX_TOKENS:-512}"
export WORLDMM_TRIPLE_MAX_TOKENS="${WORLDMM_TRIPLE_MAX_TOKENS:-4096}"

python preprocess/openie/run_openie.py \
    --chunks "output/chunks/${PERSON}/${CONDITION}/chunks.json" \
    --output-dir "output/metadata/episodic_memory/${PERSON}_${CONDITION}" \
    --model "$MODEL" \
    --shard-id "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    "$@"

echo "finished: $(date)"
