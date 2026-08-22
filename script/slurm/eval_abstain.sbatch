#!/bin/bash
# Track B - EgoLifeQA evaluation, uncertainty-aware abstention arm.
#
# Must match the baseline arm's retrieval flags exactly; the only intended
# difference is --enable-abstention. Changing --episodic-candidates,
# --visual-top-k or --max-rounds would confound the comparison.
#
# Results are checkpointed per question, so hitting the time limit is not fatal:
# resubmit without FRESH and it resumes from the first unanswered question.
#
#   FRESH=1 sbatch --dependency=afterany:<baseline_jobid> script/slurm/eval_abstain.sbatch
#   sbatch script/slurm/eval_abstain.sbatch        # resume after a timeout
#
#SBATCH --job-name="Evaluation Script"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --chdir=/parallel_scratch/zt00372/WorldMM
#SBATCH --output=/parallel_scratch/zt00372/Logs/eval_abstain_%j.log
#SBATCH --error=/parallel_scratch/zt00372/Logs/eval_abstain_%j.err

set -eo pipefail

REPO=/parallel_scratch/zt00372/WorldMM
PERSON="${PERSON:-A1_JAKE}"
MODEL="${MODEL:-qwen3vl-30b}"
THRESHOLD="${THRESHOLD:-0.75}"
FRESH="${FRESH:-0}"

# Records which card was actually allocated. A plain gpu:1 can hand back a MIG
# slice, and the resulting OOM is confusing without this line in the log.
nvidia-smi

module load CUDA/12.2.2

# cd before activating: the venv path is relative to the repo, and --chdir is
# not reliable on this cluster.
cd "$REPO"
mkdir -p /parallel_scratch/zt00372/Logs

if [ -f /parallel_scratch/zt00372/env.sh ]; then
    source /parallel_scratch/zt00372/env.sh
fi
source "$REPO/.venv/bin/activate"

# Recovers ~2.3 GiB of allocator fragmentation. Measured: without it, OOM hit
# with the request short by only ~30 MiB.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host:      $(hostname)"
echo "commit:    $(git rev-parse --short HEAD)"
echo "person:    $PERSON   model: $MODEL   threshold: $THRESHOLD"
echo "fresh:     $FRESH"
echo "started:   $(date)"

RESUME_FLAG=""
if [ "$FRESH" = "1" ]; then
    RESUME_FLAG="--fresh"
fi

/parallel_scratch/$USER/WorldMM/script/4_eval.sh \
    --person "$PERSON" \
    --retriever-model "$MODEL" \
    --respond-model "$MODEL" \
    --memory-model "$MODEL" \
    --episodic-candidates "30sec:5,3min:3,10min:2,1h:0" \
    --visual-top-k 1 \
    --max-rounds 3 \
    --enable-abstention \
    --confidence-threshold "$THRESHOLD" \
    $RESUME_FLAG "$@"

echo "finished: $(date)"
