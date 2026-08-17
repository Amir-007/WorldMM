#!/bin/bash
# Track A - frame feature extraction for event-boundary detection.
#
# CPU only, no GPU, no model. This is the pass that gates the event condition;
# once the cache exists, boundary detection and the whole threshold sweep run
# from it in seconds.
#
# Checkpointed per mp4: if this job is killed or times out, resubmit it and it
# resumes from where it stopped.
#
#   sbatch script/slurm/1_extract_features.sbatch
#   sbatch script/slurm/1_extract_features.sbatch --days 1,2      # subset first
#
#SBATCH --job-name=trackA-features
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=.log/slurm/features_%j.out
#SBATCH --error=.log/slurm/features_%j.err

set -eo pipefail

PERSON="${PERSON:-A1_JAKE}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p .log/slurm

echo "host:    $(hostname)"
echo "cpus:    ${SLURM_CPUS_PER_TASK:-1}"
echo "started: $(date)"

source .venv/bin/activate

python preprocess/chunking/extract_frame_features.py \
    --person "$PERSON" \
    --config configs/event.json \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"

echo "finished: $(date)"
