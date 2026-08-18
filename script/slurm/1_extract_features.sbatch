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
#SBATCH --time=4-00:00:00
#SBATCH --output=/parallel_scratch/ms04938/logs/features_%j.out
#SBATCH --error=/parallel_scratch/ms04938/logs/features_%j.err
#SBATCH --chdir=/parallel_scratch/ms04938/WorldMM

set -eo pipefail

PERSON="${PERSON:-A1_JAKE}"

# Navigate explicitly to the repository root
cd /parallel_scratch/ms04938/WorldMM

# Ensure your standard logs directory exists
mkdir -p /parallel_scratch/ms04938/logs

echo "host:    $(hostname)"
echo "cpus:    ${SLURM_CPUS_PER_TASK:-1}"
echo "started: $(date)"

# Load environment and fix $PATH just in case
if [ -f "/parallel_scratch/ms04938/env.sh" ]; then
    source /parallel_scratch/ms04938/env.sh
fi
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:$PATH

# Activate virtual environment with absolute path
source /parallel_scratch/ms04938/WorldMM/.venv/bin/activate

python preprocess/chunking/extract_frame_features.py \
    --person "$PERSON" \
    --config configs/event.json \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"

echo "finished: $(date)"