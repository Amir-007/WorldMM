#!/bin/bash
# Track A - Objective 3 metrics collection.
#
# CPU-only by default: sizes, compression ratios, distributions, and a
# model-free indexing benchmark. Pass --with-model to additionally time the
# real SemanticMemory.index(), which needs a GPU and the embedding model.
#
#   sbatch script/slurm/6_metrics.sbatch
#   sbatch script/slurm/6_metrics.sbatch --with-model
#
#SBATCH --job-name=trackA-metrics
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/parallel_scratch/ms04938/logs/metrics_%j.out
#SBATCH --error=/parallel_scratch/ms04938/logs/metrics_%j.err
#SBATCH --chdir=/parallel_scratch/ms04938/WorldMM

set -eo pipefail

PERSON="${PERSON:-A1_JAKE}"

cd /parallel_scratch/ms04938/WorldMM
mkdir -p /parallel_scratch/ms04938/logs

echo "host:    $(hostname)"
echo "started: $(date)"

if [ -f "/parallel_scratch/ms04938/env.sh" ]; then
    source /parallel_scratch/ms04938/env.sh
fi
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:$PATH
source /parallel_scratch/ms04938/WorldMM/.venv/bin/activate

python metrics/collect_metrics.py --person "$PERSON" "$@"
python metrics/make_figures.py --metrics-dir output/metrics || true

echo "finished: $(date)"
