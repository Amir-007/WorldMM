#!/bin/bash
# Track A - Objective 3 metrics collection.
#
# CPU-only by default: sizes, compression ratios, distributions, and a
# model-free indexing benchmark. Pass --with-model to additionally time the
# real SemanticMemory.index(), which needs a GPU and the embedding model.
#
#   sbatch script/slurm/6_metrics.sbatch
#
# The --with-model variant times the real SemanticMemory.index() and therefore
# needs a GPU, so override the partition on the command line and send it to a
# separate directory - it is a superset of the plain run, and the two would
# otherwise overwrite each other's output:
#
#   OUTDIR=output/metrics_model sbatch --partition=gpu --gres=gpu:3g.40gb:1 \
#       script/slurm/6_metrics.sbatch --with-model
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
OUTDIR="${OUTDIR:-output/metrics}"

cd /parallel_scratch/ms04938/WorldMM
mkdir -p /parallel_scratch/ms04938/logs

echo "host:    $(hostname)"
echo "started: $(date)"

if [ -f "/parallel_scratch/ms04938/env.sh" ]; then
    source /parallel_scratch/ms04938/env.sh
fi
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:$PATH
source /parallel_scratch/ms04938/WorldMM/.venv/bin/activate

python metrics/collect_metrics.py --person "$PERSON" --output-dir "$OUTDIR" "$@"
python metrics/make_figures.py --metrics-dir "$OUTDIR" || true

echo "finished: $(date)"
