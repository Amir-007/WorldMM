#!/bin/bash
# Chain semantic extraction -> merge -> consolidation for one condition.
#
# Each stage only starts if the previous one succeeded, so consolidation can no
# longer run ahead of the input it consumes. Submit and walk away:
#
#   ./script/slurm/run_stage_5.sh fixed
#   ./script/slurm/run_stage_5.sh event
#
# Walltimes are set here rather than taken from the sbatch defaults, because on
# a two-node partition holding a 7-day job, an oversized --time request is what
# keeps a short job out of backfill. Every stage is checkpointed, so a timeout
# costs one unit of work and a resubmit, not a rerun.

set -eo pipefail

CONDITION="${1:?usage: run_stage_5.sh <fixed|event>}"
PERSON="${PERSON:-A1_JAKE}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

extract=$(CONDITION="$CONDITION" PERSON="$PERSON" \
    sbatch --parsable --time=06:00:00 "$HERE/3_semantic.sbatch")
echo "semantic extract : $extract"

merge=$(CONDITION="$CONDITION" PERSON="$PERSON" \
    sbatch --parsable --time=1:00:00 --dependency=afterok:"$extract" \
    "$HERE/3_semantic.sbatch" --merge)
echo "semantic merge   : $merge  (after $extract)"

consolidate=$(CONDITION="$CONDITION" PERSON="$PERSON" \
    sbatch --parsable --time=03:00:00 --dependency=afterok:"$merge" \
    "$HERE/4_consolidate.sbatch")
echo "consolidation    : $consolidate  (after $merge)"

echo
echo "watch with: squeue -u \$USER -o '%.10i %.24j %.8T %.10M %.10l %R'"
