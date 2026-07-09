#!/bin/bash
# WorldMM Preprocessing Script
# Usage: ./script/2_preprocess.sh [--person A1_JAKE]

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

PERSON="A1_JAKE"

while [[ $# -gt 0 ]]; do
    case $1 in
        --person) PERSON="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

source .venv/bin/activate

cd "$(dirname "$0")/.."

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/preprocess/egolife/${PERSON}"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}Translating DenseCaption...${NC}"
python data/EgoLife/utils/translate_densecap.py 2>&1 | tee "$LOG_DIR/translate_densecap_$TIMESTAMP.log"

echo -e "${BLUE}Generating Sync data...${NC}"
python data/EgoLife/utils/generate_sync.py 2>&1 | tee "$LOG_DIR/generate_sync_$TIMESTAMP.log"

echo -e "${BLUE}Preprocess Done!${NC}"
