#!/bin/bash
# WorldMM Preprocessing Script
# Usage: ./script/2_preprocess.sh [--person A1_JAKE]

set -eo pipefail
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

PERSON="A1_JAKE" MODEL="qwen3vl-30b"

while [[ $# -gt 0 ]]; do
    case $1 in
        --person) PERSON="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."

source .venv/bin/activate

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/preprocess/egolife/${PERSON}"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}Translating DenseCaption...${NC}"
python data/EgoLife/utils/translate_densecap.py --model "$MODEL" 2>&1 | tee "$LOG_DIR/translate_densecap_$TIMESTAMP.log"

echo -e "${BLUE}Generating Sync data...${NC}"
python data/EgoLife/utils/generate_sync.py 2>&1 | tee "$LOG_DIR/generate_sync_$TIMESTAMP.log"

echo -e "${BLUE}Preprocess Done!${NC}"
