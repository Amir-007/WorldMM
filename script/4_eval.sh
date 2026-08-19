#!/bin/bash
# WorldMM Evaluation Script
# Usage: ./script/4_eval.sh [--person <person>] [--retriever-model qwen3vl-30b] [--respond-model gpt-5]
#        [--memory-model qwen3vl-30b] [--metadata-dir output/metadata] [--max-rounds 5]
#        [--enable-spatial] [--enable-abstention] [--confidence-threshold 0.75] [--resume|--fresh]

set -eo pipefail
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

PERSON="A1_JAKE" RET_MODEL="qwen3vl-30b" RESP_MODEL="gpt-5" MEM_MODEL="qwen3vl-30b"
MAX_ROUNDS=5 MAX_ERRORS=5 EPISODIC_K=3 SEMANTIC_K=10 VISUAL_K=3
OUTPUT_DIR="output" DATA_DIR="data/EgoLife" METADATA_DIR="output/metadata"
SPATIAL_FLAGS="" CONF_THRESHOLD="0.75" RESUME_FLAG=""
VIS_EMBED_DEV="cuda" TEXT_EMBED_DEV="cuda" VIS_MAX_FRAMES="8"

cd "$(dirname "$0")/.."

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --person) PERSON="$2"; shift 2 ;;
        --memory-model) MEM_MODEL="$2"; shift 2 ;;
        --retriever-model) RET_MODEL="$2"; shift 2 ;;
        --respond-model) RESP_MODEL="$2"; shift 2 ;;
        --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
        --max-errors) MAX_ERRORS="$2"; shift 2 ;;
        --episodic-top-k) EPISODIC_K="$2"; shift 2 ;;
        --semantic-top-k) SEMANTIC_K="$2"; shift 2 ;;
        --visual-top-k) VISUAL_K="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --metadata-dir) METADATA_DIR="$2"; shift 2 ;;
        --enable-spatial) SPATIAL_FLAGS="$SPATIAL_FLAGS --enable-spatial"; shift ;;
        --enable-abstention) SPATIAL_FLAGS="$SPATIAL_FLAGS --enable-abstention"; shift ;;
        --confidence-threshold) CONF_THRESHOLD="$2"; shift 2 ;;
        --visual-max-frames) VIS_MAX_FRAMES="$2"; shift 2 ;;
        --visual-embed-device) VIS_EMBED_DEV="$2"; shift 2 ;;
        --text-embed-device) TEXT_EMBED_DEV="$2"; shift 2 ;;
        --fresh) RESUME_FLAG="--fresh"; shift ;;
        --resume) RESUME_FLAG="--resume"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/eval/${PERSON}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_${RESP_MODEL//-/_}_$TIMESTAMP.log"

echo -e "${BLUE}Running EgoLifeQA evaluation: $PERSON with Retriever $RET_MODEL, Responder $RESP_MODEL, Memory $MEM_MODEL${NC}"
python eval/eval_egolife.py \
    --subject "$PERSON" \
    --retriever-model "$RET_MODEL" \
    --respond-model "$RESP_MODEL" \
    --memory-model "$MEM_MODEL" \
    --max-rounds "$MAX_ROUNDS" \
    --max-errors "$MAX_ERRORS" \
    --episodic-top-k "$EPISODIC_K" \
    --semantic-top-k "$SEMANTIC_K" \
    --visual-top-k "$VISUAL_K" \
    --output-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --metadata-dir "$METADATA_DIR" \
    --confidence-threshold "$CONF_THRESHOLD" \
    --visual-max-frames "$VIS_MAX_FRAMES" \
    --visual-embed-device "$VIS_EMBED_DEV" \
    --text-embed-device "$TEXT_EMBED_DEV" \
    $SPATIAL_FLAGS $RESUME_FLAG 2>&1 | tee "$LOG_FILE"

echo -e "${BLUE}Eval Done! Results: ${OUTPUT_DIR}/${RET_MODEL//-/_}_${RESP_MODEL//-/_}/egolife_eval_${PERSON}.json${NC}"
