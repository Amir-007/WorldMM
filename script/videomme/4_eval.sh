#!/bin/bash
# WorldMM Evaluation Script
# Usage: ./script/4_eval.sh [--retriever-model gpt-5-mini] [--respond-model gpt-5] [--max-rounds 5]

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

RET_MODEL="gpt-5-mini" RESP_MODEL="gpt-5"
MAX_ROUNDS=5 MAX_ERRORS=5 EPISODIC_K=3 SEMANTIC_K=10 VISUAL_K=3
OUTPUT_DIR="output"
EVAL_JSON="data/Video-MME/videomme/test.json"
CAPTION_DIR="data/Video-MME/caption"
METADATA_DIR="output/metadata/videomme"
EPISODIC_CACHE_DIR=".cache/videomme"
DURATION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --retriever-model) RET_MODEL="$2"; shift 2 ;;
        --respond-model) RESP_MODEL="$2"; shift 2 ;;
        --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
        --max-errors) MAX_ERRORS="$2"; shift 2 ;;
        --episodic-top-k) EPISODIC_K="$2"; shift 2 ;;
        --semantic-top-k) SEMANTIC_K="$2"; shift 2 ;;
        --visual-top-k) VISUAL_K="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --eval-json) EVAL_JSON="$2"; shift 2 ;;
        --caption-dir) CAPTION_DIR="$2"; shift 2 ;;
        --metadata-dir) METADATA_DIR="$2"; shift 2 ;;
        --episodic-cache-dir) EPISODIC_CACHE_DIR="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/../.."
source .venv/bin/activate

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DURATION_TAG="${DURATION:+_${DURATION}}"
LOG_DIR=".log/eval/videomme"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_${RET_MODEL//-/_}_${RESP_MODEL//-/_}${DURATION_TAG}_$TIMESTAMP.log"

DURATION_FLAG=""
if [ -n "$DURATION" ]; then
    DURATION_FLAG="--duration $DURATION"
fi

echo -e "${BLUE}Running evaluation: Retriever=$RET_MODEL Responder=$RESP_MODEL${DURATION:+ Duration=$DURATION}${NC}"
python eval/eval.py \
    --eval-json "$EVAL_JSON" \
    --caption-dir "$CAPTION_DIR" \
    --metadata-dir "$METADATA_DIR" \
    --retriever-model "$RET_MODEL" \
    --respond-model "$RESP_MODEL" \
    --max-rounds "$MAX_ROUNDS" \
    --max-errors "$MAX_ERRORS" \
    --episodic-top-k "$EPISODIC_K" \
    --semantic-top-k "$SEMANTIC_K" \
    --visual-top-k "$VISUAL_K" \
    --output-dir "$OUTPUT_DIR" \
    --episodic-cache-dir "$EPISODIC_CACHE_DIR" \
    $DURATION_FLAG 2>&1 | tee "$LOG_FILE"

echo -e "${BLUE}Eval Done! Results: ${OUTPUT_DIR}/${RET_MODEL//-/_}_${RESP_MODEL//-/_}/videomme_eval${DURATION_TAG}.json${NC}"
