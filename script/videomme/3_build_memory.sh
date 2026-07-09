#!/bin/bash
# WorldMM Memory Construction Script
# Usage: ./script/videomme/3_build_memory.sh [--step episodic|semantic|visual|all] [--gpu 0,1,2,3] [--model gpt-5-mini]
# Adjust paths for video, transcript, and output for different datasets or directory structures.

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

VIDEO_PATH="data/Video-MME/data" TRANSCRIPT_PATH="data/Video-MME/transcript" CAPTION_PATH="data/Video-MME/caption"
STEP="all" GPU_LIST="0,1,2,3" MODEL="gpt-5-mini" UNIT_TIME=10 NUM_FRAMES=10

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift 2 ;;
        --gpu) GPU_LIST="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --frames) NUM_FRAMES="$2"; shift 2 ;;
        --video-path) VIDEO_PATH="$2"; shift 2 ;;
        --transcript-path) TRANSCRIPT_PATH="$2"; shift 2 ;;
        --caption-path) CAPTION_PATH="$2"; shift 2 ;;
        --unit-time) UNIT_TIME="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/../.."
mkdir -p output/metadata/videomme/{episodic,semantic,visual}_memory

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/build_memory/videomme"
mkdir -p "$LOG_DIR"

run_episodic() {
    echo -e "${BLUE}Episodic Memory: Generating fine captions...${NC}"
    python preprocess/episodic_memory/generate_fine_caption.py \
        --video-path "$VIDEO_PATH" \
        --transcript-path "$TRANSCRIPT_PATH" \
        --output-path "$CAPTION_PATH" \
        --model "$MODEL" \
        --unit-time "$UNIT_TIME" 2>&1 | tee "$LOG_DIR/generate_fine_caption_$TIMESTAMP.log"
    echo -e "${BLUE}Episodic Memory: Generating multiscale memory...${NC}"
    python -m worldmm.memory.episodic.multiscale \
        --caption_dir "$CAPTION_PATH" \
        --model "$MODEL" \
        --windows "30,180,600" \
        --granularity_names "30sec,3min,10min" \
        --perspective general 2>&1 | tee "$LOG_DIR/multiscale_memory_$TIMESTAMP.log"
    echo -e "${BLUE}Episodic Memory: Extracting triples...${NC}"
    python preprocess/build_memory.py \
        --caption-dir "$CAPTION_PATH" \
        --output-dir "output/metadata/videomme" \
        --model "$MODEL" \
        --step episodic 2>&1 | tee "$LOG_DIR/episodic_triples_$TIMESTAMP.log"
}

run_semantic() {
    echo -e "${BLUE}Semantic Memory: Extracting triples...${NC}"
    CUDA_VISIBLE_DEVICES="${GPU_LIST%%,*}" python preprocess/build_memory.py \
        --caption-dir "$CAPTION_PATH" \
        --output-dir "output/metadata/videomme" \
        --model "$MODEL" \
        --step semantic 2>&1 | tee "$LOG_DIR/semantic_memory_$TIMESTAMP.log"
}

run_visual() {
    echo -e "${BLUE}Visual Memory: Extracting features...${NC}"
    python preprocess/build_memory.py \
        --caption-dir "$CAPTION_PATH" \
        --output-dir "output/metadata/videomme" \
        --model "$MODEL" \
        --step visual \
        --gpu "$GPU_LIST" \
        --num-frames "$NUM_FRAMES" 2>&1 | tee "$LOG_DIR/visual_features_$TIMESTAMP.log"
}

case $STEP in
    all) run_episodic; run_semantic; run_visual ;;
    episodic) run_episodic ;;
    semantic) run_semantic ;;
    visual) run_visual ;;
    *) echo "Invalid step: $STEP"; exit 1 ;;
esac

echo -e "${BLUE}Build Memory Done! Output: output/metadata/videomme/*_memory/${NC}"
