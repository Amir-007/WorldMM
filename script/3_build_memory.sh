#!/bin/bash
# WorldMM Memory Construction Script
# Usage: ./script/3_build_memory.sh [--step episodic|semantic|visual|all] [--person <person>] [--gpu 0,1,2,3] [--model gpt-5-mini]

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

PERSON="A1_JAKE" STEP="all" GPU_LIST="0,1,2,3" MODEL="gpt-5-mini" NUM_FRAMES=16

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift 2 ;;
        --person) PERSON="$2"; shift 2 ;;
        --gpu) GPU_LIST="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --frames) NUM_FRAMES="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
mkdir -p output/metadata/{episodic,semantic,visual}_memory/${PERSON}

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/build_memory/egolife/${PERSON}"
mkdir -p "$LOG_DIR"

run_episodic() {
    echo -e "${BLUE}Episodic Memory: Generating fine captions...${NC}"
    python preprocess/episodic_memory/generate_fine_caption_egolife.py \
        --person "$PERSON" \
        --sync-dir "data/EgoLife/EgoLifeCap/Sync" \
        --output "data/EgoLife/EgoLifeCap/${PERSON}/${PERSON}_30sec.json" 2>&1 | tee "$LOG_DIR/generate_fine_caption_$TIMESTAMP.log"
    echo -e "${BLUE}Episodic Memory: Generating multiscale memory...${NC}"
    python -m worldmm.memory.episodic.multiscale \
        --caption_dir "data/EgoLife/EgoLifeCap/${PERSON}" \
        --model "$MODEL" \
        --base_name "${PERSON}_30sec.json" \
        --windows "180,600,3600" \
        --granularity_names "${PERSON}_3min,${PERSON}_10min,${PERSON}_1h" \
        --perspective egocentric 2>&1 | tee "$LOG_DIR/multiscale_memory_$TIMESTAMP.log"
    echo -e "${BLUE}Episodic Memory: Extracting triples...${NC}"
    python preprocess/episodic_memory/extract_episodic_triples.py \
        --caption-file "data/EgoLife/EgoLifeCap/${PERSON}/${PERSON}_30sec.json" \
        --output-dir "output/metadata/episodic_memory/${PERSON}" \
        --model "$MODEL" 2>&1 | tee "$LOG_DIR/episodic_triples_$TIMESTAMP.log"
}

run_semantic() {
    echo -e "${BLUE}Semantic Memory: Extracting triples...${NC}"
    python preprocess/semantic_memory/extract_semantic_triples.py \
        --caption-file "data/EgoLife/EgoLifeCap/${PERSON}/${PERSON}_30sec.json" \
        --openie-file "output/metadata/episodic_memory/${PERSON}/openie_results_${MODEL}.json" \
        --output-dir "output/metadata/semantic_memory/${PERSON}" \
        --model "$MODEL" 2>&1 | tee "$LOG_DIR/semantic_extraction_$TIMESTAMP.log"
    echo -e "${BLUE}Semantic Memory: Consolidating...${NC}"
    CUDA_VISIBLE_DEVICES="${GPU_LIST%%,*}" python preprocess/semantic_memory/consolidate_semantic_memory.py \
        --semantic-file "output/metadata/semantic_memory/${PERSON}/semantic_extraction_results_${MODEL}.json" \
        --output-dir "output/metadata/semantic_memory/${PERSON}" \
        --model "$MODEL" 2>&1 | tee "$LOG_DIR/semantic_consolidation_$TIMESTAMP.log"
}

run_visual() {
    echo -e "${BLUE}Visual Memory: Extracting features...${NC}"
    bash preprocess/visual_memory/extract_visual_features.sh \
        --person "$PERSON" --gpu "$GPU_LIST" --num_frames "$NUM_FRAMES" 2>&1 | tee "$LOG_DIR/visual_features_$TIMESTAMP.log"
}

case $STEP in
    all) run_episodic; run_semantic; run_visual ;;
    episodic) run_episodic ;;
    semantic) run_semantic ;;
    visual) run_visual ;;
    *) echo "Invalid step: $STEP"; exit 1 ;;
esac

echo -e "${BLUE}Build Memory Done! Output: output/metadata/*_memory/${PERSON}/${NC}"
