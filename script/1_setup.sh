#!/bin/bash
# WorldMM Setup Script
# Usage: ./script/1_setup.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

cd "$(dirname "$0")/.."

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/setup"
mkdir -p "$LOG_DIR"

SCRATCH_BASE="/parallel_scratch/ms04938"
SCRATCH_VENV="$SCRATCH_BASE/MyEnv"
SCRATCH_DATA="$SCRATCH_BASE/data"

mkdir -p "$SCRATCH_DATA"

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo -e "${BLUE}uv could not be found, installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tee "$LOG_DIR/uv_install_$TIMESTAMP.log"
    source "$HOME/.local/bin/env" || true
fi

module load CUDA/12.2.2
export CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH

export UV_PROJECT_ENVIRONMENT="$SCRATCH_VENV"

# Set up virtual environment and install dependencies
echo -e "${BLUE}Setting up virtual environment and installing dependencies...${NC}"
MAX_JOBS=4 uv sync 2>&1 | tee "$LOG_DIR/uv_sync_$TIMESTAMP.log"

source "$SCRATCH_VENV/bin/activate"

echo -e "${BLUE}Linking WorldMM module to scratch environment...${NC}"
uv pip install -e .

export HF_HOME="$SCRATCH_DATA/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_TOKEN="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"

# Download EgoLife dataset
echo -e "${BLUE}Downloading EgoLife dataset to $SCRATCH_DATA/EgoLife...${NC}"
hf download lmms-lab/EgoLife --repo-type=dataset --local-dir "$SCRATCH_DATA/EgoLife" 2>&1 | tee "$LOG_DIR/hf_download_egolife_$TIMESTAMP.log"
unzip "$SCRATCH_DATA/EgoLife/caption.zip" -d "$SCRATCH_DATA/EgoLife" && rm "$SCRATCH_DATA/EgoLife/caption.zip"

mkdir -p data
rm -rf data/EgoLife
ln -s "$SCRATCH_DATA/EgoLife" data/EgoLife

echo -e "${BLUE}Setup Done! Dependencies installed and data downloaded.${NC}"
