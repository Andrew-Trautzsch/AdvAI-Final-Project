#!/bin/bash
# ============================================================
# Traffic Routing Demo - Launch Script
# Run: bash launch.sh
# Then open http://localhost:5000 in your browser
# ============================================================

module load miniconda3/24.1.2-py310
module load cuda/11.8.0
module load ffmpeg/6.1.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate traffic_env

# Install Flask if not already present
pip install flask --quiet

BASE_DIR="$HOME/AdvAI_Final"

echo "=================================================="
echo " Traffic Routing Demo"
echo "=================================================="
echo ""
echo " Open in browser: http://localhost:5000"
echo " Press Ctrl+C to stop"
echo ""

cd "$BASE_DIR"
python demo/app.py