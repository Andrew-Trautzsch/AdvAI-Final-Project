#!/bin/bash
# ============================================================
# Traffic Routing Project - Andrew Trautzsch
# Sets up conda environment and downloads required data
# Run: bash setup.sh
# ============================================================

set +e

echo "=================================================="
echo " Traffic Routing Project Setup"
echo "=================================================="
echo ""

# -- Base directory -----------------------------------------
BASE_DIR="$HOME/TrafficRouting"
mkdir -p "$BASE_DIR/detection/images"
mkdir -p "$BASE_DIR/detection/predictions"
mkdir -p "$BASE_DIR/detection/models"
mkdir -p "$BASE_DIR/routing"
mkdir -p "$BASE_DIR/website/results"
mkdir -p "$BASE_DIR/datasets"

# -- [1/6] Load miniconda and CUDA -------------------------
echo "[1/6] Loading miniconda and CUDA..."
module load miniconda3/24.1.2-py310
module load cuda/11.8.0
echo "Done."

# -- [2/6] Create conda environment ------------------------
echo ""
echo "[2/6] Creating conda environment 'traffic_env' (Python 3.11)..."
if conda env list | grep -q "traffic_env"; then
    echo "Environment 'traffic_env' already exists, skipping creation."
else
    conda create -n traffic_env python=3.11 -y
    echo "Environment created."
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate traffic_env
echo "Activated traffic_env."

# -- [3/6] Install dependencies ----------------------------
echo ""
echo "[3/6] Installing dependencies..."

# PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# YOLO and vision tools
pip install ultralytics
pip install opencv-python-headless matplotlib Pillow tqdm

# Pin numpy for compatibility
pip install "numpy>=1.23,<2.0"

# Routing
pip install networkx

# Jupyter
pip install ipykernel jupyter_client jupyter_core platformdirs

echo "Dependencies installed."

# -- [4/6] Register Jupyter kernel -------------------------
echo ""
echo "[4/6] Registering Jupyter kernel..."
python -m ipykernel install --user --name traffic_env --display-name "Traffic Routing"

# Overwrite kernel.json to point to the correct Python binary
KERNEL_JSON="$HOME/.local/share/jupyter/kernels/traffic_env/kernel.json"
PYTHON_PATH="$HOME/.conda/envs/traffic_env/bin/python"
cat > "$KERNEL_JSON" << EOF
{
 "argv": [
  "$PYTHON_PATH",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Traffic Routing",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
EOF
echo "Kernel registered and kernel.json fixed."

# -- [5/6] Pre-download YOLO11n weights --------------------
echo ""
echo "[5/6] Pre-downloading YOLO11x weights..."
python -c "
from ultralytics import YOLO
print('Downloading yolo11x.pt...')
YOLO('yolo11x.pt')
print('YOLO11x weights ready.')
"

# -- [6/6] Pre-download VisDrone dataset -------------------
echo ""
echo "[6/6] Downloading VisDrone 2019 DET dataset (~1.5 GB)..."
python -c "
import os
from ultralytics.data.utils import check_det_dataset

print('Triggering VisDrone auto-download via ultralytics...')
print('This may take several minutes depending on network speed.')
print()

try:
    check_det_dataset('VisDrone.yaml')
    print('VisDrone dataset ready.')
except Exception as e:
    print(f'Auto-download failed: {e}')
    print('You can trigger the download manually by running Cell 4 in detection.ipynb.')
"

# -- Summary -----------------------------------------------
echo ""
echo "=================================================="
echo " Setup Complete!"
echo "=================================================="
echo ""
echo " Environment   : traffic_env"
echo " Jupyter       : kernel 'Traffic Routing'"
echo ""
echo " Folder structure:"
echo "   ~/TrafficRouting/"
echo "     detection/images/       - Place aerial test images here (img0.jpg, img1.jpg...)"
echo "     detection/predictions/  - Saved output PNGs from notebook"
echo "     detection/models/       - Saved .pt checkpoints"
echo "     routing/                - Routing notebook (coming soon)"
echo "     website/                - Demo site"
echo ""
echo " Next steps:"
echo "   1. Add aerial images to ~/TrafficRouting/detection/images/"
echo "   2. Open detection.ipynb with the 'Traffic Routing' kernel"
echo "   3. Run cells top to bottom"
echo ""