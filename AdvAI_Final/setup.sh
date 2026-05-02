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
# Change this to match your OSC path if needed
BASE_DIR="$HOME/AdvAI_Final"
mkdir -p "$BASE_DIR/detection/images"
mkdir -p "$BASE_DIR/detection/predictions"
mkdir -p "$BASE_DIR/detection/models"
mkdir -p "$BASE_DIR/routing/predictions"
mkdir -p "$BASE_DIR/website/results"
mkdir -p "$BASE_DIR/datasets/VisDrone"

# -- [1/7] Load miniconda and CUDA -------------------------
echo "[1/7] Loading miniconda and CUDA..."
module load miniconda3/24.1.2-py310
module load cuda/11.8.0
echo "Done."

# -- [2/7] Create conda environment ------------------------
echo ""
echo "[2/7] Creating conda environment 'traffic_env' (Python 3.11)..."
if conda env list | grep -q "traffic_env"; then
    echo "Environment 'traffic_env' already exists, skipping creation."
else
    conda create -n traffic_env python=3.11 -y
    echo "Environment created."
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate traffic_env
echo "Activated traffic_env."

# -- [3/7] Install core dependencies -----------------------
echo ""
echo "[3/7] Installing core dependencies..."

# PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# YOLO and vision tools
pip install ultralytics
pip install opencv-python-headless matplotlib Pillow tqdm

# Pin numpy for compatibility
pip install "numpy>=1.23,<2.0"

# Graph and routing
pip install networkx scikit-learn scipy

# gdown for Google Drive downloads
pip install gdown

# Jupyter
pip install ipykernel jupyter_client jupyter_core platformdirs

echo "Core dependencies installed."

# -- [4/7] Install PyTorch Geometric -----------------------
echo ""
echo "[4/7] Installing PyTorch Geometric..."

# PyG requires matching the exact torch and CUDA version
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html 2>/dev/null || \
    echo "torch-scatter/sparse optional extras failed, continuing."

pip install torch-geometric

echo "PyTorch Geometric installed."

# -- [5/7] Register Jupyter kernel -------------------------
echo ""
echo "[5/7] Registering Jupyter kernel..."
python -m ipykernel install --user --name traffic_env --display-name "Traffic Routing"

# Overwrite kernel.json to guarantee correct Python binary
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

# -- [6/7] Pre-download YOLO11x weights --------------------
echo ""
echo "[6/7] Pre-downloading YOLO11x weights..."
python -c "
from ultralytics import YOLO
print('Downloading yolo11x.pt...')
YOLO('yolo11x.pt')
print('YOLO11x weights ready.')
"

# -- [7/7] Download VisDrone DET dataset -------------------
echo ""
echo "[7/8] Downloading VisDrone 2019 DET dataset (~2.3 GB)..."
python -c "
import os
from ultralytics.data.utils import check_det_dataset

print('Triggering VisDrone DET auto-download via ultralytics...')
print('This may take several minutes depending on network speed.')
print()

try:
    check_det_dataset('VisDrone.yaml')
    print('VisDrone DET dataset ready.')
except Exception as e:
    print(f'Auto-download failed: {e}')
    print('You can trigger the download manually by running Cell 4 in detection.ipynb.')
"

# -- [8/8] VisDrone VID dataset note ----------------------
echo ""
echo "[8/8] VisDrone VID dataset..."
echo "  The VID dataset Google Drive links are frequently quota-limited."
echo "  This project uses VisDrone DET sequential frames instead."
echo "  DET frames are grouped by sequence ID from their filenames"
echo "  (e.g. 0000137_02220_d_0000163.jpg = sequence 0000137, frame 02220)"
echo "  and interpolated to produce a continuous congestion time series."
echo "  No additional download required."

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
echo "   $BASE_DIR/"
echo "     detection/images/       - Place aerial test images here"
echo "     detection/predictions/  - Saved output PNGs from detection notebook"
echo "     detection/models/       - Saved .pt checkpoints"
echo "     routing/predictions/    - Saved output PNGs from routing notebook"
echo "     website/results/        - Website assets"
echo "     datasets/VisDrone/      - VisDrone DET + VID datasets"
echo ""
echo " Next steps:"
echo "   1. Run detection.ipynb to verify YOLO setup"
echo "   2. Run data_prep.ipynb to extract congestion time series from VID"
echo "   3. Run gnn_routing.ipynb to train and evaluate the GNN"
echo ""