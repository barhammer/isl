#!/bin/bash

echo "🚀 Setting up ISL environment..."

# ----------------------------------------
# 🔹 Check conda
# ----------------------------------------
if ! command -v conda &> /dev/null
then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit
fi

# ----------------------------------------
# 🔹 Create environment
# ----------------------------------------
ENV_NAME="isl"

echo "📦 Creating conda environment: $ENV_NAME"

conda create -n $ENV_NAME python=3.10 -y

# ----------------------------------------
# 🔹 Activate environment
# ----------------------------------------
echo "🔧 Activating environment..."

# Required for scripts
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# ----------------------------------------
# 🔹 Install dependencies
# ----------------------------------------
echo "📥 Installing dependencies..."

pip install --upgrade pip

pip install torch torchvision
pip install opencv-python mediapipe numpy

# ----------------------------------------
# 🔹 Verify install
# ----------------------------------------
echo "🧪 Verifying installation..."

python - <<EOF
import torch, cv2, mediapipe, numpy
print("✅ All core libraries imported successfully")
print("Torch device:", "CUDA" if torch.cuda.is_available() else "CPU")
EOF

# ----------------------------------------
# 🔹 Done
# ----------------------------------------
echo ""
echo "✅ Setup complete!"
echo "👉 Run the project with:"
echo "   conda activate $ENV_NAME"
echo "   python main.py"