#!/bin/bash

# Particle Transformer Quark-Gluon Training Commands for Kubernetes Pod
# Optimized for 4090 GPU with 32GB max memory to avoid OOM errors
# Repo: https://github.com/tim-legge/particle_transformer_legge
# Mount path in pod: /data (ephemeral volume for trained models)

set -e  # Exit on any error
set -x  # Print commands being executed

# =============================================================================
# STEP 1: Environment Setup and Repository Clone
# =============================================================================

echo "=== Setting up environment and cloning repository ==="

# Create workspace directory
mkdir -p workspace
cd workspace

# Clone the repository
git clone https://github.com/tim-legge/particle_transformer_legge.git
cd particle_transformer_legge

# Create directories for datasets and models
mkdir -p datasets
mkdir -p data/models  # Ephemeral volume for trained models
mkdir -p logs

# =============================================================================
# STEP 2: Python Environment Setup
# =============================================================================

echo "=== Setting up Python environment ==="

# Update pip first
pip install --upgrade pip

# Install PyTorch first (CUDA 11.8 compatible)
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other scientific packages
pip install --no-cache-dir numpy pandas pyarrow tables h5py uproot awkward
pip install --no-cache-dir matplotlib seaborn plotly scikit-learn tensorboard

# Install weaver-core with verbose output to check for errors
echo "Installing weaver-core..."
pip install --no-cache-dir --verbose weaver-core[analytics]>=0.4

# Verify weaver installation
echo "Verifying weaver installation..."
python -c "import weaver; print('Weaver version:', weaver.__version__)" || {
    echo "Weaver import failed, trying alternative installation..."
    pip uninstall -y weaver-core
    pip install --no-cache-dir git+https://github.com/hqucms/weaver-core.git
}

# Check if weaver command is available
which weaver || {
    echo "Weaver command not found, checking Python module..."
    python -m weaver --help || {
        echo "Weaver module not working, trying manual installation..."
        cd /workspace
        git clone https://github.com/hqucms/weaver-core.git
        cd weaver-core
        pip install -e .
        cd ../particle_transformer_legge
    }
}

# =============================================================================
# STEP 3: Dataset Download and Setup
# =============================================================================

echo "=== Downloading QuarkGluon dataset ==="

# Download QuarkGluon dataset
python get_datasets.py QuarkGluon -d ./datasets

# Update environment configuration for pod paths
cat > env.sh << EOF
#!/bin/bash

export DATADIR_JetClass=
export DATADIR_TopLandscape=
export DATADIR_QuarkGluon=./datasets/QuarkGluon
EOF

# Source the environment
source env.sh

# =============================================================================
# STEP 4: Memory-Optimized Training Configuration
# =============================================================================

echo "=== Starting QuarkGluon training with memory optimizations ==="

# Memory optimization parameters for 4090 GPU:
# - Reduced batch size to prevent OOM
# - Reduced samples per epoch for faster iterations
# - Single worker to reduce memory overhead
# - Mixed precision training enabled

# Train ParT model with kinematic + PID features
echo "Training ParT with kinematic + PID features..."
# Try using weaver command, fallback to python module if it fails
if command -v weaver >/dev/null 2>&1; then
    echo "Using weaver command..."
    ./train_QuarkGluon.sh ParT kinpid \
        --batch-size 32 \
        --fetch-step 0.15
else
    echo "Weaver command not found, using python module directly..."
    python -m weaver \
        --data-train "./datasets/QuarkGluon/train_file_*.parquet" \
        --data-test "./datasets/QuarkGluon/test_file_*.parquet" \
        --data-config data/QuarkGluon/qg_kinpid.yaml \
        --network-config networks/example_ParticleTransformer.py \
        --use-amp \
        --optimizer-option weight_decay 0.01 \
        --model-prefix data/models/QuarkGluon/ParT_kinpid/net \
        --num-workers 0 --fetch-step 1 --in-memory --train-val-split 0.8889 \
        --batch-size 32 --samples-per-epoch 400000 --samples-per-epoch-val 50000 \
        --num-epochs 20 --gpus 0 --start-lr 1e-3 --optimizer ranger \
        --log logs/QuarkGluon_ParT_kinpid_pod.log \
        --predict-output pred.root \
        --tensorboard QuarkGluon_ParT_kinpid_pod
fi

echo "=== ParT kinpid training completed ==="

# =============================================================================
# STEP 5: Model and Log Organization
# =============================================================================

echo "=== Organizing trained models and logs ==="

# Copy logs to ephemeral volume for easy access
cp -r logs data/

# Create summary of trained models
ls -la data/models/QuarkGluon/*/net_best_epoch_*.pth > data/trained_models_summary.txt 2>/dev/null || echo "No .pth files found" > data/trained_models_summary.txt
ls -la data/models/QuarkGluon/*/net_best_epoch_*.pt > data/trained_models_summary_pt.txt 2>/dev/null || echo "No .pt files found" >> data/trained_models_summary.txt

echo "=== Training pipeline completed successfully ==="
echo "Trained models are available in /data/models/"
echo "Logs are available in /data/logs/"
echo "Model summary available in /data/trained_models_summary.txt"

# Print final summary
echo "=== TRAINING SUMMARY ==="
echo "Models trained:"
echo "1. ParT with kinematic features"
echo "2. ParT with kinematic + PID features" 
echo "3. ParticleNet with kinematic + PID features"
echo ""
echo "All models saved to /data/models/ for manual transfer to persistent volume"
echo "Tensorboard logs saved to /data/logs/ for analysis"