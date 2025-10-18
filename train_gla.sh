#!/bin/bash
# Training script for GLA Transformer

# Default parameters
DATASET="hls4ml"
DATA_DIR=""
SAVE_DIR="./gla_results"
HIDDEN_SIZE=128
NUM_LAYERS=6
NUM_HEADS=8
BATCH_SIZE=64
NUM_EPOCHS=100
LEARNING_RATE=1e-3
DEVICE="auto"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --hidden_size)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --num_heads)
            NUM_HEADS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET       Dataset type [hls4ml, jetclass, top, QG] (default: hls4ml)"
            echo "  --data_dir DIR          Path to data directory (required)"
            echo "  --save_dir DIR          Output directory (default: ./gla_results)"
            echo "  --hidden_size SIZE      Hidden dimension (default: 128)"
            echo "  --num_layers LAYERS     Number of layers (default: 6)"
            echo "  --num_heads HEADS       Number of attention heads (default: 8)"
            echo "  --batch_size BATCH      Batch size (default: 64)"
            echo "  --num_epochs EPOCHS     Number of epochs (default: 100)"
            echo "  --learning_rate LR      Learning rate (default: 1e-3)"
            echo "  --device DEVICE         Device [cuda, cpu, auto] (default: auto)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dataset hls4ml --data_dir ./data/hls4ml"
            echo "  $0 --dataset jetclass --data_dir ./data/jetclass --batch_size 32 --num_epochs 200"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if data directory is provided
if [ -z "$DATA_DIR" ]; then
    echo "Error: --data_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Print configuration
echo "GLA Transformer Training Configuration:"
echo "======================================"
echo "Dataset: $DATASET"
echo "Data directory: $DATA_DIR"
echo "Save directory: $SAVE_DIR"
echo "Hidden size: $HIDDEN_SIZE"
echo "Number of layers: $NUM_LAYERS"
echo "Number of heads: $NUM_HEADS"
echo "Batch size: $BATCH_SIZE"
echo "Number of epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Device: $DEVICE"
echo ""

# Create save directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Set dataset-specific parameters
case $DATASET in
    "hls4ml")
        NUM_PARTICLES=128
        SORT_BY="pt"
        ;;
    "jetclass")
        NUM_PARTICLES=150
        SORT_BY="pt"
        ;;
    "top")
        NUM_PARTICLES=200
        SORT_BY="pt"
        ;;
    "QG")
        NUM_PARTICLES=150
        SORT_BY="pt"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Supported datasets: hls4ml, jetclass, top, QG"
        exit 1
        ;;
esac

echo "Starting training with $NUM_PARTICLES particles..."
echo ""

# Run training script
python train_gla_pytorch.py \
    --dataset "$DATASET" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --num_particles "$NUM_PARTICLES" \
    --sort_by "$SORT_BY" \
    --hidden_size "$HIDDEN_SIZE" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --device "$DEVICE" \
    --use_scheduler \
    --scheduler_type "cosine" \
    --early_stopping_patience 20 \
    --gradient_clip_norm 1.0 \
    --weight_decay 1e-4 \
    --dropout 0.1 \
    --pooling_type "attention" \
    --use_short_conv \
    --conv_size 4 \
    --seed 42

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Results saved to: $SAVE_DIR"
    echo ""
    echo "To view results:"
    echo "  - Training curves: $SAVE_DIR/*/training_curves.png"
    echo "  - ROC curves: $SAVE_DIR/*/roc_curves.png"
    echo "  - Model weights: $SAVE_DIR/*/best_model.pth"
    echo "  - Training log: $SAVE_DIR/*/train.log"
else
    echo ""
    echo "Training failed! Check the error messages above."
    exit 1
fi