#!/bin/bash
# run_data_parallel_experiment.sh - Single experiment with data parallelism on 8x RTX 4090

echo "🚀 Starting Data Parallel Training: AMAMAMAMAMAMAMAM 16L on 8x RTX 4090"
echo "================================================================"

# Check if we have at least 8 GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $AVAILABLE_GPUS -lt 8 ]; then
    echo "❌ Need at least 8 GPUs for data parallelism! Found: $AVAILABLE_GPUS"
    exit 1
fi

echo "✅ Found $AVAILABLE_GPUS GPUs - using first 8 for data parallelism"

# Create experiment directory
mkdir -p experiments_extended
mkdir -p logs_extended

# Configuration
EXPERIMENT_NAME="amama_16L_data_parallel"
PATTERN="AMAMAMAMAMAMAMAM"  # Fixed: 16 characters for 16 layers
STEPS=30000
USE_WANDB="--use_wandb"
DEBUG_FLAG=""

echo " Experiment: $EXPERIMENT_NAME"
echo "📊 Pattern: $PATTERN (16 layers)"
echo "⏱️ Steps: $STEPS"
echo "🚀 GPUs: 0-7 (Data Parallel)"
echo "📦 Batch: 16 per GPU (256 effective)"
echo "🔄 Gradient Accumulation: 2 steps"

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# Launch with torchrun for distributed training on 8 GPUs
echo ""
echo "🚀 Launching distributed training on 8 GPUs..."
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    experimental_training_extended.py \
    --pattern "$PATTERN" \
    --name "$EXPERIMENT_NAME" \
    --steps $STEPS \
    $DEBUG_FLAG \
    $USE_WANDB

echo ""
echo "✅ Training completed!"
echo "📊 Results saved in: experiments_extended/$EXPERIMENT_NAME"
echo "📝 Logs saved in: logs_extended/${EXPERIMENT_NAME}_distributed_8gpu.log"
