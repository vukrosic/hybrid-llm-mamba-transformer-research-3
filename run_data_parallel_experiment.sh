#!/bin/bash
# run_data_parallel_experiment.sh - Single experiment with data parallelism on 2x RTX 4090

echo "🚀 Starting Data Parallel Training: AMAMAMAMAMAMAM 16L on 2x RTX 4090"
echo "================================================================"

# Check if we have at least 2 GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $AVAILABLE_GPUS -lt 2 ]; then
    echo "❌ Need at least 2 GPUs for data parallelism! Found: $AVAILABLE_GPUS"
    exit 1
fi

echo "✅ Found $AVAILABLE_GPUS GPUs - using first 2 for data parallelism"

# Create experiment directory
mkdir -p experiments_extended
mkdir -p logs_extended

# Configuration
EXPERIMENT_NAME="amama_16L_data_parallel"
PATTERN="AMAMAMAMAMAMAM"
STEPS=30000
USE_WANDB="--use_wandb"
DEBUG_FLAG=""

echo "�� Experiment: $EXPERIMENT_NAME"
echo "📊 Pattern: $PATTERN (16 layers)"
echo "⏱️ Steps: $STEPS"
echo "🚀 GPUs: 0,1 (Data Parallel)"
echo "📦 Batch: 16 per GPU (32 effective)"
echo "🔄 Gradient Accumulation: 2 steps"

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# Launch with torchrun for distributed training
echo ""
echo "🚀 Launching distributed training..."
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    experimental_training_extended.py \
    --pattern "$PATTERN" \
    --name "$EXPERIMENT_NAME" \
    --steps $STEPS \
    $DEBUG_FLAG \
    $USE_WANDB \
    2>&1 | tee "logs_extended/${EXPERIMENT_NAME}_distributed.log"

echo ""
echo "✅ Training completed!"
echo "📊 Results saved in: experiments_extended/$EXPERIMENT_NAME"
echo "📝 Logs saved in: logs_extended/${EXPERIMENT_NAME}_distributed.log"
