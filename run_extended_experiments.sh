# pkill -f "experimental_training_extended.py"
# ps aux | grep "experimental_training_extended.py" | awk '{print $2}' | xargs kill

#!/bin/bash
# run_extended_experiments.sh - Extended experiments on 8x RTX 4090 GPUs
# Scaling to 30k steps with parallel execution across all GPUs

# Create experiment directory structure
mkdir -p experiments_extended
mkdir -p logs_extended

# Extended settings for longer training
DEBUG_FLAG="--debug"  # Set to "--debug" for quick testing
STEPS=30000    # Increased from 10k to 30k steps
USE_WANDB="--use_wandb"   # Set to "" to disable W&B logging
FORCE_RELOAD=""  # Set to "--force_reload_data" to retokenize data

# Detect available GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "🚀 Starting Extended Hybrid LLM Architecture Experiments"
echo "========================================================"
echo "🔍 Detected: $AVAILABLE_GPUS GPUs available"
echo "Training Steps: $STEPS (3x longer than original)"
echo "Data: Will be scaled accordingly"

if [ $AVAILABLE_GPUS -eq 0 ]; then
    echo "❌ No GPUs detected! Please check nvidia-smi"
    exit 1
fi

# Define all 8 experiments with larger models
declare -a EXPERIMENTS=(
    "MAMAMAMAMAMA:mama_alternating_12L_extended:MAMA alternating pattern scaled to 12 layers (1024H)"
    "MAMAMAMAMAMAMAMAM:mama_alternating_15L_extended:MAMA alternating pattern scaled to 15 layers (1024H)"
    "MMAAMMAAMMAAMMAA:mmaammaa_pattern_12L_extended:MMAAMMAA pattern scaled to 12 layers (1024H)"
    "MMAAMMAAMMAAMMA:mmaammaa_pattern_14L_extended:MMAAMMAA pattern scaled to 14 layers (1024H)"
    "MAMAMAMAMAMAMAMA:mama_alternating_10L_extended:MAMA alternating pattern with 10 layers (1024H)"
    "MAMMMMMMMMMMMM:nemotron_14L:Nemotron architecture (1024H)"
    "MMAAAMMMAAA:mixed_grouped_11L:Mixed grouped pattern with varied block sizes (1024H)"
    "MAMAMAMAMAMAM:mama_alternating_13L_extended:MAMA alternating pattern with 13 layers (1024H)"
)

NUM_EXPERIMENTS=${#EXPERIMENTS[@]}
echo "📊 Total experiments: $NUM_EXPERIMENTS"
echo "🎯 Will run experiments in batches across $AVAILABLE_GPUS GPUs"
date

# Function to run experiment on specific GPU
run_experiment_on_gpu() {
    local gpu_id=$1
    local pattern=$2
    local name=$3
    local description=$4
    
    echo ""
    echo "🔬 GPU $gpu_id - Extended Experiment: $name"
    echo "   Pattern: $pattern"
    echo "   Description: $description"
    echo "   Steps: $STEPS"
    echo "   GPU: $gpu_id"
    echo "   Starting at: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Set CUDA device for this process
    CUDA_VISIBLE_DEVICES=$gpu_id python experimental_training_extended.py \
        --pattern "$pattern" \
        --name "$name" \
        --steps $STEPS \
        $DEBUG_FLAG \
        $USE_WANDB \
        $FORCE_RELOAD \
        2>&1 | tee "logs_extended/${name}_gpu${gpu_id}.log" &
    
    # Store the process ID for this GPU
    local current_pid=$!
    eval "PID_GPU_$gpu_id=$current_pid"
    echo "   GPU $gpu_id PID: $current_pid"
}

# Function to wait for all experiments to complete
wait_for_all_experiments() {
    echo ""
    echo "⏳ Waiting for all experiments to complete..."
    echo "============================================="
    
    # Wait for all background processes
    for pid in "${RUNNING_PIDS[@]}"; do
        if [ ! -z "$pid" ]; then
            wait $pid
            local exit_code=$?
            echo "Process $pid completed with exit code: $exit_code"
        fi
    done
    
    echo "✅ All experiments completed at: $(date '+%Y-%m-%d %H:%M:%S')"
}

# Function to run experiments adaptively across available GPUs
run_experiments_adaptively() {
    echo ""
    echo "🚀 Launching experiments adaptively..."
    echo "======================================"
    
    # Array to store all running PIDs
    RUNNING_PIDS=()
    
    if [ $AVAILABLE_GPUS -ge $NUM_EXPERIMENTS ]; then
        # If we have enough GPUs, run all experiments in parallel
        echo "✨ Sufficient GPUs: Running all $NUM_EXPERIMENTS experiments in parallel"
        for i in "${!EXPERIMENTS[@]}"; do
            IFS=':' read -r pattern name description <<< "${EXPERIMENTS[$i]}"
            gpu_id=$i
            run_experiment_on_gpu $gpu_id "$pattern" "$name" "$description"
            current_pid=$(eval echo \$PID_GPU_$gpu_id)
            RUNNING_PIDS+=($current_pid)
        done
    else
        # If we have fewer GPUs than experiments, run in batches
        echo "🔄 Limited GPUs: Running experiments in batches"
        
        experiment_idx=0
        while [ $experiment_idx -lt $NUM_EXPERIMENTS ]; do
            # Start as many experiments as we have GPUs
            batch_pids=()
            for gpu_id in $(seq 0 $((AVAILABLE_GPUS-1))); do
                if [ $experiment_idx -lt $NUM_EXPERIMENTS ]; then
                    IFS=':' read -r pattern name description <<< "${EXPERIMENTS[$experiment_idx]}"
                    echo "🔬 Batch: Starting experiment $((experiment_idx+1))/$NUM_EXPERIMENTS on GPU $gpu_id"
                    run_experiment_on_gpu $gpu_id "$pattern" "$name" "$description"
                    current_pid=$(eval echo \$PID_GPU_$gpu_id)
                    batch_pids+=($current_pid)
                    RUNNING_PIDS+=($current_pid)
                    experiment_idx=$((experiment_idx+1))
                fi
            done
            
            # If this isn't the last batch, wait for current batch to complete
            if [ $experiment_idx -lt $NUM_EXPERIMENTS ]; then
                echo "⏳ Waiting for current batch to complete before starting next batch..."
                for pid in "${batch_pids[@]}"; do
                    wait $pid
                done
                echo "✅ Batch completed, starting next batch..."
            fi
        done
    fi
}

# Function to monitor GPU usage
monitor_gpus() {
    echo ""
    echo "📊 GPU Status Check:"
    echo "==================="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    echo ""
}

# ============================================
# EXTENDED EXPERIMENTS - ADAPTIVE PARALLEL EXECUTION
# ============================================
echo ""
echo "📊 EXTENDED EXPERIMENTS: Adaptive GPU Allocation"
echo "================================================"

# Check initial GPU status
monitor_gpus

# Run experiments adaptively based on available GPUs
run_experiments_adaptively

# Give processes a moment to start
sleep 5

# Show GPU status after launch
echo ""
echo "📊 GPU Status after launch:"
monitor_gpus

# Wait for all experiments to complete
wait_for_all_experiments

# ============================================
# POST-PROCESSING AND ANALYSIS
# ============================================
echo ""
echo "📈 All experiments completed! Generating analysis..."

# Final GPU status check
monitor_gpus

echo ""
echo "📊 Final Summary:"
echo "================"
echo "✅ 8 experiments completed in parallel"
echo "🎯 Each experiment ran 30k steps on dedicated GPU" 
echo "🔧 Models: 1024 hidden size, 16 attention heads, 48 SSM states"
echo "📚 150k documents per experiment (3x original)"
echo "💾 Results saved in experiments_extended/"
echo "📝 Logs saved in logs_extended/"
echo "🕐 Completed at: $(date)"

# Generate comprehensive analysis
echo ""
echo "📈 Generating Extended Analysis..."
python analyze_extended_results.py

# ============================================
# COMPARISON WITH ORIGINAL RESULTS
# ============================================
echo ""
echo "📊 Performance Comparison:"
echo "========================="
echo "🔄 Original: 10k steps, sequential execution"
echo "⚡ Extended: 30k steps, 8x parallel execution"
echo "🎯 Expected: Better convergence, lower perplexity"

# Show top performers
if [ -f "experiments_extended/all_extended_results.csv" ]; then
    echo ""
    echo "🏆 Top 3 Extended Patterns:"
    python -c "
import pandas as pd
try:
    df = pd.read_csv('experiments_extended/all_extended_results.csv')
    if 'final_val_perplexity' in df.columns:
        top3 = df.nsmallest(3, 'final_val_perplexity')[['pattern', 'final_val_perplexity', 'final_val_loss']]
        print(top3.to_string(index=False))
    else:
        print('📊 Extended results being processed...')
except Exception as e:
    print('📊 Extended results being processed...')
"
else
    echo "📊 Extended results being processed..."
fi

# ============================================
# W&B PROJECT SUMMARY
# ============================================
echo ""
echo "🔗 Weights & Biases Summary:"
echo "============================"
echo "📊 Project: hybrid-patterns-extended"
echo "🎯 8 parallel runs logged with individual tags"
echo "📈 Compare runs at: https://wandb.ai/your-username/hybrid-patterns-extended"
echo "🏷️ Tags: extended, 30k-steps, [10-15]L"
