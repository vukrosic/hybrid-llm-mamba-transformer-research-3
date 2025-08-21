#!/bin/bash
# run_extended_experiments.sh - Extended experiments on 8x RTX 4090 GPUs
# Scaling to 30k steps with parallel execution across all GPUs

# Create experiment directory structure
mkdir -p experiments_extended
mkdir -p logs_extended

# Extended settings for longer training
DEBUG_FLAG=""  # Set to "--debug" for quick testing
STEPS=30000    # Increased from 10k to 30k steps
USE_WANDB="--use_wandb"   # Set to "" to disable W&B logging
FORCE_RELOAD=""  # Set to "--force_reload_data" to retokenize data

echo "🚀 Starting Extended Hybrid LLM Architecture Experiments"
echo "========================================================"
echo "🎯 Running 8 experiments in PARALLEL on 8x RTX 4090 GPUs"
echo "Training Steps: $STEPS (3x longer than original)"
echo "Data: Will be scaled accordingly"
echo "GPUs: 0,1,2,3,4,5,6,7 (one experiment per GPU)"
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
    eval "PID_GPU_$gpu_id=$!"
    echo "   GPU $gpu_id PID: $(eval echo \$PID_GPU_$gpu_id)"
}

# Function to wait for all experiments to complete
wait_for_all_experiments() {
    echo ""
    echo "⏳ Waiting for all 8 experiments to complete..."
    echo "================================================"
    
    local all_pids=""
    for gpu in {0..7}; do
        local pid_var="PID_GPU_$gpu"
        local pid=$(eval echo \$$pid_var)
        if [ ! -z "$pid" ]; then
            all_pids="$all_pids $pid"
        fi
    done
    
    # Wait for all background processes
    for pid in $all_pids; do
        wait $pid
        local exit_code=$?
        echo "Process $pid completed with exit code: $exit_code"
    done
    
    echo "✅ All experiments completed at: $(date '+%Y-%m-%d %H:%M:%S')"
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
# EXTENDED EXPERIMENTS - PARALLEL EXECUTION ON 8 GPUs
# ============================================
echo ""
echo "📊 EXTENDED EXPERIMENTS: 8 Patterns Running in Parallel"
echo "======================================================="

# Check initial GPU status
monitor_gpus

# Launch all 8 experiments in parallel, one per GPU
echo "🚀 Launching all experiments..."

# GPU 0: MAMA alternating scaled to 12 layers
run_experiment_on_gpu 0 "MAMAMAMAMAMA" "mama_alternating_12L_extended" \
    "MAMA alternating pattern scaled to 12 layers"

# GPU 1: MAMA alternating scaled to 15 layers (longest)
run_experiment_on_gpu 1 "MAMAMAMAMAMAMAMAM" "mama_alternating_15L_extended" \
    "MAMA alternating pattern scaled to 15 layers"

# GPU 2: MMAAMMAA pattern scaled to 12 layers
run_experiment_on_gpu 2 "MMAAMMAAMMAAMMAA" "mmaammaa_pattern_12L_extended" \
    "MMAAMMAA pattern scaled to 12 layers"

# GPU 3: MMAAMMAA pattern scaled to 14 layers
run_experiment_on_gpu 3 "MMAAMMAAMMAAMMA" "mmaammaa_pattern_14L_extended" \
    "MMAAMMAA pattern scaled to 14 layers"

# GPU 4: MAMA alternating with 10 layers
run_experiment_on_gpu 4 "MAMAMAMAMAMAMAMA" "mama_alternating_10L_extended" \
    "MAMA alternating pattern with 10 layers"

# GPU 5: Grouped pattern - More separated M and A blocks
run_experiment_on_gpu 5 "MMMMAAAAAA" "grouped_separated_10L" \
    "Grouped pattern with separated M and A blocks"

# GPU 6: Mixed grouped pattern with varied block sizes
run_experiment_on_gpu 6 "MMAAAMMMAAA" "mixed_grouped_11L" \
    "Mixed grouped pattern with varied block sizes"

# GPU 7: Long MAMA with 13 layers
run_experiment_on_gpu 7 "MAMAMAMAMAMAM" "mama_alternating_13L_extended" \
    "MAMA alternating pattern with 13 layers"

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
