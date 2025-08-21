#!/bin/bash
# run_experiments.sh - First comprehensive experiment suite for hybrid LLM

# Create experiment directory structure
mkdir -p experiments
mkdir -p logs

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=0

# Common settings for all experiments
DEBUG_FLAG=""  # Set to "--debug" for quick testing
STEPS=10000    # Number of training steps
USE_WANDB=""   # Set to "--use_wandb" to enable W&B logging

echo "🚀 Starting Hybrid LLM Architecture Experiments"
echo "================================================"
date

# Function to run experiment and log output
run_experiment() {
    local pattern=$1
    local name=$2
    local description=$3
    
    echo ""
    echo "🔬 Experiment: $name"
    echo "   Pattern: $pattern"
    echo "   Description: $description"
    echo "   Starting at: $(date '+%Y-%m-%d %H:%M:%S')"
    
    python experiment_patterns.py \
        --pattern "$pattern" \
        --name "$name" \
        --steps $STEPS \
        $DEBUG_FLAG \
        $USE_WANDB \
        2>&1 | tee "logs/${name}.log"
    
    echo "   Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
}

# ============================================
# PHASE 1: BASELINE ARCHITECTURES
# ============================================
echo ""
echo "📊 PHASE 1: Baseline Architectures"
echo "-----------------------------------"

# Pure Attention baseline (Transformer)
run_experiment "AAAAAAAA" "baseline_attention_8L" \
    "Pure 8-layer Transformer baseline"

# Pure SSM baseline (Mamba)
run_experiment "MMMMMMMM" "baseline_mamba_8L" \
    "Pure 8-layer Mamba/SSM baseline"

# ============================================
# PHASE 2: SIMPLE ALTERNATING PATTERNS
# ============================================
echo ""
echo "📊 PHASE 2: Alternating Patterns"
echo "--------------------------------"

# Classic alternation
run_experiment "AMAMAMAM" "alternate_A_first_8L" \
    "Attention-first alternating pattern"

run_experiment "MAMAMAMA" "alternate_M_first_8L" \
    "Mamba-first alternating pattern"

# Double alternation
run_experiment "AAMMAAMMAA" "double_alternate_10L" \
    "Double alternating pattern (AA-MM)"

# ============================================
# PHASE 3: STRUCTURED PATTERNS
# ============================================
echo ""
echo "📊 PHASE 3: Structured Patterns"
echo "-------------------------------"

# Sandwich patterns (Attention at boundaries)
run_experiment "AMMMMMMA" "sandwich_8L" \
    "Sandwich pattern - Attention at start/end"

run_experiment "AAMMMMAA" "thick_sandwich_8L" \
    "Thick sandwich - 2 Attention layers at boundaries"

# Grouped patterns
run_experiment "MMMMAAAA" "grouped_M4A4" \
    "Grouped pattern - 4 Mamba then 4 Attention"

run_experiment "AAAAMMMM" "grouped_A4M4" \
    "Grouped pattern - 4 Attention then 4 Mamba"

# ============================================
# PHASE 4: INSPIRED BY RESEARCH
# ============================================
echo ""
echo "📊 PHASE 4: Research-Inspired Patterns"
echo "--------------------------------------"

# Increasing complexity pattern (more attention as we go deeper)
run_experiment "MMMMAAA" "increasing_attention_7L" \
    "Increasing attention ratio with depth"

# U-Net style (symmetric)
run_experiment "AMMMMMA" "unet_style_7L" \
    "U-Net inspired symmetric pattern"

# Fibonacci-inspired (golden ratio approximation)
run_experiment "MAMMAMMMA" "fibonacci_9L" \
    "Fibonacci-inspired pattern"

# ============================================
# PHASE 5: SCALE TESTING (Optional)
# ============================================
# if [[ "$1" == "--include-scale" ]]; then
#     echo ""
#     echo "📊 PHASE 5: Scale Testing"
#     echo "------------------------"
    
#     # Best pattern from above at different scales
#     BEST_PATTERN="AMAMAMAM"  # Update based on results
    
#     # 4 layers
#     run_experiment "AMAM" "scale_test_4L" \
#         "4-layer version of best pattern"
    
#     # 12 layers
#     run_experiment "AMAMAMAMAMA" "scale_test_12L" \
#         "12-layer version of best pattern"
    
#     # 16 layers
#     run_experiment "AMAMAMAMAMAMAMAM" "scale_test_16L" \
#         "16-layer version of best pattern"
# fi

# ============================================
# RESULTS ANALYSIS
# ============================================
echo ""
echo "📈 Generating Analysis..."
python analyze_results.py

echo ""
echo "✅ All experiments completed!"
echo "Results saved in experiments/"
echo "Logs saved in logs/"
echo "Completed at: $(date)"

# ============================================
# QUICK SUMMARY
# ============================================
echo ""
echo "📊 Quick Summary of Results:"
echo "============================"

# Find best performing model
if [ -f "experiments/all_results.csv" ]; then
    echo "Top 5 patterns by validation perplexity:"
    python -c "
import pandas as pd
df = pd.read_csv('experiments/all_results.csv')
if 'final_val_perplexity' in df.columns:
    top5 = df.nsmallest(5, 'final_val_perplexity')[['pattern', 'final_val_perplexity']]
    print(top5.to_string(index=False))
"
fi

# Optional: Send notification (uncomment and configure as needed)
# echo "Experiments completed" | mail -s "Hybrid LLM Experiments Done" your@email.com
# curl -X POST https://your-webhook-url -d "Experiments completed"