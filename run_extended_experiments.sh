#!/bin/bash
# run_extended_experiments.sh - Extended experiments based on best performers
# Scaling to 30k steps with increased data for better convergence

# Create experiment directory structure
mkdir -p experiments_extended
mkdir -p logs_extended

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=0

# Extended settings for longer training
DEBUG_FLAG=""  # Set to "--debug" for quick testing
STEPS=30000    # Increased from 10k to 30k steps
USE_WANDB="--use_wandb"   # Set to "" to disable W&B logging
FORCE_RELOAD=""  # Set to "--force_reload_data" to retokenize data

echo "🚀 Starting Extended Hybrid LLM Architecture Experiments"
echo "========================================================"
echo "Training Steps: $STEPS (3x longer than original)"
echo "Data: Will be scaled accordingly"
date

# Function to run experiment and log output
run_experiment() {
    local pattern=$1
    local name=$2
    local description=$3
    
    echo ""
    echo "🔬 Extended Experiment: $name"
    echo "   Pattern: $pattern"
    echo "   Description: $description"
    echo "   Steps: $STEPS"
    echo "   Starting at: $(date '+%Y-%m-%d %H:%M:%S')"
    
    python experimental_training_extended.py \
        --pattern "$pattern" \
        --name "$name" \
        --steps $STEPS \
        $DEBUG_FLAG \
        $USE_WANDB \
        $FORCE_RELOAD \
        2>&1 | tee "logs_extended/${name}.log"
    
    echo "   Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
}

# ============================================
# EXTENDED EXPERIMENTS - BASED ON BEST PERFORMERS
# ============================================
echo ""
echo "📊 EXTENDED EXPERIMENTS: Variations of Top Performers"
echo "====================================================="

# 1. MAMA alternating scaled to 12 layers
run_experiment "MAMAMAMAMAMA" "mama_alternating_12L_extended" \
    "MAMA alternating pattern scaled to 12 layers"

# 2. MAMA alternating scaled to 15 layers (longest)
run_experiment "MAMAMAMAMAMAMAMAM" "mama_alternating_15L_extended" \
    "MAMA alternating pattern scaled to 15 layers"

# 3. MMAAMMAA pattern scaled to 12 layers
run_experiment "MMAAMMAAMMAAMMAA" "mmaammaa_pattern_12L_extended" \
    "MMAAMMAA pattern scaled to 12 layers"

# 4. MMAAMMAA pattern scaled to 14 layers
run_experiment "MMAAMMAAMMAAMMA" "mmaammaa_pattern_14L_extended" \
    "MMAAMMAA pattern scaled to 14 layers"

# 5. MAMA alternating with 10 layers
run_experiment "MAMAMAMAMAMA" "mama_alternating_10L_extended" \
    "MAMA alternating pattern with 10 layers"

# 6. Grouped pattern: MMMMAAAAAA (10L) - More separated M and A blocks
run_experiment "MMMMAAAAAA" "grouped_separated_10L" \
    "Grouped pattern with separated M and A blocks"

# 7. Mixed grouped: MMAAAMMMAAA (11L) - Mixed grouping
run_experiment "MMAAAMMMAAA" "mixed_grouped_11L" \
    "Mixed grouped pattern with varied block sizes"

# 8. Long MAMA with 13 layers
run_experiment "MAMAMAMAMAMAM" "mama_alternating_13L_extended" \
    "MAMA alternating pattern with 13 layers"

# ============================================
# RESULTS ANALYSIS
# ============================================
echo ""
echo "📈 Generating Extended Analysis..."
python analyze_extended_results.py

echo ""
echo "✅ All extended experiments completed!"
echo "Results saved in experiments_extended/"
echo "Logs saved in logs_extended/"
echo "Completed at: $(date)"

# ============================================
# COMPARISON WITH ORIGINAL RESULTS
# ============================================
echo ""
echo "📊 Comparison Summary:"
echo "===================="
echo "Original experiments: 10k steps"
echo "Extended experiments: 30k steps (3x longer)"
echo "Expected improvements: Better convergence, lower perplexity"

# Find best performing model from extended runs
if [ -f "experiments_extended/all_results.csv" ]; then
    echo ""
    echo "Top 3 extended patterns by validation perplexity:"
    python -c "
import pandas as pd
try:
    df = pd.read_csv('experiments_extended/all_results.csv')
    if 'final_val_perplexity' in df.columns:
        top3 = df.nsmallest(3, 'final_val_perplexity')[['pattern', 'final_val_perplexity', 'final_val_loss']]
        print(top3.to_string(index=False))
    else:
        print('Extended results not yet available')
except:
    print('Extended results not yet available')
"
fi
