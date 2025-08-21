#!/bin/bash
# run_experiments.sh - Run all pattern experiments

# Core patterns to test
patterns=(
    "MMMMMMMM"  # Pure SSM
    "AAAAAAAA"  # Pure Attention
    "MAMAMAMA"  # Alternating v1
    "AMAMAMAM"  # Alternating v2
    "MMMMAAAA"  # Blocked v1
    # "AAAAMMMM"   # Blocked v2
    # "MMAMAMAM"  # Original
    # "AMMMMMMA"   # Sandwich
    # "MMAAMMAA"  # Grouped
)

# Run each pattern
for pattern in "${patterns[@]}"; do
    echo "🚀 Running experiment for pattern: $pattern"
    python experiment_patterns.py --pattern "$pattern" --use_wandb --steps 3000 --name "debug_all_3000_steps_$pattern"
    
    # Optional: Add small delay between experiments
    sleep 5
done

echo "✅ All experiments complete!"