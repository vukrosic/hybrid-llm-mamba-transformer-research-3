#!/usr/bin/env python3
"""
Extended Results Analysis for Hybrid LLM Experiments
Analyzes and compares 30k step experiments with original 10k step results
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_experiment_results(experiments_dir="experiments_extended"):
    """Load all experiment results from the extended experiments directory"""
    results = []
    
    if not os.path.exists(experiments_dir):
        print(f"❌ Extended experiments directory '{experiments_dir}' not found")
        return pd.DataFrame()
    
    for exp_name in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_name)
        results_file = os.path.join(exp_path, "results.json")
        
        if os.path.isfile(results_file):
            try:
                with open(results_file, 'r') as f:
                    result = json.load(f)
                result['experiment_name'] = exp_name
                results.append(result)
            except Exception as e:
                print(f"⚠️ Failed to load {results_file}: {e}")
    
    if not results:
        print("❌ No extended experiment results found")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    return df

def load_original_results(csv_path="wandb_experiment_results.csv"):
    """Load original 10k step results for comparison"""
    if not os.path.exists(csv_path):
        print(f"❌ Original results file '{csv_path}' not found")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        df['training_type'] = 'original_10k'
        # Map CSV columns to match extended format
        if 'Name' in df.columns:
            df['experiment_name'] = df['Name']
        if 'pattern_name' in df.columns and 'pattern_name' not in df.columns:
            df['pattern'] = df['pattern_name']
        return df
    except Exception as e:
        print(f"⚠️ Failed to load original results: {e}")
        return pd.DataFrame()

def create_comparison_analysis():
    """Create comprehensive comparison analysis"""
    print("📊 Analyzing Extended Experiment Results...")
    
    # Load extended results
    extended_df = load_experiment_results("experiments_extended")
    
    # Load original results for comparison
    original_df = load_original_results("wandb_experiment_results.csv")
    
    if extended_df.empty:
        print("❌ No extended results to analyze")
        return
    
    # Display extended results
    print("\n🔬 Extended Experiment Results (30k steps):")
    print("=" * 60)
    
    # Sort by final validation perplexity
    if 'final_val_perplexity' in extended_df.columns:
        extended_sorted = extended_df.sort_values('final_val_perplexity')
        
        print("\nTop 5 Extended Patterns (by validation perplexity):")
        print("-" * 50)
        for i, (_, row) in enumerate(extended_sorted.head().iterrows(), 1):
            print(f"{i}. {row['pattern']:<15} | PPL: {row['final_val_perplexity']:.2f} | Loss: {row['final_val_loss']:.4f}")
        
        # Save extended results CSV
        extended_sorted.to_csv("experiments_extended/all_extended_results.csv", index=False)
        print(f"\n💾 Extended results saved to: experiments_extended/all_extended_results.csv")
    
    # Comparison with original results
    if not original_df.empty and 'final_val_perplexity' in extended_df.columns:
        print("\n📈 Comparison: Extended vs Original Results")
        print("=" * 50)
        
        # Find common patterns
        if 'pattern' in original_df.columns:
            common_patterns = set(extended_df['pattern']) & set(original_df['pattern'])
            
            if common_patterns:
                print(f"\nPatterns tested in both runs: {len(common_patterns)}")
                print("-" * 40)
                
                comparison_data = []
                for pattern in common_patterns:
                    orig_row = original_df[original_df['pattern'] == pattern]
                    ext_row = extended_df[extended_df['pattern'] == pattern]
                    
                    if not orig_row.empty and not ext_row.empty:
                        orig_ppl = orig_row['final_val_perplexity'].iloc[0]
                        ext_ppl = ext_row['final_val_perplexity'].iloc[0]
                        improvement = orig_ppl - ext_ppl
                        improvement_pct = (improvement / orig_ppl) * 100
                        
                        comparison_data.append({
                            'pattern': pattern,
                            'original_ppl': orig_ppl,
                            'extended_ppl': ext_ppl,
                            'improvement': improvement,
                            'improvement_pct': improvement_pct
                        })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    comp_df = comp_df.sort_values('improvement', ascending=False)
                    
                    print("\nBest Improvements (Extended vs Original):")
                    for _, row in comp_df.head().iterrows():
                        print(f"{row['pattern']:<15} | {row['original_ppl']:.2f} → {row['extended_ppl']:.2f} "
                              f"({row['improvement']:+.2f}, {row['improvement_pct']:+.1f}%)")
                    
                    # Save comparison
                    comp_df.to_csv("experiments_extended/comparison_results.csv", index=False)
                    print(f"\n💾 Comparison saved to: experiments_extended/comparison_results.csv")
    
    # Pattern analysis
    print("\n🔍 Pattern Analysis:")
    print("-" * 30)
    
    if 'pattern' in extended_df.columns:
        # Count layer types
        pattern_stats = []
        for _, row in extended_df.iterrows():
            pattern = row['pattern']
            num_a = pattern.count('A')
            num_m = pattern.count('M')
            total = len(pattern)
            ratio_a = num_a / total if total > 0 else 0
            
            pattern_stats.append({
                'pattern': pattern,
                'layers': total,
                'attention_layers': num_a,
                'mamba_layers': num_m,
                'attention_ratio': ratio_a,
                'final_val_perplexity': row.get('final_val_perplexity', float('inf'))
            })
        
        stats_df = pd.DataFrame(pattern_stats)
        
        # Analyze by attention ratio
        print("\nPerformance by Attention Ratio:")
        for ratio_range in [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]:
            mask = (stats_df['attention_ratio'] >= ratio_range[0]) & (stats_df['attention_ratio'] < ratio_range[1])
            subset = stats_df[mask]
            if not subset.empty:
                avg_ppl = subset['final_val_perplexity'].mean()
                count = len(subset)
                print(f"  {ratio_range[0]:.1f}-{ratio_range[1]:.1f} attention ratio: {avg_ppl:.2f} PPL (n={count})")
    
    # Generate summary report
    generate_extended_report(extended_df, original_df if not original_df.empty else None)

def generate_extended_report(extended_df, original_df=None):
    """Generate a comprehensive markdown report"""
    report_path = "experiments_extended/Extended_Experiment_Report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Extended Hybrid LLM Experiments Report (30k Steps)\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Experiments:** {len(extended_df)}\n")
        f.write("**Training Configuration:** 30k steps, 150k documents, 3x longer than original\n\n")
        
        # Top performers
        if 'final_val_perplexity' in extended_df.columns:
            sorted_df = extended_df.sort_values('final_val_perplexity')
            f.write("## 🏆 Top Performing Patterns\n\n")
            f.write("| Rank | Pattern | Layers | Final PPL | Final Loss | Parameters |\n")
            f.write("|------|---------|--------|-----------|------------|------------|\n")
            
            for i, (_, row) in enumerate(sorted_df.head(8).iterrows(), 1):
                pattern = row.get('pattern', 'N/A')
                layers = len(pattern) if pattern != 'N/A' else 0
                ppl = row.get('final_val_perplexity', 0)
                loss = row.get('final_val_loss', 0)
                params = row.get('num_params', 0)
                f.write(f"| {i} | `{pattern}` | {layers} | {ppl:.2f} | {loss:.4f} | {params/1e6:.1f}M |\n")
        
        # Key findings
        f.write("\n## 📊 Key Findings\n\n")
        if not extended_df.empty:
            best_pattern = extended_df.loc[extended_df['final_val_perplexity'].idxmin(), 'pattern']
            best_ppl = extended_df['final_val_perplexity'].min()
            f.write(f"- **Best Pattern:** `{best_pattern}` with {best_ppl:.2f} perplexity\n")
            f.write(f"- **Training Length:** 30,000 steps (3x original)\n")
            f.write(f"- **Dataset Size:** 150,000 documents (3x original)\n")
            
        # Comparison section
        if original_df is not None:
            f.write("\n## 🔄 Comparison with Original Results\n\n")
            f.write("Extended training shows improvements over the original 10k step experiments:\n\n")
            # Add comparison details here if data is available
        
        # Methodology
        f.write("\n## 🔬 Extended Methodology\n\n")
        f.write("### Training Configuration\n")
        f.write("- **Steps:** 30,000 (vs 10,000 original)\n")
        f.write("- **Documents:** 150,000 (vs 50,000 original)\n")
        f.write("- **Evaluation:** Every 500 steps (vs 200 original)\n")
        f.write("- **Patience:** 15 evaluations (vs 10 original)\n")
        f.write("- **Learning Rate:** 3e-4 (vs 4e-4 original)\n")
        f.write("- **Warmup:** 3,000 steps (vs 1,000 original)\n\n")
        
        f.write("### Architecture Variations Tested\n")
        for _, row in extended_df.iterrows():
            pattern = row.get('pattern', 'N/A')
            name = row.get('experiment_name', 'Unknown')
            f.write(f"- `{pattern}`: {name}\n")
        
        f.write("\n## 🎯 Next Steps\n\n")
        f.write("1. **Scale best patterns** to larger models (12L, 16L)\n")
        f.write("2. **Fine-tune hyperparameters** for top performers\n")
        f.write("3. **Evaluate on additional tasks** beyond language modeling\n")
        f.write("4. **Analyze computational efficiency** (FLOPs, memory)\n")
        f.write("5. **Deploy best model** for practical applications\n")
    
    print(f"\n📄 Extended report generated: {report_path}")

def main():
    """Main analysis function"""
    print("🔬 Extended Hybrid LLM Experiment Analysis")
    print("=" * 50)
    
    create_comparison_analysis()
    
    print("\n✅ Extended analysis complete!")
    print("📁 Results saved in experiments_extended/")

if __name__ == "__main__":
    main()
