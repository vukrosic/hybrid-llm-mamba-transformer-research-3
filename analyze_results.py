import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def collect_results():
    """Collect all experiment results"""
    results = []
    
    for exp_dir in Path("experiments").iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            metrics_file = exp_dir / "metrics.json"
            
            if results_file.exists():
                with open(results_file) as f:
                    result = json.load(f)
                    result['exp_name'] = exp_dir.name
                    
                    # Load training metrics
                    if metrics_file.exists():
                        with open(metrics_file) as mf:
                            metrics = json.load(mf)
                                                # Add convergence speed (steps to reach loss < 3.0)
                    if 'train_loss' in metrics and 'step' in metrics:
                        for i, loss in enumerate(metrics['train_loss']):
                            if loss < 3.0:
                                result['convergence_steps'] = metrics['step'][i]
                                break
                        else:
                            result['convergence_steps'] = None
                    else:
                        result['convergence_steps'] = None
                    
                    results.append(result)
    
    return pd.DataFrame(results)

def plot_results(df):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sort by performance
    df = df.sort_values('final_val_perplexity')
    
    # Plot 1: Perplexity comparison
    axes[0, 0].barh(df['pattern_name'], df['final_val_perplexity'])
    axes[0, 0].set_xlabel('Validation Perplexity')
    axes[0, 0].set_title('Model Performance')
    
    # Plot 2: Convergence speed
    if 'convergence_steps' in df.columns and df['convergence_steps'].notna().any():
        valid_convergence = df[df['convergence_steps'].notna()]
        if len(valid_convergence) > 0:
            axes[0, 1].barh(valid_convergence['pattern_name'], valid_convergence['convergence_steps'])
            axes[0, 1].set_xlabel('Steps to Convergence')
            axes[0, 1].set_title('Training Efficiency')
        else:
            axes[0, 1].text(0.5, 0.5, 'No convergence data', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Training Efficiency')
    else:
        axes[0, 1].text(0.5, 0.5, 'No convergence data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Training Efficiency')
    
    # Plot 3: Pattern types pie chart
    pattern_types = {
        'Pure SSM': df['pattern'].str.count('M') == 8,
        'Pure Attention': df['pattern'].str.count('A') == 8,
        'Alternating': df['pattern'].str.contains('MAMA|AMAM'),
        'Blocked': df['pattern'].str.contains('MMMM|AAAA'),
        'Mixed': True  # Catch-all
    }
    
    type_counts = {}
    for name, mask in pattern_types.items():
        if name != 'Mixed':
            type_counts[name] = mask.sum()
    
    axes[1, 0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.0f%%')
    axes[1, 0].set_title('Pattern Distribution')
    
    # Plot 4: Performance vs pattern characteristics
    df['ssm_ratio'] = df['pattern'].str.count('M') / df['pattern'].str.len()
    axes[1, 1].scatter(df['ssm_ratio'], df['final_val_perplexity'])
    axes[1, 1].set_xlabel('SSM Layer Ratio')
    axes[1, 1].set_ylabel('Validation Perplexity')
    axes[1, 1].set_title('Performance vs SSM Ratio')
    
    plt.tight_layout()
    plt.savefig('experiments/pattern_comparison.png', dpi=150)
    plt.show()
    
    # Print summary table
    print("\n📊 RESULTS SUMMARY")
    print("=" * 60)
    
    # Select available columns
    available_cols = ['pattern', 'final_val_perplexity']
    if 'convergence_steps' in df.columns:
        available_cols.append('convergence_steps')
    
    summary = df[available_cols].sort_values('final_val_perplexity')
    print(summary.to_string(index=False))
    
    return df

if __name__ == "__main__":
    df = collect_results()
    if len(df) > 0:
        plot_results(df)
        df.to_csv('experiments/all_results.csv', index=False)
        print(f"\n💾 Saved results to experiments/all_results.csv")
    else:
        print("No results found yet!")