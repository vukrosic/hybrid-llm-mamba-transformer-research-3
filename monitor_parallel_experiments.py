#!/usr/bin/env python3
"""
Real-time monitoring for 8 parallel experiments running on 8 GPUs
Tracks progress, GPU usage, and experiment status
"""

import os
import time
import subprocess
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

def get_gpu_info():
    """Get GPU usage information"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) == 6:
                    gpu_info.append({
                        'gpu_id': int(parts[0]),
                        'name': parts[1],
                        'memory_used': int(parts[2]),
                        'memory_total': int(parts[3]),
                        'utilization': int(parts[4]),
                        'temperature': int(parts[5])
                    })
            return gpu_info
    except Exception as e:
        print(f"Error getting GPU info: {e}")
    return []

def get_experiment_status():
    """Check status of all 8 experiments"""
    experiments = [
        ("mama_alternating_12L_extended", "MAMAMAMAMAMA", 0),
        ("mama_alternating_15L_extended", "MAMAMAMAMAMAMAMAM", 1),
        ("mmaammaa_pattern_12L_extended", "MMAAMMAAMMAAMMAA", 2),
        ("mmaammaa_pattern_14L_extended", "MMAAMMAAMMAAMMA", 3),
        ("mama_alternating_10L_extended", "MAMAMAMAMAMAMAMA", 4),
        ("grouped_separated_10L", "MMMMAAAAAA", 5),
        ("mixed_grouped_11L", "MMAAAMMMAAA", 6),
        ("mama_alternating_13L_extended", "MAMAMAMAMAMAM", 7)
    ]
    
    status_info = []
    for name, pattern, gpu_id in experiments:
        exp_dir = f"experiments_extended/{name}"
        log_file = f"logs_extended/{name}_gpu{gpu_id}.log"
        
        status = {
            'name': name,
            'pattern': pattern,
            'gpu_id': gpu_id,
            'status': 'Not Started',
            'current_step': 0,
            'total_steps': 30000,
            'val_loss': None,
            'val_ppl': None,
            'progress': 0.0
        }
        
        # Check if experiment directory exists
        if os.path.exists(exp_dir):
            status['status'] = 'Running'
            
            # Try to read current metrics from results.json if it exists
            results_file = os.path.join(exp_dir, "results.json")
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    status['status'] = 'Completed'
                    status['current_step'] = results.get('total_steps', 30000)
                    status['val_loss'] = results.get('final_val_loss')
                    status['val_ppl'] = results.get('final_val_perplexity')
                    status['progress'] = 100.0
                except:
                    pass
            
            # If still running, try to parse log file for progress
            if status['status'] == 'Running' and os.path.exists(log_file):
                try:
                    # Read last few lines of log file to get current step
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Look for step information in recent lines
                    for line in reversed(lines[-50:]):  # Check last 50 lines
                        if 'Step ' in line and 'val_loss=' in line:
                            # Parse step number
                            step_part = line.split('Step ')[1].split(':')[0]
                            try:
                                current_step = int(step_part)
                                status['current_step'] = current_step
                                status['progress'] = (current_step / 30000) * 100
                                
                                # Try to extract val_loss and val_ppl
                                if 'val_loss=' in line:
                                    val_loss_part = line.split('val_loss=')[1].split(',')[0]
                                    status['val_loss'] = float(val_loss_part)
                                if 'val_ppl=' in line:
                                    val_ppl_part = line.split('val_ppl=')[1].split(' ')[0]
                                    status['val_ppl'] = float(val_ppl_part)
                                break
                            except:
                                continue
                except Exception as e:
                    print(f"Error reading log for {name}: {e}")
        
        status_info.append(status)
    
    return status_info

def display_status():
    """Display current status of all experiments and GPUs"""
    os.system('clear')  # Clear screen
    
    print("🔬 Parallel Hybrid LLM Experiments Monitor")
    print("=" * 80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    # Get experiment status
    exp_status = get_experiment_status()
    
    # Display experiment status
    print("📊 EXPERIMENT STATUS:")
    print("-" * 80)
    print(f"{'GPU':<3} {'Name':<25} {'Pattern':<17} {'Status':<10} {'Step':<8} {'Progress':<8} {'Val PPL':<8}")
    print("-" * 80)
    
    for status in exp_status:
        gpu_id = status['gpu_id']
        name = status['name'][:24]
        pattern = status['pattern'][:16]
        exp_status_str = status['status'][:9]
        step = f"{status['current_step']}/{status['total_steps']//1000}k"
        progress = f"{status['progress']:.1f}%"
        val_ppl = f"{status['val_ppl']:.2f}" if status['val_ppl'] else "N/A"
        
        print(f"{gpu_id:<3} {name:<25} {pattern:<17} {exp_status_str:<10} {step:<8} {progress:<8} {val_ppl:<8}")
    
    print()
    
    # Display GPU status
    print("🖥️  GPU STATUS:")
    print("-" * 80)
    print(f"{'GPU':<3} {'Name':<20} {'Memory':<15} {'Util':<6} {'Temp':<6} {'Experiment':<25}")
    print("-" * 80)
    
    for gpu in gpu_info:
        gpu_id = gpu['gpu_id']
        name = gpu['name'][:19]
        memory = f"{gpu['memory_used']}/{gpu['memory_total']}MB"
        util = f"{gpu['utilization']}%"
        temp = f"{gpu['temperature']}°C"
        
        # Find experiment running on this GPU
        exp_name = "Idle"
        for status in exp_status:
            if status['gpu_id'] == gpu_id and status['status'] in ['Running', 'Completed']:
                exp_name = status['name'][:24]
                break
        
        print(f"{gpu_id:<3} {name:<20} {memory:<15} {util:<6} {temp:<6} {exp_name:<25}")
    
    print()
    
    # Summary stats
    completed = sum(1 for s in exp_status if s['status'] == 'Completed')
    running = sum(1 for s in exp_status if s['status'] == 'Running')
    not_started = sum(1 for s in exp_status if s['status'] == 'Not Started')
    
    print("📈 SUMMARY:")
    print(f"✅ Completed: {completed}/8  🔄 Running: {running}/8  ⏳ Not Started: {not_started}/8")
    
    if completed > 0:
        completed_exps = [s for s in exp_status if s['status'] == 'Completed' and s['val_ppl']]
        if completed_exps:
            best_exp = min(completed_exps, key=lambda x: x['val_ppl'])
            print(f"🏆 Best so far: {best_exp['pattern']} (PPL: {best_exp['val_ppl']:.2f})")
    
    print("\n⏹️  Press Ctrl+C to stop monitoring")

def main():
    """Main monitoring loop"""
    print("🚀 Starting parallel experiment monitor...")
    print("📊 Monitoring 8 experiments across 8 GPUs")
    print("🔄 Updates every 30 seconds")
    
    try:
        while True:
            display_status()
            time.sleep(30)  # Update every 30 seconds
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped.")
        
        # Final summary
        exp_status = get_experiment_status()
        completed = sum(1 for s in exp_status if s['status'] == 'Completed')
        print(f"\n📊 Final Status: {completed}/8 experiments completed")
        
        if completed > 0:
            completed_exps = [s for s in exp_status if s['status'] == 'Completed' and s['val_ppl']]
            if completed_exps:
                print("\n🏆 Completed Experiments (by validation perplexity):")
                completed_exps.sort(key=lambda x: x['val_ppl'])
                for i, exp in enumerate(completed_exps[:3], 1):
                    print(f"  {i}. {exp['pattern']} - PPL: {exp['val_ppl']:.2f}")

if __name__ == "__main__":
    main()
