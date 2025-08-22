#!/usr/bin/env python3
"""Find available model checkpoints in the workspace"""

import os
import glob
from pathlib import Path

def find_checkpoints():
    """Find all available model checkpoints"""
    print("🔍 Searching for model checkpoints...\n")
    
    # Common checkpoint locations
    search_patterns = [
        "experiments_extended/*/checkpoints/*.pt",
        "checkpoints/*.pt",
        "models/*.pt",
        "*.pt"
    ]
    
    found_checkpoints = []
    
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        for match in matches:
            if os.path.isfile(match):
                # Get file size
                size_mb = os.path.getsize(match) / (1024 * 1024)
                
                # Try to get checkpoint info
                info = get_checkpoint_info(match)
                
                found_checkpoints.append({
                    'path': match,
                    'size_mb': size_mb,
                    'info': info
                })
    
    if not found_checkpoints:
        print("❌ No checkpoints found!")
        return
    
    # Sort by modification time (newest first)
    found_checkpoints.sort(key=lambda x: os.path.getmtime(x['path']), reverse=True)
    
    print(f"✅ Found {len(found_checkpoints)} checkpoint(s):\n")
    
    for i, cp in enumerate(found_checkpoints, 1):
        print(f"{i}. 📁 {cp['path']}")
        print(f"   �� Size: {cp['size_mb']:.1f} MB")
        if cp['info']:
            print(f"   ℹ️  {cp['info']}")
        print()

def get_checkpoint_info(checkpoint_path):
    """Extract basic info from checkpoint file"""
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info_parts = []
        
        if 'step' in checkpoint:
            info_parts.append(f"Step {checkpoint['step']}")
        
        if 'val_loss' in checkpoint:
            info_parts.append(f"Val Loss: {checkpoint['val_loss']:.4f}")
        
        if 'config' in checkpoint and 'pattern_name' in checkpoint['config']:
            info_parts.append(f"Pattern: {checkpoint['config']['pattern_name']}")
        
        if 'config' in checkpoint and 'num_layers' in checkpoint['config']:
            info_parts.append(f"{checkpoint['config']['num_layers']}L")
        
        return " | ".join(info_parts) if info_parts else None
        
    except Exception:
        return None

if __name__ == "__main__":
    find_checkpoints()
