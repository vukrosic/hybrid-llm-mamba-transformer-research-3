#!/usr/bin/env python3
"""
Test script to verify progress bar functionality
"""

from tqdm import tqdm
import time

def test_progress_bar():
    print("Testing progress bar functionality...")
    
    # Test 1: Basic progress bar
    print("\n1. Basic progress bar:")
    for i in tqdm(range(100), desc="Basic Test", ncols=80):
        time.sleep(0.01)
    
    # Test 2: Progress bar with postfix updates
    print("\n2. Progress bar with postfix updates:")
    pbar = tqdm(range(100), desc="Postfix Test", ncols=80)
    for i in pbar:
        pbar.set_postfix({'loss': f'{i/100:.3f}', 'step': i})
        time.sleep(0.01)
    pbar.close()
    
    # Test 3: Progress bar with write (should not interfere)
    print("\n3. Progress bar with write calls:")
    pbar = tqdm(range(100), desc="Write Test", ncols=80)
    for i in pbar:
        if i % 20 == 0:
            pbar.write(f"Checkpoint at step {i}")
        pbar.set_postfix({'progress': f'{i}%'})
        time.sleep(0.01)
    pbar.close()
    
    print("\n✅ Progress bar tests completed!")

if __name__ == "__main__":
    test_progress_bar()
