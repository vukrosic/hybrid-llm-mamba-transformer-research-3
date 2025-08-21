#!/usr/bin/env python3
"""
Clear the cached tokenized datasets.
Use this when you want to force retokenization of data.
"""

from shared_data import shared_data_manager

if __name__ == "__main__":
    print("🗑️ Clearing data cache...")
    shared_data_manager.clear_cache()
    print("✅ Cache cleared! Next run will retokenize all data.")
