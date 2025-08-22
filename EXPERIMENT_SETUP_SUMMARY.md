# Extended Hybrid LLM Experiments - Setup Summary

## 🎯 Experiment Overview
- **Goal**: Train hybrid Transformer-Mamba models for 30,000 steps (3x longer than original)
- **Models**: 8 different layer patterns with 10-15 layers
- **Model Size**: 1024 hidden dimensions (2.7x larger than original 384)

## 🔧 Key Fixes Applied

### 1. Early Stopping Issue - RESOLVED ✅
**Problem**: Experiments were stopping at ~7,500 steps instead of 30,000
- **Root Cause**: `eval_every=500` × `patience=15` = 7,500 steps max
- **Solution**: 
  - `eval_every=1000` (every 1k steps)
  - `patience=30` (30 evaluations)
  - **Result**: Can now train for full 30,000 steps

### 2. Model Size Increased - UPGRADED 🚀
**Original**: 384 hidden size, 8 attention heads, 16 SSM states
**New**: 1024 hidden size, 16 attention heads, 48 SSM states
- **Hidden Size**: 2.7x larger (384 → 1024)
- **Attention Heads**: 2x more (8 → 16)
- **SSM States**: 3x larger (16 → 48)
- **Total Parameters**: ~4-5x more parameters

## 📊 Current Configuration

### Training Parameters
- **Steps**: 30,000 (vs 10,000 original)
- **Documents**: 150,000 (vs 50,000 original)
- **Evaluation**: Every 1,000 steps (vs 200 original)
- **Patience**: 30 evaluations (vs 10 original)
- **Learning Rate**: 3e-4 (vs 4e-4 original)
- **Warmup**: 3,000 steps (vs 1,000 original)

### Model Architecture
- **Base Config**: 768 hidden size (2x original)
- **Extended Config**: 1024 hidden size (2.7x original)
- **Attention**: Multi-head scaled dot-product
- **SSM**: Improved parallel state-space model
- **Patterns**: 8 different M/A layer combinations

## 🚀 How to Run

### 1. Single Experiment (Debug)
```bash
python experimental_training_extended.py --pattern "MAMAMAMAMAMA" --debug
```

### 2. Full Extended Experiment
```bash
python experimental_training_extended.py --pattern "MAMAMAMAMAMA" --use_wandb
```

### 3. All 8 Experiments in Parallel
```bash
bash run_extended_experiments.sh
```

## 📁 File Structure
```
transformer-mamba-llm-research/
├── experimental_training_extended.py  # Main extended training script
├── train_hybrid_llm.py               # Base model architecture
├── shared_data.py                     # Data loading and caching
├── run_extended_experiments.sh        # Parallel execution script
├── analyze_extended_results.py        # Results analysis
└── experiments_extended/              # Results directory (created when running)
```

## 🎯 Expected Results
- **Training Time**: ~3-4x longer per experiment (due to larger models + more steps)
- **Memory Usage**: ~4-5x more GPU memory (due to larger models)
- **Performance**: Better convergence and lower perplexity (due to longer training)
- **Model Quality**: More capable models due to increased capacity

## ⚠️ Requirements
- **GPU Memory**: Minimum 16GB per GPU (24GB+ recommended for 15L models)
- **Time**: Each experiment takes ~6-12 hours depending on GPU
- **Storage**: ~2-5GB per experiment for checkpoints and logs

## 🔍 Monitoring
- **Progress**: Real-time progress bars with loss/metrics
- **Logging**: Comprehensive logging to `logs_extended/`
- **W&B**: Optional Weights & Biases integration
- **Checkpoints**: Saved every 2,000 steps + best model

## ✅ Status
- **Early Stopping**: ✅ Fixed
- **Model Size**: ✅ Increased
- **Training Length**: ✅ 30k steps enabled
- **File Cleanup**: ✅ Unnecessary files removed
- **Ready to Run**: ✅ All systems go!
