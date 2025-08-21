# 🚀 Transformer-Mamba Hybrid LLM Research

LLM combining **Transformer attention** and **Mamba SSM** architectures for efficient language modeling & research.

## ✨ Features

- 🧑‍💻 **Only 252 Lines of Code**
- 🔄 **Hybrid Architecture**: Alternating Transformer and Mamba layers
- ⚡ **Efficient Training**: Mixed precision, gradient scaling, and optimized data loading
- 🎯 **Flexible Patterns**: Configurable layer arrangements (e.g., "MMAMAMAM")
- 🚀 **Multi-GPU Support**: Automatic DataParallel for multi GPUs (only 1 tested)
- 💾 **Easy Inference**: Interactive chat mode and text generation

## ️ Architecture

The model alternates between:
- **Mamba SSM**: State space model with convolution and simplified parallel processing
- **Transformer**: Multi-head attention with causal masking

##  Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python hybrid_llm.py
```

### 3. Run Inference
```bash
python inference.py --interactive
```
or

```bash
python inference.py --prompt "The future of AI is"
```

### 4. To upload to Hugging Face automatically, first run
echo "HF_TOKEN=your_hf_token_with_write_permission" > .env

## 📤 Uploading Models to Hugging Face

After training, you can upload your model to Hugging Face Hub using the included script:

```bash
# Install required package
pip install huggingface_hub

# Upload a saved model
python upload_model.py --model_path experiments/my_experiment/checkpoints/best_model.pt --repo_name username/repo-name
```

The script automatically creates the proper HF Hub structure and uploads your model using your API token.

## 📊 Model Configuration

- **Hidden Size**: 384
- **Layers**: 8 (configurable pattern)
- **Sequence Length**: 512
- **Vocabulary**: Auto-detected from tokenizer
- **Parameters**: ~2.5M (configurable)

## 🎮 Usage Examples

### Interactive Chat
```bash
python inference.py --model model.pt --interactive
```

### Text Generation
```bash
python inference.py --model model.pt --prompt "The future of AI is"
```

## 📁 Project Structure

```
├── hybrid_llm.py    # Training script and model definition
├── inference.py     # Inference and chat interface
├── gpu_monitor.py  # GPU monitoring utilities
├── upload_model.py # Script to upload models to HF Hub
└── requirements.txt # Python dependencies
```

##  Customization

Modify `HybridConfig` in `hybrid_llm.py` to adjust:
- Model dimensions
- Layer patterns
- Training parameters
- Architecture choices

## 📈 Performance

- **Training**: Optimized with AMP, gradient clipping, and efficient data loading
- **Inference**: Fast generation with temperature and top-k sampling
- **Memory**: Efficient attention and SSM implementations

##  Contributing

This is a research project - feel free to experiment with different architectures and configurations!

## 📄 License

See [LICENSE](LICENSE) file for details.