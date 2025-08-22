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
        print(f"    Size: {cp['size_mb']:.1f} MB")
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

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional
import sys

# Add the current directory to path to import your model classes
sys.path.append('.')

from train_hybrid_llm import HybridConfig, HybridModel
from experimental_training_extended import ImprovedSSM


class InteractiveInference:
    """Interactive inference for the trained hybrid Mamba-Transformer model"""
    
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.device = self._setup_device(device)
        self.model, self.config, self.tokenizer = self._load_model(checkpoint_path)
        self.max_length = self.config.max_seq_len
        self.temperature = 1.0
        self.top_p = 0.9
        self.top_k = 50
        self.max_new_tokens = 100
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup the device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🎯 Using CUDA: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                print("💻 Using CPU")
        else:
            print(f"🎯 Using device: {device}")
        
        return torch.device(device)
    
    def _load_model(self, checkpoint_path: str):
        """Load the trained model from checkpoint"""
        print(f"🎯 Loading model from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            config = HybridConfig(**config_dict)
        else:
            # Fallback to default config if not in checkpoint
            config = HybridConfig()
            print("⚠️ No config found in checkpoint, using defaults")
        
        # Create model
        model = HybridModel(config).to(self.device)
        
        # Replace SSM layers with improved version if needed
        for i, layer in enumerate(model.layers):
            if config.layer_pattern[i] == 'M':
                layer.mixer = ImprovedSSM(config).to(self.device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model weights loaded successfully")
        else:
            print("⚠️ No model weights found in checkpoint")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
            print("✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load tokenizer: {e}")
            print("🔄 Using basic tokenizer...")
            # Fallback to basic tokenizer
            tokenizer = self._create_basic_tokenizer()
        
        model.eval()
        return model, config, tokenizer
    
    def _create_basic_tokenizer(self):
        """Create a basic tokenizer as fallback"""
        class BasicTokenizer:
            def __init__(self):
                self.vocab_size = 32000
                self.pad_token_id = 0
                self.eos_token_id = 2
                self.unk_token_id = 1
                
            def encode(self, text, add_special_tokens=True):
                # Simple character-level tokenization for testing
                tokens = [ord(c) % self.vocab_size for c in text]
                if add_special_tokens:
                    tokens = [self.unk_token_id] + tokens + [self.eos_token_id]
                return tokens
                
            def decode(self, tokens):
                # Simple character-level detokenization
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                # Remove special tokens and convert back to characters
                tokens = [t for t in tokens if t not in [self.pad_token_id, self.eos_token_id, self.unk_token_id]]
                return ''.join([chr(t) if t < 128 else '?' for t in tokens])
        
        return BasicTokenizer()
    
    def generate_text(self, prompt: str, max_new_tokens: int = 100, 
                     temperature: Optional[float] = None, 
                     top_p: Optional[float] = None,
                     top_k: Optional[int] = None) -> str:
        """Generate text continuation from prompt"""
        
        # Use instance defaults if not specified
        temp = temperature if temperature is not None else self.temperature
        tp = top_p if top_p is not None else self.top_p
        tk = top_k if top_k is not None else self.top_k
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        
        # Ensure input doesn't exceed max length
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, -self.max_length:]
        
        print(f"📝 Prompt tokens: {input_ids.shape[1]}")
        print(f"🎯 Generating {max_new_tokens} new tokens...")
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model predictions
                logits, _ = self.model(generated_ids)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temp
                
                # Apply top-k filtering
                if tk > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, tk)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Apply top-p (nucleus) filtering
                if tp < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > tp
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit the end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Stop if we exceed max length
                if generated_ids.shape[1] >= self.max_length:
                    break
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n" + "="*60)
        print("🤖 Interactive Inference Session Started!")
        print("="*60)
        print("💡 Type your prompts and press Enter to generate text")
        print("🔧 Commands:")
        print("   /temp <value> - Set temperature (0.1-2.0)")
        print("   /top_p <value> - Set top-p (0.1-1.0)")
        print("   /top_k <value> - Set top-k (1-100)")
        print("   /tokens <value> - Set max new tokens")
        print("   /quit or /exit - Exit the session")
        print("   /help - Show this help")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if self._handle_command(user_input):
                        continue
                    else:
                        break
                
                # Generate text
                print("\n🤖 AI: ", end="", flush=True)
                generated_text = self.generate_text(
                    user_input, 
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k
                )
                
                # Remove the original prompt from output
                if generated_text.startswith(user_input):
                    generated_text = generated_text[len(user_input):].strip()
                
                print(generated_text)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _handle_command(self, command: str) -> bool:
        """Handle interactive commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/quit' or cmd == '/exit':
            print("👋 Goodbye!")
            return False
        
        elif cmd == '/help':
            print("💡 Available commands:")
            print("   /temp <value> - Set temperature (0.1-2.0)")
            print("   /top_p <value> - Set top-p (0.1-1.0)")
            print("   /top_k <value> - Set top-k (1-100)")
            print("   /tokens <value> - Set max new tokens")
            print("   /quit or /exit - Exit the session")
            print("   /help - Show this help")
        
        elif cmd == '/temp' and len(parts) > 1:
            try:
                value = float(parts[1])
                if 0.1 <= value <= 2.0:
                    self.temperature = value
                    print(f"🌡️ Temperature set to: {value}")
                else:
                    print("❌ Temperature must be between 0.1 and 2.0")
            except ValueError:
                print("❌ Invalid temperature value")
        
        elif cmd == '/top_p' and len(parts) > 1:
            try:
                value = float(parts[1])
                if 0.1 <= value <= 1.0:
                    self.top_p = value
                    print(f"🎯 Top-p set to: {value}")
                else:
                    print("❌ Top-p must be between 0.1 and 1.0")
            except ValueError:
                print("❌ Invalid top-p value")
        
        elif cmd == '/top_k' and len(parts) > 1:
            try:
                value = int(parts[1])
                if 1 <= value <= 100:
                    self.top_k = value
                    print(f"🔝 Top-k set to: {value}")
                else:
                    print("❌ Top-k must be between 1 and 100")
            except ValueError:
                print("❌ Invalid top-k value")
        
        elif cmd == '/tokens' and len(parts) > 1:
            try:
                value = int(parts[1])
                if 1 <= value <= 1000:
                    self.max_new_tokens = value
                    print(f"🎯 Max new tokens set to: {value}")
                else:
                    print("❌ Max new tokens must be between 1 and 1000")
            except ValueError:
                print("❌ Invalid token count")
        
        else:
            print("❌ Unknown command. Type /help for available commands.")
        
        return True
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts"""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"📝 Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            result = self.generate_text(prompt, **kwargs)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description='Interactive Inference for Hybrid Mamba-Transformer Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--prompt', type=str, help='Single prompt for generation (non-interactive)')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("💡 Available checkpoints:")
        
        # Look for checkpoints in common locations
        checkpoint_dirs = [
            "experiments_extended/*/checkpoints/",
            "checkpoints/",
            "models/",
            "."
        ]
        
        for pattern in checkpoint_dirs:
            import glob
            checkpoints = glob.glob(pattern)
            for cp in checkpoints:
                if os.path.isfile(cp) and cp.endswith('.pt'):
                    print(f"   {cp}")
                elif os.path.isdir(cp):
                    for file in os.listdir(cp):
                        if file.endswith('.pt'):
                            print(f"   {os.path.join(cp, file)}")
        
        return
    
    try:
        # Initialize inference
        inference = InteractiveInference(args.checkpoint, args.device)
        
        # Single prompt mode
        if args.prompt:
            print(f"📝 Prompt: {args.prompt}")
            print(f"🎯 Generating {args.max_tokens} tokens...")
            result = inference.generate_text(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k
            )
            print(f"\n🎯 Generated text:\n{result}")
        
        # Interactive mode
        else:
            inference.interactive_chat()
    
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
