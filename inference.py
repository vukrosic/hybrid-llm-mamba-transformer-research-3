import torch
import torch.nn.functional as F
from hybrid_llm import HybridModel, HybridConfig
from transformers import AutoTokenizer
import argparse

def load_model(model_path, device):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {model_path}...")
    
    # First, load the checkpoint to get the actual vocabulary size
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract vocab size from the embed.weight shape
    vocab_size = checkpoint['embed.weight'].shape[0]
    print(f"Detected vocabulary size: {vocab_size}")
    
    # Create config with the correct vocab size
    config = HybridConfig()
    config.vocab_size = vocab_size
    
    # Create model instance
    model = HybridModel(config)
    
    # Load trained weights
    model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, device='cuda'):
    """Generate text using the loaded model"""
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"Prompt: {prompt}")
    print(f"Generating...")
    
    with torch.no_grad():
        for i in range(max_length):
            # Get model predictions
            logits, _ = model(input_ids)
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Decode and print progress
            if i % 10 == 0:
                current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                print(f"Step {i}: {current_text[-50:]}...")
    
    # Decode final result
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def interactive_mode(model, tokenizer, device):
    """Run interactive chat mode"""
    print("\n=== Interactive Mode ===")
    print("Type 'quit' to exit, 'clear' to start new conversation")
    print("-" * 50)
    
    conversation_history = ""
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            conversation_history = ""
            print("Conversation cleared!")
            continue
        
        # Build full prompt with history
        full_prompt = conversation_history + user_input if conversation_history else user_input
        
        try:
            generated = generate_text(
                model, tokenizer, full_prompt, 
                max_length=50, temperature=0.7, top_k=40, device=device
            )
            
            # Extract only the new generated part
            new_text = generated[len(full_prompt):]
            print(f"\nAI: {new_text}")
            
            # Update conversation history
            conversation_history = generated + "\n"
            
        except Exception as e:
            print(f"Error during generation: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inference script for Hybrid Transformer-Mamba model")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Text prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
        print("Tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Load model
    try:
        model = load_model(args.model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if args.interactive:
        interactive_mode(model, tokenizer, device)
    else:
        # Single generation
        try:
            generated_text = generate_text(
                model, tokenizer, args.prompt, 
                max_length=args.max_length, 
                temperature=args.temperature, 
                top_k=args.top_k, 
                device=device
            )
            
            print("\n" + "="*50)
            print("FINAL GENERATED TEXT:")
            print("="*50)
            print(generated_text)
            
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
