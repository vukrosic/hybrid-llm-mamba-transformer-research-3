import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from typing import Dict, List
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import argparse
from datetime import datetime
import wandb

# Import your existing classes
from train import HybridConfig, SimpleSSM, SimpleAttention, HybridBlock, HybridModel, TextDataset

@dataclass
class ExperimentConfig(HybridConfig):
    """Extended config for experiments"""
    experiment_name: str = "pattern_ablation"
    pattern_name: str = "MMAMAMAM"
    eval_every: int = 500  # steps
    save_every: int = 2000
    num_eval_batches: int = 50
    
    # Reduced for faster experiments
    num_documents: int = 2000
    num_steps: int = 5000
    
    # HF Hub
    hf_repo: str = "vukrosic/hybrid-llm"
    
    def get_run_name(self):
        return f"{self.pattern_name}_{self.num_layers}L_{self.hidden_size}H_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class MetricsTracker:
    """Simple metrics tracking"""
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_perplexity': [],
            'grad_norm': [],
            'learning_rate': [],
            'tokens_per_second': [],
            'memory_gb': [],
            'step': []
        }
    
    def update(self, step: int, **kwargs):
        self.metrics['step'].append(step)
        for k, v in kwargs.items():
            if k in self.metrics:
                self.metrics[k].append(v)
    
    def get_current_metrics(self) -> Dict:
        return {k: v[-1] if v else 0 for k, v in self.metrics.items()}
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)


def evaluate_model(model, eval_loader, config, device, num_batches=50):
    """Quick evaluation"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= num_batches:
                break
            batch = batch.to(device)
            with autocast():
                _, loss = model(batch, labels=batch)
            batch_size = batch.numel()
            total_loss += loss.item() * batch_size
            total_tokens += batch_size
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    model.train()
    return avg_loss, perplexity


def get_gpu_memory():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str, default='MMAMAMAM', help='Layer pattern string')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--debug', action='store_true', help='Debug mode (fewer steps)')
    parser.add_argument('--use_wandb', action='store_true', help='Log to W&B')
    args = parser.parse_args()
    
    # Setup config
    config = ExperimentConfig(
        layer_pattern=args.pattern,
        pattern_name=args.pattern,
        num_layers=len(args.pattern),
    )
    
    if args.debug:
        config.num_steps = 500
        config.eval_every = 100
        config.num_documents = 500
    
    run_name = args.name or config.get_run_name()
    
    # Create experiment directory
    exp_dir = f"experiments/{run_name}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    
    # Save config
    with open(f"{exp_dir}/config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="hybrid-patterns",
            name=run_name,
            config=asdict(config)
        )
    
    # Setup
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    
    print(f"🔬 Experiment: {run_name}")
    print(f"📊 Pattern: {config.layer_pattern} ({config.num_layers} layers)")
    
    # Load data
    print("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 
                          split="train", streaming=True)
    
    # Tokenize
    all_tokens = []
    for i, item in enumerate(tqdm(dataset, total=config.num_documents, desc="Tokenizing")):
        if i >= config.num_documents:
            break
        tokens = tokenizer.encode(item["text"][:3000], add_special_tokens=False)
        all_tokens.extend(tokens)
    
    # Split train/val (90/10)
    split_idx = int(len(all_tokens) * 0.9)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    config.vocab_size = tokenizer.vocab_size
    
    # Create datasets
    train_dataset = TextDataset(train_tokens, config.max_seq_len)
    val_dataset = TextDataset(val_tokens, config.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model
    model = HybridModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"💾 Model: {total_params/1e6:.1f}M parameters ({trainable_params/1e6:.1f}M trainable)")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_steps)
    scaler = GradScaler()
    
    # Metrics
    metrics = MetricsTracker()
    
    # Training loop
    model.train()
    step = 0
    pbar = tqdm(total=config.num_steps, desc="Training")
    
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    
    while step < config.num_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        batch = batch.to(device, non_blocking=True)
        
        # Forward
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        
        with autocast():
            _, loss = model(batch, labels=batch)
        
        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        t1.record()
        torch.cuda.synchronize()
        time_ms = t0.elapsed_time(t1)
        tokens_per_sec = (batch.numel() / time_ms) * 1000
        
        step += 1
        
        # Track metrics
        if step % config.log_every == 0:
            metrics.update(
                step=step,
                train_loss=loss.item(),
                grad_norm=grad_norm.item(),
                learning_rate=scheduler.get_last_lr()[0],
                tokens_per_second=tokens_per_sec,
                memory_gb=get_gpu_memory()
            )
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'grad': f'{grad_norm.item():.2f}',
                'tok/s': f'{tokens_per_sec:.0f}',
                'mem': f'{get_gpu_memory():.1f}GB'
            })
            
            if args.use_wandb:
                wandb.log(metrics.get_current_metrics(), step=step)
        
        # Evaluate
        if step % config.eval_every == 0:
            val_loss, val_ppl = evaluate_model(model, val_loader, config, device, config.num_eval_batches)
            metrics.update(step=step, val_loss=val_loss, val_perplexity=val_ppl)
            
            print(f"\n📈 Step {step}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': asdict(config),
                    'step': step,
                    'val_loss': val_loss,
                    'val_perplexity': val_ppl
                }, f"{exp_dir}/checkpoints/best_model.pt")
        
        # Save checkpoint
        if step % config.save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': asdict(config),
                'step': step,
                'metrics': metrics.metrics
            }, f"{exp_dir}/checkpoints/checkpoint_{step}.pt")
        
        pbar.update(1)
    
    pbar.close()
    
    # Final evaluation
    print("\n🏁 Final evaluation...")
    final_val_loss, final_val_ppl = evaluate_model(model, val_loader, config, device, 200)
    
    # Save final results
    results = {
        'pattern': config.layer_pattern,
        'pattern_name': config.pattern_name,
        'num_params': total_params,
        'final_val_loss': final_val_loss,
        'final_val_perplexity': final_val_ppl,
        'best_val_loss': best_val_loss,
        'total_steps': step,
        'total_time_minutes': pbar.format_dict['elapsed'] / 60
    }
    
    with open(f"{exp_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    metrics.save(f"{exp_dir}/metrics.json")
    
    print(f"\n✅ Experiment complete!")
    print(f"📊 Results: val_loss={final_val_loss:.4f}, val_ppl={final_val_ppl:.2f}")
    print(f"💾 Saved to {exp_dir}/")
    
    if args.use_wandb:
        wandb.log(results)
        wandb.finish()
    
    return results


if __name__ == "__main__":
    main()