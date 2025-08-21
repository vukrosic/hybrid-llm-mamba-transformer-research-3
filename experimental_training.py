import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import argparse
from datetime import datetime
import wandb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import your existing classes
from train_hybrid_llm import HybridConfig, SimpleSSM, SimpleAttention, HybridBlock, HybridModel

class TextDataset(Dataset):
    def __init__(self, tokens, max_length, stride=None, pad_token_id=0):
        self.tokens = tokens
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length
        self.pad_token_id = pad_token_id

    def __len__(self):
        return max(1, (len(self.tokens) - self.max_length) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = min(start + self.max_length, len(self.tokens))
        chunk = self.tokens[start:end]
        if len(chunk) < self.max_length:
            chunk = chunk + [self.pad_token_id] * (self.max_length - len(chunk))
        return torch.tensor(chunk, dtype=torch.long)

@dataclass
class ExperimentConfig(HybridConfig):
    """Extended config for experiments"""
    experiment_name: str = "pattern_ablation"
    pattern_name: str = "MMAMAMAM"
    eval_every: int = 200  # More frequent evaluation
    save_every: int = 1000
    num_eval_batches: int = 100  # More batches for stable eval
    
    # Increased data for better training
    num_documents: int = 5000  # Increased from 2000
    num_steps: int = 10000  # Increased from 5000
    
    # Better hyperparameters
    dropout: float = 0.1  # Reduced back to 0.1
    learning_rate: float = 3e-4  # Increased from 1e-4
    weight_decay: float = 0.01  # Add weight decay
    warmup_steps: int = 500  # Add warmup
    
    # Better batch size for stability
    batch_size: int = 16  # Increased from 8
    gradient_accumulation_steps: int = 2  # Add gradient accumulation
    
    # HF Hub
    hf_repo: str = "vukrosic/hybrid-llm"
    
    def get_run_name(self):
        return f"{self.pattern_name}_{self.num_layers}L_{self.hidden_size}H_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class ImprovedSSM(nn.Module):
    """Improved SSM with better numerical stability"""
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.ssm_state_size
        
        self.in_proj = nn.Linear(config.hidden_size, self.intermediate_size * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.intermediate_size, self.intermediate_size,
            kernel_size=config.conv_kernel, groups=self.intermediate_size,
            padding=config.conv_kernel - 1, bias=False
        )
        self.x_proj = nn.Linear(self.intermediate_size, config.ssm_state_size * 2 + 1, bias=False)
        
        # Better initialization for A
        self.A = nn.Parameter(torch.randn(self.intermediate_size, self.ssm_state_size) * 0.1)
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        
        # Add layer norm for stability
        self.norm = nn.LayerNorm(self.intermediate_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = self.conv1d(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x = F.silu(x)
        
        # Add normalization for stability
        x = self.norm(x)
        
        x_proj = self.x_proj(x)
        delta, B, C = x_proj.split([1, self.ssm_state_size, self.ssm_state_size], dim=-1)
        
        # Improved stability with clamping
        delta = F.softplus(delta).clamp(min=1e-6, max=10)
        
        # More stable A computation
        A = -F.softplus(self.A).clamp(min=0.1, max=10)
        
        # Compute SSM with better numerical stability
        decay = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        decay = decay.clamp(min=1e-6, max=1.0)  # Prevent numerical issues
        
        states = x.unsqueeze(-1) * B.unsqueeze(2) * decay
        y = (states * C.unsqueeze(2)).sum(dim=-1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return self.out_proj(y * F.silu(z))


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
        
        # Add moving averages for smoother tracking
        self.train_loss_ema = None
        self.ema_alpha = 0.95
    
    def update(self, step: int, **kwargs):
        self.metrics['step'].append(step)
        
        # Update EMA for train loss
        if 'train_loss' in kwargs:
            if self.train_loss_ema is None:
                self.train_loss_ema = kwargs['train_loss']
            else:
                self.train_loss_ema = self.ema_alpha * self.train_loss_ema + (1 - self.ema_alpha) * kwargs['train_loss']
            kwargs['train_loss_ema'] = self.train_loss_ema
        
        for k, v in kwargs.items():
            if k in self.metrics:
                self.metrics[k].append(v)
    
    def get_current_metrics(self) -> Dict:
        return {k: v[-1] if v else 0 for k, v in self.metrics.items()}
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)


def evaluate_model(model, eval_loader, config, device, num_batches=100):
    """Improved evaluation with proper loss calculation"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= num_batches:
                break
            batch = batch.to(device)
            
            # Create attention mask for padding
            if config.pad_token_id is not None:
                attention_mask = (batch != config.pad_token_id).float()
            else:
                attention_mask = torch.ones_like(batch).float()
            
            with autocast('cuda'):
                logits, _ = model(batch)
                
                # Compute loss manually with proper masking
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch[..., 1:].contiguous()
                shift_mask = attention_mask[..., 1:].contiguous()
                
                # Flatten for loss computation
                loss = F.cross_entropy(
                    shift_logits.view(-1, config.vocab_size),
                    shift_labels.view(-1),
                    reduction='none'
                )
                
                # Apply mask and average
                loss = (loss.view_as(shift_labels) * shift_mask).sum() / shift_mask.sum()
            
            valid_tokens = shift_mask.sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 10))  # Cap to avoid overflow
    model.train()
    return avg_loss, perplexity


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str, default='MMAMAMAM', help='Layer pattern string')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--debug', action='store_true', help='Debug mode (fewer documents)')
    parser.add_argument('--steps', type=int, default=None, help='Number of training steps')
    parser.add_argument('--use_wandb', action='store_true', help='Log to W&B')
    args = parser.parse_args()
    
    # Setup config
    config = ExperimentConfig(
        layer_pattern=args.pattern,
        pattern_name=args.pattern,
        num_layers=len(args.pattern),
    )
    
    if args.debug:
        config.num_documents = 1000
        config.num_steps = 1000
        config.eval_every = 100
    
    if args.steps:
        config.num_steps = args.steps
    
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
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    
    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 
                          split="train", streaming=True)
    
    # Tokenize documents
    all_documents = []
    for i, item in enumerate(tqdm(dataset, total=config.num_documents, desc="Tokenizing")):
        if i >= config.num_documents:
            break
        tokens = tokenizer.encode(item["text"][:4000], add_special_tokens=False)  # Increased from 3000
        all_documents.append(tokens)
    
    # Better train/val split
    n_train = int(len(all_documents) * 0.85)  # 85/15 split instead of 90/10
    train_docs = all_documents[:n_train]
    val_docs = all_documents[n_train:]
    
    # Flatten
    train_tokens = [token for doc in train_docs for token in doc]
    val_tokens = [token for doc in val_docs for token in doc]
    
    config.vocab_size = tokenizer.vocab_size
    
    # Create datasets - reduced overlap for training
    train_dataset = TextDataset(
        train_tokens, 
        config.max_seq_len, 
        stride=int(config.max_seq_len * 0.8),  # 80% stride instead of 50%
        pad_token_id=config.pad_token_id
    )
    val_dataset = TextDataset(
        val_tokens, 
        config.max_seq_len, 
        stride=config.max_seq_len, 
        pad_token_id=config.pad_token_id
    )
    
    print(f"📚 Data: {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens")
    print(f"📊 Data: {len(train_dataset)} train sequences, {len(val_dataset)} val sequences")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Create model with improved SSM if needed
    model = HybridModel(config).to(device)
    
    # Replace SSM layers with improved version
    for i, layer in enumerate(model.layers):
        if config.layer_pattern[i] == 'M':
            layer.mixer = ImprovedSSM(config)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🔧 Model: {total_params/1e6:.1f}M parameters ({trainable_params/1e6:.1f}M trainable)")
    
    # Improved optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),  # Better betas for language modeling
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.num_steps
    )
    
    scaler = GradScaler('cuda')
    
    # Metrics
    metrics = MetricsTracker()
    
    # Training loop
    model.train()
    step = 0
    accumulation_counter = 0
    accumulated_loss = 0
    pbar = tqdm(total=config.num_steps, desc="Training")
    
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10  # Early stopping patience
    
    while step < config.num_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        batch = batch.to(device, non_blocking=True)
        
        # Forward with gradient accumulation
        with autocast('cuda'):
            _, loss = model(batch, labels=batch)
            loss = loss / config.gradient_accumulation_steps
        
        accumulated_loss += loss.item()
        
        # Backward
        scaler.scale(loss).backward()
        
        accumulation_counter += 1
        
        # Update weights after accumulation
        if accumulation_counter % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            step += 1
            
            # Track metrics
            if step % config.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                metrics.update(
                    step=step,
                    train_loss=accumulated_loss,
                    grad_norm=grad_norm.item(),
                    learning_rate=current_lr,
                    memory_gb=torch.cuda.max_memory_allocated() / 1024**3
                )
                
                pbar.set_postfix({
                    'loss': f'{accumulated_loss:.4f}',
                    'grad': f'{grad_norm.item():.2f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                if args.use_wandb:
                    wandb.log(metrics.get_current_metrics(), step=step)
                
                accumulated_loss = 0
            
            # Evaluate
            if step % config.eval_every == 0:
                val_loss, val_ppl = evaluate_model(model, val_loader, config, device, config.num_eval_batches)
                metrics.update(step=step, val_loss=val_loss, val_perplexity=val_ppl)
                
                print(f"\n📈 Step {step}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': asdict(config),
                        'step': step,
                        'val_loss': val_loss,
                        'val_perplexity': val_ppl
                    }, f"{exp_dir}/checkpoints/best_model.pt")
                    print(f"  ✓ New best model saved!")
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print(f"\n⚠️ Early stopping triggered after {patience_counter} evaluations without improvement")
                        break
            
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
    
    # Final evaluation with more batches
    print("\n🔬 Final evaluation...")
    final_val_loss, final_val_ppl = evaluate_model(model, val_loader, config, device, len(val_loader))
    
    # Save final results
    results = {
        'pattern': config.layer_pattern,
        'pattern_name': config.pattern_name,
        'num_params': total_params,
        'final_val_loss': final_val_loss,
        'final_val_perplexity': final_val_ppl,
        'best_val_loss': best_val_loss,
        'total_steps': step,
        'early_stopped': patience_counter >= max_patience
    }
    
    with open(f"{exp_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    metrics.save(f"{exp_dir}/metrics.json")
    
    print(f"\n✅ Experiment complete!")
    print(f"📊 Results: val_loss={final_val_loss:.4f}, val_ppl={final_val_ppl:.2f}")
    print(f"🔧 Best val_loss={best_val_loss:.4f}")
    print(f"💾 Saved to {exp_dir}/")
    
    if args.use_wandb:
        wandb.log(results)
        wandb.finish()
    
    return results


if __name__ == "__main__":
    main()