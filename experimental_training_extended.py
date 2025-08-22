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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Load environment variables from .env file
load_dotenv()

# Import your existing classes
from train_hybrid_llm import HybridConfig, SimpleSSM, SimpleAttention, HybridBlock, HybridModel
from shared_data import shared_data_manager


@dataclass
class ExtendedExperimentConfig(HybridConfig):
    """Extended config for longer experiments with more data"""
    experiment_name: str = "extended_pattern_ablation"
    pattern_name: str = "AMAMAMAMAMAMAMAM"  # Fixed: 16 characters for 16 layers
    eval_every: int = 1000  # Less frequent evaluation for longer runs (every 1k steps)
    save_every: int = 2000
    num_eval_batches: int = 100  # More eval batches for better estimates
    
    # Significantly increased data for 30k step training
    num_documents: int = 150000  # 3x increase from 50k for longer training
    num_steps: int = 30000  # 3x increase from 10k
    
    # Larger model for extended experiments
    hidden_size: int = 1024  # Increased from 768 (larger than base config)
    num_heads: int = 16  # Increased from 12 (proportional to hidden_size)
    ssm_state_size: int = 48  # Increased from 32 (larger SSM states)
    num_layers: int = 16  # Changed to 16 layers
    
    # Optimized hyperparameters for longer training
    dropout: float = 0.1
    learning_rate: float = 3e-4  # Slightly reduced for longer training
    weight_decay: float = 0.01
    warmup_steps: int = 3000  # 3x increase for longer schedule
    
    # Reduced batch size for data parallelism across 2 GPUs
    batch_size: int = 16  # Reduced from 32 to 16 per GPU
    gradient_accumulation_steps: int = 2  # Increased to maintain effective batch size
    
    # HF Hub
    hf_repo: str = "vukrosic/hybrid-llm-extended"
    
    def get_run_name(self):
        return f"{self.pattern_name}_{self.num_layers}L_extended_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


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
    """Enhanced metrics tracking for longer runs"""
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
        self.ema_alpha = 0.98  # Slower EMA for longer runs
    
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


def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Extended Hybrid LLM Training with Data Parallelism')
    parser.add_argument('--pattern', type=str, default='AMAMAMAMAMAMAM', help='Layer pattern')
    parser.add_argument('--name', type=str, default='amama_16L_extended', help='Experiment name')
    parser.add_argument('--steps', type=int, default=30000, help='Number of training steps')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--force_reload_data', action='store_true', help='Force reload data')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Only main process should create experiment directory and log
    if rank == 0:
        print(f"🚀 Starting Data Parallel Training on {world_size} GPUs")
        print(f"🔬 Pattern: {args.pattern} ({len(args.pattern)} layers)")
        print(f"📊 Batch size: {16} per GPU, {16 * world_size * 2} effective")
        
        # Create experiment directory
        exp_dir = f"experiments_extended/{args.name}"
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
        
        # Save config
        config = ExtendedExperimentConfig(
            layer_pattern=args.pattern,
            pattern_name=args.name,
            num_steps=args.steps
        )
        
        with open(f"{exp_dir}/config.json", 'w') as f:
            json.dump(asdict(config), f, indent=2)
    
    # Synchronize all processes
    if dist.is_initialized():
        dist.barrier()
    
    # Load config on all processes
    if rank != 0:
        exp_dir = f"experiments_extended/{args.name}"
        with open(f"{exp_dir}/config.json", 'r') as f:
            config_dict = json.load(f)
        config = ExtendedExperimentConfig(**config_dict)
    else:
        config = ExtendedExperimentConfig(
            layer_pattern=args.pattern,
            pattern_name=args.name,
            num_steps=args.steps
        )
    
    # Initialize wandb only on main process
    if args.use_wandb and rank == 0:
        wandb.init(
            project="hybrid-patterns-extended",
            name=config.get_run_name(),
            config=asdict(config),
            tags=["extended", "30k-steps", "16L", "data-parallel", "2x4090"]
        )
    
    # Setup
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print(f"🔬 Extended Experiment: {config.get_run_name()}")
        print(f"📊 Pattern: {config.layer_pattern} ({config.num_layers} layers)")
        print(f"⏱️ Steps: {config.num_steps} (30k extended)")
        print(f"📚 Documents: {config.num_documents} (3x increase)")
        print(f"🚀 Data Parallel: {world_size} GPUs, {config.batch_size} batch per GPU")
    
    # Load data using shared data manager
    if rank == 0:
        print("Loading extended dataset...")
    
    train_loader, val_loader = shared_data_manager.load_or_create_datasets(config, force_reload=args.force_reload_data)
    
    # Wrap with DistributedSampler for data parallelism
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_loader.dataset, shuffle=True)
        val_sampler = DistributedSampler(val_loader.dataset, shuffle=False)
        
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
    
    # Get tokenizer from shared manager
    tokenizer = shared_data_manager.get_tokenizer()
    config.pad_token_id = tokenizer.pad_token_id
    config.vocab_size = tokenizer.vocab_size
    
    # Create model with improved SSM if needed
    model = HybridModel(config).to(device)
    
    # Replace SSM layers with improved version AND ensure they're on the correct device
    for i, layer in enumerate(model.layers):
        if config.layer_pattern[i] == 'M':
            layer.mixer = ImprovedSSM(config).to(device)  # Explicitly move to device
    
    # Double-check all parameters are on the correct device
    model = model.to(device)
    
    # Verify all parameters are on the correct device before DDP
    for name, param in model.named_parameters():
        if param.device != device:
            print(f"⚠️ Parameter {name} is on {param.device}, moving to {device}")
            param.data = param.data.to(device)
    
    # Wrap with DDP for data parallelism
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if rank == 0:
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
    pbar = tqdm(total=config.num_steps, desc="Extended Training", ncols=140, leave=True, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 30  # Increased patience for longer runs (30k steps / 1k eval = 30 evaluations max)
    
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
                    train_loss=accumulated_loss / config.log_every,
                    grad_norm=grad_norm.item(),
                    learning_rate=current_lr,
                    memory_gb=torch.cuda.max_memory_allocated() / 1024**3
                )
                
                # Update progress bar with current metrics
                pbar.set_postfix({
                    'loss': f'{accumulated_loss / config.log_every:.4f}',
                    'grad': f'{grad_norm.item():.2f}',
                    'lr': f'{current_lr:.2e}',
                    'step': f'{step}/{config.num_steps}'
                }, refresh=True)
                
                if args.use_wandb:
                    wandb.log(metrics.get_current_metrics(), step=step)
                
                accumulated_loss = 0
            
            # Evaluate
            if step % config.eval_every == 0:
                val_loss, val_ppl = evaluate_model(model, val_loader, config, device, config.num_eval_batches)
                metrics.update(step=step, val_loss=val_loss, val_perplexity=val_ppl)
                
                # Update progress bar with validation info
                pbar.set_postfix({
                    'loss': f'{accumulated_loss / config.log_every:.4f}',
                    'grad': f'{grad_norm.item():.2f}',
                    'lr': f'{current_lr:.2e}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_ppl': f'{val_ppl:.2f}',
                    'best': f'{best_val_loss:.4f}'
                })
                
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
                    pbar.write(f"📈 Step {step}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f} ✓ New best!")
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        pbar.write(f"⚠️ Early stopping after {patience_counter} evals without improvement")
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
            
            # Ensure progress bar is visible
            if step % 200 == 0:  # Force refresh every 200 steps
                pbar.refresh()
    
    pbar.close()
    
    # Final evaluation with more batches
    print("\n🔬 Final extended evaluation...")
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
        'early_stopped': patience_counter >= max_patience,
        'training_type': 'extended_30k'
    }
    
    with open(f"{exp_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    metrics.save(f"{exp_dir}/metrics.json")
    
    print(f"\n✅ Extended experiment complete!")
    print(f"📊 Results: val_loss={final_val_loss:.4f}, val_ppl={final_val_ppl:.2f}")
    print(f"🔧 Best val_loss={best_val_loss:.4f}")
    print(f"💾 Saved to {exp_dir}/")
    
    if args.use_wandb:
        # Log final results
        wandb.log({
            'final_val_loss': final_val_loss,
            'final_val_perplexity': final_val_ppl,
            'best_val_loss': best_val_loss,
            'num_params': total_params,
            'total_steps': step
        })
        wandb.finish()
    
    # Cleanup
    cleanup_distributed()
    return results


if __name__ == "__main__":
    main()
