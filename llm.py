import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
import time
import os

@dataclass
class HybridConfig:
    vocab_size: int = 50257
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    
    # Mamba specific
    ssm_state_size: int = 16
    conv_kernel: int = 4
    expand_factor: int = 2
    
    # Layer pattern: "M" for Mamba, "A" for Attention
    layer_pattern: str = "MAMAMM"  # Mix of Mamba and Attention
    
    # Training - EXTENDED OPTIONS
    max_seq_len: int = 512
    batch_size: int = 8
    num_documents: int = 5000  # Increased from 100
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5  # For cosine annealing
    
    # EXTENDED TRAINING OPTIONS
    num_epochs: int = 10  # Increased from 1
    max_steps: int = None  # If set, overrides num_epochs
    gradient_accumulation_steps: int = 4  # Accumulate gradients
    
    # Evaluation and checkpointing
    eval_every: int = 500  # Evaluate every N steps
    save_every: int = 1000  # Save checkpoint every N steps
    log_every: int = 10  # Log metrics every N steps
    
    # Regularization
    dropout: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 500
    
    def __post_init__(self):
        assert len(self.layer_pattern) == self.num_layers
        self.intermediate_size = self.expand_factor * self.hidden_size


# [Previous SimpleSSM, SimpleAttention, HybridBlock classes remain the same...]
# I'll include just the key parts that change

class SimpleSSM(nn.Module):
    """Simplified State Space Model (Mamba-like) block"""
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel = config.conv_kernel
        
        # Input projection
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
            bias=False
        )
        
        # SSM parameters projection
        self.x_proj = nn.Linear(self.intermediate_size, config.ssm_state_size * 2 + 1, bias=False)
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(self.intermediate_size, self.ssm_state_size))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        
        # Output projection
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Input gating
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # 2. Convolution
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        
        # 3. Apply activation
        x_conv = F.silu(x_conv)
        
        # 4. SSM projection
        x_proj = self.x_proj(x_conv)
        delta, B, C = torch.split(
            x_proj, 
            [1, self.ssm_state_size, self.ssm_state_size], 
            dim=-1
        )
        
        # 5. Discretization
        delta = F.softplus(delta)
        
        # 6. Simplified SSM computation (for efficiency in training)
        A = -torch.exp(self.A)
        
        # Parallel computation (approximation for training efficiency)
        delta_expanded = delta.unsqueeze(-1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)
        
        # Compute decay factors
        decay = torch.exp(delta_expanded * A_expanded)
        
        # Input contribution
        x_expanded = x_conv.unsqueeze(-1)
        B_expanded = B.unsqueeze(2)
        input_contrib = x_expanded * B_expanded
        
        # Simplified state computation
        states = input_contrib * decay
        
        # Output computation
        C_expanded = C.unsqueeze(2)
        y = (states * C_expanded).sum(dim=-1)
        
        # Add skip connection
        D_expanded = self.D.unsqueeze(0).unsqueeze(0)
        y = y + x_conv * D_expanded
        
        # 7. Output gating
        y = y * F.silu(z)
        
        # 8. Output projection
        output = self.out_proj(y)
        
        final_state = states[:, -1, :, :] if state is not None else None
            
        return output, final_state


class SimpleAttention(nn.Module):
    """Simplified Multi-Head Attention"""
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True if mask is None else False
        )
        
        # Reshape and project output
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn)
        
        return output


class HybridBlock(nn.Module):
    """A block that can be either Mamba or Attention"""
    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = config.layer_pattern[layer_idx]
        
        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # Choose mixer based on pattern
        if self.block_type == 'M':
            self.mixer = SimpleSSM(config)
            self.is_mamba = True
        elif self.block_type == 'A':
            self.mixer = SimpleAttention(config)
            self.is_mamba = False
        else:
            raise ValueError(f"Unknown block type: {self.block_type}")
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        residual = x
        x = self.norm(x)
        
        if self.is_mamba:
            output, new_state = self.mixer(x, state)
        else:
            output = self.mixer(x)
            new_state = None
        
        output = self.dropout(output)
        output = residual + output
        
        return output, new_state


class HybridModel(nn.Module):
    """Minimal Hybrid Mamba-Transformer Model"""
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Hybrid layers
        self.layers = nn.ModuleList([
            HybridBlock(config, idx) for idx in range(config.num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed.weight
        
        # Initialize
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # Embed tokens
        x = self.embed(input_ids) * math.sqrt(self.config.hidden_size)
        x = self.embed_dropout(x)
        
        # Process through layers
        for layer in self.layers:
            x, _ = layer(x, None)
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100  # Ignore padding tokens
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 0.8):
        """Simple generation method"""
        self.eval()
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Get logits
            logits, _ = self(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Break on EOS
            if next_token.item() == 2:
                break
        
        self.train()
        return generated


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts
        print("Tokenizing texts...")
        self.examples = []
        for text in tqdm(texts):
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            self.examples.append(tokens["input_ids"].squeeze(0))
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_data(config: HybridConfig):
    """Load tokenizer and dataset"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading {config.num_documents} documents...")
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", 
        "cosmopedia-v2", 
        split="train", 
        streaming=True, 
        token=False
    )
    
    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])
        if (i + 1) % 500 == 0:
            print(f"  Loaded {i + 1} documents...")
    
    print(f"Loaded {len(texts)} documents")
    
    # Update vocab size in config
    config.vocab_size = tokenizer.vocab_size
    
    return texts, tokenizer


class Trainer:
    """Extended trainer with more features"""
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Calculate total steps
        steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
        total_steps = steps_per_epoch * config.num_epochs
        
        if config.max_steps:
            total_steps = min(total_steps, config.max_steps)
        
        # Learning rate scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=config.min_learning_rate / config.learning_rate
        )
        
        # Tracking
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            batch = batch.to(self.device)
            _, loss = self.model(batch, labels=batch)
            
            batch_tokens = (batch != -100).sum()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 20))
        
        self.model.train()
        return avg_loss, perplexity
    
    def save_checkpoint(self, path=None):
        """Save model checkpoint"""
        if path is None:
            path = f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        # Fix for PyTorch 2.6+ compatibility
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            # Fallback: try with safe globals
            import torch.serialization
            torch.serialization.add_safe_globals(['__main__.HybridConfig'])
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        print(f"  Loaded checkpoint from {path} (step {self.global_step})")
    
    def train(self):
        """Extended training loop"""
        self.model.train()
        
        # Calculate max steps
        max_steps = self.config.max_steps if self.config.max_steps else len(self.train_loader) * self.config.num_epochs
        
        accumulation_counter = 0
        accumulated_loss = 0
        
        # Create progress bar for all steps
        pbar = tqdm(total=max_steps, desc="Training", initial=self.global_step)
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            epoch_tokens = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                if self.global_step >= max_steps:
                    break
                
                batch = batch.to(self.device)
                
                # Forward pass
                _, loss = self.model(batch, labels=batch)
                loss = loss / self.config.gradient_accumulation_steps
                accumulated_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                accumulation_counter += 1
                
                # Update weights after accumulation
                if accumulation_counter % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update metrics
                    self.global_step += 1
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    # Log metrics
                    if self.global_step % self.config.log_every == 0:
                        avg_loss = accumulated_loss
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{current_lr:.2e}',
                            'epoch': epoch + 1
                        })
                        self.training_history['train_loss'].append(avg_loss)
                        self.training_history['learning_rate'].append(current_lr)
                        accumulated_loss = 0
                    
                    # Evaluate
                    if self.global_step % self.config.eval_every == 0:
                        val_loss, perplexity = self.evaluate()
                        self.training_history['val_loss'].append(val_loss)
                        
                        print(f"\n  Step {self.global_step}: Val Loss = {val_loss:.4f}, Perplexity = {perplexity:.2f}")
                        
                        # Save best model
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self.save_checkpoint("best_model.pt")
                    
                    # Regular checkpoint
                    if self.global_step % self.config.save_every == 0:
                        self.save_checkpoint()
                    
                    pbar.update(1)
                
                # Early stopping based on max_steps
                if self.global_step >= max_steps:
                    break
            
            if self.global_step >= max_steps:
                break
            
            # End of epoch evaluation
            val_loss, perplexity = self.evaluate()
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}: Val Loss = {val_loss:.4f}, Perplexity = {perplexity:.2f}")
        
        pbar.close()
        return self.training_history


def main():
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Config - EXTENDED FOR LONGER TRAINING
    config = HybridConfig(
        hidden_size=384,  # Increased model size
        num_layers=8,  # More layers
        num_heads=8,
        layer_pattern="MMAMAMAM",  # 8 layers pattern
        
        # Data
        batch_size=16,
        num_documents=5000,  # Much more data
        max_seq_len=512,
        
        # Training duration
        num_epochs=10,  # Train for 10 epochs
        max_steps=10000,  # Or stop at 10k steps
        
        # Optimization
        learning_rate=5e-4,
        min_learning_rate=5e-5,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        weight_decay=0.01,
        
        # Checkpointing
        eval_every=250,
        save_every=500,
        log_every=10
    )
    
    # Load data
    texts, tokenizer = load_data(config)
    
    # Split data 90/10
    val_size = len(texts) // 10
    train_texts = texts[:-val_size]
    val_texts = texts[-val_size:]
    
    print(f"Train: {len(train_texts)} documents, Val: {len(val_texts)} documents")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length=config.max_seq_len)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=config.max_seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Larger batch for evaluation
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = HybridModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n{'='*60}")
    print(f"Model Configuration:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Layer pattern: {config.layer_pattern}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate} -> {config.min_learning_rate}")
    print(f"{'='*60}\n")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Optional: Load from checkpoint to continue training
    if os.path.exists("checkpoint_latest.pt"):
        trainer.load_checkpoint("checkpoint_latest.pt")
        print("Resuming from checkpoint...")
    
    # Train
    print("Starting training...")
    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Training completed in {training_time/3600:.2f} hours")
    print(f"Final best validation loss: {trainer.best_loss:.4f}")
    print(f"{'='*60}\n")
    
    # Generate samples with best model
    trainer.load_checkpoint("best_model.pt")
    model.eval()
    
    prompts = [
        "The universe is",
        "In the beginning",
        "Once upon a time",
        "The key to success is"
    ]
    
    print("Generating samples from best model:")
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
    
    # Save final model
    trainer.save_checkpoint("final_model.pt")
    print("\nTraining complete! Models saved.")


if __name__ == "__main__":
    main()