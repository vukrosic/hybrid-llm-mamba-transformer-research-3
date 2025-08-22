import os
# Set this before importing transformers to avoid the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from torch.amp import autocast, GradScaler

@dataclass
class HybridConfig:
    vocab_size: int = 50257
    hidden_size: int = 768  # Increased from 384 (2x larger)
    num_layers: int = 8
    num_heads: int = 12  # Increased from 8 (proportional to hidden_size)
    ssm_state_size: int = 32  # Increased from 16 (2x larger)
    conv_kernel: int = 4
    expand_factor: int = 2
    layer_pattern: str = "MMAMAMAM"
    
    # Training
    max_seq_len: int = 512
    batch_size: int = 32  # Increased for better GPU utilization
    num_documents: int = 50000
    learning_rate: float = 5e-4
    num_steps: int = 10000
    
    dropout: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 50
    
    def __post_init__(self):
        self.intermediate_size = self.expand_factor * self.hidden_size



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




class SimpleAttention(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout
        
    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.out_proj(attn.transpose(1, 2).reshape(B, L, -1))


class HybridBlock(nn.Module):
    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.mixer = ImprovedSSM(config) if config.layer_pattern[layer_idx] == 'M' else SimpleAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        return x + self.dropout(self.mixer(self.norm(x)))


@torch.compile  # JIT compile the model for faster execution
class HybridModel(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HybridBlock(config, i) for i in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Tie weights
        
        # Initialize
        self.apply(lambda m: torch.nn.init.normal_(m.weight, 0, 0.02) if isinstance(m, (nn.Linear, nn.Embedding)) else None)
    
    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids) * math.sqrt(self.config.hidden_size)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, self.config.vocab_size), labels[..., 1:].reshape(-1))
        return logits, loss


class TextDataset(Dataset):
    def __init__(self, tokens, max_length, stride=None):
        self.tokens = tokens
        self.max_length = max_length
        # Use stride for overlapping sequences (better for LM training)
        self.stride = stride if stride is not None else max_length
        
    def __len__(self):
        return max(1, (len(self.tokens) - self.max_length) // self.stride + 1)
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = min(start + self.max_length, len(self.tokens))
        # Pad if necessary
        chunk = self.tokens[start:end]
        if len(chunk) < self.max_length:
            chunk = chunk + [0] * (self.max_length - len(chunk))  # pad with 0
        return torch.tensor(chunk, dtype=torch.long)


def main():
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    torch.set_float32_matmul_precision('high')  # Use TF32 on Ampere GPUs
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Config
    config = HybridConfig()
    
    # Load data
    print("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    
    # Tokenize documents separately (better for train/val split)
    all_documents = []
    for i, item in enumerate(tqdm(dataset, total=config.num_documents, desc="Tokenizing")):
        if i >= config.num_documents:
            break
        tokens = tokenizer.encode(item["text"][:3000], add_special_tokens=False)
        all_documents.append(tokens)
    
    # Split by DOCUMENTS, not tokens (more realistic evaluation)
    n_train = int(len(all_documents) * 0.9)
    train_docs = all_documents[:n_train]
    val_docs = all_documents[n_train:]
    
    # Flatten
    train_tokens = [token for doc in train_docs for token in doc]
    val_tokens = [token for doc in val_docs for token in doc]
    
    config.vocab_size = tokenizer.vocab_size
    
    # Create datasets with overlapping sequences (stride = max_len // 2)
    train_dataset = TextDataset(train_tokens, config.max_seq_len, stride=config.max_seq_len // 2)
    val_dataset = TextDataset(val_tokens, config.max_seq_len, stride=config.max_seq_len)
    
    print(f" Data: {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens")
    print(f"📊 Data: {len(train_dataset)} train sequences, {len(val_dataset)} val sequences")
    
    # Create DataLoader for training
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,  # Reduced from 8 to 2 for better compatibility
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Create model - use DataParallel for multi-GPU
    model = HybridModel(config)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters, {config.num_layers} layers ({config.layer_pattern})")
    
    # Optimizer and AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, fused=True)  # Fused optimizer
    scaler = GradScaler('cuda')
    
    # Training loop
    model.train()
    step = 0
    pbar = tqdm(total=config.num_steps, desc="Training")
    
    while step < config.num_steps:
        for batch in train_loader:
            if step >= config.num_steps:
                break
                
            batch = batch.to(device, non_blocking=True)  # Async transfer
            
            # Mixed precision training
            with autocast('cuda'):
                _, loss = model(batch, labels=batch)
            
            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            step += 1
            if step % config.log_every == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            pbar.update(1)
    
    pbar.close()
    
    # Save model
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), "model.pt")
    print("Model saved to model.pt")
    
    # Quick generation test
    model.eval()
    with torch.no_grad():
        prompt = tokenizer.encode("The future of AI is", return_tensors="pt").to(device)
        with autocast('cuda'):
            for _ in range(30):
                logits, _ = model(prompt)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                prompt = torch.cat([prompt, next_token], dim=1)
        
        print("\nGenerated:", tokenizer.decode(prompt[0]))


if __name__ == "__main__":
    main()