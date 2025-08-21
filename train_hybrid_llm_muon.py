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
from torch.cuda.amp import autocast, GradScaler

@dataclass
class HybridConfig:
    vocab_size: int = 50257
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 8
    ssm_state_size: int = 16
    conv_kernel: int = 4
    expand_factor: int = 2
    layer_pattern: str = "MMAMAMAM"
    
    # Training
    max_seq_len: int = 512
    batch_size: int = 32
    num_documents: int = 5000
    muon_lr: float = 0.02  # Muon learning rate
    adamw_lr: float = 2e-3  # AdamW learning rate for embeddings/norms
    num_steps: int = 10000
    
    dropout: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 50
    
    def __post_init__(self):
        self.intermediate_size = self.expand_factor * self.hidden_size


# Add Newton-Schulz iteration function
@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


# Add Muon optimizer
class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


class SimpleSSM(nn.Module):
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
        self.A = nn.Parameter(torch.randn(self.intermediate_size, self.ssm_state_size))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = self.conv1d(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x = F.silu(x)
        
        x_proj = self.x_proj(x)
        delta, B, C = x_proj.split([1, self.ssm_state_size, self.ssm_state_size], dim=-1)
        delta = F.softplus(delta)
        
        # Simplified parallel SSM
        A = -torch.exp(self.A)
        decay = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
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
        self.mixer = SimpleSSM(config) if config.layer_pattern[layer_idx] == 'M' else SimpleAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        return x + self.dropout(self.mixer(self.norm(x)))


@torch.compile
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
    def __init__(self, tokens, max_length):
        self.tokens = tokens
        self.max_length = max_length
        
    def __len__(self):
        return len(self.tokens) // self.max_length
    
    def __getitem__(self, idx):
        start = idx * self.max_length
        return torch.tensor(self.tokens[start:start + self.max_length], dtype=torch.long)


def setup_hybrid_optimizers(model, config):
    """Setup Muon optimizer for 2D matrices and AdamW for others"""
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Use Muon for 2D weight matrices (except embeddings and small params)
            if (param.ndim == 2 and 
                'embed' not in name and 
                'norm' not in name and
                param.numel() > 1024):  # Skip very small matrices
                muon_params.append(param)
            else:
                adamw_params.append(param)
    
    print(f"Muon params: {sum(p.numel() for p in muon_params):,}")
    print(f"AdamW params: {sum(p.numel() for p in adamw_params):,}")
    
    optimizers = []
    if muon_params:
        optimizers.append(Muon(muon_params, lr=config.muon_lr, momentum=0.95))
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=config.adamw_lr, fused=True))
    
    return optimizers


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Config
    config = HybridConfig()
    
    # Load data
    print("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    
    # Tokenize
    all_tokens = []
    for i, item in enumerate(tqdm(dataset, total=config.num_documents, desc="Tokenizing")):
        if i >= config.num_documents:
            break
        tokens = tokenizer.encode(item["text"][:3000], add_special_tokens=False)
        all_tokens.extend(tokens)
    
    config.vocab_size = tokenizer.vocab_size
    
    # Create dataset
    train_dataset = TextDataset(all_tokens, config.max_seq_len)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Create model
    model = HybridModel(config)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters, {config.num_layers} layers ({config.layer_pattern})")
    
    # Setup hybrid optimizers (Muon + AdamW)
    base_model = model.module if hasattr(model, 'module') else model
    optimizers = setup_hybrid_optimizers(base_model, config)
    scaler = GradScaler()
    
    # Training loop
    model.train()
    step = 0
    pbar = tqdm(total=config.num_steps, desc="Training")
    
    while step < config.num_steps:
        for batch in train_loader:
            if step >= config.num_steps:
                break
                
            batch = batch.to(device, non_blocking=True)
            
            # Mixed precision training
            with autocast():
                _, loss = model(batch, labels=batch)
            
            # Backward
            scaler.scale(loss).backward()
            
            # Unscale and clip gradients
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # Step optimizers
            for optimizer in optimizers:
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
            
            scaler.update()
            
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
        with autocast():
            for _ in range(30):
                logits, _ = model(prompt)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                prompt = torch.cat([prompt, next_token], dim=1)
        
        print("\nGenerated:", tokenizer.decode(prompt[0]))


if __name__ == "__main__":
    main()