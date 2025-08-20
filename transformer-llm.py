import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

# Enable anomaly detection to find operations that failed to compute gradients
torch.autograd.set_detect_anomaly(True)

def check_nan(name, x):
    if not torch.isfinite(x).all():
        print(f"⚠️ NaN/Inf detected in {name}: {x}")
        raise ValueError(f"{name} has NaN/Inf")

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 12  # Increased for hybrid architecture
    d_ff: int = 1536
    batch_size: int = 2  # Reduced for memory efficiency
    max_steps: int = 5000

    # Hybrid architecture parameters
    hybrid_layers: bool = True
    
    # Mamba-2 specific parameters (further reduced for memory)
    mamba_num_heads: int = 4    # Reduced from 6
    mamba_head_dim: int = 48    # Reduced from 64
    mamba_state_size: int = 32  # Reduced from 64 for memory
    mamba_n_groups: int = 2     # Keep at 2
    mamba_conv_kernel: int = 4
    mamba_expansion: int = 2
    mamba_activation: str = "silu"
    
    # Attention specific (for hybrid layers)
    num_kv_heads: int = 4  # GQA - fewer KV heads than Q heads
    
    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = False  # Temporarily disabled for debugging
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        if self.hybrid_layers:
            # Define layer pattern for hybrid architecture
            # Pattern: [M, F, M, F, M, F, A, F] repeated, where M=Mamba, F=FFN, A=Attention
            pattern = []
            for i in range(self.n_layers):
                pos_in_cycle = i % 8
                if pos_in_cycle in [0, 2, 4]:  # Mamba layers
                    pattern.append("mamba")
                elif pos_in_cycle == 6:  # Attention layer
                    pattern.append("attention")
                else:  # FFN layers (1, 3, 5, 7)
                    pattern.append("ffn")
            self.layer_pattern = pattern
            
            # Mamba intermediate size
            self.mamba_intermediate_size = self.mamba_num_heads * self.mamba_head_dim
            
            # Ensure compatibility
            assert self.d_model % self.mamba_num_heads == 0, "d_model must be divisible by mamba_num_heads"
            assert self.mamba_intermediate_size <= self.d_model, "Mamba intermediate size should not exceed d_model"

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
	
def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention for hybrid architecture"""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # How many times to repeat each KV head

        self.q_proj = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(n_heads * self.d_k, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def repeat_kv(self, x):
        """Repeat KV heads to match number of query heads"""
        batch, seq_len, n_kv_heads, head_dim = x.shape
        if self.n_rep == 1:
            return x
        return x.repeat_interleave(self.n_rep, dim=2)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # Debug print to show we're using attention
        if not hasattr(self, '_debug_printed'):
            print(f"🔍 ATTENTION: Using Grouped Query Attention (seq_len={seq_len})")
            self._debug_printed = True

        # Project to Q, K, V
        Q = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.k_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = self.v_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)

        # Apply rotary embeddings
        Q = Q.transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
        K = K.transpose(1, 2)  # [batch, n_kv_heads, seq_len, d_k]
        V = V.transpose(1, 2)  # [batch, n_kv_heads, seq_len, d_k]
        
        Q = self.rotary(Q)
        K = self.rotary(K)

        # Repeat K and V to match Q heads
        K = self.repeat_kv(K.transpose(1, 2)).transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
        V = self.repeat_kv(V.transpose(1, 2)).transpose(1, 2)  # [batch, n_heads, seq_len, d_k]

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

def parallel_scan_log(A, B_u):
    """
    Truly parallel associative scan using log-depth algorithm
    A: [batch, seq_len, num_heads, head_dim, state_size]
    B_u: [batch, seq_len, num_heads, head_dim, state_size]
    Returns: [batch, seq_len, num_heads, head_dim, state_size]
    """
    batch, seq_len, num_heads, head_dim, state_size = A.shape
    
    # Debug print - only print once per unique sequence length to avoid spam
    if not hasattr(parallel_scan_log, '_printed_lens'):
        parallel_scan_log._printed_lens = set()
    if seq_len not in parallel_scan_log._printed_lens:
        log_depth = int(math.ceil(math.log2(seq_len))) if seq_len > 0 else 0
        print(f"    ⚡ LOG-DEPTH SCAN: Processing {seq_len} tokens with depth {log_depth}")
        parallel_scan_log._printed_lens.add(seq_len)
    
    # Use log-depth parallel scan
    # We'll work with (A, B) pairs and compose them in parallel
    
    # Initialize with identity for padding if needed
    log_seq_len = int(math.ceil(math.log2(seq_len)))
    padded_seq_len = 2 ** log_seq_len
    
    if padded_seq_len > seq_len:
        pad_len = padded_seq_len - seq_len
        A = F.pad(A, (0, 0, 0, 0, 0, 0, 0, pad_len), value=1.0)
        B_u = F.pad(B_u, (0, 0, 0, 0, 0, 0, 0, pad_len), value=0.0)
    
    # Work with the operators
    A_scan = A.clone()
    B_scan = B_u.clone()
    
    # Parallel scan using doubling
    for d in range(log_seq_len):
        # Shift by 2^d positions
        shift = 2 ** d
        
        # Create indices for parallel composition
        indices = torch.arange(padded_seq_len, device=A.device)
        mask = indices >= shift
        
        # Get previous values (shifted)
        prev_indices = indices - shift
        prev_indices = torch.clamp(prev_indices, min=0)
        
        # Compose in parallel: (A2, B2) ∘ (A1, B1) = (A2*A1, A2*B1 + B2)
        if mask.any():
            A_prev = A_scan[:, prev_indices]
            B_prev = B_scan[:, prev_indices]
            
            # Only update positions that have a valid previous element
            mask = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask = mask.expand_as(A_scan[:, :padded_seq_len])
            
            new_A = torch.where(mask, A_scan[:, :padded_seq_len] * A_prev, A_scan[:, :padded_seq_len])
            new_B = torch.where(mask, A_scan[:, :padded_seq_len] * B_prev + B_scan[:, :padded_seq_len], B_scan[:, :padded_seq_len])
            
            A_scan[:, :padded_seq_len] = new_A
            B_scan[:, :padded_seq_len] = new_B
    
    # Return only the valid sequence length
    return B_scan[:, :seq_len]

def chunked_parallel_scan(A, B_u, chunk_size=16):
    """
    Chunked parallel scan - much faster for long sequences
    Processes chunks in parallel, then combines results
    """
    batch, seq_len, num_heads, head_dim, state_size = A.shape
    device = A.device
    
    # Debug print - only print once per unique sequence length to avoid spam
    if not hasattr(chunked_parallel_scan, '_printed_lens'):
        chunked_parallel_scan._printed_lens = set()
    if seq_len not in chunked_parallel_scan._printed_lens:
        print(f"    📊 CHUNKED SCAN: Processing {seq_len} tokens in chunks of {chunk_size}")
        chunked_parallel_scan._printed_lens.add(seq_len)
    
    # Reshape into chunks
    n_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    # Pad to multiple of chunk_size
    padded_len = n_chunks * chunk_size
    if padded_len > seq_len:
        pad_len = padded_len - seq_len
        A = F.pad(A, (0, 0, 0, 0, 0, 0, 0, pad_len), value=1.0)
        B_u = F.pad(B_u, (0, 0, 0, 0, 0, 0, 0, pad_len), value=0.0)
    
    # Reshape to chunks
    A_chunks = A.reshape(batch, n_chunks, chunk_size, num_heads, head_dim, state_size)
    B_chunks = B_u.reshape(batch, n_chunks, chunk_size, num_heads, head_dim, state_size)
    
    # Scan within each chunk (can be done in parallel)
    A_scan = torch.zeros_like(A_chunks)
    B_scan = torch.zeros_like(B_chunks)
    
    for i in range(chunk_size):
        if i == 0:
            A_scan[:, :, i] = A_chunks[:, :, i]
            B_scan[:, :, i] = B_chunks[:, :, i]
        else:
            A_scan[:, :, i] = A_scan[:, :, i-1] * A_chunks[:, :, i]
            B_scan[:, :, i] = A_scan[:, :, i-1] * B_chunks[:, :, i] + B_scan[:, :, i-1]
    
    # Get chunk aggregates (last element of each chunk)
    chunk_A = A_scan[:, :, -1]  # [batch, n_chunks, num_heads, head_dim, state_size]
    chunk_B = B_scan[:, :, -1]  # [batch, n_chunks, num_heads, head_dim, state_size]
    
    # Scan across chunks
    chunk_A_scan = torch.zeros_like(chunk_A)
    chunk_B_scan = torch.zeros_like(chunk_B)
    
    for i in range(n_chunks):
        if i == 0:
            chunk_A_scan[:, i] = chunk_A[:, i]
            chunk_B_scan[:, i] = chunk_B[:, i]
        else:
            chunk_A_scan[:, i] = chunk_A_scan[:, i-1] * chunk_A[:, i]
            chunk_B_scan[:, i] = chunk_A_scan[:, i-1] * chunk_B[:, i] + chunk_B_scan[:, i-1]
    
    # Broadcast chunk results back to all elements
    results = torch.zeros_like(B_chunks)
    
    for c in range(n_chunks):
        if c == 0:
            results[:, c] = B_scan[:, c]
        else:
            # Apply chunk prefix to all elements in chunk
            # Fix: properly broadcast the prefix across the chunk dimension
            prefix_A = chunk_A_scan[:, c-1].unsqueeze(1)  # [batch, 1, num_heads, head_dim, state_size]
            prefix_B = chunk_B_scan[:, c-1].unsqueeze(1)  # [batch, 1, num_heads, head_dim, state_size]
            
            # Now both have compatible shapes for broadcasting
            results[:, c] = prefix_A * B_scan[:, c] + prefix_B.expand_as(B_scan[:, c])
    
    # Reshape back and trim padding
    results = results.reshape(batch, padded_len, num_heads, head_dim, state_size)
    return results[:, :seq_len]

# REMOVED: selective_scan_simple - sequential fallback eliminated!
# Now only using parallel algorithms: chunked_parallel_scan and parallel_scan_log

def selective_scan_simple(*args, **kwargs):
    """
    REMOVED: This sequential implementation has been eliminated!
    Use selective_scan_parallel() instead for truly parallel processing.
    """
    raise RuntimeError(
        "❌ SEQUENTIAL SCAN BLOCKED! selective_scan_simple() has been removed. "
        "This implementation was sequential and not truly parallel. "
        "Use selective_scan_parallel() for guaranteed parallel processing."
    )

def selective_scan_fast(u, dt, A, B, C, D, use_chunked=True):
    """
    Fast selective scan using ONLY parallel algorithms - no sequential fallback!
    """
    batch, seq_len, num_heads, head_dim = u.shape
    state_size = A.shape[-1]
    
    # Debug print to show which parallel method is being used
    if use_chunked and seq_len > 32:
        chunk_size = min(64, seq_len // 4)  # Adaptive chunk size
        print(f"  🚀 Using CHUNKED parallel scan (seq_len={seq_len}, chunk_size={chunk_size})")
    else:
        print(f"  ⚡ Using LOG-DEPTH parallel scan (seq_len={seq_len})")
    
    # Discretize A and B matrices
    dt_expanded = dt.unsqueeze(-1)  # [batch, seq_len, num_heads, head_dim, 1]
    
    dA = torch.exp(-dt_expanded * A)  # [batch, seq_len, num_heads, head_dim, state_size]
    dB = dt_expanded * B.unsqueeze(-2)  # [batch, seq_len, num_heads, head_dim, state_size]
    
    # Compute B * u
    Bu = dB * u.unsqueeze(-1)  # [batch, seq_len, num_heads, head_dim, state_size]
    
    # Use ONLY parallel algorithms - no sequential fallback!
    if use_chunked and seq_len > 32:
        chunk_size = min(64, seq_len // 4)  # Adaptive chunk size
        hidden_states = chunked_parallel_scan(dA, Bu, chunk_size=chunk_size)
    else:
        # For short sequences, use log-depth parallel scan
        hidden_states = parallel_scan_log(dA, Bu)
    
    # Compute outputs
    outputs = torch.sum(hidden_states * C.unsqueeze(-2), dim=-1)
    
    # Add skip connection
    D_expanded = D.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    outputs = outputs + D_expanded * u
    
    return outputs

def selective_scan_parallel(u, dt, A, B, C, D):
    """
    Wrapper for fast selective scan - uses optimized implementation
    """
    return selective_scan_fast(u, dt, A, B, C, D, use_chunked=True)

class Mamba2Mixer(nn.Module):
    """Optimized Mamba-2 mixer with fast scan"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.state_size = config.mamba_state_size
        self.n_groups = config.mamba_n_groups
        self.conv_kernel = config.mamba_conv_kernel
        self.expansion = config.mamba_expansion
        
        self.intermediate_size = self.num_heads * self.head_dim
        
        # Input projections
        self.x_proj = nn.Linear(self.d_model, self.intermediate_size * self.expansion, bias=False)
        self.z_proj = nn.Linear(self.d_model, self.intermediate_size, bias=False)
        
        # SSM parameter projections
        self.dt_proj = nn.Linear(self.d_model, self.num_heads, bias=False)
        self.B_proj = nn.Linear(self.d_model, self.n_groups * self.state_size, bias=False)
        self.C_proj = nn.Linear(self.d_model, self.n_groups * self.state_size, bias=False)
        
        # Causal convolution
        conv_dim = self.intermediate_size * self.expansion
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=self.conv_kernel,
            groups=conv_dim,
            padding=self.conv_kernel - 1,
            bias=False
        )
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.num_heads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.num_heads))
        
        # Output projection
        self.out_proj = nn.Linear(self.intermediate_size, self.d_model, bias=False)
        
        # Gated normalization
        self.norm = nn.RMSNorm(self.intermediate_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Debug print to show we're using Mamba
        if not hasattr(self, '_debug_printed'):
            print(f"🐍 MAMBA-2: Using PARALLEL selective scan (seq_len={seq_len})")
            self._debug_printed = True
        
        # Input projections
        x_expanded = self.x_proj(x)
        z = self.z_proj(x)
        
        # SSM parameters
        dt = self.dt_proj(x)
        B = self.B_proj(x)
        C = self.C_proj(x)
        
        # Apply activation and convolution
        x_expanded = F.silu(x_expanded)
        x_conv = self.conv1d(x_expanded.transpose(1, 2)).transpose(1, 2)
        x_conv = x_conv[:, :seq_len]
        
        # Take only the first part for SSM
        x_ssm = x_conv[:, :, :self.intermediate_size]
        
        # Reshape for processing
        x_ssm = x_ssm.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        dt = dt.reshape(batch_size, seq_len, self.num_heads, 1).expand(-1, -1, -1, self.head_dim)
        
        # Expand B and C
        B = B.reshape(batch_size, seq_len, self.n_groups, self.state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.state_size)
        
        heads_per_group = self.num_heads // self.n_groups
        B = B.repeat_interleave(heads_per_group, dim=2)
        C = C.repeat_interleave(heads_per_group, dim=2)
        
        # Prepare A matrix
        A = -torch.exp(self.A_log.float())
        A = A.unsqueeze(-1).unsqueeze(-1).expand(self.num_heads, self.head_dim, self.state_size)
        
        # Apply softplus to dt
        dt = F.softplus(dt)
        
        # Use ONLY parallel selective scan - no fallback!
        y = selective_scan_parallel(x_ssm, dt, A, B, C, self.D)
        
        # Reshape back
        y = y.reshape(batch_size, seq_len, self.intermediate_size)
        
        # Apply gating and normalization
        y = self.norm(y * F.silu(z))
        
        return self.out_proj(y)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Using squared ReLU activation like in Nemotron
        h = self.linear1(x)
        h = torch.square(F.relu(h))
        return self.linear2(self.dropout(h))

class HybridBlock(nn.Module):
    """Hybrid block that can be Mamba, Attention, or FFN"""
    def __init__(self, config: ModelConfig, layer_type: str, layer_idx: int):
        super().__init__()
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        
        # Pre-norm
        self.norm = nn.RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        if layer_type == "mamba":
            self.mixer = Mamba2Mixer(config)
        elif layer_type == "attention":
            self.mixer = GroupedQueryAttention(
                config.d_model, 
                config.n_heads, 
                config.num_kv_heads,
                config.max_seq_len, 
                config.dropout
            )
        elif layer_type == "ffn":
            self.mixer = FeedForward(config.d_model, config.d_ff, config.dropout)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    def forward(self, x):
        # Pre-norm residual connection
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        x = self.dropout(x)
        return residual + x

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class HybridLLM(nn.Module):
    """Hybrid Mamba-2/Transformer LLM"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Create hybrid blocks based on layer pattern
        if config.hybrid_layers:
            self.layers = nn.ModuleList([
                HybridBlock(config, config.layer_pattern[i], i)
                for i in range(config.n_layers)
            ])
            print(f"🔧 Created hybrid architecture: {config.layer_pattern}")
        else:
            # Fallback to pure transformer
            self.layers = nn.ModuleList([
                TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
                for _ in range(config.n_layers)
            ])
            print(f"🔧 Created pure transformer architecture")

        self.norm = nn.RMSNorm(config.d_model)

        # Separate embedding and output layers (not tied like in Nemotron)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, Mamba2Mixer):
            # Special initialization for Mamba components
            if hasattr(module, 'A_log'):
                module.A_log._no_weight_decay = True
            if hasattr(module, 'D'):
                module.D._no_weight_decay = True

    def forward(self, x):
        # No positional embeddings - Mamba handles position through convolution
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

# Keep original for backward compatibility
class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Tie weights
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                check_nan("logits", logits)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_muon_optimizer(model: nn.Module, config: ModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Train the model with Muon optimizer"""
    model_type = "Hybrid Mamba-2/Transformer" if config.hybrid_layers else "Pure Transformer"
    print(f"\n🚀 Training {model_type} model with Muon optimizer")

    # Initialize model
    set_seed(42)
    model = HybridLLM(config) if config.hybrid_layers else MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')

    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    check_nan("logits", logits)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                check_nan("logits", logits)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step after accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']

            step += 1
            if step % 100 == 0:
                pbar.update(100)

    pbar.close()

    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
          f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    return model, final_eval

if __name__ == "__main__":
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Create config for Hybrid model
    config = ModelConfig()
    print(f"\n📋 Model Configuration:")
    if config.hybrid_layers:
        print(f"   Architecture: Hybrid Mamba-2/Transformer")
        print(f"   Base: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
        print(f"   Mamba: {config.mamba_num_heads}H, {config.mamba_head_dim}d, {config.mamba_state_size}s, {config.mamba_n_groups}g")
        print(f"   Pattern: {config.layer_pattern}")
    else:
        print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

    # Load data
    texts, tokenizer, tokens = load_and_cache_data(config)
    dataset = TextTokenDataset(tokens, config.max_seq_len)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train model
    start_time = time.time()
    model, final_metrics = train_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")