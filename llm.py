import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class HybridConfig:
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    
    # Mamba specific
    ssm_state_size: int = 16
    conv_kernel: int = 4
    expand_factor: int = 2
    
    # Layer pattern: "M" for Mamba, "A" for Attention
    # Example: "MAMAMAMA" alternates between Mamba and Attention
    layer_pattern: str = "MMAAMMAAMMAA"  # 12 layers: mostly Mamba with some attention
    
    max_seq_len: int = 2048
    dropout: float = 0.1
    
    def __post_init__(self):
        assert len(self.layer_pattern) == self.num_layers
        self.intermediate_size = self.expand_factor * self.hidden_size


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
        xz = self.in_proj(x)  # [B, L, 2*intermediate]
        x, z = xz.chunk(2, dim=-1)  # Each is [B, L, intermediate]
        
        # 2. Convolution
        x_conv = x.transpose(1, 2)  # [B, intermediate, L]
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Remove padding
        x_conv = x_conv.transpose(1, 2)  # [B, L, intermediate]
        
        # 3. Apply activation
        x_conv = F.silu(x_conv)
        
        # 4. SSM projection
        x_proj = self.x_proj(x_conv)  # [B, L, state_size*2 + 1]
        delta, B, C = torch.split(
            x_proj, 
            [1, self.ssm_state_size, self.ssm_state_size], 
            dim=-1
        )
        
        # 5. Discretization
        delta = F.softplus(delta)  # [B, L, 1]
        
        # 6. SSM step (simplified - no proper state tracking for brevity)
        # This is a very simplified version of the SSM computation
        A = -torch.exp(self.A)  # [intermediate, state_size]
        
        # Initialize state if needed
        if state is None:
            state = torch.zeros(
                batch_size, self.intermediate_size, self.ssm_state_size,
                device=x.device, dtype=x.dtype
            )
        
        # Simple SSM: for each timestep (this is inefficient but clear)
        outputs = []
        for t in range(seq_len):
            # Update state: state = state * exp(delta * A) + x * B
            # Expand dimensions properly for broadcasting
            delta_t = delta[:, t, :].unsqueeze(1)  # [B, 1, 1]
            A_exp = torch.exp(delta_t * A.unsqueeze(0))  # [B, intermediate, state_size]
            
            x_t = x_conv[:, t, :].unsqueeze(-1)  # [B, intermediate, 1]
            B_t = B[:, t, :].unsqueeze(1)  # [B, 1, state_size]
            
            state = state * A_exp + x_t * B_t
            
            # Compute output: y = state * C
            C_t = C[:, t, :].unsqueeze(1)  # [B, 1, state_size]
            y_t = (state * C_t).sum(dim=-1)  # [B, intermediate]
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [B, L, intermediate]
        
        # Add skip connection with proper broadcasting
        D_expanded = self.D.unsqueeze(0).unsqueeze(0)  # [1, 1, intermediate]
        y = y + x_conv * D_expanded
        
        # 7. Output gating
        y = y * F.silu(z)
        
        # 8. Output projection
        output = self.out_proj(y)
        
        return output, state


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
    """A block that can be either Mamba or Attention based on configuration"""
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
        # Pre-norm architecture
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
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        states: Optional[list] = None,
        return_states: bool = False
    ):
        # Embed tokens
        x = self.embed(input_ids) * math.sqrt(self.config.hidden_size)
        x = self.embed_dropout(x)
        
        # Initialize states if needed
        if states is None:
            states = [None] * self.config.num_layers
        
        # Process through layers
        new_states = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, states[i])
            new_states.append(state)
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        if return_states:
            return logits, new_states
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0):
        """Simple generation method with temperature"""
        states = None
        generated = input_ids
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Get logits for the last position
                if states is None:
                    # First iteration - process full sequence
                    logits, states = self(generated, states, return_states=True)
                    next_token_logits = logits[:, -1, :]
                else:
                    # Subsequent iterations - only process new token
                    last_token = generated[:, -1:]
                    logits, states = self(last_token, states, return_states=True)
                    next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create config with hybrid pattern
    config = HybridConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=6,
        num_heads=8,
        layer_pattern="MAMAMM",  # Mix of Mamba and Attention layers
        max_seq_len=512
    )
    
    # Create model
    model = HybridModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Layer pattern: {config.layer_pattern}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    # Forward pass
    logits = model(input_ids)
    print(f"Output shape: {logits.shape}")
    
    # Compute loss
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    prompt = torch.tensor([[1, 2, 3]]).to(device)
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated}")
    
    # Quick training test
    print("\nQuick training test:")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for step in range(5):
        # Random batch
        inputs = torch.randint(0, config.vocab_size, (4, 64)).to(device)
        targets = torch.randint(0, config.vocab_size, (4, 64)).to(device)
        
        # Forward
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Step {step}: Loss = {loss.item():.4f}")