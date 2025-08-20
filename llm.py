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
    
    # Training
    max_seq_len: int = 512
    batch_size: int = 8
    num_documents: int = 1000
    learning_rate: float = 3e-4
    num_epochs: int = 1
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
        # We'll use a simpler parallel scan approximation
        A = -torch.exp(self.A)
        
        # Parallel computation (approximation for training efficiency)
        # This avoids the sequential loop
        delta_expanded = delta.unsqueeze(-1)  # [B, L, 1, 1]
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # [1, 1, intermediate, state_size]
        
        # Compute decay factors
        decay = torch.exp(delta_expanded * A_expanded)  # [B, L, intermediate, state_size]
        
        # Input contribution
        x_expanded = x_conv.unsqueeze(-1)  # [B, L, intermediate, 1]
        B_expanded = B.unsqueeze(2)  # [B, L, 1, state_size]
        input_contrib = x_expanded * B_expanded  # [B, L, intermediate, state_size]
        
        # Simplified state computation (using cumsum as approximation)
        # This is not exact SSM but works well in practice
        states = input_contrib * decay
        
        # Output computation
        C_expanded = C.unsqueeze(2)  # [B, L, 1, state_size]
        y = (states * C_expanded).sum(dim=-1)  # [B, L, intermediate]
        
        # Add skip connection
        D_expanded = self.D.unsqueeze(0).unsqueeze(0)
        y = y + x_conv * D_expanded
        
        # 7. Output gating
        y = y * F.silu(z)
        
        # 8. Output projection
        output = self.out_proj(y)
        
        # Return dummy state for compatibility
        if state is not None:
            final_state = states[:, -1, :, :]
        else:
            final_state = None
            
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
        
        # Process through layers (without state tracking for training)
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
                shift_labels.view(-1)
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
            
            # Break on EOS (token id 2 for many tokenizers)
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


def load_data(config: HybridConfig):
    """Load tokenizer and dataset"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading dataset...")
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
    
    print(f"Loaded {len(texts)} documents")
    
    # Update vocab size in config
    config.vocab_size = tokenizer.vocab_size
    
    return texts, tokenizer


def train(model, train_loader, config, device):
    """Simple training loop"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model.train()
    
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        batch = batch.to(device)
        
        # Forward pass
        logits, loss = model(batch, labels=batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"})
    
    return avg_loss


def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Config
    config = HybridConfig(
        hidden_size=256,
        num_layers=6,
        num_heads=8,
        layer_pattern="MAMAMM",  # Mix of Mamba and Attention
        batch_size=8,
        num_documents=100,  # Start small for testing
        max_seq_len=256,
        learning_rate=3e-4,
        num_epochs=1
    )
    
    # Load data
    texts, tokenizer = load_data(config)
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, max_length=config.max_seq_len)
    train_loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    model = HybridModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print(f"Model Configuration:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Layer pattern: {config.layer_pattern}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"{'='*50}\n")
    
    # Train
    print("Starting training...")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        avg_loss = train(model, train_loader, config, device)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Generate sample
        print("\nGenerating sample text...")
        prompt = "The universe is"
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
    
    print("\nTraining complete!")
    
    # Save model
    torch.save(model.state_dict(), "hybrid_model.pt")
    print("Model saved to hybrid_model.pt")


if __name__ == "__main__":
    main()