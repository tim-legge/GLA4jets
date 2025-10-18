#!/usr/bin/env python3
"""
Gated Linear Attention Transformer for Jet Classification
Based on the flash-linear-attention package and DeltaNet-inspired architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import numpy as np

try:
    from fla.layers.gla import GatedLinearAttention
    from fla.modules import RMSNorm
    from fla.modules.activations import ACT2FN
except ImportError:
    print("Warning: flash-linear-attention package not found. Using custom implementations.")
    
    # Fallback implementations if fla is not available
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
            
        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return self.weight * x
    
    class GatedLinearAttention(nn.Module):
        """Simplified GLA implementation for fallback"""
        def __init__(self, hidden_size=128, num_heads=8, **kwargs):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            
            self.gate_norm = RMSNorm(self.head_dim)
            
        def forward(self, x, **kwargs):
            B, L, D = x.shape
            
            q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            g = self.g_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Simple linear attention approximation
            q = F.silu(q) / math.sqrt(self.head_dim)
            k = F.silu(k) / math.sqrt(self.head_dim)
            
            # Compute attention via linear operations
            kv = torch.einsum('bhld,bhlv->bhdv', k, v)
            qkv = torch.einsum('bhld,bhdv->bhlv', q, kv)
            
            # Apply gate
            g = torch.sigmoid(g)
            out = self.gate_norm(qkv) * g
            
            # Merge heads and project
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            return self.o_proj(out), None, None

class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation function"""
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.dim_out = dim_out
        # Use 2/3 rule for hidden dimension in SwiGLU
        hidden_dim = int(2 * dim_out / 3)
        hidden_dim = ((hidden_dim + 7) // 8) * 8  # Round to nearest 8 for efficiency
        
        self.w1 = nn.Linear(dim_in, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim_in, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim_out, bias=bias)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GLABlock(nn.Module):
    """
    A single GLA transformer block with pre-normalization and SwiGLU FFN
    """
    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 8,
        expand_ratio: float = 4.0,
        dropout: float = 0.1,
        layer_idx: Optional[int] = None,
        **gla_kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        
        # Pre-norm for attention
        self.attn_norm = RMSNorm(hidden_size)
        
        # Gated Linear Attention layer
        self.gla = GatedLinearAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_idx=layer_idx,
            **gla_kwargs
        )
        
        # Pre-norm for FFN
        self.ffn_norm = RMSNorm(hidden_size)
        
        # Feed-forward network with SwiGLU
        ffn_dim = int(hidden_size * expand_ratio)
        self.ffn = SwiGLU(hidden_size, ffn_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x, attention_mask=None, **kwargs):
        # Self-attention with residual connection
        residual = x
        x = self.attn_norm(x)
        attn_out, _, _ = self.gla(x, attention_mask=attention_mask, **kwargs)
        x = residual + self.dropout(attn_out)
        
        # FFN with residual connection  
        residual = x
        x = self.ffn_norm(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        
        return x


class InputEmbedding(nn.Module):
    """
    Input embedding layer for particle features
    """
    def __init__(
        self,
        input_dim: int = 17,
        hidden_size: int = 128,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        # Feature embedding
        self.feature_projection = nn.Linear(input_dim, hidden_size)
        
        # Positional embedding (learnable)
        #self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_size) * 0.02)
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input features [batch_size, seq_len, input_dim]  
            mask: Attention mask [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project features to hidden dimension
        x = self.feature_projection(x)
        
        # Add positional embeddings
        #pos_emb = self.pos_embedding[:, :seq_len, :]
        #x = x + pos_emb
        
        # Apply normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        # Apply mask if provided (set masked positions to zero)
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(x)
            x = x * mask
        
        return x

class GLATransformer(nn.Module):
    """
    Complete Gated Linear Attention Transformer for jet classification
    
    Architecture:
    - Input embedding (features + positional encoding)
    - Stack of GLA blocks with RMSNorm and SwiGLU
    - Particle-level pooling
    - Classification head
    """
    def __init__(
        self,
        input_dim: int = 17,
        hidden_size: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        num_classes: int = 5,
        max_seq_len: int = 128,
        expand_ratio: float = 4.0,
        dropout: float = 0.1,
        use_short_conv: bool = True,
        conv_size: int = 4,
        **gla_kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input embedding
        self.embedding = InputEmbedding(
            input_dim=input_dim,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Stack of GLA blocks
        self.blocks = nn.ModuleList([
            GLABlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                expand_ratio=expand_ratio,
                dropout=dropout,
                layer_idx=i,
                use_short_conv=use_short_conv,
                conv_size=conv_size,
                **gla_kwargs
            )
            for i in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(hidden_size)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            # Use Xavier/Glorot initialization for linear layers
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            # Initialize parameters with small random values
            torch.nn.init.normal_(module, mean=0.0, std=0.02)
            
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input particle features [batch_size, seq_len, input_dim]
            mask: Attention mask [batch_size, seq_len] (1 for valid, 0 for padded)
            return_features: Whether to return intermediate features
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
            features: Pooled features if return_features=True
        """
        # Input embedding
        x = self.embedding(x, mask)
        
        # Pass through GLA blocks
        for block in self.blocks:
            x = block(x, attention_mask=mask)
            
        # Final normalization
        x = self.final_norm(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_gla_model(
    dataset: str = "hls4ml",
    input_dim: int = 17,
    hidden_size: int = 128,
    num_layers: int = 6,
    num_heads: int = 8,
    **kwargs
) -> GLATransformer:
    """
    Factory function to create GLA models for different datasets
    """
    if dataset == "hls4ml":
        num_classes = 5
        max_seq_len = 150
    elif dataset == "jetclass":
        num_classes = 10
        max_seq_len = 128
        input_dim = 17  # Assuming same feature dimension
    elif dataset in ["top", "QG"]:
        num_classes = 1  # Binary classification
        max_seq_len = 150 if dataset == "QG" else 200
        input_dim = 3  # pt, eta, phi
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    model = GLATransformer(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        **kwargs
    )
    
    print(f"Created GLA model for {dataset} dataset:")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Num layers: {num_layers}")
    print(f"  - Num heads: {num_heads}")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Max sequence length: {max_seq_len}")
    print(f"  - Total parameters: {model.get_num_trainable_params():,}")
    
    return model


if __name__ == "__main__":
    # Example usage and testing
    print("Testing GLA Transformer...")
    torch.set_default_device(('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Test with HLS4ML configuration
    model = create_gla_model(
        dataset="jetclass",
        hidden_size=128,
        num_layers=6,
        num_heads=8
    )
    
    # Create dummy input
    batch_size = 4
    seq_len = 128
    input_dim = 17
    
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len).bool()
    
    # Forward pass
    with torch.no_grad():
        logits = model(x, mask)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {logits.shape}")
        
    print("GLA Transformer test completed successfully!")