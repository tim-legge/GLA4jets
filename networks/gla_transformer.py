#!/usr/bin/env python3
"""
Gated Linear Attention Transformer for Jet Classification
Based on the flash-linear-attention package and DeltaNet-inspired architecture
"""
import weaver
import copy
import random
from weaver.utils.logger import _logger
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
        def __init__(self, embed_dim, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(embed_dim))
            self.eps = eps
            
        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return self.weight * x
    
    class GatedLinearAttention(nn.Module):
        """Simplified GLA implementation for fallback"""
        def __init__(self, embed_dim=128, num_heads=8, **kwargs):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.g_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            
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

class SequenceTrimmer(nn.Module):

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        # (Not used in this implementation:)
        # v: (N, 4, P) [px,py,pz,energy]
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, random.uniform(*self.target))
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, mask


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
        embed_dim: int = 128,
        num_heads: int = 8,
        expand_ratio: float = 4.0,
        dropout: float = 0.1,
        layer_idx: Optional[int] = None,
        **gla_kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx
        
        # Pre-norm for attention
        self.attn_norm = RMSNorm(embed_dim)
        
        # Gated Linear Attention layer
        self.gla = GatedLinearAttention(
            hidden_size=embed_dim,
            num_heads=num_heads,
            layer_idx=layer_idx,
            **gla_kwargs
        )
        
        # Pre-norm for FFN
        self.ffn_norm = RMSNorm(embed_dim)
        
        # Feed-forward network with SwiGLU
        #ffn_dim = int(embed_dim * expand_ratio)
        self.ffn = SwiGLU(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x, attention_mask=None, **kwargs):
        # Self-GLA with residual connection
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
        embed_dim: int = 128,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Feature embedding
        self.feature_projection = nn.Linear(input_dim, embed_dim)
        
        # Positional embedding (learnable)
        #self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input features [batch_size, input_dim, seq_len]  
            mask: Attention mask [batch_size, seq_len], 1 for valid
        
        Returns:
            embedded features of shape `(batch_size, seq_len, embed_dim)`
        """
        batch_size, _, seq_len = x.shape
        
        # Project features to hidden dimension
        x = x.permute(0, 2, 1).contiguous()  # (B, L, input_dim)
        x = self.feature_projection(x)
        
        # Add positional embeddings
        #pos_emb = self.pos_embedding[:, :seq_len, :]
        #x = x + pos_emb
        
        # Apply normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        # Apply mask if provided (set masked positions to zero)
        if mask is not None:
            mask = mask.expand_as(x)
            x = x * mask
        
        return x # (B, seq_L, embed_dim)

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
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        num_classes: int = 5,
        max_seq_len: int = 128,
        expand_ratio: float = 4.0,
        dropout: float = 0.1,
        use_short_conv: bool = True,
        conv_size: int = 4,
        trim=True,
        for_inference: bool = False,
        **gla_kwargs
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Token trimming
        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        # Input embedding
        self.embedding = InputEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Stack of GLA blocks
        self.blocks = nn.ModuleList([
            GLABlock(
                embed_dim=embed_dim,
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
        self.final_norm = RMSNorm(embed_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
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
            x: Input particle features `[batch_size, input_dim, seq_len]`
            mask: Attention mask `[batch_size, 1, seq_len]` (1 for valid, 0 for padded)
            return_features: Whether to return intermediate features
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
            features: Pooled features if return_features=True
        """

        with torch.no_grad():
            x, mask = self.trimmer(x, mask=mask)
            padding_mask = mask.squeeze(1)

        # Input embedding
        x = self.embedding(x, mask)
        
        # Pass through GLA blocks
        for block in self.blocks:
            x = block(x, attention_mask=padding_mask)
            
        # Final normalization
        x = self.final_norm(x)
        
        # Classification
        logits = self.classifier(x)
        #output = torch.softmax(logits, dim=-1)
        print(logits.shape)
        return logits
    
    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class GLATransformerWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = GLATransformer(**kwargs)

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(x=features, mask=mask)


def create_gla_model(
    dataset: str = "hls4ml",
    input_dim: int = 17,
    embed_dim: int = 128,
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
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        **kwargs
    )
    
    print(f"Created GLA model for {dataset} dataset:")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Embedding dimension: {embed_dim}")
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
        embed_dim=128,
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


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        #pair_input_dim=4,
        #use_pre_activation_pair=False,
        embed_dim=128,
        #pair_embed_dims=[64, 64, 64],
        num_heads=4,
        num_layers=4,
        #num_cls_layers=2,
        dropout=0.2,
        #cls_block_params={'dropout': 0.2, 'attn_dropout': 0.2, 'activation_dropout': 0.2},
        #fc_params=[],
        #activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = GLATransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
        return torch.nn.CrossEntropyLoss()
