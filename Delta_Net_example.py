# Delta Net Layer 

class DeltaNetLayer:
    """
    Full DeltaNet layer with chunkwise parallel training.
    """
    def __init__(self, d_model, num_heads, chunk_size=64, conv_kernel=4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.chunk_size = chunk_size
        self.conv_kernel = conv_kernel
        
        # Projections
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_beta = Linear(d_model, num_heads) 
        
        self.conv_Q = DepthwiseConv1d(d_model, kernel_size=conv_kernel)
        self.conv_K = DepthwiseConv1d(d_model, kernel_size=conv_kernel)
        self.conv_V = DepthwiseConv1d(d_model, kernel_size=conv_kernel)
        
        # Output projection
        self.W_O = Linear(d_model, d_model)
        
    def feature_map_with_norm(self, x):
        """
        Feature map: SiLU activation + L2 normalization
        φ(x) = SiLU(x) / ||SiLU(x)||_2
        """
        x = silu(x)
        norm = l2_norm(x, dim=-1, keepdim=True)
        return x / (norm + 1e-8)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Linear projections
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        beta_logits = self.W_beta(x)
        
        # 2. Depthwise convolutions
        Q = self.conv_Q(Q)
        K = self.conv_K(K)
        V = self.conv_V(V)
        
        # 3. Apply feature map and normalization
        Q = self.feature_map_with_norm(Q)
        K = self.feature_map_with_norm(K)
        
        # 4. Compute β (learning rate/write strength)
        beta = sigmoid(beta_logits)  # ∈ (0,1)
        
        # 5. Reshape for multi-head
        Q = rearrange(Q, 'b l (h d) -> b h l d', h=self.num_heads)
        K = rearrange(K, 'b l (h d) -> b h l d', h=self.num_heads)
        V = rearrange(V, 'b l (h d) -> b h l d', h=self.num_heads)
        beta = rearrange(beta, 'b l h -> b h l')
        
        # 6. Chunkwise parallel computation
        O = self.chunkwise_deltanet(Q, K, V, beta)
        
        # 7. Merge heads and project
        O = rearrange(O, 'b h l d -> b l (h d)')
        return self.W_O(O)
    
    def chunkwise_deltanet(self, Q, K, V, beta):
        """
        Chunkwise parallel DeltaNet computation.
        
        Args:
            Q, K, V: [batch, num_heads, seq_len, d_head]
            beta: [batch, num_heads, seq_len]
        """
        batch, num_heads, seq_len, d_head = Q.shape
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        # Pad to multiple of chunk_size
        pad_len = num_chunks * self.chunk_size - seq_len
        if pad_len > 0:
            Q = F.pad(Q, (0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))
            beta = F.pad(beta, (0, pad_len))
        
        # Reshape into chunks
        Q = rearrange(Q, 'b h (n c) d -> b h n c d', c=self.chunk_size)
        K = rearrange(K, 'b h (n c) d -> b h n c d', c=self.chunk_size)
        V = rearrange(V, 'b h (n c) d -> b h n c d', c=self.chunk_size)
        beta = rearrange(beta, 'b h (n c) -> b h n c', c=self.chunk_size)
        
        # Initialize output
        O = torch.zeros_like(Q)
        
        # Initialize state: S[0] = 0
        S = torch.zeros(batch, num_heads, d_head, d_head, 
                       device=Q.device, dtype=Q.dtype)
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            Q_chunk = Q[:, :, chunk_idx]  # [b, h, C, d]
            K_chunk = K[:, :, chunk_idx]
            V_chunk = V[:, :, chunk_idx]
            beta_chunk = beta[:, :, chunk_idx]  # [b, h, C]
            
            # Compute W and U for this chunk using UT transform
            W_chunk, U_chunk = self.compute_UT_transform(
                K_chunk, V_chunk, beta_chunk
            )
            
            # Compute intra-chunk attention
            O_intra = self.intra_chunk_attention(
                Q_chunk, K_chunk, U_chunk, W_chunk, S
            )
            
            # Inter-chunk contribution
            O_inter = torch.einsum('bhcd,bhde->bhce', Q_chunk, S)
            
            # Combine
            O[:, :, chunk_idx] = O_intra + O_inter
            
            # Update state for next chunk
            delta = U_chunk - torch.einsum('bhcd,bhde->bhce', W_chunk, S)
            S = S + torch.einsum('bhce,bhcf->bhef', delta, K_chunk)
        
        # Reshape back and remove padding
        O = rearrange(O, 'b h n c d -> b h (n c) d')
        O = O[:, :, :seq_len] 
        
        return O
    
    def compute_UT_transform(self, K, V, beta):
        """
        Compute W and U using UT transform.
        
        Args:
            K, V: [batch, num_heads, C, d_head]
            beta: [batch, num_heads, C]
        
        Returns:
            W, U: [batch, num_heads, C, d_head]
        """
        batch, num_heads, C, d_head = K.shape
        
        # Compute K_beta and V_beta
        K_beta = K * beta.unsqueeze(-1)  # [b, h, C, d]
        V_beta = V * beta.unsqueeze(-1)
        
        # Compute lower triangular matrix: K_beta @ K^T
        KK_T = torch.einsum('bhcd,bhce->bhde', K_beta, K)  # [b, h, C, C]
        
        # Make it strictly lower triangular (offset -1)
        lower_KK_T = torch.tril(KK_T, diagonal=-1)
        
        # T = (I + lower_KK_T)^(-1) (solved via forward substitution)
        I = torch.eye(C, device=K.device, dtype=K.dtype)
        I = I.unsqueeze(0).unsqueeze(0)  # [1, 1, C, C]
        
        # Forward substitution to solve (I + L)T = I
        T = self.forward_substitution(I + lower_KK_T, I)
        
        # Apply beta scaling
        beta_diag = torch.diag_embed(beta)  # [b, h, C, C]
        T = torch.einsum('bhij,bhjk->bhik', T, beta_diag)
        
        # Compute W and U
        W = torch.einsum('bhij,bhjd->bhid', T, K_beta)
        U = torch.einsum('bhij,bhjd->bhid', T, V_beta)
        
        return W, U
    
    def forward_substitution(self, L, B):
        """
        Solve LX = B for lower triangular L using forward substitution.
        
        Args:
            L: [batch, num_heads, C, C] lower triangular
            B: [1, 1, C, C] identity matrix
        
        Returns:
            X: [batch, num_heads, C, C]
        """
        batch, num_heads, C, _ = L.shape
        X = torch.zeros_like(L)
        
        for i in range(C):
            # X[i] = (B[i] - sum(L[i,j] * X[j] for j < i)) / L[i,i]
            if i == 0:
                X[:, :, i, :] = B[:, :, i, :] / L[:, :, i, i:i+1]
            else:
                sum_term = torch.einsum('bhij,bhj...->bhi...',
                                       L[:, :, i, :i], X[:, :, :i])
                X[:, :, i, :] = (B[:, :, i, :] - sum_term) / L[:, :, i, i:i+1]
        
        return X
    
    def intra_chunk_attention(self, Q, K, U, W, S):
        """
        Compute intra-chunk attention.
        
        Args:
            Q, K: [batch, num_heads, C, d_head]
            U, W: [batch, num_heads, C, d_head]
            S: [batch, num_heads, d_head, d_head]
        
        Returns:
            O: [batch, num_heads, C, d_head]
        """
        # Attention scores: Q @ K^T
        scores = torch.einsum('bhcd,bhce->bhde', Q, K) 
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(self.chunk_size, self.chunk_size,
                                            device=Q.device, dtype=Q.dtype))
        scores = scores * causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Compute pseudo-values: U - W @ S^T
        pseudo_values = U - torch.einsum('bhcd,bhde->bhce', W, S)
        
        # Apply attention to pseudo-values
        O = torch.einsum('bhde,bhec->bhdc', scores, pseudo_values)
        
        return O
    
    
class DeltaNetTransformerBlock(nn.Module):
    """
    Single transformer block with DeltaNet attention.
    """
    def __init__(self, d_model, num_heads, d_ff, chunk_size=64, 
                 conv_kernel=4, dropout=0.1):
        super().__init__()
        
        # 1. DeltaNet attention layer
        self.deltanet = DeltaNetLayer(d_model, num_heads, chunk_size, conv_kernel)
        
        # 2. RMSNorm (before attention)
        self.norm1 = RMSNorm(d_model)
        
        # 3. Feed-forward network with SwiGLU
        self.ffn = SwiGLU(d_model, d_ff)
        
        # 4. RMSNorm (before FFN)
        self.norm2 = RMSNorm(d_model)
        
        # 5. Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Attention block with pre-norm
        attn_out = self.deltanet(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # FFN block with pre-norm
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU activation: FFN_SwiGLU(x) = (xW1 ⊙ SiLU(xW2))W3
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)
        self.W3 = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        return self.W3(F.silu(self.W1(x)) * self.W2(x))


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms
class DeltaNetTransformer(nn.Module):
    """
    Complete DeltaNet transformer for language modeling.
    """
    def __init__(self, vocab_size, d_model=2048, num_layers=16, 
                 num_heads=16, d_ff=8192, max_seq_len=8192,
                 chunk_size=64, conv_kernel=4, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DeltaNetTransformerBlock(
                d_model, num_heads, d_ff, chunk_size, conv_kernel, dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm_final = RMSNorm(d_model)
        
        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (embedding = output projection)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm_final(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0):
        """
        Autoregressive generation with constant memory.
        Uses recurrent form for O(1) memory per step.
        """
        for _ in range(max_new_tokens):
            # Forward pass (only need last position for next token)
            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids