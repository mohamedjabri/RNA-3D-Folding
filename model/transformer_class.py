import torch
import torch.nn as nn
import math

VOCAB_SIZE = 5     # A, C, G, U, PAD
PAD_IDX = 4
D_MODEL = 128

class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=32, max_distance=128, num_heads=8):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads

        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position):
        """
        relative_position: (L, L) with values >= 0
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        # Half for exact positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Log-scaled buckets for large distances
        large_pos = max_exact + (
            torch.log(relative_position.float() / max_exact + 1e-6)
            / torch.log(torch.tensor(max_distance / max_exact))
            * (num_buckets - max_exact)
        ).long()

        large_pos = torch.min(
            large_pos,
            torch.full_like(large_pos, num_buckets - 1)
        )

        return torch.where(is_small, relative_position, large_pos)

    def forward(self, L, device):
        """
        returns: (num_heads, L, L)
        """
        # positions
        context_pos = torch.arange(L, device=device)[:, None]
        memory_pos = torch.arange(L, device=device)[None, :]
        relative_position = torch.abs(context_pos - memory_pos)

        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket)  # (L, L, H)

        return values.permute(2, 0, 1)  # (H, L, L)
    

class RelativeMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1,
                 num_buckets=32, max_distance=128):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rel_pos_bias = RelativePositionBias(
            num_buckets=num_buckets,
            max_distance=max_distance,
            num_heads=num_heads
        )

    def forward(self, x, mask=None):
        """
        x: (B, L, D)
        mask: (B, L)
        """
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, H, Hd)

        q = q.transpose(1, 2)  # (B, H, L, Hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (B, H, L, L)

        # relative positional bias
        bias = self.rel_pos_bias(L, x.device)  # (H, L, L)
        scores = scores + bias.unsqueeze(0)

        if mask is not None:
            mask = mask[:, None, None, :]  # (B, 1, 1, L)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, L, Hd)
        out = out.transpose(1, 2).reshape(B, L, D)

        return self.out(out)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # x: (B, L, d_model)
        B, L, _ = x.shape
        device = x.device

        position = torch.arange(L, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device)
            * (-math.log(10000.0) / self.d_model)
        )

        pe = torch.zeros(L, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return x + pe.unsqueeze(0)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = RelativeMultiheadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x
    
def build_transformer_encoder(d_model, num_layers=7, num_heads=8):
    return nn.ModuleList([
        TransformerBlock(d_model, num_heads)
        for _ in range(num_layers)
    ])

class RNATransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL):
        super().__init__()

        # Token embedding
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=PAD_IDX
        )

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)

        # Transformer encoder
        self.encoder = build_transformer_encoder(d_model)

        # Delta coordinate head
        self.delta_head = nn.Linear(d_model, 3)

        # auxiliary distance head
        self.dist_proj = nn.Linear(d_model, 32)

    def forward(self, tokens, mask=None):
        """
        tokens: (B, L)
        returns: (B, L, 3)
        """
        x = self.embedding(tokens)      # (B, L, D)
        x = self.pos_encoding(x)        # (B, L, D)
        for layer in self.encoder:
            x = layer(x, mask)
            
        # Predict deltas
        delta = self.delta_head(x)      # (B, L, 3)

        # Mask padding deltas
        if mask is not None:
            delta = delta * mask.unsqueeze(-1)

        # Reconstruct coordinates via cumulative sum
        coords = torch.cumsum(delta, dim=1)  # (B, L, 3)
        
        Z = self.dist_proj(x)               # (B, L, 32)
    
        return coords, Z

