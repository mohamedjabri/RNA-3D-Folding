import torch
import torch.nn as nn
import math

VOCAB_SIZE = 5     # A, C, G, U, PAD
PAD_IDX = 4
D_MODEL = 128
    
class RNABaselineModel(nn.Module):
    def __init__(self, vocab_size=4, embed_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        self.proj = nn.Linear(embed_dim, 3)

    def forward(self, tokens):
        """
        tokens: (B, L)
        """
        x = self.embedding(tokens)   # (B, L, D)
        coords = self.proj(x)         # (B, L, 3)
        return coords

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
    
def build_transformer_encoder(d_model):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=8,
        dim_feedforward=4 * d_model,
        dropout=0.1,
        batch_first=True
    )

    encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=4
    )

    return encoder

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

        # Output head: embedding → (x, y, z)
        self.coord_head = nn.Linear(d_model, 3)

    def forward(self, tokens):
        """
        tokens: (B, L)
        returns: (B, L, 3)
        """

        x = self.embedding(tokens)      # (B, L, D)
        x = self.pos_encoding(x)        # (B, L, D)
        x = self.encoder(x)             # (B, L, D)
        coords = self.coord_head(x)     # (B, L, 3)

        return coords


