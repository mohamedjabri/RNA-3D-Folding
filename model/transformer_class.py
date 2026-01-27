import torch
import torch.nn as nn

    
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

