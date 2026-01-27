import torch
import torch.nn as nn

NUC_TO_ID = {
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3
}

def tokenize_sequence(seq: str):
    return torch.tensor([NUC_TO_ID[c] for c in seq], dtype=torch.long)

def masked_mse_loss(pred, target, mask):
    """
    pred:   (B, L, 3)
    target: (B, L, 3)
    mask:   (B, L)
    """
    valid_coords = torch.isfinite(target).all(dim=-1)
    final_mask = mask & valid_coords

    final_mask = final_mask.unsqueeze(-1)

    valid = final_mask.sum()
    if valid == 0:
        return torch.tensor(0.0, device=pred.device)

    diff = (pred - target) ** 2
    diff = diff * final_mask

    return diff.sum() / valid

def center_coords(coords, mask):
    """
    coords: (B, L, 3)
    mask:   (B, L)
    """
    mask_f = mask.unsqueeze(-1)

    centroid = (coords * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True)
    return coords - centroid