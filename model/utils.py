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

def pairwise_distance_loss(pred, target, mask, max_dist=20.0):
    """
    pred, target: (B, L, 3)
    mask: (B, L)
    """
    B, L, _ = pred.shape
    device = pred.device

    # Expand for pairwise computation
    pred_i = pred.unsqueeze(2)        # (B, L, 1, 3)
    pred_j = pred.unsqueeze(1)        # (B, 1, L, 3)
    target_i = target.unsqueeze(2)
    target_j = target.unsqueeze(1)

    # Pairwise distances
    dist_pred = torch.norm(pred_i - pred_j, dim=-1)      # (B, L, L)
    dist_true = torch.norm(target_i - target_j, dim=-1)

    # Distance mask
    pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)    # (B, L, L)

    # Optional: ignore very large distances
    pair_mask = pair_mask * (dist_true < max_dist)

    loss = ((dist_pred - dist_true) ** 2) * pair_mask

    return loss.sum() / pair_mask.sum().clamp(min=1)

 

def center_coords(coords, mask):
    """
    coords: (B, L, 3)
    mask:   (B, L)
    """
    mask_f = mask.unsqueeze(-1)

    centroid = (coords * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True)
    return coords - centroid