import torch
import torch.nn as nn

def pairwise_distance_loss(pred, target, mask):
    # pred, target: (L, 3)
    # mask: (L,)
    valid = mask > 0
    P = pred[valid]
    Q = target[valid]

    if P.shape[0] < 3:
        return torch.tensor(0.0, device=pred.device)

    Dp = torch.cdist(P, P)
    Dq = torch.cdist(Q, Q)

    max_d = 20.0
    Dp = torch.clamp(Dp, max=max_d)
    Dq = torch.clamp(Dq, max=max_d)

    loss = torch.mean((Dp - Dq) ** 2)

    return loss


def kabsch_align(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Aligns P to Q using Kabsch (autograd-safe).
    """
    assert P.shape == Q.shape
    assert P.shape[-1] == 3

    centroid_P = P.mean(dim=0)
    centroid_Q = Q.mean(dim=0)

    p = P - centroid_P
    q = Q - centroid_Q

    H = p.T @ q
    U, S, Vt = torch.linalg.svd(H)

    # Reflection-safe correction (NO inplace ops)
    det = torch.det(Vt.T @ U.T)

    D = torch.eye(3, device=P.device, dtype=P.dtype)
    D[-1, -1] = torch.where(det < 0, -1.0, 1.0)

    R = Vt.T @ D @ U.T

    P_aligned = p @ R + centroid_Q
    return P_aligned

def center_coords(coords, mask):
    """
    coords: (B, L, 3)
    mask:   (B, L)
    """
    mask_f = mask.unsqueeze(-1)

    centroid = (coords * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True)
    return coords - centroid

def combined_loss(pred, target, mask,
                  Z=None,
                  w_kabsch=1.0,
                  w_dist=0.1,
                  w_aux=0.05,
                  w_smooth=0.01):
    """
    pred, target: (B, L, 3)
    mask: (B, L)
    Z: (B, L, d_aux) or None
    """
    loss = 0.0
    B = pred.shape[0]

    for b in range(B):
        valid = mask[b] > 0

        P = pred[b, valid]     # (L', 3)
        Q = target[b, valid]   # (L', 3)

        if P.shape[0] < 3:
            continue

        # --- 1. Kabsch-aligned MSE ---
        P_aligned = kabsch_align(P, Q)
        loss_kabsch = torch.mean((P_aligned - Q) ** 2)

        # --- 2. Pairwise distance loss ---
        loss_pair = pairwise_distance_loss(
            P,
            Q,
            torch.ones(len(P), device=P.device)
        )

        # --- 3. Delta smoothness loss ---
        d1 = P[1:] - P[:-1]
        d2 = d1[1:] - d1[:-1]
        loss_smooth = torch.mean(torch.abs(d2).sum(dim=-1))

        total = (
            w_kabsch * loss_kabsch
            + w_dist * loss_pair
            + w_smooth * loss_smooth
        )

        # --- 4. Auxiliary distance loss ---
        if Z is not None:
            Zb = Z[b, valid]

            D_pred = torch.cdist(Zb, Zb)
            D_true = torch.cdist(Q, Q)

            loss_aux = torch.mean(torch.abs(D_pred - D_true))
            total = total + w_aux * loss_aux

        loss += total

    return loss / B