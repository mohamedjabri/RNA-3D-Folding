import torch
import torch.nn as nn

def local_pairwise_distance_loss(P, Q, k=4):
    """
    Only enforce distances to k nearest neighbors along the chain
    """
    L = P.shape[0]

    if L < k + 1:
        return P.new_tensor(0.0)

    loss = 0.0
    count = 0

    for i in range(L):
        j_max = min(L, i + k + 1)
        Dp = torch.norm(P[i] - P[i+1:j_max], dim=-1)
        Dq = torch.norm(Q[i] - Q[i+1:j_max], dim=-1)

        loss += torch.mean(torch.log1p(torch.abs(Dp - Dq)))
        count += 1

    return loss / count


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

def center_preds_multi(preds, mask, eps=1e-8):
    """
    coords: (B, K, L, 3)
    mask:   (B, L)
    """
    B, K, L, _ = preds.shape

    # (B, 1, L, 1)
    mask_f = mask[:, None, :, None].float()

    denom = mask_f.sum(dim=2, keepdim=True).clamp_min(eps)

    centroid = (preds * mask_f).sum(dim=2, keepdim=True) / denom

    return preds - centroid

def center_coords_multi(coords, mask, eps=1e-8):
    """
    coords: (B, K, L, 3)
    mask:   (B, L)
    """
    B, L, _ = coords.shape

    # (B, 1, L, 1)
    mask_f = mask[:, None, :, None].float()

    denom = mask_f.sum(dim=2, keepdim=True).clamp_min(eps)

    centroid = (coords * mask_f).sum(dim=2, keepdim=True) / denom

    return coords - centroid

def combined_loss_multi(
    pred, target, mask, Z=None,
    w_kabsch=1.0,
    w_dist=0.1,
    w_aux=0.05,
    w_smooth=0.01,
    clamp=100.0
):
    """
    pred   : (B, K, L, 3)
    target : (B, T, L, 3)
    mask   : (B, L)
    Z      : (B, K, L, d_aux) or None
    """
    B, K, L, _ = pred.shape
    T = target.shape[1]

    total_loss = 0.0

    for b in range(B):
        valid = mask[b] > 0

        if valid.sum() < 3:
            continue

        for k in range(K):
            P = pred[b, k, valid]
            P = torch.clamp(P, -clamp, clamp)

            # ---- best GT match ----
            best_kabsch = None
            best_dist   = None

            for t in range(T):
                Q = target[b, t, valid]
                Q = torch.clamp(Q, -clamp, clamp)

                # center
                Pc = center_coords_multi(P, valid)
                Qc = center_coords_multi(Q, valid)

                # Kabsch
                P_aligned = kabsch_align(Pc, Qc)
                loss_k = torch.mean((P_aligned - Qc) ** 2)

                # Pairwise distances
                Dp = torch.cdist(P, P)
                Dq = torch.cdist(Q, Q)
                loss_d = local_pairwise_distance_loss(Dp, Dq)

                if best_kabsch is None or loss_k < best_kabsch:
                    best_kabsch = loss_k
                    best_dist   = loss_d

            # ---- smoothness ----
            d1 = P[1:] - P[:-1]
            d2 = d1[1:] - d1[:-1]
            loss_smooth = torch.mean(torch.norm(d2, dim=-1))

            loss_k_total = (
                w_kabsch * best_kabsch
                + w_dist   * best_dist
                + w_smooth * loss_smooth
            )

            # ---- auxiliary latent distance ----
            if Z is not None:
                Zk = Z[b, k, valid]
                Dz = torch.cdist(Zk, Zk)
                Dq = torch.cdist(Q, Q)
                loss_aux = torch.mean(torch.abs(Dz - Dq))
                loss_k_total = loss_k_total + w_aux * loss_aux

            total_loss += loss_k_total

    return total_loss / (B * K)