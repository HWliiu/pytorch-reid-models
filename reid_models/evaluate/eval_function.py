"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import torch
import torch.nn.functional as F


def eval_function(
    q_feats: torch.Tensor,
    g_feats: torch.Tensor,
    q_pids: torch.Tensor,
    g_pids: torch.Tensor,
    q_camids: torch.Tensor,
    g_camids: torch.Tensor,
    max_rank: int = 20,
    reduction=True,
):
    """Evaluation with reid metric (CUDA accelerate! very fast!)
    note: for each query identity, its gallery images from the same camera view are discarded.
    """
    # make sure to be on the same device
    q_feats = q_feats.half()
    device = q_feats.device
    g_feats = g_feats.half().to(device)
    q_pids, g_pids = q_pids.to(device), g_pids.to(device)
    q_camids, g_camids = q_camids.to(device), g_camids.to(device)

    # if the distance matrix is too large, it is offloaded to the CPU
    distmat_mem = (len(q_feats) * len(g_feats) * 4) / (2**30)
    mem_threshold = 0.5
    if q_feats.is_cuda and distmat_mem > mem_threshold:
        distmat = []
        for q_feat in torch.chunk(q_feats, int(distmat_mem // mem_threshold)):
            distmat.append(
                (1.0 - torch.mm(F.normalize(q_feat), F.normalize(g_feats).T)).cpu()
            )
        distmat = torch.cat(distmat)
    else:
        distmat = 1.0 - torch.mm(F.normalize(q_feats), F.normalize(g_feats).T)
    del q_feats, g_feats

    assert len(distmat.shape) == 2
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    metrics = {"CMC": [], "AP": [], "INP": [], "valids": []}
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = torch.argsort(distmat[q_idx].to(device))
        keep = ~((g_pids[order] == q_pid) & (g_camids[order] == q_camid))
        match = g_pids[order] == q_pid
        # binary vector, positions with value 1 are correct matches
        orig_cmc = match[keep].int()

        # this condition is true when query identity does not appear in gallery
        if not torch.any(orig_cmc):
            metrics["CMC"].append(torch.zeros(max_rank, device=device, dtype=torch.int))
            metrics["AP"].append(torch.zeros(1, device=device))
            metrics["INP"].append(torch.zeros(1, device=device))
            metrics["valids"].append(torch.zeros(1, device=device, dtype=torch.bool))
            continue

        # compute inverse negative penalty
        cum_cmc = orig_cmc.cumsum(dim=0)
        max_pos_idx = cum_cmc.argmax(dim=0)
        INP = cum_cmc[max_pos_idx] / (max_pos_idx + 1.0)

        # compute cmc curve
        cmc = cum_cmc[:max_rank].clone()
        cmc[cmc > 1] = 1
        CMC = cmc

        # compute average precision
        tmp_cmc = cum_cmc.clone()
        tmp_cmc = tmp_cmc / torch.arange(1.0, tmp_cmc.shape[0] + 1, device=device)
        tmp_cmc = tmp_cmc * orig_cmc
        AP = tmp_cmc.sum() / orig_cmc.sum()

        metrics["CMC"].append(CMC)
        metrics["AP"].append(AP)
        metrics["INP"].append(INP)
        metrics["valids"].append(torch.ones(1, device=device, dtype=torch.bool))

    metrics = {k: torch.stack(v) for k, v in metrics.items()}

    if reduction:
        all_CMC, all_AP, all_INP, valids = list(metrics.values())
        cmc = all_CMC.sum(dim=0) / valids.sum(dim=0)
        mAP = all_AP.mean(dim=0)
        mINP = all_INP.mean(dim=0)
        valids = valids.sum(dim=0)
        metrics = {"CMC": cmc, "AP": mAP, "INP": mINP, "valids": valids}

    return metrics
