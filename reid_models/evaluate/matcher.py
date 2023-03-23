"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tqdm.auto import tqdm


@dataclass
class MatchResults:
    matches: torch.Tensor
    sims: torch.Tensor
    pids: torch.Tensor
    camids: torch.Tensor
    indices: torch.Tensor


class Matcher(nn.Module):
    def __init__(
        self, model, g_dataset, topk=10, rm_same_cam=True, rm_origin_feats=True
    ):
        super().__init__()
        self.training = False
        self.model = model.eval()
        self.g_dataset = g_dataset
        self.topk = topk
        self.rm_same_cam = rm_same_cam

        self.device = next(model.parameters()).device
        self.g_feats, self.g_pids, self.g_camids = self._extract_feats(g_dataset)
        self.g_feats_optimized = F.normalize(self.g_feats).T.contiguous()
        # For save memory
        if rm_origin_feats:
            del self.g_feats

    def _extract_feats(self, dataset):
        data_loader = data.DataLoader(dataset, batch_size=128, num_workers=8)

        all_feats, all_pids, all_camids = [], [], []
        for imgs, pids, camids in tqdm(
            data_loader, desc="Extracting features", leave=False
        ):
            imgs = imgs.to(self.device)
            pids = pids.to(self.device)
            camids = camids.to(self.device)
            with torch.no_grad():
                all_feats.append(self._forward(imgs, pids, camids))
            all_pids.append(pids)
            all_camids.append(camids)

        all_feats = torch.cat(all_feats)
        all_pids = torch.cat(all_pids)
        all_camids = torch.cat(all_camids)

        return all_feats, all_pids, all_camids

    def _forward(self, imgs, pids, camids):
        if self.model.name in ["vit_transreid", "deit_transreid"]:
            # cam_label starts from 0
            feats = self.model(imgs, cam_label=camids - 1)
        else:
            feats = self.model(imgs)

        return feats

    def forward(self, imgs, pids=None, camids=None) -> MatchResults:
        q_feats = F.normalize(self._forward(imgs, pids, camids))
        sim_mat = torch.mm(q_feats, self.g_feats_optimized)

        if self.rm_same_cam:
            assert pids is not None and camids is not None
            assert len(imgs) == len(pids) == len(camids)
            # remove gallery samples that have the same pid and camid with query
            sim_mat[
                (pids.view(-1, 1) == self.g_pids)
                & (camids.view(-1, 1) == self.g_camids)
            ] = -1

        topk_sim, topk_index = torch.topk(sim_mat, k=self.topk)
        match_pids = self.g_pids[topk_index]
        match_camids = self.g_camids[topk_index]
        matches = (
            pids.view(-1, 1) == match_pids
            if pids is not None
            else torch.zeros_like(topk_sim, dtype=torch.bool)
        )
        return MatchResults(matches, topk_sim, match_pids, match_camids, topk_index)
