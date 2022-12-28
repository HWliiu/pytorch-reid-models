"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import torch
import torch.nn.functional as F
from torch.utils import data
from tqdm.auto import tqdm


class Matcher:
    def __init__(self, model, g_dataset, topk=10, rm_same_cam=True):
        self.model = model.eval()
        self.g_dataset = g_dataset
        self.topk = topk
        self.rm_same_cam = rm_same_cam

        self.device = next(model.parameters()).device
        self.g_feats, self.g_pids, self.g_camids = self._extract_feats(g_dataset)
        self.g_feats = F.normalize(self.g_feats).T.contiguous()

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

    def __call__(self, imgs, pids=None, camids=None):
        q_feats = F.normalize(self._forward(imgs, pids, camids))
        sim_mat = torch.mm(q_feats, self.g_feats)
        if self.rm_same_cam:
            assert pids is not None and camids is not None
            assert len(imgs) == len(pids) == len(camids)
            # remove gallery samples that have the same pid and camid with query
            sim_mat[
                (pids.view(-1, 1) == self.g_pids)
                & (camids.view(-1, 1) == self.g_camids)
            ] = -1
        topk_sim, topk_index = torch.topk(sim_mat, k=self.topk)
        return topk_sim, topk_index
