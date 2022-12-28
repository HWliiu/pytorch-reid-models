"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import torch
from tqdm.auto import tqdm

from .eval_function import eval_function


class Estimator:
    def __init__(self, model, g_data_loader):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.g_data_loader = g_data_loader
        self.g_feats, self.g_pids, self.g_camids = self._extract_feats(g_data_loader)

    def _extract_feats(self, data_loader):
        all_feats = []
        all_pids = []
        all_camids = []
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

    def __call__(self, q_data_loader):
        q_feats, q_pids, q_camids = self._extract_feats(q_data_loader)
        metrics = eval_function(
            q_feats, self.g_feats, q_pids, self.g_pids, q_camids, self.g_camids
        )

        cmc, mAP, mINP, _ = list(metrics.values())
        return cmc.tolist(), mAP.item(), mINP.item()
