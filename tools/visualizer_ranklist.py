"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""
import functools
from pathlib import Path

import accelerate
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from torchvision.transforms.functional import to_pil_image

from reid_models.data import build_test_dataloaders
from reid_models.evaluate import Matcher
from reid_models.modeling import build_reid_model
from reid_models.utils import setup_logger


class VisualizerRankList:
    def __init__(
        self,
        g_dataset,
        model,
        vis_topk=10,
        rm_same_cam=True,
        save_dir="logs/",
    ):
        self.g_dataset = g_dataset
        self.device = next(model.parameters()).device
        self.model = model.eval()
        self.vis_topk = vis_topk
        self.rm_same_cam = rm_same_cam

        self.matcher = Matcher(
            self.model, self.g_dataset, self.vis_topk, self.rm_same_cam
        )
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @functools.lru_cache()
    def _get_figure_and_axes(self, vis_row):
        fig = plt.figure(
            figsize=((self.vis_topk + 1) * 1.5, vis_row * 3),
            layout="constrained",
            dpi=100,
        )
        axes = fig.subplots(vis_row, self.vis_topk + 1)
        for ax in axes.flat:
            for sp in ax.spines.values():
                sp.set_visible(False)
        return fig, axes

    @staticmethod
    def _plot_single_image(
        ax, img=None, title=None, xlabel=None, ylabel=None, edge_color=None
    ):
        ax.clear()
        ax.set(xticks=[], yticks=[])

        if img is not None:
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img)
            ax.imshow(img)

        if title is not None:
            ax.set_title(title, fontsize="x-large")
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize="large")
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize="x-large")
        if edge_color is not None:
            ax.patch.set(alpha=0.5, ec=edge_color, lw=5)

    def __call__(self, imgs, pids, camids, save_fname):
        with torch.no_grad():
            match_results = self.matcher(imgs, pids, camids)
        fig, axes = self._get_figure_and_axes(len(imgs))

        m_loader = data.DataLoader(
            self.g_dataset,
            batch_sampler=match_results.indices.cpu(),
            # FIXME: fix bug when num_workers!=0
            num_workers=0,
        )
        for i, (m_imgs, m_pids, m_camids) in enumerate(m_loader):
            q_img, q_pid, q_camid = imgs[i], pids[i], camids[i]
            self._plot_single_image(
                axes[i, 0],
                q_img,
                ylabel="Query",
                xlabel=f"p:{q_pid.item():0>4} c:{q_camid.item():0>2}",
            )
            for j in range(self.vis_topk):
                self._plot_single_image(
                    axes[i, j + 1],
                    m_imgs[j],
                    title=f"sim:{match_results.sims[i,j]:.2f}",
                    ylabel=f"Match" if j == 0 else None,
                    xlabel=f"p:{m_pids[j].item():0>4} c:{m_camids[j].item():0>2}",
                    edge_color=("g" if q_pid == m_pids[j] else "r")
                    if self.rm_same_cam
                    else None,
                )
        fig.savefig(self.save_dir / f"{save_fname}.png")


def main():
    setup_logger(name="reid_models")

    test_dataloaders = build_test_dataloaders(
        dataset_names=["dukemtmcreid", "market1501", "msmt17"],
        query_batch_size=4,
        query_num=32,
    )

    accelerator = accelerate.Accelerator(mixed_precision="fp16")

    model_names = ["bagtricks_R50_fastreid"]
    for model_name in model_names:
        for dataset_name, (q_data_loader, g_data_loader) in test_dataloaders.items():
            model = build_reid_model(model_name, dataset_name)
            q_data_loader, g_data_loader, model = accelerator.prepare(
                q_data_loader, g_data_loader, model
            )

            visualizer = VisualizerRankList(
                g_data_loader.dataset,
                model,
                save_dir=f"logs/visualizer_ranklist/{dataset_name}-{model_name}",
            )
            for i, (imgs, pids, camids) in enumerate(q_data_loader):
                visualizer(imgs, pids, camids, save_fname=f"{i:0>4}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
