"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""
import random
from pathlib import Path

import accelerate
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
# from e2i import EmbeddingsProjector
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.manifold import TSNE
from torch.utils import data
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from reid_models.data import build_test_datasets
from reid_models.modeling import build_reid_model
from reid_models.utils import setup_logger


class VisualizerTSNE:
    def __init__(self, model, save_dir="logs/"):
        self.device = next(model.parameters()).device
        self.model = model.eval()
        self.tsne = TSNE(n_components=2, init="pca", perplexity=30)
        # self.projector = EmbeddingsProjector()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

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
                all_feats.append(self.model(imgs))
            all_pids.append(pids)
            all_camids.append(camids)

        all_feats = torch.cat(all_feats)
        all_pids = torch.cat(all_pids)
        all_camids = torch.cat(all_camids)

        all_feats = F.normalize(all_feats)

        return all_feats.cpu().numpy(), all_pids.cpu().numpy(), all_camids.cpu().numpy()

    @staticmethod
    def _imscatter(x, y, images, ax, zoom=1):
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0, image in zip(x, y, images):
            im = cv2.imread(image)
            im = cv2.resize(im, (64, 128))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_f = OffsetImage(im, zoom=zoom)
            ab = AnnotationBbox(im_f, (x0, y0), xycoords="data", frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        return artists

    def draw_tsne_with_image(self, features, imgs):
        Y = self.tsne.fit_transform(features)

        fig, ax = plt.subplots(figsize=(30, 40), layout="constrained")
        ax.axis("off")
        self._imscatter(Y[:, 0], Y[:, 1], imgs, ax, zoom=0.55)
        fig.savefig(
            fname=self.save_dir / "tsne_with_image.png",
            bbox_inches="tight",
            pad_inches=0.01,
        )

    # TODO: Sampling part of pids for drawing
    # def draw_tsne_with_label(self, features, pids, camids):
    #     X_tsne = self.tsne.fit_transform(features)

    #     fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    #     axes[0].set_title("tsne with pid", fontsize="x-large")
    #     axes[1].set_title("tsne with camid", fontsize="x-large")

    #     scatter0 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=pids)
    #     legend0 = axes[0].legend(
    #         *scatter0.legend_elements(), loc="lower left", title="pids"
    #     )
    #     axes[0].add_artist(legend0)

    #     scatter1 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=camids)
    #     legend1 = axes[1].legend(
    #         *scatter1.legend_elements(), loc="lower left", title="camids"
    #     )
    #     axes[1].add_artist(legend1)

    #     fig.savefig(fname=self.save_dir / "tsne_with_label.png")

    def __call__(self, dataset, show_num=None):
        if show_num is not None:
            dataset.samples = random.sample(dataset.samples, show_num)
        data_loader = data.DataLoader(dataset, batch_size=128, num_workers=8)
        all_feats, all_pids, all_camids = self._extract_feats(data_loader)
        # self.draw_tsne_with_label(all_feats, all_pids, all_camids)
        imgs_path = [data[0] for data in dataset.samples]
        self.draw_tsne_with_image(all_feats, imgs_path)


def main():
    setup_logger(name="reid_models")

    test_datasets = build_test_datasets(
        dataset_names=["dukemtmcreid", "market1501", "msmt17"],
    )

    accelerator = accelerate.Accelerator(mixed_precision="fp16")

    model_names = ["bagtricks_R50_fastreid"]
    for model_name in model_names:
        for dataset_name, (_, g_dataset) in test_datasets.items():
            model = build_reid_model(model_name, dataset_name)
            model = accelerator.prepare(model)

            visualizer = VisualizerTSNE(
                model,
                save_dir=f"logs/visualizer_tsne/{dataset_name}-{model_name}",
            )
            visualizer(g_dataset, 1500)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
