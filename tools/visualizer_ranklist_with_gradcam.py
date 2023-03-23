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
from einops import rearrange, reduce, repeat
from fvcore.common.registry import Registry
from pytorch_grad_cam import (
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMElementWise,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    XGradCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils import data
from torchvision.transforms.functional import to_pil_image

from reid_models.data import build_test_dataloaders
from reid_models.evaluate import Matcher
from reid_models.modeling import build_reid_model
from reid_models.utils import setup_logger
from tools.visualizer_ranklist import VisualizerRankList


def register_target_layer():
    TARGET_LAYER_REGISTRY = Registry("target layer and reshape transform registry")

    @TARGET_LAYER_REGISTRY.register()
    def resnet50_dpr(model):
        return [model.get_submodule("1.layer4.2")], None

    @TARGET_LAYER_REGISTRY.register()
    def resnet50_abd(model):
        branch1 = model.get_submodule("1.branches.0.0.backbone")
        # branch2 = model.get_submodule("1.branches.1.1.sum_conv")
        return [branch1], None

    @TARGET_LAYER_REGISTRY.register()
    def densenet121_abd(model):
        return resnet50_abd(model)

    @TARGET_LAYER_REGISTRY.register()
    def resnet50_agw(model):
        return [model._modules["1"].base.layer4[-1]], None

    @TARGET_LAYER_REGISTRY.register()
    def resnet50_ap(model):
        return [model._modules["1"].BN5], None

    @TARGET_LAYER_REGISTRY.register()
    def osnet_x1_0_dpr(model):
        return [model._modules["1"].conv5], None

    @TARGET_LAYER_REGISTRY.register()
    def osnet_ain_x1_0_dpr(model):
        return [model._modules["1"].conv5], None

    @TARGET_LAYER_REGISTRY.register()
    def osnet_ibn_x1_0_dpr(model):
        return [model._modules["1"].conv5], None

    @TARGET_LAYER_REGISTRY.register()
    def resnet50_bot(model):
        return [model.get_submodule("1.base.layer4.2")], None

    @TARGET_LAYER_REGISTRY.register()
    def bagtricks_osnet_x1_0_fastreid(model):
        return [model._modules["1"].backbone.conv5], None

    @TARGET_LAYER_REGISTRY.register()
    def bagtricks_osnet_ibn_x1_0_fastreid(model):
        return [model._modules["1"].backbone.conv5], None

    @TARGET_LAYER_REGISTRY.register()
    def bagtricks_R50_fastreid(model):
        return [model.get_submodule("1.backbone.layer4.2")], None

    @TARGET_LAYER_REGISTRY.register()
    def bagtricks_R50_ibn_fastreid(model):
        return [model.get_submodule("1.backbone.layer4.2")], None

    @TARGET_LAYER_REGISTRY.register()
    def bagtricks_SeR50_fastreid(model):
        return [
            model._modules["1"].backbone.layer4[-1],
        ], None

    @TARGET_LAYER_REGISTRY.register()
    def bagtricks_S50_fastreid(model):
        return [model._modules["1"].backbone.layer4[-1]], None

    @TARGET_LAYER_REGISTRY.register()
    def bagtricks_mobilenet_v3_large_fastreid(model):
        return [model._modules["1"].backbone.features[-1]], None

    @TARGET_LAYER_REGISTRY.register()
    def bagtricks_densenet121_fastreid(model):
        return [model._modules["1"].backbone._modules["features"].norm5], None

    @TARGET_LAYER_REGISTRY.register()
    def bagtricks_inception_resnet_v2_fastreid(model):
        return [model._modules["1"].backbone.conv2d_7b], None

    @TARGET_LAYER_REGISTRY.register()
    def bagtricks_inception_v3_fastreid(model):
        return [model._modules["1"].backbone.Mixed_7c], None

    @TARGET_LAYER_REGISTRY.register()
    def bagtricks_inception_v4_fastreid(model):
        return [model._modules["1"].backbone.features[21]], None

    @TARGET_LAYER_REGISTRY.register()
    def bagtricks_convnext_tiny_fastreid(model):
        return [model._modules["1"].backbone.stages[3]], None

    @TARGET_LAYER_REGISTRY.register()
    def sbs_R50_fastreid(model):
        return [model._modules["1"].backbone.layer4[-1]], None

    @TARGET_LAYER_REGISTRY.register()
    def sbs_R50_ibn_fastreid(model):
        return [model._modules["1"].backbone.layer4[-1]], None

    @TARGET_LAYER_REGISTRY.register()
    def vit_base_transreid(model):
        def reshape_transform(tensor):
            return rearrange(tensor[:, 1:], "b (h w) c->b c h w", h=16, w=8)

        return [
            model._modules["1"].base.blocks[-1].norm1,
        ], reshape_transform

    # TODO: Add more...
    return TARGET_LAYER_REGISTRY


class SimilarityTarget:
    def __init__(self, target_feature):
        self.target_feature = target_feature.clone()

    def __call__(self, input_feature):
        return torch.nn.functional.cosine_similarity(
            input_feature, self.target_feature, dim=0
        )


class VisualizerRankListWithGradCAM(VisualizerRankList):
    def __init__(
        self,
        g_dataset,
        model,
        vis_topk=10,
        rm_same_cam=True,
        save_dir="logs/",
    ):
        super().__init__(g_dataset, model, vis_topk, rm_same_cam, save_dir)
        self.target_layer_registry = register_target_layer()
        self.cam = self._get_gradcam()
        self.fig, self.axes = self._get_figure_and_axes()

    def _get_gradcam(self):
        # prepare gradcam
        try:
            # get target_layer and reshape transform
            target_layers, reshape_transform = self.target_layer_registry.get(
                self.model.name
            )(self.model)
        except KeyError:
            raise ValueError(
                "please check `register_target_layer` for correct configuration"
            )
        for target_layer in target_layers:
            target_layer.requires_grad_(True)
        cam = GradCAM(
            model=self.model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
            use_cuda=True,
        )

        return cam

    def _get_figure_and_axes(self):
        fig = plt.figure(
            figsize=((self.vis_topk + 1) * 1.5, 3 * 3),
            layout="constrained",
            dpi=100,
        )
        axes = fig.subplots(3, self.vis_topk + 1)
        fig.delaxes(axes[1, 0])
        fig.delaxes(axes[2, 0])
        for ax in axes.flat:
            for sp in ax.spines.values():
                sp.set_visible(False)
        return fig, axes

    def _forward(self, imgs, pids, camids):
        if self.model.name in ["vit_transreid", "deit_transreid"]:
            # cam_label starts from 0
            feats = self.model(imgs, cam_label=camids - 1)
        else:
            feats = self.model(imgs)

        return feats

    def __call__(self, imgs, pids, camids, save_fname):
        with torch.no_grad():
            match_results = self.matcher(imgs, pids, camids)
            q_feats = self._forward(imgs, pids, camids)

        m_loader = data.DataLoader(
            self.g_dataset,
            batch_sampler=match_results.indices.cpu(),
            # FIXME: fix bug when num_workers!=0
            num_workers=0,
        )
        for i, (m_imgs, m_pids, m_camids) in enumerate(m_loader):
            q_img, q_pid, q_camid = imgs[i].cpu(), pids[i].cpu(), camids[i].cpu()
            self._plot_single_image(
                self.axes[0, 0],
                q_img,
                ylabel="Query",
                xlabel=f"p:{q_pid.item():0>4} c:{q_camid.item():0>2}",
            )
            m_cam_target = [SimilarityTarget(q_feats[i]) for _ in range(self.vis_topk)]
            m_imgs_cam = self.cam(m_imgs, targets=m_cam_target, aug_smooth=True)

            with torch.no_grad():
                g_feats = self._forward(
                    m_imgs.to(imgs.device),
                    m_pids.to(imgs.device),
                    m_camids.to(imgs.device),
                )
            q_cam_target = [SimilarityTarget(g_feats[k]) for k in range(self.vis_topk)]
            q_imgs_cam = self.cam(
                q_img.expand_as(m_imgs), targets=q_cam_target, aug_smooth=True
            )

            for j in range(self.vis_topk):
                self._plot_single_image(
                    self.axes[0, j + 1],
                    m_imgs[j],
                    title=f"sim:{match_results.sims[i,j]:.2f}",
                    ylabel=f"Match" if j == 0 else None,
                    xlabel=f"p:{m_pids[j].item():0>4} c:{m_camids[j].item():0>2}",
                    edge_color=("g" if q_pid == m_pids[j] else "r")
                    if self.rm_same_cam
                    else None,
                )

                g_img_vis = show_cam_on_image(
                    m_imgs[j].numpy().transpose(1, 2, 0),
                    m_imgs_cam[j],
                    use_rgb=True,
                    image_weight=0.6,
                )
                self._plot_single_image(
                    self.axes[1, j + 1],
                    g_img_vis,
                    ylabel=f"Match CAM" if j == 0 else None,
                )

                q_img_vis = show_cam_on_image(
                    q_img.numpy().transpose(1, 2, 0),
                    q_imgs_cam[j],
                    use_rgb=True,
                    image_weight=0.6,
                )
                self._plot_single_image(
                    self.axes[2, j + 1],
                    q_img_vis,
                    ylabel=f"Query CAM" if j == 0 else None,
                )

            self.fig.savefig(self.save_dir / f"{save_fname}_{i:0>4}.png")


def main():
    setup_logger(name="reid_models")

    test_dataloaders = build_test_dataloaders(
        dataset_names=["dukemtmcreid", "market1501", "msmt17"],
        query_batch_size=4,
        query_num=32,
    )

    accelerator = accelerate.Accelerator(mixed_precision="no")

    model_names = ["bagtricks_R50_fastreid"]
    for model_name in model_names:
        for dataset_name, (q_data_loader, g_data_loader) in test_dataloaders.items():
            model = build_reid_model(model_name, dataset_name)
            q_data_loader, g_data_loader, model = accelerator.prepare(
                q_data_loader, g_data_loader, model
            )

            visualizer = VisualizerRankListWithGradCAM(
                g_data_loader.dataset,
                model,
                save_dir=f"logs/visualizer_ranklist_with_gradcam/{dataset_name}-{model_name}",
            )
            for i, (imgs, pids, camids) in enumerate(q_data_loader):
                visualizer(imgs, pids, camids, save_fname=f"{i:0>4}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
