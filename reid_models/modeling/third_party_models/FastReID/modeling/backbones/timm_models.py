import timm
import torch
from torch import nn
from torch.nn import functional as F

from .build import BACKBONE_REGISTRY

__all__ = [
    "build_densenet_backbone",
    "build_inception_resnet_v2_backbone",
    "build_inception_v3_backbone",
    "build_inception_v4_backbone",
    "build_convnext_backbone",
    "build_vgg_backbone",
]


@BACKBONE_REGISTRY.register()
def build_densenet_backbone(cfg):
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    depth = cfg.MODEL.BACKBONE.DEPTH

    if depth == "121x":
        model = timm.create_model(
            "densenet121",
            pretrained=pretrain,
            num_classes=0,
            global_pool="",
            scriptable=True,
            exportable=True,
        )
    else:
        raise ValueError("Unsupported model type {}".format(depth))

    return model


@BACKBONE_REGISTRY.register()
def build_inception_resnet_v2_backbone(cfg):
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN

    model = timm.create_model(
        "inception_resnet_v2",
        pretrained=pretrain,
        num_classes=0,
        global_pool="",
        scriptable=True,
        exportable=True,
    )

    return model


@BACKBONE_REGISTRY.register()
def build_inception_v3_backbone(cfg):
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN

    model = timm.create_model(
        "inception_v3",
        pretrained=pretrain,
        num_classes=0,
        global_pool="",
        scriptable=True,
        exportable=True,
    )

    return model


@BACKBONE_REGISTRY.register()
def build_inception_v4_backbone(cfg):
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN

    model = timm.create_model(
        "inception_v4",
        pretrained=pretrain,
        num_classes=0,
        global_pool="",
        scriptable=True,
        exportable=True,
    )

    return model


@BACKBONE_REGISTRY.register()
def build_convnext_backbone(cfg):
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    depth = cfg.MODEL.BACKBONE.DEPTH

    if depth == "base":
        model = timm.create_model(
            "convnext_base",
            pretrained=pretrain,
            num_classes=0,
            global_pool="",
            output_stride=16,
            scriptable=True,
            exportable=True,
        )
    elif depth == "tiny":
        model = timm.create_model(
            "convnext_tiny",
            pretrained=pretrain,
            num_classes=0,
            global_pool="",
            output_stride=16,
            scriptable=True,
            exportable=True,
        )
    else:
        raise ValueError("Unsupported model type {}".format(depth))

    return model


@BACKBONE_REGISTRY.register()
def build_vgg_backbone(cfg):
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    depth = cfg.MODEL.BACKBONE.DEPTH

    if depth == "16x":
        model = timm.create_model(
            "vgg16_bn",
            pretrained=pretrain,
            num_classes=0,
            global_pool="",
            scriptable=True,
            exportable=True,
        )
    else:
        raise ValueError("Unsupported model type {}".format(depth))

    return model
