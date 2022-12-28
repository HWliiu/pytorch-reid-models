# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from . import losses
from .backbones import BACKBONE_REGISTRY, build_backbone, build_resnet_backbone
from .heads import REID_HEADS_REGISTRY, EmbeddingHead, build_heads
from .meta_arch import META_ARCH_REGISTRY, build_model

__all__ = [k for k in globals().keys() if not k.startswith("_")]
