"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

# Project imported from https://github.com/damo-cv/TransReID
from yacs.config import CfgNode

from .. import REID_MODEL_BUILDER_REGISTRY
from .model import make_model

__all__ = ["vit_base_transreid", "vit_transreid", "deit_transreid"]

base_config = """
                INPUT:
                  SIZE_TRAIN:
                  - 256
                  - 128
                MODEL:
                  ATT_DROP_RATE: 0.0
                  COS_LAYER: false
                  DEVIDE_LENGTH: 4
                  DROP_OUT: 0.0
                  DROP_PATH: 0.1
                  ID_LOSS_TYPE: softmax
                  JPM: false
                  LAST_STRIDE: 1
                  NAME: transformer
                  NECK: bnneck
                  PRETRAIN_CHOICE: self
                  PRETRAIN_PATH: ''
                  RE_ARRANGE: true
                  SHIFT_NUM: 5
                  SHUFFLE_GROUP: 2
                  SIE_CAMERA: false
                  SIE_COE: 3.0
                  SIE_VIEW: false
                  STRIDE_SIZE:
                  - 16
                  - 16
                  TRANSFORMER_TYPE: vit_base_patch16_224_TransReID
                TEST:
                  NECK_FEAT: before"""
cfg = CfgNode.load_cfg(base_config)


@REID_MODEL_BUILDER_REGISTRY.register()
def vit_base_transreid(num_classes, **kwargs):
    model = make_model(cfg, num_class=num_classes, camera_num=0, view_num=0)
    return model


@REID_MODEL_BUILDER_REGISTRY.register()
def vit_transreid(num_classes, camera_num, **kwargs):
    cfg.MODEL.STRIDE_SIZE = [12, 12]
    cfg.MODEL.SIE_CAMERA = True
    cfg.MODEL.JPM = True
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=0)
    return model


@REID_MODEL_BUILDER_REGISTRY.register()
def deit_transreid(num_classes, camera_num, **kwargs):
    """Structure same as vit_transreid, but with different weights"""
    return vit_transreid(num_classes, camera_num, **kwargs)
