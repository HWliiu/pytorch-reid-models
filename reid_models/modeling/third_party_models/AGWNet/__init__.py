"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

# Project imported from https://github.com/mangye16/ReID-Survey/
from .. import REID_MODEL_BUILDER_REGISTRY
from .modeling import Baseline


@REID_MODEL_BUILDER_REGISTRY.register()
def resnet50_agw(num_classes, **kwargs):
    model = Baseline(
        num_classes,
        last_stride=1,
        model_path=None,
        model_name="resnet50_nl",
        gem_pool="on",
        pretrain_choice=None,
    )
    return model
