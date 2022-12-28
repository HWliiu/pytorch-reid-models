"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

# Project imported from https://github.com/KaiyangZhou/deep-person-reid
import sys

import torch

from .. import REID_MODEL_BUILDER_REGISTRY, _set_function_name
from .models import __model_factory, build_model

__all__ = [name + "_dpr" for name in __model_factory.keys()]

_module = sys.modules[__name__]

for model_name in __all__:

    @REID_MODEL_BUILDER_REGISTRY.register()
    @_set_function_name(model_name)
    def _(num_classes, *, _name=model_name.removesuffix("_dpr"), **kwargs):
        # default `_name` parameter for avoid delay binding
        model = build_model(
            name=_name,
            num_classes=num_classes,
            pretrained=True,
            use_gpu=torch.cuda.is_available(),
        )
        return model

    setattr(_module, model_name, _)
