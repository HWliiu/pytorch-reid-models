"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

# Project imported from https://github.com/VITA-Group/ABD-Net
import sys

from .. import REID_MODEL_BUILDER_REGISTRY, _set_function_name
from .models import get_names, init_model

model_names = list(name + "_abd" for name in get_names())

__all__ = model_names

_module = sys.modules[__name__]

for model_name in model_names:

    @REID_MODEL_BUILDER_REGISTRY.register()
    @_set_function_name(model_name)
    def _(num_classes, *, _name=model_name.removesuffix("_abd"), **kwargs):
        # default `_name` parameter for avoid delay binding

        # I don't know which one is used!
        args = {
            "compatibility": False,
            "branches": ["global", "abd"],
            "dropout": 0.5,
            "global_dim": 1024,
            "global_max_pooling": False,
            "abd_dim": 1024,
            "abd_np": 2,
            "abd_dan": ["cam", "pam"],
            "abd_dan_no_head": False,
            "shallow_cam": True,
            "np_dim": 1024,
            "np_np": 2,
            "np_with_global": False,
            "np_max_pooling": False,
            "dan_dim": 1024,
            "dan_dan": [],
            "dan_dan_no_head": False,
            "use_of": True,
            "of_beta": 1e-06,
            "of_start_epoch": 0,
            "of_position": ["before", "after", "cam", "pam", "intermediate"],
            "use_ow": True,
            "ow_beta": 0.001,
        }
        model = init_model(name=_name, num_classes=num_classes, args=args, **kwargs)
        return model

    setattr(_module, model_name, _)
