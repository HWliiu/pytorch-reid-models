"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

# Project imported from https://github.com/JDAI-CV/fast-reid
import sys
from pathlib import Path

from .. import REID_MODEL_BUILDER_REGISTRY, _set_function_name
from .config import get_cfg
from .modeling import build_model

default_cfg = get_cfg()
configs_path = (Path(__file__).parent / "configs/model_configs").glob("*.yml")

__all__ = []
_module = sys.modules[__name__]
for config_path in configs_path:
    model_name = config_path.stem + "_fastreid"
    __all__.append(model_name)

    @REID_MODEL_BUILDER_REGISTRY.register()
    @_set_function_name(model_name)
    def _(num_classes, *, _config_path=config_path, **kwargs):
        # default `_config_path` parameter for avoid delay binding
        cfg = default_cfg.clone()
        cfg.merge_from_file(_config_path)
        cfg.MODEL.HEADS.NUM_CLASSES = num_classes
        model = build_model(cfg)
        return model

    setattr(_module, model_name, _)
