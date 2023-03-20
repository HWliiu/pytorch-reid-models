import torch
import torch.nn as nn

from .baseline import Baseline  # ,Baseline_SE,Baseline_DENSE


def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == "resnet50":
        model = Baseline(
            num_classes,
            cfg.MODEL.LAST_STRIDE,
            cfg.MODEL.PRETRAIN_PATH,
            cfg.APNET.LEVEL,
            cfg.APNET.MSMT,
        )
        return model
    else:
        raise RuntimeError("'{}' is not available".format(cfg.MODEL.NAME))
