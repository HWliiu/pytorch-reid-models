# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline


def build_model(num_classes, model_name):
    model = Baseline(
        num_classes,
        last_stride=1,
        model_path=None,
        neck="bnneck",
        neck_feat="after",
        model_name=model_name,
        pretrain_choice="self",
    )
    return model
