from __future__ import absolute_import

from . import densenet, resnet

__model_factory = {
    **densenet.model_mapping,
    "resnet50": resnet.resnet50,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)
