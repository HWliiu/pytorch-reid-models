"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

from fvcore.common.registry import Registry

REID_MODEL_BUILDER_REGISTRY = Registry("REID_MODEL")
REID_MODEL_BUILDER_REGISTRY.__doc__ = """Registry for reid model"""


def _set_function_name(function_name):
    """For dynamically regist model builder."""

    def decorator(func):
        func.__name__ = function_name
        func.__qualname__ = function_name
        return func

    return decorator


from .ABDNet import *
from .AGWNet import *
from .APNet import *
from .DeepPersonReid import *
from .FastReID import *
from .ReidStrongBaseline import *
from .TransReID import *


def list_models():
    """List all registered models."""
    return list(REID_MODEL_BUILDER_REGISTRY._obj_map.keys())
