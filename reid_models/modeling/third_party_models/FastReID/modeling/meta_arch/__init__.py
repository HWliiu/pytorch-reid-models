# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .build import META_ARCH_REGISTRY, build_model
from .distiller import Distiller
from .mgn import MGN
from .moco import MoCo
