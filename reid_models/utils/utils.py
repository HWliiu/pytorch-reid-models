"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
