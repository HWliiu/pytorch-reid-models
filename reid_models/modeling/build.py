"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import logging
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import kornia as K
import torch
import torch.nn as nn
import yaml
from fvcore.common.checkpoint import (get_missing_parameters_message,
                                      get_unexpected_parameters_message)
from termcolor import colored

from .third_party_models import REID_MODEL_BUILDER_REGISTRY, list_models
from .utils import HiddenPrints, MutiInputSequential, download_url


def build_reid_model(model_name: str, pretrain_dataset: str):
    logger = logging.getLogger(__name__)
    try:
        with open(Path(__file__).parent / "models_config.yaml") as file:
            _models_config = yaml.load(file, Loader=yaml.FullLoader)
        model_config = _models_config[model_name][pretrain_dataset]

        weights_dir = Path(model_config.pop("weights_dir"))
        weights_dir.mkdir(parents=True, exist_ok=True)
        url = model_config.pop("url")
        file_name = Path(url).name
        weights_path = weights_dir / file_name
        if not weights_path.is_file():
            logger.warning(
                colored("Downloading model weights ... ... ", color="yellow")
            )
            try:
                download_url(url, weights_path)
            except:
                logger.error(colored("Download model weights failed. ", color="red"))
                weights_path.unlink(missing_ok=True)
                raise

        mean = model_config.pop("mean")
        std = model_config.pop("std")
    except KeyError:
        logger.error(
            colored(
                "Please check the `models_config.yaml` file for a valid configuration.",
                color="red",
            )
        )
        raise

    return _build_reid_model(model_name, weights_path, mean, std, **model_config)


def _build_reid_model(
    model_name: str,
    weights_path: str = None,
    mean: tuple[float] = (0.485, 0.456, 0.406),
    std: tuple[float] = (0.229, 0.224, 0.225),
    **kwargs,
) -> nn.Module:
    logger = logging.getLogger(__name__)

    if model_name not in list_models():
        raise ValueError(
            f"Model {model_name} not found, available models are {list_models()}"
        )

    # build reid model
    model_builder = REID_MODEL_BUILDER_REGISTRY.get(model_name)

    logger.info(colored(f"Building reid model `{model_name}`", attrs=["bold"]))
    with HiddenPrints():
        # kwargs are the parameter of each model builder
        reid_model = model_builder(**kwargs)

    # load model weights
    if weights_path is not None:
        loaded_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        logger.info(
            colored(f"Loading pretrained model from {weights_path}", attrs=["bold"])
        )

        new_dict = OrderedDict()
        origin_dict = reid_model.state_dict()
        deleted_keys = list()
        for k, v in loaded_dict.items():
            # fixed key name of state dict
            if k.startswith("module."):
                k = k[7:]
            # classify layer is unused
            if k in origin_dict and v.shape != origin_dict[k].shape:
                # for debugging only
                deleted_keys.append(k)
            else:
                new_dict[k] = v

        incompatible = reid_model.load_state_dict(new_dict, strict=False)
        del loaded_dict, origin_dict, new_dict

        if incompatible.missing_keys:
            logger.warn(get_missing_parameters_message(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            logger.warn(get_unexpected_parameters_message(incompatible.unexpected_keys))

    # warp model
    reid_model = MutiInputSequential(K.enhance.Normalize(mean, std), reid_model).eval()
    setattr(reid_model, "name", model_name)

    return reid_model
