"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

from torch.nn import ModuleList
import torchvision.transforms as T

_transforms_factory_before = {
    "autoaug": T.RandomApply(ModuleList([T.AutoAugment()]), p=0.1),
    "randomflip": T.RandomHorizontalFlip(p=0.5),
    "randomcrop": T.RandomCrop((256, 128), padding=10),
    "colorjitor": T.ColorJitter(
        brightness=0.25, contrast=0.25, saturation=0.25, hue=0.15
    ),
    "augmix": T.RandomApply(ModuleList([T.AugMix()]), p=0.1),
}

_transforms_factory_after = {"rea": T.RandomErasing()}


def build_transforms(image_size, transforms_list, **kwargs):
    if image_size is None:
        image_size = (256, 128)
    if transforms_list is None:
        transforms_list = []

    for transform in transforms_list:
        assert (
            transform in _transforms_factory_before.keys()
            or transform in _transforms_factory_after.keys()
        ), "Expect transforms in {} and {}, got {}".format(
            _transforms_factory_before.keys(),
            _transforms_factory_after.keys(),
            transform,
        )

    trans = [T.Resize(image_size)]
    for transform in transforms_list:
        if transform in _transforms_factory_before.keys():
            trans.append(_transforms_factory_before[transform])

    trans.append(T.ToTensor())

    for transform in transforms_list:
        if transform in _transforms_factory_after.keys():
            trans.append(_transforms_factory_after[transform])

    return T.Compose(trans)
