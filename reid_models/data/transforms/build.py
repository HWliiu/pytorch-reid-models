"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import torchvision.transforms as transforms

_transforms_factory_before = {
    "autoaug": transforms.AutoAugment(),
    "randomflip": transforms.RandomHorizontalFlip(p=0.5),
    "randomcrop": transforms.RandomCrop((256, 128), padding=10),
    "colorjitor": transforms.ColorJitter(
        brightness=0.25, contrast=0.15, saturation=0.25, hue=0
    ),
    "augmix": transforms.AugMix(),
}

_transforms_factory_after = {"rea": transforms.RandomErasing()}


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

    trans = [transforms.Resize(image_size)]
    for transform in transforms_list:
        if transform in _transforms_factory_before.keys():
            trans.append(_transforms_factory_before[transform])

    trans.append(transforms.ToTensor())

    for transform in transforms_list:
        if transform in _transforms_factory_after.keys():
            trans.append(_transforms_factory_after[transform])

    return transforms.Compose(trans)
