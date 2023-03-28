"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import logging
from copy import deepcopy
from typing import Callable, Dict, Tuple, Union

import numpy as np
from PIL import Image
from termcolor import colored
from torch.utils import data

from .datasets import build_dataset_samples
from .samplers import PKSampler
from .transforms import build_transforms

__all__ = [
    "build_train_dataset",
    "build_test_datasets",
    "build_train_dataloader",
    "build_test_dataloaders",
]


class ReIDDataset(data.Dataset):
    def __init__(self, samples, transform):
        self.samples = deepcopy(samples)
        self.transform = deepcopy(transform)

    def __getitem__(self, index):
        sample = self.samples[index]
        img = self._loader(sample[0])
        if self.transform is not None:
            img = self.transform(img)
        return (img, *sample[1:])

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert("RGB")


def _combine(samples_list):
    """combine more than one samples (e.g. market.train and duke.train) as a samples"""
    all_samples = []
    max_pid, max_cid = 0, 0
    for samples in samples_list:
        for a_sample in samples:
            img_path = a_sample[0]
            pid = max_pid + a_sample[1]
            cid = max_cid + a_sample[2]
            all_samples.append([img_path, pid, cid])
        max_pid = max([sample[1] for sample in all_samples])
        max_cid = max([sample[2] for sample in all_samples])
    return all_samples


def build_train_dataset(
    dataset_names: Tuple[str],
    image_size: Tuple[int, int] = None,
    transforms: Union[Tuple[str], Callable] = None,
    combineall: bool = False,
    per_dataset_num: int = None,
    **kwargs
):
    logger = logging.getLogger(__name__)
    logger.info(colored("building train datasets ... ... ", attrs=["bold"]))
    samples = build_dataset_samples(dataset_names, combineall)

    if per_dataset_num is not None:
        for sample in samples:
            total_num = len(sample.train)
            index = np.linspace(0, total_num, per_dataset_num, False).astype(int)
            sample.train = [train for i, train in enumerate(sample.train) if i in index]

    if not callable(transforms):
        transforms = build_transforms(
            image_size=image_size, transforms_list=transforms, **kwargs
        )
    train_sample = _combine([sample.train for sample in samples])
    train_dataset = ReIDDataset(train_sample, transform=transforms)
    train_dataset.name = "_".join(dataset_names)

    return train_dataset


def build_train_dataloader(
    batch_size: int = 32,
    sampler: Union[str, data.Sampler] = "random",
    train_dataset: data.Dataset = None,
    **kwargs
):
    if train_dataset is None:
        train_dataset = build_train_dataset(**kwargs)

    shuffle = None
    if isinstance(sampler, str):
        if sampler == "random":
            sampler = None
            shuffle = True
        elif sampler == "pk":
            assert (
                "num_instance" in kwargs.keys()
            ), "param num_instance(int) must be given when sample='pk'"
            num_instance = kwargs["num_instance"]
            assert (
                batch_size % num_instance == 0
            ), "batch_size must be divided by num_instance"
            sampler = PKSampler(train_dataset, k=num_instance)
        else:
            raise ValueError("Unsupported sampler.")

    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=kwargs.get("persistent_workers", False),
    )

    return train_data_loader


def build_test_datasets(
    dataset_names: Tuple[str],
    image_size: Tuple[int, int] = None,
    transforms: Union[Tuple[str], Callable] = None,
    query_num: int = None,
    **kwargs
):
    logger = logging.getLogger(__name__)
    logger.info(colored("building test datasets ... ... ", attrs=["bold"]))
    samples = build_dataset_samples(dataset_names, combineall=False)
    if not callable(transforms):
        transforms = build_transforms(
            image_size=image_size, transforms_list=transforms, **kwargs
        )

    query_gallery_datasets = {}
    for dataset_name, sample in zip(dataset_names, samples):
        if query_num is not None:
            index = np.linspace(0, len(sample.query), query_num, False).astype(int)
            sample.query = [query for i, query in enumerate(sample.query) if i in index]
        query_dataset = ReIDDataset(sample.query, transform=transforms)
        gallery_dataset = ReIDDataset(sample.gallery, transform=transforms)
        query_dataset.name = gallery_dataset.name = dataset_name
        query_gallery_datasets[dataset_name] = (query_dataset, gallery_dataset)

    return query_gallery_datasets


def build_test_dataloaders(
    query_batch_size: int = 128,
    gallery_batch_size: int = 128,
    query_sampler: data.Sampler = None,
    gallery_sampler: data.Sampler = None,
    test_datasets: Dict[str, data.Dataset] = None,
    **kwargs
):
    if test_datasets is None:
        test_datasets = build_test_datasets(**kwargs)

    query_gallery_data_loaders = {}
    for dataset_name, (query_dataset, gallery_dataset) in test_datasets.items():
        query_data_loader = data.DataLoader(
            query_dataset,
            batch_size=query_batch_size,
            num_workers=8,
            sampler=query_sampler,
            pin_memory=True,
        )
        gallery_data_loader = data.DataLoader(
            gallery_dataset,
            batch_size=gallery_batch_size,
            num_workers=8,
            sampler=gallery_sampler,
            pin_memory=True,
        )
        query_gallery_data_loaders[dataset_name] = (
            query_data_loader,
            gallery_data_loader,
        )

    return query_gallery_data_loaders
