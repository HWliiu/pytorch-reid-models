"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import os
import sys

import requests
import torch.nn as nn
from tqdm import tqdm


class HiddenPrints:
    """Context manager that suppresses printing"""

    def __init__(self, activated=True):
        self.activated = activated
        self.original_stdout = None

    def open(self):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()


class MutiInputSequential(nn.Sequential):
    """Only process the first input except the last module and the last module supports muti-input"""

    def forward(self, input, *args, **kwargs):
        for i, module in enumerate(self):
            if i != len(self) - 1:
                input = module(input)
            else:
                input = module(input, *args, **kwargs)
        return input


def download_url(url: str, fname: str):
    with requests.get(url, stream=True, allow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(fname, "wb") as file, tqdm(
            desc="Downloading",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
