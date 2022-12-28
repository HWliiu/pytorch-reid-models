"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import functools
import logging
import sys
import time
from pathlib import Path


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(name: str, distributed_rank: int = 0, log_dir: str = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if distributed_rank == 0 and not logger.handlers:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(ch)

        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_name = time.strftime("%Y-%m-%d_%H-%M", time.localtime()) + ".log"
            fh = logging.FileHandler(log_dir / file_name, mode="a")
            fh.setLevel(logging.INFO)
            fh.setFormatter(
                logging.Formatter("%(asctime)s %(name)s %(levelname)s:\n %(message)s")
            )
            logger.addHandler(fh)
    else:
        logging.disable(logging.CRITICAL)

    return logger
