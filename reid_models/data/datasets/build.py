"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

from os.path import dirname, join

import yaml

from .airportalert import AirportAlert
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .njust365 import NJUST365, NJUST365SPR, NJUST365WIN
from .occluded_reid import OccludedReID
from .partial_ilids import PartialILIDS
from .partial_reid import PartialReID
from .prid import PRID
from .rap import RAP
from .wildtrack import WildTrackCrop

__all__ = [
    "build_dataset_samples",
]


__datasets_factory = {
    "market1501": Market1501,
    "dukemtmcreid": DukeMTMCreID,
    "msmt17": MSMT17,
    "cuhk03": CUHK03,
    "wildtrack_crop": WildTrackCrop,
    "rap": RAP,
    "njust365": NJUST365,
    "njust365spr": NJUST365SPR,
    "njust365win": NJUST365WIN,
    "airportalert": AirportAlert,
    "prid": PRID,
    "occludedreid": OccludedReID,
    "partialreid": PartialReID,
    "partialilids": PartialILIDS,
}


def build_dataset_samples(dataset_names, combineall=False):
    # init dataset paths
    with open(join(dirname(__file__), "dataset_paths.yaml")) as file:
        __datasets_config = yaml.load(file, Loader=yaml.FullLoader)

    samples = []
    for dataset_name in dataset_names:
        assert (
            dataset_name in __datasets_factory.keys()
        ), "expect dataset in {}, but got {}".format(
            __datasets_factory.keys(), dataset_name
        )
        dataset_folder = join(
            __datasets_config[dataset_name]["path"],
            __datasets_config[dataset_name]["folder"],
        )
        download = __datasets_config[dataset_name]["download"]
        sample = __datasets_factory[dataset_name](
            dataset_folder, combineall=combineall, download=download
        )
        samples.append(sample)
    return samples
