from typing import Any, Dict, Union

import numpy as np
import yaml
from torchvision.datasets import ImageFolder, VisionDataset
import os
import json
from pathlib import Path


def read_yaml(cfg: Union[str, Dict[str, Any]]):
    if not isinstance(cfg, dict):
        with open(cfg) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = cfg
    return config


def write_yaml(cfg: Union[str, Dict[str, Any]], name, path=""):
    if isinstance(cfg, dict):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, name + ".yaml"), "w") as f:
            yaml.dump(cfg, f)
    else:
        ValueError


def get_label_counts(dataset_path: str):
    """Counts for each label."""
    if not dataset_path:
        return None
    td = ImageFolder(root=dataset_path)
    # get label distribution
    label_counts = [0] * len(td.classes)
    for p, l in td.samples:
        label_counts[l] += 1
    return label_counts


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
