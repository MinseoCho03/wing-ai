import os
from typing import Dict, Any
import yaml
import torch
import random
import numpy as np

def load_config(config_path: str = "config.yaml", local_path: str = "config.local.yaml") -> Dict[str, Any]:
    def _load(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    base = _load(config_path) if os.path.exists(config_path) else {}
    if os.path.exists(local_path):
        local = _load(local_path)
        base = _deep_merge(base, local)
    return base

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = _deep_merge(a[k], v)
        else:
            a[k] = v
    return a

def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_cfg == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
