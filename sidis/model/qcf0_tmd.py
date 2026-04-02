"""
Non-perturbative TMD wrapper with version tracking.

This wrapper directly subclasses the unified `fNPManager` and only adds:
- `trainable` mode toggle
- `forward_evolution` alias used by `TMDBuilder`
- parameter version tracking for cache invalidation
"""

import torch
from typing import List, Dict, Any
from omegaconf import OmegaConf
from .fnp_manager import fNPManager


class TrainablefNP(fNPManager):
    """fNP manager wrapper with optional trainability and version tracking."""

    def __init__(self, fnp_config: Dict[str, Any], trainable: bool = True):
        if hasattr(fnp_config, "keys") and not isinstance(fnp_config, dict):
            config_dict = OmegaConf.to_container(fnp_config, resolve=True)
        else:
            config_dict = fnp_config

        super().__init__(config=config_dict)

        for param in self.parameters():
            param.requires_grad = bool(trainable)

    def version(self) -> List[int]:
        versions = []
        for _, param in self.named_parameters():
            if hasattr(param, "_version"):
                versions.append(param._version)
            else:
                versions.append(id(param.data))
        return versions
