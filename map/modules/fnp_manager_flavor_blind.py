"""
Flavor-blind fNP manager module for TMD PDFs and FFs.

This module provides a unified interface for managing flavor-blind TMD PDF and FF
non-perturbative functions where ALL flavors share the same parameters. Unlike the
standard fNP manager where each flavor has its own parameter set, this implementation
uses single parameter sets that apply identically to all flavors.

Key differences from standard fNPManager:
- Single TMDPDFFlavorBlind instance serves all PDF flavors
- Single TMDFFFlavorBlind instance serves all FF flavors
- All flavors evolve together when parameters change
- Significant reduction in total parameter count

Author: Chiara Bissolotti (cbissolotti@anl.gov)
Based on MAP22 parameterization from NangaParbat
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union
import yaml

from .fnp_base_flavor_blind import (
    fNP_evolution,
    TMDPDFFlavorBlind,
    TMDFFFlavorBlind,
    MAP22_DEFAULT_EVOLUTION_FLAVOR_BLIND,
    MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND,
    MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND,
)


class fNPManagerFlavorBlind(nn.Module):
    """
    Flavor-blind unified manager for TMD PDF and FF non-perturbative functions.

    This class provides a single interface for managing flavor-blind TMD PDFs and FFs
    where all flavors share identical parameters. Key features:

    - Single parameter set for ALL PDF flavors (u, d, s, ubar, dbar, sbar, etc.)
    - Single parameter set for ALL FF flavors
    - Shared evolution factor across PDFs and FFs
    - Unified parameter optimization with dramatic parameter reduction
    - Configuration-driven parameter setup with masking

    Parameter count comparison:
    - Standard system: ~160 parameters (8 flavors x 11 PDF + 8 flavors x 9 FF + 1 evolution)
    - Flavor-blind system: 21 parameters (11 PDF + 9 FF + 1 evolution)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize flavor-blind fNP manager from configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - hadron: Hadron type (e.g., "proton")
                - zeta: Reference rapidity scale
                - evolution: Evolution configuration
                - pdf: Single PDF configuration (applies to all flavors)
                - ff: Single FF configuration (applies to all flavors)
        """
        super().__init__()

        # Store configuration
        self.config = config
        self.hadron = config.get("hadron", "proton")
        self.zeta = torch.tensor(config.get("zeta", 1.0), dtype=torch.float32)
        # Make zeta accessible as self._zeta.
        # If we do just self._zeta = tensor without registering, PyTorch wouldn’t
        # know to move/save it. As a buffer, it’s tracked like running stats in
        # BatchNorm, masks, constants, lookup tables, etc.
        # We are marking zeta as a non-trainable tensor that should live with
        # the model (move, save, load) just like real parameters do, minus gradient
        # updates, meaning it is not a trainable parameter.
        # The underscore in _zeta indicates it is intended for internal use, it's a
        # convention.
        self.register_buffer("_zeta", self.zeta)

        # Define all supported flavors (same as standard system)
        self.flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]

        # Setup evolution (shared across PDFs and FFs)
        self._setup_evolution(config.get("evolution", {}))

        # Setup single PDF module (serves all flavors)
        self._setup_pdf_module(config.get("pdf", {}))

        # Setup single FF module (serves all flavors)
        self._setup_ff_module(config.get("ff", {}))

        print(f"\033[92m[fNPManagerFlavorBlind] Initialized flavor-blind fNP manager")
        print(f"  Hadron: {self.hadron}")
        print(f"  Zeta: {self.zeta.item():.1f} GeV²")
        print(f"  All {len(self.flavor_keys)} flavors share identical parameters")
        print(f"  Total parameter count: {self.count_parameters()}\033[0m")

    def _setup_evolution(self, evolution_config: Dict[str, Any]):
        """
        Setup shared evolution factor.
        Args:
            evolution_config (Dict[str, Any]): Evolution configuration dictionary
        Returns: None

        In the initialization of the fNPManagerFlavorBlind class, this method is
        called with the content of the "evolution" key from the config dictionary
        as argument. If the "evolution" key is missing or empty, it defaults to
        MAP22_DEFAULT_EVOLUTION_FLAVOR_BLIND parameters.
        """
        # If there is no evolution key in the fNP config, use defaults.
        if not evolution_config:
            # Set the evolution config as a copy of the default module-level dictionary.
            # Using copy() avoids modifying the original default dictionary in case
            # we later modify evolution_config.
            evolution_config = MAP22_DEFAULT_EVOLUTION_FLAVOR_BLIND.copy()
            print(
                f"\033[93m[fNPManagerFlavorBlind] Using default evolution parameters\033[0m"
            )

        # Set self.evolution as an instance of fNP_evolution class.
        self.evolution = fNP_evolution(
            init_g2=evolution_config.get("init_g2", 0.12840),
            free_mask=evolution_config.get("free_mask", [True]),
        )

        print(
            f"\033[92m[fNPManagerFlavorBlind] Initialized shared evolution module\033[0m"
        )

    def _setup_pdf_module(self, pdf_config: Dict[str, Any]):
        """Setup single PDF module that serves all flavors."""

        if not pdf_config:
            pdf_config = MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND.copy()
            print(
                f"\033[93m[fNPManagerFlavorBlind] Using default PDF parameters\033[0m"
            )

        # Create single PDF module for ALL flavors
        self.pdf_module = TMDPDFFlavorBlind(
            init_params=pdf_config.get(
                "init_params", MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND["init_params"]
            ),
            free_mask=pdf_config.get(
                "free_mask", MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND["free_mask"]
            ),
        )

        print(
            f"\033[92m[fNPManagerFlavorBlind] Initialized single PDF module serving all {len(self.flavor_keys)} flavors\033[0m"
        )

    def _setup_ff_module(self, ff_config: Dict[str, Any]):
        """Setup single FF module that serves all flavors."""
        if not ff_config:
            ff_config = MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND.copy()
            print(f"\033[93m[fNPManagerFlavorBlind] Using default FF parameters\033[0m")

        # Create single FF module for ALL flavors
        self.ff_module = TMDFFFlavorBlind(
            init_params=ff_config.get(
                "init_params", MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND["init_params"]
            ),
            free_mask=ff_config.get(
                "free_mask", MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND["free_mask"]
            ),
        )

        print(
            f"\033[92m[fNPManagerFlavorBlind] Initialized single FF module serving all {len(self.flavor_keys)} flavors\033[0m"
        )

    def forward_pdf(
        self, x: torch.Tensor, b: torch.Tensor, flavors: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate TMD PDFs for specified flavors using shared parameters.

        Args:
            x (torch.Tensor): Bjorken x values
            b (torch.Tensor): Impact parameter values
            flavors (Optional[List[str]]): List of flavors to evaluate (None = all)

        Returns:
            Dict[str, torch.Tensor]: PDF values for each requested flavor (all identical)
        """
        if flavors is None:
            flavors = self.flavor_keys

        # Compute shared evolution factor
        shared_evol = self.evolution(b, self._zeta)

        # Evaluate PDF using shared parameters (same result for all flavors)
        shared_pdf_result = self.pdf_module(x, b, shared_evol)

        # Return the same result for all requested flavors
        outputs = {}
        for flavor in flavors:
            if flavor in self.flavor_keys:
                outputs[flavor] = shared_pdf_result
            else:
                raise ValueError(f"Unknown PDF flavor: {flavor}")

        return outputs

    def forward_ff(
        self, z: torch.Tensor, b: torch.Tensor, flavors: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate TMD FFs for specified flavors using shared parameters.

        Args:
            z (torch.Tensor): Energy fraction z values
            b (torch.Tensor): Impact parameter values
            flavors (Optional[List[str]]): List of flavors to evaluate (None = all)

        Returns:
            Dict[str, torch.Tensor]: FF values for each requested flavor (all identical)
        """
        if flavors is None:
            flavors = self.flavor_keys

        # Compute shared evolution factor
        shared_evol = self.evolution(b, self._zeta)

        # Evaluate FF using shared parameters (same result for all flavors)
        shared_ff_result = self.ff_module(z, b, shared_evol)

        # Return the same result for all requested flavors
        outputs = {}
        for flavor in flavors:
            if flavor in self.flavor_keys:
                outputs[flavor] = shared_ff_result
            else:
                raise ValueError(f"Unknown FF flavor: {flavor}")

        return outputs

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        pdf_flavors: Optional[List[str]] = None,
        ff_flavors: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Evaluate both TMD PDFs and FFs simultaneously using shared parameters.

        Args:
            x (torch.Tensor): Bjorken x values for PDFs
            z (torch.Tensor): Energy fraction z values for FFs
            b (torch.Tensor): Impact parameter values
            pdf_flavors (Optional[List[str]]): PDF flavors to evaluate
            ff_flavors (Optional[List[str]]): FF flavors to evaluate

        Returns:
            Dict containing 'pdfs' and 'ffs' sub-dictionaries with flavor results
        """
        return {
            "pdfs": self.forward_pdf(x, b, pdf_flavors),
            "ffs": self.forward_ff(z, b, ff_flavors),
        }

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_info(self) -> Dict[str, Any]:
        """
        Get detailed parameter information for analysis and debugging.

        Returns:
            Dict containing parameter counts, current values, and trainability info
        """
        info = {
            "total_parameters": self.count_parameters(),
            "parameter_breakdown": {},
            "current_values": {},
            "trainable_flags": {},
        }

        # Evolution parameters
        evolution_params = sum(
            p.numel() for p in self.evolution.parameters() if p.requires_grad
        )
        info["parameter_breakdown"]["evolution"] = evolution_params
        info["current_values"]["evolution_g2"] = (
            self.evolution.g2.detach().cpu().numpy().tolist()
        )
        info["trainable_flags"]["evolution"] = (
            self.evolution.mask.detach().cpu().numpy().tolist()
        )

        # PDF parameters (shared across all flavors)
        pdf_params = sum(
            p.numel() for p in self.pdf_module.parameters() if p.requires_grad
        )
        info["parameter_breakdown"]["pdf_shared"] = pdf_params
        info["current_values"]["pdf_params"] = (
            self.pdf_module.get_params_tensor.detach().cpu().numpy().tolist()
        )
        info["trainable_flags"]["pdf"] = (
            self.pdf_module.mask.detach().cpu().numpy().tolist()
        )

        # FF parameters (shared across all flavors)
        ff_params = sum(
            p.numel() for p in self.ff_module.parameters() if p.requires_grad
        )
        info["parameter_breakdown"]["ff_shared"] = ff_params
        info["current_values"]["ff_params"] = (
            self.ff_module.get_params_tensor.detach().cpu().numpy().tolist()
        )
        info["trainable_flags"]["ff"] = (
            self.ff_module.mask.detach().cpu().numpy().tolist()
        )

        # Flavor information
        info["flavors"] = {
            "supported_flavors": self.flavor_keys,
            "total_flavors": len(self.flavor_keys),
            "parameters_per_flavor": "SHARED - all flavors use identical parameters",
        }

        return info

    def summary(self):
        """Print a human-readable summary of the flavor-blind fNP manager."""
        info = self.get_parameter_info()

        print("\n" + "=" * 70)
        print("FLAVOR-BLIND fNP MANAGER SUMMARY")
        print("=" * 70)
        print(f"Hadron: {self.hadron}")
        print(f"Reference scale ζ: {self.zeta.item():.1f} GeV²")
        print(f"Supported flavors: {', '.join(self.flavor_keys)}")
        print(f"\nParameter Structure:")
        print(f"  • Evolution: {info['parameter_breakdown']['evolution']} parameters")
        print(
            f"  • PDF (shared): {info['parameter_breakdown']['pdf_shared']} parameters"
        )
        print(f"  • FF (shared): {info['parameter_breakdown']['ff_shared']} parameters")
        print(f"  • Total: {info['total_parameters']} parameters")

        print(
            f"\nKey Feature: ALL {len(self.flavor_keys)} flavors share identical parameters!"
        )
        print(
            f"Standard system would have ~{len(self.flavor_keys) * (11 + 9) + 1} parameters"
        )
        print(f"Flavor-blind system has only {info['total_parameters']} parameters")
        print(
            f"Parameter reduction: {100 * (1 - info['total_parameters'] / (len(self.flavor_keys) * 20 + 1)):.1f}%"
        )
        print("=" * 70)


def load_flavor_blind_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate flavor-blind fNP configuration from YAML file.

    Args:
        config_path (str): Path to YAML configuration file

    Returns:
        Dict[str, Any]: Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}")

    # Validate required sections
    required_sections = ["hadron", "zeta"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Set defaults for optional sections
    if "evolution" not in config:
        config["evolution"] = MAP22_DEFAULT_EVOLUTION_FLAVOR_BLIND.copy()
        print(
            f"\033[93m[load_flavor_blind_config] Using default evolution configuration\033[0m"
        )

    if "pdf" not in config:
        config["pdf"] = MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND.copy()
        print(
            f"\033[93m[load_flavor_blind_config] Using default PDF configuration\033[0m"
        )

    if "ff" not in config:
        config["ff"] = MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND.copy()
        print(
            f"\033[93m[load_flavor_blind_config] Using default FF configuration\033[0m"
        )

    return config
