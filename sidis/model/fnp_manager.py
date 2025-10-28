"""
Top-level fNP manager module for TMD PDFs and FFs.

This module provides the main interface for managing both TMD PDF and FF
non-perturbative functions within a unified PyTorch framework. It handles:
- Simultaneous optimization of PDF and FF parameters
- Unified parameter management and masking
- Configuration-driven setup for both PDFs and FFs
- Consistent evolution factor across PDFs and FFs

Author: Chiara Bissolotti (cbissolotti@anl.gov)
Based on MAP22 parameterization from NangaParbat
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union
import yaml

from .fnp_base import fNP_evolution, TMDPDFBase, TMDFFBase


class fNPManager(nn.Module):
    """
    Unified manager for TMD PDF and FF non-perturbative functions.

    This class provides a single interface for managing both TMD PDFs and FFs
    with shared evolution and unified parameter optimization. It enables:

    - Joint optimization of PDF and FF parameters
    - Shared evolution factor across PDFs and FFs
    - Flavor-specific parameterizations for both PDFs and FFs
    - Configuration-driven parameter setup with masking
    - Unified parameter analysis and summary tools
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the fNP manager with unified PDF/FF configuration.

        Args:
            config (Dict): Configuration dictionary containing:
                - hadron: target hadron type
                - zeta: reference rapidity scale
                - evolution: shared evolution parameters
                - pdfs: PDF flavor configurations
                - ffs: FF flavor configurations
        """
        # First, call the constructor of the parent class (nn.Module)
        super().__init__()

        print("\033[94m[fNPManager] Initializing PDF and FF fNP modules\033[0m")

        # Extract global settings
        self.hadron = config.get("hadron", "proton")
        self.zeta = torch.tensor(config.get("zeta", 1.0), dtype=torch.float32)

        # Initialize shared evolution factor
        self._setup_evolution(config.get("evolution", {}))

        # Initialize PDF modules
        self._setup_pdf_modules(config.get("pdfs", {}))

        # Initialize FF modules
        self._setup_ff_modules(config.get("ffs", {}))

        print("\033[92m✅ fNP manager initialization completed\033[0m\n")

    def _setup_evolution(self, evolution_config: Dict[str, Any]):
        """Setup shared evolution factor module."""
        init_g2 = evolution_config.get("init_g2", 0.12840)  # MAP22 default
        free_mask = evolution_config.get("free_mask", [True])

        print(
            f"\033[94m[fNPManager] Setting up shared evolution: g2={init_g2}, trainable={free_mask[0]}\033[0m"
        )

        self.evolution = fNP_evolution(init_g2=init_g2, free_mask=free_mask)

    def _setup_pdf_modules(self, pdf_config: Dict[str, Any]):
        """Setup TMD PDF modules for all flavors."""
        # Define supported TMD PDF flavors
        self.pdf_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]

        pdf_modules = {}

        for flavor in self.pdf_flavor_keys:
            flavor_cfg = pdf_config.get(flavor, None)

            if flavor_cfg is None:
                # Use MAP22 defaults for undefined flavors
                from .fnp_base import MAP22_DEFAULT_PDF_PARAMS

                flavor_cfg = MAP22_DEFAULT_PDF_PARAMS.copy()
                print(
                    f"\033[93m[fNPManager] \033[92mUsing MAP22 defaults for PDF flavor '{flavor}'\033[0m"
                )
            # TODO: add here the check the else statement if the flavor_cfg is not None

            # Create TMD PDF module for this flavor
            pdf_modules[flavor] = TMDPDFBase(
                n_flavors=1,
                init_params=flavor_cfg["init_params"],
                free_mask=flavor_cfg["free_mask"],
            )

        self.pdf_modules = nn.ModuleDict(pdf_modules)
        print(
            f"\033[92m[fNPManager] \033[92mInitialized {len(self.pdf_modules)} PDF flavor modules\033[0m"
        )

    def _setup_ff_modules(self, ff_config: Dict[str, Any]):
        """Setup TMD FF modules for all flavors."""
        # Define supported FF flavors (same as PDFs for consistency)
        self.ff_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]

        ff_modules = {}

        for flavor in self.ff_flavor_keys:
            flavor_cfg = ff_config.get(flavor, None)

            if flavor_cfg is None:
                # Use MAP22 defaults for undefined flavors
                from .fnp_base import MAP22_DEFAULT_FF_PARAMS

                flavor_cfg = MAP22_DEFAULT_FF_PARAMS.copy()
                print(
                    f"\033[93m[fNPManager] Using MAP22 defaults for FF flavor '{flavor}'\033[0m"
                )

            # Create TMD FF module for this flavor
            ff_modules[flavor] = TMDFFBase(
                n_flavors=1,
                init_params=flavor_cfg["init_params"],
                free_mask=flavor_cfg["free_mask"],
            )

        self.ff_modules = nn.ModuleDict(ff_modules)
        print(
            f"\033[92m[fNPManager] Initialized {len(self.ff_modules)} FF flavor modules\033[0m"
        )

    def forward_pdf(
        self, x: torch.Tensor, b: torch.Tensor, flavors: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate TMD PDFs for specified flavors.

        Args:
            x (torch.Tensor): Bjorken x values
            b (torch.Tensor): Impact parameter values
            flavors (Optional[List[str]]): List of flavors to evaluate (None = all)

        Returns:
            Dict[str, torch.Tensor]: PDF values for each requested flavor
        """
        if flavors is None:
            flavors = self.pdf_flavor_keys

        # Compute shared evolution factor
        shared_evol = self.evolution(b, self.zeta)

        # Evaluate each requested PDF flavor
        outputs = {}
        for flavor in flavors:
            if flavor in self.pdf_modules:
                outputs[flavor] = self.pdf_modules[flavor](x, b, shared_evol, 0)
            else:
                raise ValueError(f"Unknown PDF flavor: {flavor}")

        return outputs

    def forward_ff(
        self, z: torch.Tensor, b: torch.Tensor, flavors: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate TMD FFs for specified flavors.

        Args:
            z (torch.Tensor): Energy fraction z values
            b (torch.Tensor): Impact parameter values
            flavors (Optional[List[str]]): List of flavors to evaluate (None = all)

        Returns:
            Dict[str, torch.Tensor]: FF values for each requested flavor
        """
        if flavors is None:
            flavors = self.ff_flavor_keys

        # Compute shared evolution factor
        shared_evol = self.evolution(b, self.zeta)

        # Evaluate each requested FF flavor
        outputs = {}
        for flavor in flavors:
            if flavor in self.ff_modules:
                outputs[flavor] = self.ff_modules[flavor](z, b, shared_evol, 0)
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
        Evaluate both TMD PDFs and FFs simultaneously.

        Args:
            x (torch.Tensor): Bjorken x values for PDFs
            z (torch.Tensor): Energy fraction z values for FFs
            b (torch.Tensor): Impact parameter values
            pdf_flavors (Optional[List[str]]): PDF flavors to evaluate
            ff_flavors (Optional[List[str]]): FF flavors to evaluate

        Returns:
            Dict containing 'pdfs' and 'ffs' sub-dictionaries with flavor results
        """
        # Unsqueeze to ensure correct tensor shapes for x, z. We want (n_events, 1)
        # so that broadcasting works correctly when combined with b of shape (n_events, n_b).
        x = x.unsqueeze(-1)
        z = z.unsqueeze(-1)

        return {
            "pdfs": self.forward_pdf(x, b, pdf_flavors),
            "ffs": self.forward_ff(z, b, ff_flavors),
        }

    def get_parameter_info(self) -> Dict[str, Any]:
        """
        Get comprehensive parameter information for both PDFs and FFs.

        Returns:
            Dict containing parameter counts, trainability info, and breakdowns
        """
        info = {
            "total_parameters": 0,
            "truly_trainable_parameters": 0,
            "pytorch_trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "evolution": {},
            "pdfs": {},
            "ffs": {},
        }

        # Evolution parameter
        if hasattr(self.evolution, "g2_mask"):
            is_trainable = bool(self.evolution.g2_mask[0].item())
            info["evolution"] = {
                "total": 1,
                "trainable": 1 if is_trainable else 0,
                "fixed": 0 if is_trainable else 1,
                "mask": [is_trainable],
            }
            info["total_parameters"] += 1
            if is_trainable:
                info["truly_trainable_parameters"] += 1

        # PDF parameters
        for flavor in self.pdf_flavor_keys:
            if flavor in self.pdf_modules:
                module = self.pdf_modules[flavor]
                n_params = module.n_params
                info["total_parameters"] += n_params

                if hasattr(module, "mask"):
                    free_count = int(module.mask.sum().item())
                    fixed_count = n_params - free_count
                    info["truly_trainable_parameters"] += free_count

                    info["pdfs"][flavor] = {
                        "total": n_params,
                        "trainable": free_count,
                        "fixed": fixed_count,
                        "mask": module.mask.squeeze().tolist(),
                    }

        # FF parameters
        for flavor in self.ff_flavor_keys:
            if flavor in self.ff_modules:
                module = self.ff_modules[flavor]
                n_params = module.n_params
                info["total_parameters"] += n_params

                if hasattr(module, "mask"):
                    free_count = int(module.mask.sum().item())
                    fixed_count = n_params - free_count
                    info["truly_trainable_parameters"] += free_count

                    info["ffs"][flavor] = {
                        "total": n_params,
                        "trainable": free_count,
                        "fixed": fixed_count,
                        "mask": module.mask.squeeze().tolist(),
                    }

        return info

    def print_parameter_summary(self):
        """Print a formatted summary of all parameters."""
        info = self.get_parameter_info()

        print("\n" + "=" * 70)
        print("fNP MANAGER PARAMETER SUMMARY (PDFs + FFs)")
        print("=" * 70)
        print(f"Total parameters: {info['total_parameters']}")
        print(f"Truly trainable parameters: {info['truly_trainable_parameters']}")
        print(f"PyTorch trainable parameters: {info['pytorch_trainable_parameters']}")
        print(
            f"Fixed parameters: {info['total_parameters'] - info['truly_trainable_parameters']}"
        )

        # Evolution
        print(f"\nShared Evolution Parameter:")
        evo = info["evolution"]
        print(
            f"  g2: {evo['total']} total ({evo['trainable']} trainable, {evo['fixed']} fixed)"
        )

        # PDFs
        print(f"\nTMD PDF Parameters:")
        for flavor, data in info["pdfs"].items():
            print(
                f"  {flavor}: {data['total']} total ({data['trainable']} trainable, {data['fixed']} fixed)"
            )
            if data["total"] <= 15:  # Show mask for reasonable parameter counts
                mask_str = ["T" if m else "F" for m in data["mask"]]
                print(f"      mask: [{', '.join(mask_str)}]")

        # FFs
        print(f"\nTMD FF Parameters:")
        for flavor, data in info["ffs"].items():
            print(
                f"  {flavor}: {data['total']} total ({data['trainable']} trainable, {data['fixed']} fixed)"
            )
            if data["total"] <= 15:  # Show mask for reasonable parameter counts
                mask_str = ["T" if m else "F" for m in data["mask"]]
                print(f"      mask: [{', '.join(mask_str)}]")

        print("=" * 70 + "\n")

    def get_trainable_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """
        Extract all trainable parameters for optimization.

        Returns:
            Dict mapping parameter paths to trainable parameter tensors
        """
        trainable_params = {}

        # Evolution parameter
        if hasattr(self.evolution, "g2") and hasattr(self.evolution, "g2_mask"):
            if self.evolution.g2_mask[0].item():  # If trainable
                trainable_params["evolution.g2"] = self.evolution.g2.data.clone()

        # PDF parameters
        for flavor in self.pdf_flavor_keys:
            if flavor in self.pdf_modules:
                module = self.pdf_modules[flavor]
                if hasattr(module, "mask"):
                    full_params = module.get_params_tensor.squeeze()
                    mask = module.mask.squeeze().bool()
                    trainable_values = full_params[mask]
                    trainable_params[f"pdfs.{flavor}.trainable"] = (
                        trainable_values.clone()
                    )

        # FF parameters
        for flavor in self.ff_flavor_keys:
            if flavor in self.ff_modules:
                module = self.ff_modules[flavor]
                if hasattr(module, "mask"):
                    full_params = module.get_params_tensor.squeeze()
                    mask = module.mask.squeeze().bool()
                    trainable_values = full_params[mask]
                    trainable_params[f"ffs.{flavor}.trainable"] = (
                        trainable_values.clone()
                    )

        return trainable_params

    def set_trainable_parameters_dict(self, trainable_params: Dict[str, torch.Tensor]):
        """
        Update model with trainable parameters only.

        Args:
            trainable_params: Dict from get_trainable_parameters_dict()
        """
        # Update evolution parameter
        if "evolution.g2" in trainable_params:
            # Directly update the g2 value through the property setter
            new_g2 = trainable_params["evolution.g2"]
            self.evolution.free_g2.data.copy_(new_g2.unsqueeze(0))

        # Update PDF parameters
        for flavor in self.pdf_flavor_keys:
            param_key = f"pdfs.{flavor}.trainable"
            if param_key in trainable_params and flavor in self.pdf_modules:
                module = self.pdf_modules[flavor]
                trainable_values = trainable_params[param_key]

                if hasattr(module, "mask"):
                    full_params = module.get_params_tensor.squeeze()
                    mask = module.mask.squeeze().bool()

                    # Update only trainable parameters
                    full_params[mask] = trainable_values

                    # Update module's parameters
                    fixed_part = full_params * (~mask).float()
                    free_part = full_params * mask.float()

                    module.fixed_params.data.copy_(fixed_part.unsqueeze(0))
                    module.free_params.data.copy_(free_part.unsqueeze(0))

        # Update FF parameters
        for flavor in self.ff_flavor_keys:
            param_key = f"ffs.{flavor}.trainable"
            if param_key in trainable_params and flavor in self.ff_modules:
                module = self.ff_modules[flavor]
                trainable_values = trainable_params[param_key]

                if hasattr(module, "mask"):
                    full_params = module.get_params_tensor.squeeze()
                    mask = module.mask.squeeze().bool()

                    # Update only trainable parameters
                    full_params[mask] = trainable_values

                    # Update module's parameters
                    fixed_part = full_params * (~mask).float()
                    free_part = full_params * mask.float()

                    module.fixed_params.data.copy_(fixed_part.unsqueeze(0))
                    module.free_params.data.copy_(free_part.unsqueeze(0))
