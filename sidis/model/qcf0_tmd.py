"""
Non-perturbative TMD factor (fNP) wrappers with version tracking.

This module provides wrappers around the fNP manager for both ground truth
and trainable scenarios, analogous to the parametricqcf0.py in the DIS code.

Author: Refactored from original TrainableModel
"""

import torch
from typing import List, Dict
from .fnp_factory import create_fnp_manager


class TruefNP(torch.nn.Module):
    """Ground truth fNP manager with fixed parameters.
    
    This class wraps the fNP manager created from configuration,
    but does not have trainable parameters. Used for generating
    ground truth or validating against known parametrizations.
    
    Args:
        fnp_config: Configuration dictionary for fNP manager
    """
    
    def __init__(self, fnp_config: Dict):
        super().__init__()
        
        # Create fNP manager from config
        # fNPManager is a proper nn.Module, so register it as submodule
        self.fnp_manager = create_fnp_manager(config_dict=fnp_config)
        
        # For ground truth, we don't want parameters to be trainable
        # Freeze all parameters
        for param in self.fnp_manager.parameters():
            param.requires_grad = False

    def forward_evolution(self, bT: torch.Tensor, Q:torch.Tensor) -> torch.Tensor:
        """Compute non-perturbative evolution factor.
        
        Args:
            bT: Fourier conjugate of transverse momentum values, shape (n_events, n_bT)
            Q: Hard scale, shape (n_events,)
            
        Returns:
            Evolution factor tensor (CS kernel)
        """
        return self.fnp_manager.get_evolution(bT, Q)
        
    def forward_pdf(self, x: torch.Tensor, bT: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute non-perturbative TMD PDF factors for all flavors.
        
        Args:
            x: momentum fraction values, shape (n_events,)
            bT: Fourier conjugate of transverse momentum values, shape (n_events, n_bT)
            
        Returns:
            Dictionary mapping flavor names to fNP tensors
        """
        return self.fnp_manager.forward_pdf(x, bT)
    
    def forward_ff(self, z: torch.Tensor, bT: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute non-perturbative fragmentation function factors for all flavors.
        
        Args:
            z: Fragmentation fraction, shape (n_events,)
            bT: Fourier conjugate of transverse momentum values, shape (n_events, n_bT)
            
        Returns:
            Dictionary mapping flavor names to fNP tensors
        """
        return self.fnp_manager.forward_ff(z, bT)
    
    def forward_sivers(self, x: torch.Tensor, bT: torch.Tensor) -> torch.Tensor:
        """Compute non-perturbative Sivers function factor.
        
        Note: Currently does not have flavor dependence.
        
        Args:
            x: momentum fraction values, shape (n_events,)
            bT: Fourier conjugate of transverse momentum values, shape (n_events, n_bT)
            
        Returns:
            Sivers fNP tensor (no flavor dependence yet)
        """
        return self.fnp_manager.forward_sivers(x, bT)


class TrainablefNP(TruefNP):
    """Trainable fNP manager with parameter version tracking.
    
    Inherits from TruefNP but enables trainable parameters and adds
    version tracking for cache invalidation (analogous to DIS pattern).
    
    When parameters are updated via optimizer, the version numbers change,
    signaling that cached values need to be recomputed.
    
    Args:
        fnp_config: Configuration dictionary for fNP manager
    """
    
    def __init__(self, fnp_config: Dict):
        # Initialize parent (creates fnp_manager)
        super().__init__(fnp_config)
        
        # Re-enable gradients for all parameters (parent froze them)
        for param in self.fnp_manager.parameters():
            param.requires_grad = True
    
    def version(self) -> List[int]:
        """Get current version numbers of all parameters.
        
        Returns:
            List of version integers, one per parameter
            
        Note:
            When parameters are updated by optimizer, these versions change.
            The parent model uses this to detect when to invalidate caches.
        """
        versions = []
        for name, param in self.fnp_manager.named_parameters():
            # PyTorch tracks parameter updates via param._version
            if hasattr(param, '_version'):
                versions.append(param._version)
            else:
                # Fallback: use data pointer as version indicator
                versions.append(id(param.data))
        return versions
