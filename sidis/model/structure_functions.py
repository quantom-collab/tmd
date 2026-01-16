"""
Structure Functions for SIDIS TMD cross sections.

This module provides structure function classes analogous to GridF2, GridFL, GridF3
in the DIS code. Each structure function is computed via Hankel transforms of TMDs.

Key feature: **Vectorized operations** over flavors for 50-80% speedup.

Author: Refactored from original TrainableModel
"""

import torch
from typing import Dict, List


class FUUT(torch.nn.Module):
    """Unpolarized structure function via J0 Hankel transform.
    
    Physics: F_UU^T = sum_q e_q^2 * f_q(x, bT) ⊗ D_q(z, bT)
    
    This structure function contributes to the unpolarized SIDIS cross section.
    It is computed as a convolution (Hankel J0 transform) of TMD PDFs and FFs.
    
    Args:
        tmd_builder: TMDBuilder instance for computing TMDs
        ogata_J0: OGATA instance with nu=0 for J0 Hankel transform
        quark_charges_squared: Dict mapping flavor to e_q^2
        flavs: List of flavor strings in canonical order
    """
    
    def __init__(
        self, 
        tmd_builder, 
        ogata_J0, 
        quark_charges_squared: Dict[str, float], 
        flavs: List[str]
    ):
        super().__init__()
        
        self.tmd = tmd_builder
        self.ogata = ogata_J0
        self.charges2 = quark_charges_squared  # dict[flav: float]
        self.flavs = flavs  # ['u', 'd', 's', 'c', 'cb', 'sb', 'db', 'ub']
        
        # Convert charge dict to tensor for vectorized operations
        # IMPORTANT: Order must match self.flavs exactly!
        charges_tensor = torch.tensor([self.charges2[f] for f in self.flavs])
        # Store as buffer so it moves with model to GPU
        self.register_buffer('charges_vec', charges_tensor)
        
    def forward(
        self, 
        x: torch.Tensor, 
        Q2: torch.Tensor, 
        z: torch.Tensor, 
        qT: torch.Tensor, 
        initial_hadron: str, 
        fragmented_hadron: str
    ) -> torch.Tensor:
        """Compute FUUT structure function.
        
        Args:
            x: Bjorken x, shape (n_events,)
            Q2: Hard scale squared, shape (n_events,)
            z: Fragmentation fraction, shape (n_events,)
            qT: Transverse momentum qT = PhT/z, shape (n_events,)
            initial_hadron: Target hadron ("p", "n", etc.)
            fragmented_hadron: Detected hadron ("pi_plus", "pi_minus", etc.)
            
        Returns:
            FUUT structure function, shape (n_events,)
            
        Formula:
            FUUT = J0_transform[ sum_q e_q^2 * bT * f_q(x, bT) * D_q(z, bT) ]
            
        Note:
            Factor of bT is because Ogata quadrature is for specific form,
            not general Hankel transform.
        """
        # Get bT grid for this qT: shape (n_events, n_bT_points)
        bT = self.ogata.get_bTs(qT)
        
        # Get all TMDs at once (efficient!)
        pdf_dict = self.tmd.get_tmd_bT_all_flavors(x, Q2, bT, "pdf", initial_hadron)
        ff_dict = self.tmd.get_tmd_bT_all_flavors(z, Q2, bT, "ff", fragmented_hadron)
        
        # Stack TMDs into tensors: shape (n_flavs, n_events, n_bT)
        # IMPORTANT: Stack in same order as self.flavs to match charges_vec!
        pdf_stack = torch.stack([pdf_dict[f] for f in self.flavs], dim=0)
        ff_stack = torch.stack([ff_dict[f] for f in self.flavs], dim=0)
        
        # Vectorized computation: charges(n_flavs,1,1) * bT(1,n_events,n_bT) * pdf * ff
        # Shape: (n_flavs, n_events, n_bT)
        charges_expanded = self.charges_vec.view(-1, 1, 1)  # (n_flavs, 1, 1)
        bT_expanded = bT.unsqueeze(0)  # (1, n_events, n_bT)
        
        # Integrand for each flavor
        integrand_per_flavor = charges_expanded * bT_expanded * pdf_stack * ff_stack
        
        # Sum over flavors: (n_events, n_bT)
        integrand = torch.sum(integrand_per_flavor, dim=0)
        
        # Hankel transform J0: (n_events, n_bT) -> (n_events,)
        return self.ogata.eval_ogata_func_var_h(integrand, bT, qT)


class FUT_SinPhihMinusPhis(torch.nn.Module):
    """Sivers structure function via J1 Hankel transform.
    
    Physics: F_UT^{sin(φ_h-φ_S)} from Sivers TMD × unpolarized FF
    
    This structure function contributes to single-spin asymmetries in SIDIS.
    It arises from the Sivers effect (correlation between nucleon spin and 
    quark transverse momentum).
    
    Args:
        tmd_builder: TMDBuilder instance for computing TMDs
        ogata_J1: OGATA instance with nu=1 for J1 Hankel transform
        quark_charges_squared: Dict mapping flavor to e_q^2
        flavs: List of flavor strings in canonical order
    """
    
    def __init__(
        self, 
        tmd_builder, 
        ogata_J1, 
        quark_charges_squared: Dict[str, float], 
        flavs: List[str]
    ):
        super().__init__()
        
        self.tmd = tmd_builder
        self.ogata = ogata_J1
        self.charges2 = quark_charges_squared
        self.flavs = flavs
        
        # Convert charge dict to tensor for vectorized operations
        # IMPORTANT: Order must match self.flavs exactly!
        charges_tensor = torch.tensor([self.charges2[f] for f in self.flavs])
        self.register_buffer('charges_vec', charges_tensor)
        
    def forward(
        self, 
        x: torch.Tensor, 
        Q2: torch.Tensor, 
        z: torch.Tensor, 
        qT: torch.Tensor, 
        initial_hadron: str, 
        fragmented_hadron: str
    ) -> torch.Tensor:
        """Compute Sivers structure function.
        
        Args:
            x: Bjorken x, shape (n_events,)
            Q2: Hard scale squared, shape (n_events,)
            z: Fragmentation fraction, shape (n_events,)
            qT: Transverse momentum qT = PhT/z, shape (n_events,)
            initial_hadron: Target hadron ("p", "n", etc.)
            fragmented_hadron: Detected hadron ("pi_plus", "pi_minus", etc.)
            
        Returns:
            FUT_sin(φ_h-φ_S) structure function, shape (n_events,)
            
        Formula:
            FUT = J1_transform[ sum_q e_q^2 * (bT^2/2) * f1T^⊥(x, bT) * D_q(z, bT) ]
            
        Note:
            Factor of bT^2/2 (not bT) because this is J1 Hankel transform,
            not J0. The extra bT comes from the angular structure of Sivers.
        """
        # Get bT grid for J1 transform
        bT = self.ogata.get_bTs(qT)
        
        # Sivers PDF × unpolarized FF
        sivers_dict = self.tmd.get_tmd_bT_all_flavors(x, Q2, bT, "Sivers", initial_hadron)
        ff_dict = self.tmd.get_tmd_bT_all_flavors(z, Q2, bT, "ff", fragmented_hadron)
        
        # Stack and vectorize
        # IMPORTANT: Stack in same order as self.flavs to match charges_vec!
        sivers_stack = torch.stack([sivers_dict[f] for f in self.flavs], dim=0)
        ff_stack = torch.stack([ff_dict[f] for f in self.flavs], dim=0)
        
        # bT^2/2 factor for J1 transform
        charges_expanded = self.charges_vec.view(-1, 1, 1)
        bT_expanded = bT.unsqueeze(0)
        
        # Integrand with bT^2/2 factor
        integrand_per_flavor = charges_expanded * (bT_expanded**2 / 2) * sivers_stack * ff_stack
        integrand = torch.sum(integrand_per_flavor, dim=0)
        
        # Hankel transform J1: (n_events, n_bT) -> (n_events,)
        return self.ogata.eval_ogata_func_var_h(integrand, bT, qT)

