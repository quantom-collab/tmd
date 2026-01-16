"""
TMD Builder: Constructs TMDs from OPE × evolution × fNP

This module is analogous to idis.py in the DIS code, but for TMD construction.
It provides both vectorized (all flavors at once) and single-flavor interfaces.

Key formula: TMD(xi, Q², bT) = OPE(xi, bT; Q₀) × evolution(bT, Q₀→Q) × fNP(xi, bT, Q)

Author: Refactored from original TrainableModel
"""

import torch
from typing import Dict, List


class TMDBuilder(torch.nn.Module):
    """Builds TMDs in bT-space from component parts.
    
    This class combines three ingredients to construct TMDs:
    1. OPE: Operator Product Expansion (interpolated from pre-computed grids)
    2. Evolution: Perturbative evolution from Q₀ to Q
    3. Non-perturbative evolution: Non-perturbative evolution factor (CS kernel)
    4. fNP: Non-perturbative function
    
    Key Methods:
        get_tmd_bT_all_flavors(): Efficient batch computation for all flavors
        get_tmd_bT(): Convenience method for single flavor (analysis/plotting)
    
    Caching Strategy:
        - OPE interpolators: Already loaded as grids (no caching needed)
        - Evolution factors: NOT cached (see note below)
        - Non-perturbative evolution factor: NOT cached (see note below)
        - TMDs: NOT cached (depend on xi which varies per event)
        
    Note on Evolution Caching:
        Evolution caching by shape was buggy - different batches with same
        shape but different Q2 VALUES would get wrong cached results.
        Proper caching would require caching by Q2 values, but:
        - Each batch typically has different Q2 values
        - Cache would rarely hit in practice
        - Evolution is fast enough without caching
        Cache can be re-added later if profiling shows it's needed.
        
    Args:
        ope_dict: Nested dict of OPE interpolators [type][hadron][flav]
        evo: PERTURBATIVE_EVOLUTION instance
        fnp: fNP manager (TruefNP or TrainablefNP); this includes the non-perturbative evolution factor (CS kernel)
        Q20: Initial scale squared
        flavs: List of flavor strings ['u', 'd', 's', 'c', 'cb', 'sb', 'db', 'ub']
    """
    
    def __init__(self, ope_dict, evo, fnp, Q20, flavs: List[str]):
        super().__init__()
        
        self.ope    = ope_dict  # Pre-loaded OPE grids
        self.evo    = evo       # PERTURBATIVE_EVOLUTION instance
        self.fnp    = fnp       # fNP manager (truth or trainable); this includes the non-perturbative evolution factor (CS kernel)

        self.Q20   = Q20     # Initial scale squared
        self.flavs = flavs   # List of flavor strings
        
    def get_tmd_bT_all_flavors(
        self, 
        xi: torch.Tensor, 
        Q2: torch.Tensor, 
        bT: torch.Tensor, 
        type: str, 
        hadron: str
    ) -> Dict[str, torch.Tensor]:
        """Compute TMDs for all flavors at once (efficient for structure functions).
        
        This is the **primary method** for structure function calculations.
        It computes fNP once for all flavors, maximizing efficiency.
        
        Args:
            xi: x or z (depending on type), shape (n_events,)
            Q2: Hard scale squared, shape (n_events,)
            bT: Impact parameter, shape (n_events, n_bT)
            type: "pdf", "ff", or "Sivers"
            hadron: "p", "n", "pi_plus", "pi_minus", etc.
            
        Returns:
            Dictionary mapping flavor names to TMD tensors
            Each TMD has shape (n_events, n_bT)
            
        Formula:
            TMD[flav] = OPE[flav](xi, bT) × fNP[flav](xi, bT, Q) × evolution(bT, Q₀→Q) × non_perturbative_evolution(bT, Q)
        """
        # Compute evolution factor (no caching - see class docstring)
        evolution = self.evo(bT, self.Q20, Q2)
        non_perturbative_evolution = self.fnp.forward_evolution(bT, Q2**0.5)

        # Compute fNP once for all flavors (this is the key optimization!)
        if type == "pdf":
            fNP_dict = self.fnp.forward_pdf(xi, bT)  # Returns dict
        elif type == "ff":
            fNP_dict = self.fnp.forward_ff(xi, bT)  # Returns dict
        elif type == "Sivers":
            # Sivers doesn't have flavor dependence yet
            fNP_sivers = self.fnp.forward_sivers(xi, bT)
            # Create dict with same value for all flavors
            fNP_dict = {flav: fNP_sivers for flav in self.flavs}
        else:
            raise ValueError(f"Unknown type: {type}")
        
        # Compute TMD for each flavor: OPE × fNP × evolution × non_perturbative_evolution
        tmd_dict = {}
        for flav in self.flavs:
            # Get OPE value for this flavor
            ope_tmd = self.ope[type][hadron][flav](xi, bT)
            
            # Handle flavor name conversion for fNP dict lookup
            # Convert 'ub', 'db', etc. to 'ubar', 'dbar', etc.
            if 'b' in flav and len(flav) > 1:
                npflav = flav[0] + 'bar'
            else:
                npflav = flav
            
            # Get fNP value for this flavor (use converted name)
            if type == "Sivers":
                # Sivers uses the same value for all flavors
                fNP = fNP_dict[flav]
            else:
                fNP = fNP_dict[npflav]
            
            # Compute TMD: OPE × fNP × evolution × non-perturbative evolution
            tmd_dict[flav] = ope_tmd * fNP * evolution * non_perturbative_evolution
            
        return tmd_dict
        
    def get_tmd_bT(
        self, 
        xi: torch.Tensor, 
        Q2: torch.Tensor, 
        bT: torch.Tensor, 
        type: str, 
        hadron: str, 
        flav: str
    ) -> torch.Tensor:
        """Get TMD for a single flavor (convenience for analysis/plotting).
        
        This method is useful for:
        - Plotting TMDs as function of bT, xi, or Q
        - Extracting TMD values for specific flavor
        - Analysis workflows that focus on one flavor at a time
        
        Args:
            xi: x or z (depending on type), shape (n_events,)
            Q2: Hard scale squared, shape (n_events,)
            bT: Impact parameter, shape (n_events, n_bT)
            type: "pdf", "ff", or "Sivers"
            hadron: "p", "n", "pi_plus", "pi_minus", etc.
            flav: Specific flavor to extract: 'u', 'd', 's', etc.
            
        Returns:
            TMD tensor for specified flavor, shape (n_events, n_bT)
            
        Note:
            This method calls get_tmd_bT_all_flavors() internally,
            so if you need multiple flavors, call that method directly
            for better efficiency.
        """
        tmd_dict = self.get_tmd_bT_all_flavors(xi, Q2, bT, type, hadron)
        return tmd_dict[flav]

