"""
OPE (Operator Product Expansion) calculations for TMD PDFs and FFs

This module contains classes for computing OPE matching coefficients in Mellin space
and inverting to x-space (PDFs) or z-space (FFs).

Classes:
- PDF_OPE: OPE for parton distribution functions
- FF_OPE: OPE for fragmentation functions

Future extensions could include:
- Qiu-Sterman OPE for twist-3 functions
- Other OPE schemes
"""

import numpy as np
import torch
from scipy.special import zeta

from ..qcdlib import params
from ..qcdlib import config_loader as cfg
from ..qcdlib import special


def _to_numpy(x):
    """Convert torch tensor to numpy, handle both scalars and arrays"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _to_torch(x, dtype=torch.float64):
    """Convert numpy/scalar to torch tensor"""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(x, dtype=dtype)


class PDF_OPE:
    """
    OPE for TMD PDFs at the scale mu_b* and zeta_b*
    
    Computes the OPE in Mellin space and inverts to x-space.
    Works with numpy-based Mellin inversion and torch-based evolution.
    """
    
    def __init__(self, pdf, alphaS, tmdmodel):
        """
        Args:
            pdf: PDF object with DGLAP evolution
            alphaS: ALPHAS object for strong coupling
            tmdmodel: MODEL_TORCH object for b* prescription (can handle torch or numpy)
        """
        self.iorder = cfg.tmd_order  # LO: 0, NLO: 1
        
        self.pdf = pdf
        self.mellin = pdf.mellin
        
        self.tmdmodel = tmdmodel
        self.alphaS = alphaS
        
        # QCD parameters
        self.CF = params.CF
        self.CA = params.CA
        self.TR = params.TR
        self.TF = params.TF
        
        # Quark charges (for future use if needed)
        quarkcharges = np.zeros(11)
        quarkcharges[0] = 0      # gluon
        quarkcharges[1] = 2/3    # u
        quarkcharges[2] = -1/3   # d
        quarkcharges[3] = -1/3   # s
        quarkcharges[4] = 2/3    # c
        quarkcharges[5] = -1/3   # b
        quarkcharges[-1] = -2/3  # ubar
        quarkcharges[-2] = 1/3   # dbar
        quarkcharges[-3] = 1/3   # sbar
        quarkcharges[-4] = -2/3  # cbar
        quarkcharges[-5] = 1/3   # bbar
        self.quarkcharges = quarkcharges
        
        # Storage and flavor mapping
        self.storage_ope = {}
        self.fmap = {1:'u', 2:'d', 3:'s', 4:'c', 5:'b',
                    -5:'bb', -4:'cb', -3:'sb', -2:'db', -1:'ub', 0:'g'}
        
        self.setup()

    def reset_storage_ope(self):
        """Clear cached OPE evaluations"""
        self.storage_ope = {}

    def setup(self):
        """Pre-compute Mellin-space quantities for NLO matching"""
        N = self.mellin.N
        self.N = N
        
        psi0N = special.get_psi(0, N)
        
        # NLO quark coefficient function pieces
        # M transform of Pqq = (2/(1-x)_+ - 1 - x)
        self.NLOq1 = -1/N - 1/(N+1) - 2*special.euler - 2*psi0N
        # M transform of (1-x)
        self.NLOq2 = 1/N - 1/(N+1)
        
        # NLO gluon coefficient function pieces
        # M transform of Pgq = 1 - 2x(1-x)
        self.NLOg1 = 1/N - 2/(N+1) + 2/(N+2)
        # M transform of x(1-x)
        self.NLOg2 = 1/(N+1) - 1/(N+2)
     
    def get_OPE_TMDPDF(self, x, bT):
        """
        Compute OPE for TMD PDF at given x and bT
        
        Args:
            x: Bjorken x (float or numpy scalar)
            bT: Transverse distance in GeV^-1 (float or numpy scalar)
        
        Returns:
            numpy array of shape (11,) with OPE values for each flavor
            [g, u, d, s, c, b, bb, cb, sb, db, ub]
        """
        # Convert to numpy if needed (tmdmodel might return torch tensors)
        bT_np = _to_numpy(bT)
        x_np = _to_numpy(x)
        
        mub = self.tmdmodel.get_mub(bT_np)
        mub_np = _to_numpy(mub)
        
        zeta = mub_np**2
        bstar = self.tmdmodel.get_bstar(bT_np)
        bstar_np = _to_numpy(bstar)
        
        key = (float(x_np), float(bT_np), float(zeta))

        if key in self.storage_ope:
            if float(mub_np**2) not in self.pdf.storage:
                self.reset_storage_ope()
        
        if key not in self.storage_ope:
            N = self.mellin.N           

            aS = self.alphaS.get_a(mub_np**2)  # alphaS/(4 pi)
            
            Log = self.tmdmodel.get_Log(bstar_np, mub_np)
            Log_np = _to_numpy(Log)

            self.pdf.evolve(mub_np**2)
            
            moments = self.pdf.storage[mub_np**2]

            # Coefficient functions in Mellin space
            Cq = np.ones(N.size, dtype=np.complex128)
            Cg = np.zeros(N.size, dtype=np.complex128)

            if self.iorder >= 1:
                # NLO corrections
                Cq += aS * self.CF * (
                    -4*Log_np**2 
                    - 4*Log_np*np.log(zeta/mub_np**2)
                    - 4*Log_np*self.NLOq1 
                    + 2*self.NLOq2 
                    - np.pi**2/6
                )
                Cg += aS * self.TF * (-4*Log_np*self.NLOg1 + 4*self.NLOg2)

            # Convolution in Mellin space
            mellin_convolutions = np.zeros((11, N.size), dtype=np.complex128)
            for i in [1, 2, 3, 4, 5]:
                mellin_convolutions[+i] = Cq*moments[self.fmap[+i]] + Cg*moments['g']
                mellin_convolutions[-i] = Cq*moments[self.fmap[-i]] + Cg*moments['g']

            # Invert to x-space
            xspace_convolutions = np.zeros(11)
            for i in [1, 2, 3, 4, 5]:
                xspace_convolutions[+i] = self.mellin.invert(x_np, mellin_convolutions[+i])
                xspace_convolutions[-i] = self.mellin.invert(x_np, mellin_convolutions[-i])
            
            self.storage_ope[key] = xspace_convolutions

        return self.storage_ope[key]


class FF_OPE:
    """
    OPE for TMD FFs at the scale mu_b* and zeta_b*
    
    Computes the OPE in Mellin space and inverts to z-space.
    Works with numpy-based Mellin inversion and torch-based evolution.
    """
    
    def __init__(self, ff, alphaS, tmdmodel):
        """
        Args:
            ff: FF object with DGLAP evolution
            alphaS: ALPHAS object for strong coupling
            tmdmodel: MODEL_TORCH object for b* prescription (can handle torch or numpy)
        """
        self.iorder = cfg.tmd_order  # LO: 0, NLO: 1
        
        self.ff = ff
        self.mellin = ff.mellin
        
        self.tmdmodel = tmdmodel
        self.alphaS = alphaS
        
        # QCD parameters
        self.CF = params.CF
        self.CA = params.CA
        self.TR = params.TR
        self.TF = params.TF
        
        # Quark charges (for future use if needed)
        quarkcharges = np.zeros(11)
        quarkcharges[0] = 0      # gluon
        quarkcharges[1] = 2/3    # u
        quarkcharges[2] = -1/3   # d
        quarkcharges[3] = -1/3   # s
        quarkcharges[4] = 2/3    # c
        quarkcharges[5] = -1/3   # b
        quarkcharges[-1] = -2/3  # ubar
        quarkcharges[-2] = 1/3   # dbar
        quarkcharges[-3] = 1/3   # sbar
        quarkcharges[-4] = -2/3  # cbar
        quarkcharges[-5] = 1/3   # bbar
        self.quarkcharges = quarkcharges
        
        # Storage and flavor mapping
        self.storage_ope = {}
        self.fmap = {1:'u', 2:'d', 3:'s', 4:'c', 5:'b',
                    -5:'bb', -4:'cb', -3:'sb', -2:'db', -1:'ub', 0:'g'}
        
        self.setup()

    def reset_storage_ope(self):
        """Clear cached OPE evaluations"""
        self.storage_ope = {}

    def setup(self):
        """Pre-compute Mellin-space quantities for NLO matching"""
        N = self.mellin.N
        self.N = N
        
        S2Nm1 = zeta(2) - special.get_psi(1, N-1+1)
        S2Np1 = zeta(2) - special.get_psi(1, N+1+1)

        # Quark coefficient function pieces
        # Mellin transform of [z^2 \mathbb{C}_{q->q}(z)] from 1604.07869:
        # CF * (2*(1-z) + 4*(1+z^2)*log(z)/(1-z) - delta(1-z)*pi^2/6)
        # We take out CF and pi^2/6 factors
        self.NLOq = 4*(S2Nm1 - zeta(2) + S2Np1 - zeta(2)) + 2/N - 2/(N+1)

        # Gluon coefficient function pieces
        # Mellin transform of [z^2 \mathbb{C}_{q->g}(z)] from 1604.07869:
        # CF * (2*z + 4*(1+(1-z)^2)/z * log(z))
        # We take out CF factor
        self.NLOg = 2/(N+1) - 8/(N-1)**2 + 8/N**2 - 4/(N+1)**2

    def get_OPE_TMDFF(self, z, bT):
        """
        Compute OPE for TMD FF at given z and bT
        
        Args:
            z: Momentum fraction (float or numpy scalar)
            bT: Transverse distance in GeV^-1 (float or numpy scalar)
        
        Returns:
            numpy array of shape (11,) with OPE values for each flavor
            [g, u, d, s, c, b, bb, cb, sb, db, ub]
        """
        # Convert to numpy if needed (tmdmodel might return torch tensors)
        bT_np = _to_numpy(bT)
        z_np = _to_numpy(z)
        
        mub = self.tmdmodel.get_mub(bT_np)
        mub_np = _to_numpy(mub)
        
        zeta = mub_np**2
        bstar = self.tmdmodel.get_bstar(bT_np)
        bstar_np = _to_numpy(bstar)
        
        key = (float(z_np), float(bT_np), float(zeta))

        if key in self.storage_ope:
            if float(mub_np**2) not in self.ff.storage:
                self.reset_storage_ope()
        
        if key not in self.storage_ope:
            N = self.mellin.N           

            aS = self.alphaS.get_a(mub_np**2)  # alphaS/(4 pi)
            
            Log = self.tmdmodel.get_Log(bstar_np, mub_np)
            Log_np = _to_numpy(Log)

            self.ff.evolve(mub_np**2)
            
            moments = self.ff.storage[mub_np**2]

            # Coefficient functions in Mellin space
            Cq = np.ones(N.size, dtype=np.complex128)
            Cg = np.zeros(N.size, dtype=np.complex128)

            if self.iorder >= 1:
                # NLO corrections
                Cq += aS * self.CF * (self.NLOq - np.pi**2/6)
                Cg += aS * self.CF * self.NLOg

            # Convolution in Mellin space
            mellin_convolutions = np.zeros((11, N.size), dtype=np.complex128)
            for i in [1, 2, 3, 4, 5]:
                mellin_convolutions[+i] = Cq*moments[self.fmap[+i]] + Cg*moments['g']
                mellin_convolutions[-i] = Cq*moments[self.fmap[-i]] + Cg*moments['g']

            # Invert to z-space with 1/z^2 factor
            zspace_convolutions = np.zeros(11)
            for i in [1, 2, 3, 4, 5]:
                zspace_convolutions[+i] = (1/z_np**2) * self.mellin.invert(z_np, mellin_convolutions[+i])
                zspace_convolutions[-i] = (1/z_np**2) * self.mellin.invert(z_np, mellin_convolutions[-i])
                
            self.storage_ope[key] = zspace_convolutions

        return self.storage_ope[key]

