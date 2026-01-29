"""
Generate OPE grids for TMD PDFs and FFs at Q0 scale with proper evolution

This script generates OPE-based TMD grids for use in SIDIS cross section calculations.
The grids are computed at a reference scale Q0 = mc = 1.28 GeV with:
- NLO OPE matching (O(alphaS))
- NNLL TMD evolution (gamma_K at O(alphaS^2), gamma_F at O(alphaS))
- JAM collinear PDFs/FFs as input
- Nf = 4 flavor scheme

For detailed documentation, see:
- README_OPE_grids.md: Full physics description and usage
- ../config.yaml: All configuration parameters

Output: 16 grid files (8 PDF + 8 FF flavors) in grids/grids/ directory
"""

import os
import sys
import time
import copy
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import interpolate
from scipy.integrate import fixed_quad, quad 
from scipy.special import gamma, zeta, jv, jn_zeros, kv
from scipy.interpolate import griddata, RegularGridInterpolator, CubicSpline
from tqdm import tqdm

from mpmath import fp

# Torch
import torch
import torch.nn as nn
import torch.optim as optim

# Get absolute paths based on this script's location
SCRIPT_DIR = Path(__file__).resolve().parent  # sidis/ope/
SIDIS_DIR = SCRIPT_DIR.parent                 # sidis/
TMD_DIR = SIDIS_DIR.parent                    # tmd/
GRIDS_DIR = TMD_DIR / 'grids' / 'grids'       # tmd/grids/grids/

# Add project path
sys.path.append(str(TMD_DIR))

# Import from sidis/one_d
from sidis.one_d.qcd_qcf_1d import PDF
from sidis.one_d.qcd_ff_1d import FF_PIP

# Import from sidis
from sidis.model.evolution import PERTURBATIVE_EVOLUTION
from sidis.qcdlib.tmdmodel import MODEL_TORCH
from sidis.qcdlib.mellin import MELLIN
from sidis.qcdlib.alphaS import ALPHAS
import sidis.qcdlib.params as params
from sidis.qcdlib import config_loader as cfg

# Import OPE classes
from sidis.ope.OPE import PDF_OPE, FF_OPE

# Helper function for inline printing
def lprint(msg):
    sys.stdout.write('\r')
    sys.stdout.write(msg)
    sys.stdout.flush()


def load_grid_axes(grid_file):
    """
    Load unique x and bT values from existing grid file.
    
    Args:
        grid_file: path to grid file (e.g., 'tmd/grids/grids/tmdpdf_u_Q_1.txt')
    
    Returns:
        x_grid: 1D array of unique x values
        bT_grid: 1D array of unique bT values
    """
    data = np.loadtxt(grid_file, skiprows=1)
    
    # Grid is organized as: for each x, cycle through all bT values
    # So first 500 rows have same x, different bT
    n_bT = np.sum(data[:, 0] == data[0, 0])  # Count rows with first x value
    n_x = len(data) // n_bT
    
    # Extract unique values
    bT_grid = data[:n_bT, 1]  # First n_bT rows give all bT values
    x_grid = data[::n_bT, 0]  # Every n_bT-th row gives unique x values
    
    print(f"📊 Loaded grid axes from {grid_file}")
    print(f"   ✓ x: {len(x_grid)} points from {x_grid[0]:.6g} to {x_grid[-1]:.6g}")
    print(f"   ✓ bT: {len(bT_grid)} points from {bT_grid[0]:.6g} to {bT_grid[-1]:.6g} GeV^-1")
    
    return x_grid, bT_grid


def main():
    """
    Main function to generate OPE grids according to these steps:
    1. Define x and bT grids
    2. Compute the bstar correctly; this defines also the mub* scale
    3. Evaluate the evolution factor for these scales from mub* to Q0 scale; store in a table
    4. Evaluate the OPE for the given x and bT grids for all flavors in one step (no need to loop over)
    5. Multiply the OPE by the evolution factor to get the OPE at Q0 scale
    6. Save the OPE grids for each flavor at Q0 scale in a text file according to tmd/grids/tmdpdf_u_Q_1.txt format
    """
    print("=" * 60)
    print("🚀 OPE Grid Generation")
    print("=" * 60)

    Q0 = torch.sqrt(torch.tensor(cfg.Q20))
    
    # 1. Define x and bT grids (load from existing grid to match format)
    # reference_grid = '/Users/barry/work/QuantOm/workspace/tmd/grids/grids/tmdpdf_u_Q_1.txt'
    # x_grid, bT_grid = load_grid_axes(reference_grid)
    # print('xs axes: ', x_grid)
    # print('bT axes: ', bT_grid)

    # do from my own choice
    #x_grid = 10**torch.linspace(-3, 0, 500, dtype=torch.float64)  # 500 points from 0.001 to 1
    x_grid = 10**torch.linspace(torch.log10(torch.tensor(5e-5)),torch.log10(torch.tensor(0.0999)),300,dtype=torch.float64)  # 300 points from 5e-5 to 0.0999
    x_grid = torch.cat((x_grid, torch.linspace(0.1,1,200,dtype=torch.float64)))  # 200 points from 0.1 to 1
    bT_grid = 10**torch.linspace(-3, np.log10(20), 500, dtype=torch.float64)  # 500 points from 0.001 to 20 GeV^-1
    z_grid = torch.linspace(0.2, 0.9, 500, dtype=torch.float64)  # 500 points from 0.2 to 0.9
    # print('xs axes: ', x_grid)
    # print('bT axes: ', bT_grid)
    
    # 2. Compute the bstar correctly; this defines also the mub* scale
    # Use MODEL_TORCH which can handle both torch and numpy
    tmdmodel = MODEL_TORCH()
    bstar = tmdmodel.get_bstar(bT_grid)
    mubstar = tmdmodel.get_mub(bT_grid)
    # print('bstar: ', bstar)
    
    # 3. Evaluate the evolution factor for these scales from mub* to Q0 scale; store in a table
    print('tmd_resummation_order: ', cfg.tmd_resummation_order)
    evo = PERTURBATIVE_EVOLUTION(order=cfg.tmd_resummation_order)

    # -- take the rapidity evolution factor from mub* to Q0 scale
    rap_evo = torch.exp(evo.get_Ktilde(bstar, mubstar) * torch.log(Q0/mubstar))

    # -- build the pieces of the evolution factor for the RGE evolution
    """
    In going from mub* to Q0 scale, we have the following pieces of the RGE evolution factor:
    exp(\int_{mub*}^{Q0} d\mu^\prime [\gamma_F(\alpha_S(mu^\prime);1) + \gamma_K(\alpha_S(mu^\prime)) * log(\sqrt{\frac{\mu^\prime^2}{Q^2}})])).

    This amounts to the following pieces from the evolution code (minus because of the integration from mub* to Q0 instead of Q0 to mub*):
    exp(-K_gamma(\alpha_S(mub*),\alpha_S(Q0)) - K_Gamma(\alpha_S(Q0),\alpha_S(mub*)))
    """
    alphaS_mub = evo.alphaS.get_alphaS(mubstar**2)
    alphaS_Q0 = evo.alphaS.get_alphaS(Q0**2)
    Nf0, Nf = 4, 4  #--this is because we are taking Nf=4 everywhere. We need to think about how to handle this later.
    eta_Gamma, K_gamma, K_Gamma = evo.compute_evolution_components(alphaS_mub, alphaS_Q0, Nf0, Nf)
    RGE_factor = torch.exp(-K_gamma - K_Gamma).real

    # print('evolution factor: ', evolution_factor)
    evolution_factor = rap_evo * RGE_factor
    #print('evolution factor: ', evolution_factor)



    # 4. Evaluate the OPE for the given x and bT grids for all flavors in one step (no need to loop over)
    mellin = MELLIN(extended=True)
    pdf = PDF(mellin, evo.alphaS)
    ope_pdf_vals = {}
    ope_pdf = PDF_OPE(pdf, evo.alphaS, tmdmodel)
    flav_idx = {'u':1, 'd':2, 's':3, 'c':4, 'b':5, 'bb':6, 'cb':7, 'sb':8, 'db':9, 'ub':10}  #--while b and bb are technically here, we don't want them since we are assuming Nf=4 everywhere.

    for flavor in ['u', 'd', 's', 'c', 'cb', 'sb', 'db', 'ub']:
        ope_pdf_vals[flavor] = np.zeros((len(x_grid), len(bT_grid)))
    for i in range(len(x_grid)):
        if x_grid[i] == 1: continue
        lprint('progress:%i/%i'%(i, len(x_grid)))
        for j in range(len(bT_grid)):
            opes = ope_pdf.get_OPE_TMDPDF(x_grid[i].item(), bT_grid[j].item())
            for flavor in ['u', 'd', 's', 'c', 'cb', 'sb', 'db', 'ub']:
                if opes[flav_idx[flavor]] < 0: 
                    print('negative OPE value: ', ope_pdf_vals[flavor][i,j],'for flavor: ', flavor, 'at x: ', x_grid[i].item(), 'and bT: ', bT_grid[j].item())
                    ope_pdf_vals[flavor][i,j] = 0
                else:
                    ope_pdf_vals[flavor][i,j] = opes[flav_idx[flavor]]


    # # do for FFs
    ff = FF_PIP(mellin, evo.alphaS)
    ope_ff_vals = {}
    ope_ff = FF_OPE(ff, evo.alphaS, tmdmodel)
    flav_idx = {'u':1, 'd':2, 's':3, 'c':4, 'b':5, 'bb':6, 'cb':7, 'sb':8, 'db':9, 'ub':10}  #--while b and bb are technically here, we don't want them since we are assuming Nf=4 everywhere.

    for flavor in ['u', 'd', 's', 'c', 'cb', 'sb', 'db', 'ub']:
        ope_ff_vals[flavor] = np.zeros((len(z_grid), len(bT_grid)))
    for i in range(len(z_grid)):
        if z_grid[i] == 1: continue
        lprint('progress:%i/%i'%(i, len(z_grid)))
        for j in range(len(bT_grid)):
            opes = ope_ff.get_OPE_TMDFF(z_grid[i].item(), bT_grid[j].item())
            for flavor in ['u', 'd', 's', 'c', 'cb', 'sb', 'db', 'ub']:
                if opes[flav_idx[flavor]] < 0: 
                    print('negative OPE value: ', ope_ff_vals[flavor][i,j],'for flavor: ', flavor, 'at z: ', z_grid[i].item(), 'and bT: ', bT_grid[j].item())
                    ope_ff_vals[flavor][i,j] = 0
                else:
                    ope_ff_vals[flavor][i,j] = opes[flav_idx[flavor]]


    # 5. Multiply the OPE by the evolution factor to get the OPE at Q0 scale
    # evolution_factor has shape (len(bT_grid),)
    # ope_pdf_vals[flavor] has shape (len(x_grid), len(bT_grid))
    # We need to broadcast: ope_pdf_vals[flavor] * evolution_factor[None, :]
    # This multiplies each row (constant x) by the evolution_factor array
    print("\n🔄 Applying evolution factor...")
    ope_pdf_vals_Q0 = {}
    for flavor in ['u', 'd', 's', 'c', 'cb', 'sb', 'db', 'ub']:
        # Convert evolution_factor to numpy for multiplication
        evo_factor_np = evolution_factor.detach().cpu().numpy()
        # Broadcast: (n_x, n_bT) * (1, n_bT) -> (n_x, n_bT)
        ope_pdf_vals_Q0[flavor] = ope_pdf_vals[flavor] * evo_factor_np[None, :]
    
    ope_ff_vals_Q0 = {}
    for flavor in ['u', 'd', 's', 'c', 'cb', 'sb', 'db', 'ub']:
        evo_factor_np = evolution_factor.detach().cpu().numpy()
        # Broadcast: (n_z, n_bT) * (1, n_bT) -> (n_z, n_bT)
        ope_ff_vals_Q0[flavor] = ope_ff_vals[flavor] * evo_factor_np[None, :]
    
    # 6. Save the OPE grids for each flavor at Q0 scale in a text file 
    # Format: x(or z) <tab> bT <tab> TMD (one row per x-bT combination)
    print("\n💾 Saving grids...")
    
    # Convert grids to numpy for saving
    x_grid_np = x_grid.detach().cpu().numpy()
    bT_grid_np = bT_grid.detach().cpu().numpy()
    z_grid_np = z_grid.detach().cpu().numpy()
    
    # Create output directory if it doesn't exist
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save PDF grids
    for flavor in ['u', 'd', 's', 'c', 'cb', 'sb', 'db', 'ub']:
        filename = GRIDS_DIR / f'tmdpdf_{flavor}_Q_{Q0.item():.2f}.txt'
        with open(filename, 'w') as f:
            # Write header
            f.write('x\tbT\tTMD\n')
            # Write data: for each x, cycle through all bT values
            for i in range(len(x_grid_np)):
                for j in range(len(bT_grid_np)):
                    f.write(f'{x_grid_np[i]:.6g}\t{bT_grid_np[j]:.6g}\t{ope_pdf_vals_Q0[flavor][i,j]:.6g}\n')
        print(f'   ✓ Saved {filename}')
    
    # Save FF grids
    for flavor in ['u', 'd', 's', 'c', 'cb', 'sb', 'db', 'ub']:
        filename = GRIDS_DIR / f'tmdff_{flavor}_Q_{Q0.item():.2f}.txt'
        with open(filename, 'w') as f:
            # Write header
            f.write('z\tbT\tTMD\n')
            # Write data: for each z, cycle through all bT values
            for i in range(len(z_grid_np)):
                for j in range(len(bT_grid_np)):
                    f.write(f'{z_grid_np[i]:.6g}\t{bT_grid_np[j]:.6g}\t{ope_ff_vals_Q0[flavor][i,j]:.6g}\n')
        print(f'   ✓ Saved {filename}')
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()

