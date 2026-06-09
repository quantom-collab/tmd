"""QCD constants for spin-dependent collinear evolution."""

import numpy as np

# PDG masses (GeV)
M = 0.93891897  # proton mass; for -2 M T_F collinear Sivers observable
mc = 1.28
mb = 4.18
mZ = 91.1876
mc2 = mc**2
mb2 = mb**2
mZ2 = mZ**2
alphaSMZ = 0.118

# Casimir invariants
CF = 4.0 / 3.0
CA = 3.0
NC = CA  # number of colors; default eta for modified Qiu-Sterman evolution
TF = 0.5

# RGE / alpha_s setup (NLO running by default)
alphaS_order = 1
alphaS_mu20 = 1.0

# Nonperturbative Sivers width (GeV^2) for b-space TMD factors (see Spin2.qiu_sterman).
# Not used by collinear x-space DGLAP in Spin2.dglap / Spin2.kernels.
g1T_default = 0.180
