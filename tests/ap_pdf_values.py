import os
import apfelpy as ap
import numpy as np
import lhapdf as lh


# Output folder
output_folder = "_pdfs"
os.makedirs(output_folder, exist_ok=True)
print(f"Creating folder '{output_folder}' to store the output.\n")

# LHAPDF set
lhapdf_set = "MMHT2014nnlo68cl"

# Initalise LHAPDF set
pdf = lh.mkPDF(lhapdf_set, 0)

# x-space grid
g = ap.Grid([ap.SubGrid(100,1e-5,3), ap.SubGrid(60,1e-1,3), ap.SubGrid(50,6e-1,3), ap.SubGrid(50,8e-1,3)])

# Initial scale. Can be chosen as the charm mass (= np.sqrt(2)) or higher.
# Can also be taken from the LHAPDF set, with mu0 = np.sqrt(pdf.q2Min())
mu0 = np.sqrt(pdf.q2Min)

# Final scale (Q)
mu = 100

# Vectors of masses and thresholds. The original default choice are:
# Thresholds = [0, 0, 0, np.sqrt(2), 4.5, 175]
# Masses = [0, 0, 0, pdf.quarkThreshold(4), pdf.quarkThreshold(5)]
Thresholds = [0, 0, 0, np.sqrt(2), 200, 200]
Masses     = [0, 0, 0, np.sqrt(2), 200, 200]

# Perturbative order. Make sure it is consistent with the PDF set. 
# Here I choose to set the pertubative order manually to have more control. It is used 
# to compute the apfelxx evolution objects. The cons is that the consistency must
# be checked by hand. The perturbative order can also be obtained from 
# the PDF set with PerturbativeOrder = pdf.orderQCD
PerturbativeOrder = pdf.orderQCD

# Running coupling from the LHAPDF set. Can also be obtained from apfelxx set with:
# a = ap.AlphaQCD(0.35, np.sqrt(2), Thresholds, PerturbativeOrder)
# Alphas = ap.TabulateObject(a, 100, 0.9, 1001, 3)
a = lambda mu: pdf.alphasQ(mu)
Alphas = ap.TabulateObject(
    a,                           # Running coupling function
    100,                         # Number of points in the mu grid
    np.sqrt(pdf.q2Min) * 0.9,    # Minimum mu value (QMin)
    np.sqrt(pdf.q2Max),          # Maximum mu value (QMax)
    3,                           # Interpolation degree
    Thresholds                   # Quark mass thresholds
)

# Initialize QCD evolution objects
# DglapObj = ap.initializers.InitializeDglapObjectsQCD(g, Thresholds)
DglapObj = ap.initializers.InitializeDglapObjectsQCD(g, Masses, Thresholds)

# Construct the DGLAP objects
EvolvedPDFs = ap.builders.BuildDglap(DglapObj, lambda x, mu: ap.utilities.PhysToQCDEv(pdf.xfxQ(x, mu)), mu0, PerturbativeOrder, Alphas.Evaluate)

# Tabulate PDFs
TabulatedPDFs = ap.TabulateObjectSetD(EvolvedPDFs, 500, 1, 1000, 3)

# Tabulate Operators
# TabulatedOps = ap.TabulateObjectSetO(EvolvedOps, 500, 1, 1000, 3)

# Evaluate PDFs at final scale mu
pdfs = ap.utilities.QCDEvToPhys(EvolvedPDFs.Evaluate(mu).GetObjects())

# x grid: logarithmically spaced
x_values = np.logspace(-5, 0, num=3000)

# Output file
output_file = os.path.join(output_folder, f"{lhapdf_set}_pdfs_at_Q_{mu}.txt")

# Map to get flavor names
flavor_names = {
    -5: 'bbar',
    -4: 'cbar',
    -3: 'sbar',
    -2: 'ubar',
    -1: 'dbar',
     0: 'gluon',
     1: 'd',
     2: 'u',
     3: 's',
     4: 'c',
     5: 'b',
}

# Write to file
with open(output_file, 'w') as f:

    # Write header
    f.write("x")
    for i in range(-5, 6):
        f.write(f" xf_{flavor_names[i]}")
    f.write("\n")

    # Write data
    for x in x_values:
        f.write(f"{x}")
        for i in range(-5, 6):
            value = pdfs[i].Evaluate(x)
            f.write(f" {value}")
        f.write("\n")

print(f"\nPDFs at Q = {mu} GeV have been written to {output_file}")

### --- Initial PDFs at Q = mu0 from the lhapdf set --- ###

# Output file for initial PDFs
output_file_initial = os.path.join(output_folder, f"{lhapdf_set}_at_init_scale_Q_{mu0}.txt")

# Write to file
with open(output_file_initial, 'w') as f:

    # Write header
    f.write("x")
    for i in range(-5, 6):
        f.write(f" xf_{flavor_names[i]}")
    f.write("\n")

    # Write data
    for x in x_values:
        f.write(f"{x}")
        for i in range(-5, 6):
            value = pdf.xfxQ(i, x, mu0)  # Get xf_i(x, mu0) from LHAPDF
            f.write(f" {value}")
        f.write("\n")

print(f"Initial PDFs at Q = {mu0} GeV have been written to {output_file_initial}")

### --- Print the running coupling at different Q values --- ###

# Q values
Q_values = np.linspace(2, 100, num=10)

# Dictionary to store the perturbative order
pto = {0 : "LO", 1 : "NLO", 2 : "NNLO"}

# Output file for initial PDFs
output_file_alphas = os.path.join(output_folder, f"alphas_at_{pto[PerturbativeOrder]}.txt")

# Write to file
with open(output_file_alphas, 'w') as f:

    # Write header
    f.write("Q")
    f.write(f" alpha_s(Q^2)")
    f.write("\n")

    # Write data
    for Q in Q_values:
        f.write(f"{Q}")
        value = Alphas.Evaluate(Q)
        f.write(f" {value}")
        f.write("\n")

# Check: print the running coupling at MZ^2
print(f"AlphaS(MZ^2) = {Alphas.Evaluate(ap.constants.ZMass)}\n")
