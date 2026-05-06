# OPE Grid Generation for TMD PDFs and FFs

This directory contains the code and configuration for generating Operator Product Expansion (OPE) grids for Transverse Momentum Dependent (TMD) parton distribution functions (PDFs) and fragmentation functions (FFs).

## Overview

The OPE grids provide the collinear input for TMD calculations at a reference scale Q₀. These grids are computed by:

1. Evaluating the OPE in Mellin space at the scale μ_b* and ζ_b* = μ_b*²
2. Evolving from μ_b* to Q₀ using TMD evolution equations
3. Inverting to x-space (for PDFs) or z-space (for FFs)
4. Storing on a 2D grid in (x or z) × b_T

## Physics Details

### Collinear Input

**Source**: JAM collaboration collinear PDFs and FFs  
**Reference**: Replica from arXiv:2501.00665  
**Flavor scheme**: N_f = 4 (up, down, strange, charm + antiquarks)

The collinear PDFs and FFs are parametrized as:
```
f(x) = N × x^a × (1-x)^b × (1 + c×x^0.5 + d×x)
D(z) = N × z^a × (1-z)^b × (1 + c×z^0.5 + d×z)
```

Parameters are stored in `pdf_ff_params.txt` and loaded into:
- `one_d/qcf/qcd_qcf_1d.py` (PDF class)
- `one_d/qcf/qcd_ff_1d.py` (FF_PIP class for π⁺)

### Reference Scale

**Q₀ = m_c = 1.28 GeV**

This choice sets the reference scale at the charm quark mass, ensuring we remain in the N_f = 4 region.

### b* Prescription

The b* prescription regulates the perturbative behavior at large b_T:

```
b* = b_T / √(1 + b_T²/b_max²)
μ_b* = C₁/b*
```

where:
- **C₁ = 2 exp(-γ_E) ≈ 1.123** (γ_E = Euler-Mascheroni constant)
- **b_max = C₁/Q₀** ensures μ_b* ≥ Q₀

This prescription ensures:
- At small b_T: b* ≈ b_T (perturbative region)
- At large b_T: b* ≈ b_max (frozen coupling)

### OPE Calculation

The OPE is computed in Mellin space at **O(α_s)** (NLO):

**For PDFs:**
```
f̃_OPE(x, b_T) = ∫ dN/(2πi) x^(-N) [C_q(N) ⊗ f_q(N, μ_b*²) + C_g(N) ⊗ f_g(N, μ_b*²)]
```

where the coefficient functions include:
- **LO**: C_q = 1, C_g = 0
- **NLO**: O(α_s) corrections from matching onto collinear PDFs

**For FFs:**
```
D̃_OPE(z, b_T) = (1/z²) ∫ dN/(2πi) z^(-N) [C_q(N) ⊗ D_q(N, μ_b*²) + C_g(N) ⊗ D_g(N, μ_b*²)]
```

The collinear PDFs/FFs are evolved to μ_b*² using DGLAP evolution.

### TMD Evolution

Evolution from μ_b* to Q₀ is performed at **NNLL accuracy**:

```
f̃(x, b_T; Q₀) = f̃_OPE(x, b_T; μ_b*) × exp[S_pert(Q₀, μ_b*, b*)]
```

The perturbative Sudakov factor consists of two pieces:

**1. Rapidity Evolution:**
```
S_rap = K̃(b*, μ_b*) × ln(Q₀/μ_b*)
```

where K̃ is the Collins-Soper kernel expanded to **O(α_s²)** (NNLO).

**2. Renormalization Group Evolution:**
```
S_RGE = -∫_{μ_b*}^{Q₀} dμ'/μ' [γ_F(α_s(μ')) + γ_K(α_s(μ')) ln(μ'/Q₀)]
```

where:
- **γ_F**: Anomalous dimension of TMD operator, expanded to **O(α_s)** (NLO)
- **γ_K**: Cusp anomalous dimension, expanded to **O(α_s²)** (NNLO)

This combination gives **NNLL** accuracy:
- Leading logs: all orders
- Next-to-leading logs: all orders  
- Next-to-next-to-leading logs: all orders

### Running Coupling

The strong coupling α_s is evolved using the β-function at the order consistent with the TMD evolution:
- **β₀, β₁, β₂** coefficients included
- Matching at flavor thresholds (though we stay in N_f = 4 region)
- Initial value: α_s(M_Z) from global fits

## Grid Specifications

### PDF Grids

**Filename format**: `grids/grids/tmdpdf_{flavor}_Q_{Q0:.2f}.txt`

**Flavors**: u, d, s, c, ub, db, sb, cb

**Grid dimensions**:
- x: 500 points total, using a two-piece grid  
  - 300 points, logarithmically spaced from 5×10⁻⁵ to 0.0999  
  - 200 points, linearly spaced from 0.1 to 1  
- b_T: 500 points, logarithmically spaced from 10⁻³ to 20 GeV⁻¹

**File format**:
```
x       bT      TMD
0.001   0.001   <value>
0.001   0.00126 <value>
...
```

Each file contains 250,000 rows (500 × 500).

### FF Grids

**Filename format**: `grids/grids/tmdff_{flavor}_Q_{Q0:.2f}.txt`

**Flavors**: u, d, s, c, ub, db, sb, cb (for π⁺)

**Grid dimensions**:
- z: 500 points, linearly spaced from 0.2 to 0.9
- b_T: 500 points, logarithmically spaced from 10⁻³ to 20 GeV⁻¹

**File format**:
```
z       bT      TMD
0.2     0.001   <value>
0.2     0.00126 <value>
...
```

Each file contains 250,000 rows (500 × 500).

## Usage in Cross Section Calculations

These OPE grids serve as input for SIDIS (Semi-Inclusive Deep Inelastic Scattering) cross section calculations:

```
dσ/dx dQ² dz dP_T² ∝ Σ_q e_q² f̃_q(x, b_T; Q) D̃_q(z, b_T; Q) × [Fourier transform to P_T]
```

The grids can be:
1. **Interpolated** at arbitrary (x or z, b_T) points
2. **Fourier transformed** to momentum space (P_T) for cross sections
3. **Evolved** to different Q scales if needed (using the same evolution equations)

## Code Structure

- `generate_grids.py`: Main script for grid generation
- `config_ope.yaml`: Configuration file with all physics parameters
- `pdf_ff_params.txt`: Collinear PDF/FF parameters from JAM
- `one_d/qcf/qcd_qcf_1d.py`: PDF class with DGLAP evolution
- `one_d/qcf/qcd_ff_1d.py`: FF class with DGLAP evolution
- `sidis/model/evolution.py`: TMD evolution implementation

## Running the Code

```bash
cd /Users/barry/work/QuantOm/workspace/tmd/ope
python generate_grids.py
```

The script will:
1. Load collinear PDFs and FFs
2. Set up x/z and b_T grids
3. Compute evolution factors for all b_T points
4. Evaluate OPE for all (x or z, b_T) combinations
5. Apply evolution to get grids at Q₀
6. Save 16 grid files (8 PDF + 8 FF flavors)

**Expected runtime**: ~1-3 minutes depending on hardware

## Configuration

All physics parameters are stored in `../config.yaml` (the main SIDIS configuration file). To generate grids with different settings:

1. Edit `../config.yaml` (adjust Q02, perturbative orders, etc.)
2. Run `generate_grids.py`
3. Grids will be saved with Q₀ value in filename

## References

1. **TMD Formalism**: Collins, "Foundations of Perturbative QCD" (2011)
2. **Evolution Equations**: Ebert, JHEP 02 (2022) 136 [arXiv:2110.11360]
3. **OPE Matching**: Echevarria, Idilbi, Scimemi, Phys.Rev.D 96, 054011 (2017) [arXiv:1604.07869]
4. **JAM PDFs/FFs**: arXiv:2501.00665
5. **b* Prescription**: Collins, Soper, Sterman, Nucl.Phys.B 250, 199 (1985)

## Notes

- Negative OPE values are set to zero (unphysical, occurs at boundaries)
- Grid spacing is optimized for interpolation accuracy vs. file size
- All grids use the same b_T grid for consistency in Fourier transforms
- The x=1 and z=1 endpoints are excluded (singular behavior)

## Contact

For questions about these grids or the calculation procedure, please refer to the main TMD project documentation.

