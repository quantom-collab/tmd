# Understanding `fit_fnp_synthetic.py`: PyTorch Autograd Validation for fNP Parameters

## Table of Contents

- [Understanding `fit_fnp_synthetic.py`: PyTorch Autograd Validation for fNP Parameters](#understanding-fit_fnp_syntheticpy-pytorch-autograd-validation-for-fnp-parameters)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Usage Requirements](#usage-requirements)
    - [Prerequisites](#prerequisites)
    - [Running the Script](#running-the-script)
  - [What the Script Does](#what-the-script-does)
    - [Primary Purpose](#primary-purpose)
  - [Locate right paths from imports](#locate-right-paths-from-imports)
  - [The b-Grid and Its Role](#the-b-grid-and-its-role)
    - [What is the b-Grid?](#what-is-the-b-grid)
    - [Why the b-Grid Was Needed](#why-the-b-grid-was-needed)
      - [Historical Context: Ogata Quadrature](#historical-context-ogata-quadrature)
      - [The Autograd Problem](#the-autograd-problem)
      - [The b-Grid Solution](#the-b-grid-solution)
    - [b-Grid Configuration](#b-grid-configuration)
  - [How the Fourier Transform is Performed](#how-the-fourier-transform-is-performed)
  - [Synthetic Fit Workflow](#synthetic-fit-workflow)
    - [Step-by-Step Process](#step-by-step-process)

## Overview

The `fit_fnp_synthetic.py` script is a **gradient validation tool** that performs a synthetic fit to verify that the SIDIS cross-section computation correctly propagates gradients through the non-perturbative function (fNP) parameters using PyTorch autograd. This is essential for ensuring that parameter optimization works correctly in actual fits to experimental data.

## Usage Requirements

This script requires Python 3.10 and the full TMD machinery including LHAPDF and APFEL++ packages.

### Prerequisites

- **Python 3.10**: Strict requirement for LHAPDF/APFEL++ compatibility
- **LHAPDF**: Installed and configured with PDF/FF data
- **APFEL++**: TMD evolution library with Python bindings
- **PyTorch**: Recent version with autograd support
- **YAML configuration files**: Properly configured for your setup

### Running the Script

```bash
# Ensure you're using Python 3.10
python3.10 fit_fnp_synthetic.py [config.yaml] [kinematics.yaml] [fNPconfig.yaml]
```

The script will immediately check the Python version and exit with a clear error if not using 3.10.

## What the Script Does

### Primary Purpose

The script performs a **synthetic fit sanity check** by:

1. **Verifying Python 3.10**: Ensures the correct Python version for LHAPDF/APFEL++ compatibility
2. Loading a pre-configured fNP model with initial parameters
3. Computing SIDIS cross sections for a small set of kinematic points using the **full TMD machinery** (APFEL++ + PyTorch)
4. Comparing against synthetic "target" cross sections (currently set to simple constants)
5. Using PyTorch's automatic differentiation to compute gradients
6. Performing gradient descent to update fNP parameters
7. Verifying that parameters change in response to the loss function

## Locate right paths from imports

What the code is doing:

```python
# Initialize repository paths and set up import system
# First import the utilities module using a fallback method
try:
    # Try direct import if we're already in the right place
    from modules.utilities import ensure_repo_on_syspath
except ImportError:
    # Fallback: manually add map to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_dir = os.path.dirname(script_dir)
    if map_dir not in sys.path:
        sys.path.insert(0, map_dir)
    from modules.utilities import ensure_repo_on_syspath
```

It first tries a **normal absolute import**: from `modules.utilities import ensure_repo_on_syspath`.

This will only work if Python can find a top-level package named `modules` on `sys.path` (i.e., a folder `modules/` with an `__init__.py`, or a namespace package).

If that fails with `ImportError`, it assumes the current file is inside `map/<something>/this_file.py` (e.g., `map/tests/...` or `map/scripts/...`), so it:

- Computes the directory of the current file (`script_dir`)
- Goes one level up to get `map_dir`
- Temporarily adds `map_dir` to the front of `sys.path`

Retries the same import, which now resolves to `map/modules/utilities.py` (because adding `map_dir` makes `modules` importable as a top-level package).

- `os.path.abspath(__file__)` — absolute path of the current file.
- `os.path.dirname(path)` — the directory containing path.
- `script_dir = dirname(abspath(__file__))` → **directory of this file**.
- `map_dir = dirname(script_dir)` → **the file’s parent directory** (expected to be map/).
- `sys.path` — the list of directories Python searches for imports.
- `sys.path.insert(0, map_dir)` pushes `map_dir` to the front, making it highest priority.

---

## The b-Grid and Its Role

### What is the b-Grid?

The **b-grid** is a discretized set of impact parameter values `b` (in GeV⁻¹) used for numerical integration in the transverse momentum space. In TMD factorization, the SIDIS cross section involves a Fourier transform:

```md
σ(qT) = ∫ db b J₀(qT·b) F(b)
```

where:

- `b` is the impact parameter (conjugate to transverse momentum qT)
- `J₀` is the zeroth-order Bessel function
- `F(b)` is the integrand containing TMD PDFs, FFs, Sudakov factors, and fNP functions

### Why the b-Grid Was Needed

#### Historical Context: Ogata Quadrature

Previously, the code used **Ogata quadrature** exclusively for the b-space integration. Ogata quadrature is:

- **Extremely accurate** for oscillatory integrals like the Fourier-Bessel transform
- **Non-differentiable** because it uses a callback-based C++ implementation that PyTorch autograd cannot trace through
- **Perfect for physics accuracy** but incompatible with gradient-based optimization

#### The Autograd Problem

When performing parameter fits with PyTorch, we need **end-to-end differentiability**:

```python
parameters → fNP(x,b) → SIDIS σ(qT) → loss → gradients → parameter updates
```

The Ogata integration broke this chain because:

1. It calls a Python function (`b_integrand`) that converts tensors to floats
2. The integration happens in C++ code that PyTorch cannot differentiate through
3. Results are returned as Python floats, severing the gradient connection

#### The b-Grid Solution

To enable autograd while maintaining numerical accuracy, we introduced a **dual approach**:

1. **Fixed b-grid**: A predefined set of logarithmically-spaced b values
2. **PyTorch trapezoidal integration**: Using `torch.trapz()` which is fully differentiable
3. **Pre-computed luminosity**: APFEL-driven factors computed once as constants on the b-grid
4. **Tensor-based integrand**: fNP evaluations kept as tensors throughout

### b-Grid Configuration

The grid is configurable via the `bgrid` section in `config.yaml`:

```yaml
bgrid:
  b_min: 1.0e-2    # Minimum b value (GeV⁻¹)
  b_max: 2.0       # Maximum b value (GeV⁻¹)  
  Nb: 256          # Number of grid points
```

Default values: `[1e-2, 2]` GeV⁻¹ with 256 logarithmically-spaced points.

## How the Fourier Transform is Performed

## Synthetic Fit Workflow

### Step-by-Step Process

1. **Python Version Check**: Verify that Python 3.10 is being used
2. **Dependency Validation**: Ensure LHAPDF and APFEL++ are available
3. **Setup**: Load configuration files and initialize SIDIS computation object
4. **Kinematics**: Load kinematic points (x, Q², z, PhT) from YAML
5. **Target Generation**: Create synthetic target cross sections using initial parameters
6. **Parameter Randomization**: Perturb fNP parameters to create a fitting challenge
7. **Forward Pass**: Compute theoretical cross sections using full SIDIS machinery
8. **Loss Calculation**: Compare theory vs synthetic targets (chi-squared loss)
9. **Backward Pass**: Compute gradients w.r.t. fNP parameters via autograd
10. **Parameter Update**: Apply gradient descent step using Adam optimizer
11. **Validation**: Check parameter changes and convergence metrics

The `fit_fnp_synthetic.py` script now serves as a robust, requirement-enforced validation tool that ensures both the computing environment and the TMD physics implementation are ready for production parameter fitting. By requiring Python 3.10 and the full dependency stack, it guarantees that validation results are directly applicable to real experimental data analysis.
