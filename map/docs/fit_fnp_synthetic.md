# Understanding `fit_fnp_synthetic.py`: PyTorch Autograd Validation for fNP Parameters

## Overview

The `fit_fnp_synthetic.py` script is a **gradient validation tool** that performs a synthetic fit to verify that the SIDIS cross-section computation correctly propagates gradients through the non-perturbative function (fNP) parameters using PyTorch autograd. This is essential for ensuring that parameter optimization works correctly in actual fits to experimental data.

## What the Script Does

### Primary Purpose

The script performs a **synthetic fit sanity check** by:

1. Loading a pre-configured fNP model with initial parameters
2. Computing SIDIS cross sections for a small set of kinematic points using the **full TMD machinery** (APFEL++ + PyTorch)
3. Comparing against synthetic "target" cross sections (currently set to simple constants)
4. Using PyTorch's automatic differentiation to compute gradients
5. Performing gradient descent to update fNP parameters
6. Verifying that parameters change in response to the loss function

### Dual Backend Architecture

The script can operate in two modes:

- **Full backend**: Uses `map.sidis_crossect_torch` with complete SIDIS computation (requires LHAPDF/APFEL++, Python 3.10)
- **Toy backend**: Fallback mode with simplified mock computation if dependencies are missing

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

### Modern Differentiable Approach

The script now performs the Fourier-Bessel transform using PyTorch operations:

```python
# 1. Pre-compute APFEL luminosity on fixed b-grid (no gradients needed)
L_b = self._precompute_luminosity_constants(x, z, Q, Yp)

# 2. Evaluate fNP as PyTorch tensors (gradients preserved)
fnp_pdf = comp.compute_fnp_pytorch(x.expand_as(b), b, pdf_flavor)
fnp_ff = comp.compute_fnp_pytorch(z.expand_as(b), b, ff_flavor)

# 3. Bessel function evaluation (device-aware)
J0 = comp._bessel_j0_torch(qT * b)

# 4. Build integrand preserving gradients
integrand = b * J0 * fnp_pdf * fnp_ff * L_b

# 5. Differentiable integration
xs = torch.trapz(integrand, b)
```

### Key Technical Details

#### Bessel Function Implementation

- **Preferred**: `torch.special.bessel_j0()` when available
- **MPS fallback**: Power series expansion for Apple Metal GPU compatibility
- **CPU/CUDA**: Native PyTorch implementation

#### Precision Strategy

- **Integration dtype**: `float64` (except MPS which uses `float32`)
- **Model dtype**: `float32` for memory efficiency
- **Type conversion**: Careful casting to maintain gradients

#### Device Compatibility

- **CUDA**: Full functionality
- **MPS (Apple Metal)**: Custom workarounds for unsupported operations
- **CPU**: Fallback for maximum compatibility

### Comparison: Ogata vs PyTorch Integration

| Aspect | Ogata Quadrature | PyTorch Integration |
|--------|------------------|-------------------|
| **Accuracy** | Excellent (adaptive) | Good (fixed grid) |
| **Speed** | Fast | Moderate |
| **Autograd** | ❌ Not differentiable | ✅ Fully differentiable |
| **GPU Support** | CPU only | Full GPU support |
| **Precision** | Double by default | Configurable |

### Integration Validation

The script validates the PyTorch approach by:

1. Comparing results against known values when possible
2. Monitoring gradient magnitudes to ensure they're reasonable
3. Verifying parameter updates occur in expected directions
4. Testing numerical stability across different kinematic regions

## Synthetic Fit Workflow

### Step-by-Step Process

1. **Setup**: Load configuration files and initialize SIDIS computation object
2. **Kinematics**: Load kinematic points (x, Q², z, PhT) from YAML
3. **Target Generation**: Create synthetic target cross sections
4. **Forward Pass**: Compute theoretical cross sections using full SIDIS machinery
5. **Loss Calculation**: Compare theory vs synthetic targets (MSE loss)
6. **Backward Pass**: Compute gradients w.r.t. fNP parameters via autograd
7. **Parameter Update**: Apply gradient descent step
8. **Validation**: Check parameter changes and convergence metrics

### Example Output

```bash
Epoch    1/1  chi2 = 3.2680e-03
Param L2 diff: 1.2960e+00 (rel 1.2223e+00)
WARNING: Parameters did not fully converge. Consider more epochs, LR tuning, or point count.
```

This output confirms:

- ✅ Forward computation completed successfully
- ✅ Gradients computed without errors
- ✅ Parameters updated (L2 difference > 0)
- ⚠️ More epochs needed for full convergence (expected for quick test)

## Why This Validation Matters

### Scientific Importance

1. **Fit Reliability**: Ensures that actual experimental fits will converge properly
2. **Gradient Accuracy**: Validates that computed gradients reflect true sensitivities
3. **Numerical Stability**: Tests the computation across realistic kinematic ranges
4. **Code Correctness**: Catches integration bugs that could affect physics results

### Technical Benefits

1. **Debug Tool**: Identifies autograd breaks in complex physics computations
2. **Performance Benchmark**: Measures computational overhead of differentiable approach
3. **Device Testing**: Validates GPU compatibility across different hardware
4. **Precision Analysis**: Guides choice of numerical precision for stability vs speed

## Future Enhancements

### Planned Improvements

1. **Hybrid Integration**: Option to use Ogata for final results, PyTorch for gradients
2. **Adaptive Grids**: Dynamic b-grid refinement based on integrand behavior
3. **Multi-Point Batching**: Vectorized computation across kinematic points
4. **Advanced Optimizers**: Integration with Adam, L-BFGS for better convergence

### Physics Extensions

1. **Multi-Hadron Fits**: Extension to multiple hadron species simultaneously
2. **Nuclear Targets**: Validation with realistic nuclear TMD modifications
3. **Systematic Uncertainties**: Gradient-based uncertainty propagation
4. **Scale Variations**: Differentiable scale variation for theoretical uncertainties

The `fit_fnp_synthetic.py` script thus serves as a crucial validation tool that bridges the gap between theoretical TMD physics and practical machine learning optimization, ensuring that our parameter extraction methods are both scientifically sound and computationally robust.
