# PyTorch autodiff compatibility of `map/modules/sidis.py`

The SIDIS cross-section computation now supports **dual integration modes**:

1. **Ogata quadrature**: Highly accurate, non-differentiable (for production results)
2. **PyTorch trapezoidal**: Differentiable, moderate accuracy (for parameter fitting)

You can choose the integration method via the `--use-ogata` command-line flag or the `use_ogata` parameter in the API.

---

## Current Implementation Status

### ✅ **Fully Differentiable Path (Default)**

The main computation path is **end-to-end differentiable** with PyTorch Autograd:

- fNP parameters → fNP(x,b) → SIDIS σ(qT) → loss → gradients → parameter updates
- Uses PyTorch trapezoidal integration on a fixed b-grid
- Supports GPU acceleration (CUDA/MPS)
- All kinematic operations preserve gradients

### ✅ **High-Accuracy Path (Optional)**

An alternative **Ogata quadrature path** provides maximum numerical accuracy:

- Uses adaptive Ogata-Hankel integration for the b-space transform
- Non-differentiable but extremely accurate for oscillatory integrals

❌ **AT THE MOMENT OGATA DOES NOT WORK - always gives zero**

---

## Integration Method Comparison

| Aspect | PyTorch Trapezoidal | Ogata Quadrature |
|:--------|:-------------------:|:------------------:|
| **Accuracy** | Good (fixed grid) | Excellent (adaptive) |
| **Speed** | Fast (GPU) | Moderate (CPU) |
| **Autograd** | ✅ Fully differentiable | ❌ Not fully differentiable |
| **Use Case** | Parameter fitting | Production results |
| **Device Support** | CUDA/MPS/CPU | CPU only |
| **Grid** | Configurable via YAML | Adaptive algorithm |

---

## b-Grid Configuration

The PyTorch integration path uses a **configurable b-grid** defined in `config.yaml`:

```yaml
bgrid:
  b_min: 1.0e-2    # Minimum b value (GeV^-1)
  b_max: 2.0       # Maximum b value (GeV^-1)  
  Nb: 256          # Number of grid points
```

**Grid Properties:**

- **Logarithmic spacing**: Better samples the physics scales
- **Device adaptive**: Automatically moves to GPU if available
- **Precision control**: Uses float64 where supported (MPS falls back to float32 because Apple GPU does not have support for float64)

**Tuning Guidelines:**

- `b_min`: Should cover perturbative region (~1/Q)
- `b_max`: Should include non-perturbative physics (~1 GeV⁻¹)
- `Nb`: More points = better accuracy but slower computation

---

## How the Fourier Transform Works

---

## When Autograd is Preserved vs Broken

### ✅ **Differentiable Operations**

- fNP model evaluation: `fNP(x,b)` → gradients flow to fNP parameters
- Tensor arithmetic: `b * J0 * fnp_pdf * fnp_ff * L_b`
- PyTorch integration: `torch.trapz(integrand, b)`
- Loss computation: `MSE(theory, target)`
- Optimization: `loss.backward()` → parameter updates

### ❌ **Non-Differentiable Operations**

- APFEL++ function calls: `TabMatchTMDPDFs.EvaluatexQ()` (returns Python floats)
- Ogata integration: `DEObj.transform()` (C++ callback system)
- Tensor extraction: `tensor.item()` (converts to Python scalar)
- NumPy operations: Mixed tensor/array arithmetic

### 🔧 **Hybrid Approach**

The current implementation uses a **hybrid strategy**:

1. **APFEL results as constants**: Pre-compute TMD PDFs/FFs, Sudakov factors on b-grid
2. **fNP as tensors**: Keep non-perturbative functions in autograd graph
3. **PyTorch integration**: Use differentiable quadrature for the final transform

APFEL-provided TMD PDFs/FFs, Sudakov, hard factors, and the running couplings are used as constants in the Torch graph. Gradients are not needed for fNP-only fits.

---

## Usage Examples

### For Parameter Fitting (Differentiable)

```bash
# Uses PyTorch integration
python3.10 tests/fit_fnp_synthetic.py
```

---

## Device Compatibility Notes

### Apple Metal (MPS)

- **Supported**: Basic tensor operations, custom Bessel series
- **Workarounds**: logspace on CPU then move, float32 for integration
- **Limitations**: No native Bessel functions, limited float64 support

### CUDA

- **Fully supported**: All operations including native Bessel functions
- **Optimal performance**: float64 integration, tensor memory management

### CPU

- **Reference platform**: All features available
- **Fallback mode**: When GPU operations fail
