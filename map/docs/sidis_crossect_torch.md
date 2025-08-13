# PyTorch autodiff compatibility of `map/sidis_crossect_torch.py`

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
- Recommended for final production results and validation

---

## Integration Method Comparison

| Aspect | PyTorch Trapezoidal | Ogata Quadrature |
|--------|-------------------|------------------|
| **Accuracy** | Good (fixed grid) | Excellent (adaptive) |
| **Speed** | Fast (GPU) | Moderate (CPU) |
| **Autograd** | ✅ Fully differentiable | ❌ Not differentiable |
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
- **Precision control**: Uses float64 where supported (MPS falls back to float32)

**Tuning Guidelines:**

- `b_min`: Should cover perturbative region (~1/Q)
- `b_max`: Should include non-perturbative physics (~1 GeV⁻¹)
- `Nb`: More points = better accuracy but slower computation

---

## How the Fourier Transform Works

### PyTorch Integration (Differentiable)

The Fourier-Bessel transform is computed as:

```python
# 1. Pre-compute APFEL luminosity on fixed b-grid (constants, no gradients)
L_b = self._precompute_luminosity_constants(x, z, Q, Yp)

# 2. Evaluate fNP preserving gradients
fnp_pdf = self.compute_fnp_pytorch(x.expand_as(b), b, pdf_flavor)
fnp_ff = self.compute_fnp_pytorch(z.expand_as(b), b, ff_flavor)

# 3. Bessel function (device-compatible)
J0 = self._bessel_j0_torch(qT * b)

# 4. Build integrand
integrand = b * J0 * fnp_pdf * fnp_ff * L_b

# 5. Trapezoidal integration
xs = torch.trapz(integrand, b)
```

**Key Features:**

- **Gradient preservation**: fNP tensors remain connected to autograd graph
- **Pre-computed luminosity**: APFEL factors computed once as constants
- **Device compatibility**: Handles CUDA/MPS/CPU differences
- **Bessel fallbacks**: Series expansion for unsupported devices

### Ogata Integration (High Accuracy)

The traditional method using callback-based integration:

```python
def b_integrand(b_val: float) -> float:
    # All components computed as Python floats
    fnp_pdf = float(self.compute_fnp_pytorch(...).item())
    fnp_ff = float(self.compute_fnp_pytorch(...).item())
    luminosity = compute_tmf_luminosity(b_val)  # APFEL
    return b_val * fnp_pdf * fnp_ff * luminosity

# Ogata quadrature handles Bessel transform internally
result = self.DEObj.transform(b_integrand, qT)
```

**Characteristics:**

- **Adaptive sampling**: Algorithm chooses optimal b-points
- **High precision**: Designed for oscillatory Hankel transforms  
- **Physics validation**: Gold standard for numerical accuracy
- **No gradients**: `.item()` calls sever autograd connections

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

---

## Usage Examples

### For Parameter Fitting (Differentiable)

```bash
python3.10 sidis_crossect_torch.py config.yaml data.yaml fnp_config.yaml results/ output.yaml
# Uses PyTorch integration by default
```

### For Production Results (High Accuracy)

```bash
python3.10 sidis_crossect_torch.py config.yaml data.yaml fnp_config.yaml results/ output.yaml --use-ogata
# Uses Ogata quadrature for maximum accuracy
```

### In Python API

```python
# Differentiable path
sidis_comp.compute_sidis_cross_section_pytorch(data_file, output_file, use_ogata=False)

# High-accuracy path  
sidis_comp.compute_sidis_cross_section_pytorch(data_file, output_file, use_ogata=True)
```

---

## Validation and Best Practices

### Numerical Validation

1. **Compare integration methods**: Run same data with both Ogata and PyTorch
2. **Check convergence**: Increase `Nb` until results stabilize
3. **Monitor gradients**: Ensure non-zero gradients for fNP parameters

### Physics Validation

1. **Positive cross sections**: Check for unphysical negative values
2. **Smooth qT dependence**: Look for numerical discontinuities
3. **Kinematic limits**: Verify behavior at small/large qT

### Optimization Workflow

```python
# 1. Develop fit with PyTorch (fast, differentiable)
model.train()
for epoch in range(epochs):
    loss = compute_loss_pytorch(use_ogata=False)
    loss.backward()
    optimizer.step()

# 2. Validate with Ogata (accurate, non-differentiable)
model.eval()
final_results = compute_cross_sections(use_ogata=True)
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

---

## File Features Summary

### Core Capabilities

- **Dual integration**: Ogata (accurate) vs PyTorch (differentiable)
- **Device management**: Automatic CUDA/MPS/CPU selection
- **Configuration driven**: YAML-based setup for all parameters
- **TMD physics**: Full APFEL++ integration for evolution and matching
- **fNP modeling**: PyTorch-based non-perturbative functions

### Recent Enhancements

- **b-grid configuration**: Tunable via `config.yaml`
- **Integration method selection**: Command-line and API control
- **Improved documentation**: Clear usage guidelines
- **Gradient validation**: Tools for checking autodiff functionality

The implementation now provides the best of both worlds: **differentiable computation for parameter fitting** and **high-accuracy integration for production results**.

---

## Historical Note

This document previously described the limitations of the original implementation where autograd was broken by Ogata integration. The current version has resolved these issues by implementing a **dual integration strategy** that maintains both accuracy and differentiability as needed.

---

## Where the autograd graph breaks today

The class stores kinematics in tensors and loads a PyTorch fNP model, but several operations detach from Autograd:

1. Ogata integration callback returns Python floats
   - `self.DEObj.transform(b_integrand, qTm)` calls a Python function `b_integrand` that builds the integrand using Python floats and returns a float. Autograd cannot trace through this C++/Python-only integration.

2. fNP outputs are converted to scalars
   - Inside `b_integrand`:
     - `fnp1_val = float(self.compute_fnp_pytorch(...).item())`
     - `fnp2_val = float(self.compute_fnp_pytorch(...).item())`
   - Calling `.item()` and then `float(...)` severs gradients from fNP parameters to the final cross section.

3. NumPy usage inside integrand
   - `bstar_min` uses NumPy; various intermediate values are Python floats. Even if they don’t need gradients themselves, mixing forces a break in the tensor path.
   - A tensor-safe `bstar_min_pytorch` exists but isn’t used in the integrand.

4. APFEL objects produce numeric values (non-Torch)
   - `TabMatchTMDPDFs.EvaluatexQ`, `TabMatchTMDFFs.EvaluatexQ`, `QuarkSudakov`, `Hf`, and `TabAlphaem` return numbers from APFEL++. They do not need gradients w.r.t. fNP, but when the full integrand is built with Python floats the whole computation exits the Autograd domain.

Result: although the fNP model itself is differentiable, the final observable assigned into `theo_xsec` is computed from detached scalars, so `loss.backward()` won’t produce gradients for fNP parameters.

---

## What can be differentiable

- fNP model alone: `fNP(x, b, flavors)` returns tensors with gradients if its parameters have `requires_grad=True`.
- Cross-section w.r.t. fNP parameters: achievable if the b-integral and the multiplication by the APFEL-based factors are performed in PyTorch tensors, keeping fNP values as tensors and not converting to Python scalars.
- Gradients w.r.t. APFEL inputs (PDF/FF parameters, alpha_s, etc.) are not available unless they’re implemented in PyTorch or wrapped in custom autograd Functions. For fNP fits, that’s not required—treat APFEL pieces as constants.

---

## Recommended refactor for autodiff fits (minimal changes)

Goal: Differentiate only through fNP parameters while using APFEL results as frozen constants.

1. Keep tensors end-to-end

   - Avoid `.item()` and `float(...)` on any quantity that depends (directly or indirectly) on fNP parameters.
   - Use `bstar_min_pytorch` within the integrand.

2. Pre-tabulate APFEL-driven “luminosity” as constants

   - For each kinematic point and a chosen b-grid: precompute
     - `L(b; x, z, Q, qT) := Yp/x * sum_q [ e_q^2 * f1_q(x, b; mu, zeta) * D1_q(z, b; mu, zeta) ] * Sudakov(b; mu, zeta)^2 * alpha_em(Q)^2 * H(mu) / (Q^3 * z)`
   - Compute `L(b, ...)` with APFEL (Python floats), then store as a 1D torch tensor on the same device with `requires_grad=False` (use `.detach()` or create from `torch.tensor(L)`), one per kinematic point or batched.

3. Do the b-integral in PyTorch

   - Choose a b-grid (e.g., logarithmic in [b_min, b_max]) and weights (trapz/Simpson). Ogata is excellent for accuracy, but its callback-based implementation is non-differentiable.
   - Compute the integrand in PyTorch:
     - `I(b) = b * J0(qT * b) * fNP_pdf(x, b) * fNP_ff(z, b) * L(b)`
     - Use `torch.special.j0` for the Bessel function and `torch.trapz(I, b)` (or Simpson’s rule you implement) for integration.
   - This preserves gradients to fNP parameters.

4. Vectorize over points

   - Stack b-grids to shape `[Npts, Nb]` and compute all integrands at once for speed. Use broadcasting for `qT`, `x`, `z`, `Q`.

5. Precision and stability

   - Consider `dtype=torch.float64` for the integrand and integration to reduce numerical noise in gradients.
   - Validate the b-range and resolution by comparing against Ogata results offline.

---

## Sketch: differentiable b-integral in PyTorch

This is illustrative; it shows the idea of keeping fNP tensors alive and integrating in PyTorch while using APFEL-derived factors as constants.

```python
import torch

# b-grid (vector on device)
b = torch.logspace(-2, 0.3, steps=256, device=device, dtype=torch.float64)  # ~[1e-2, 2]

# Precompute APFEL luminosity on this b-grid (Python side, no grads), then:
L_b = torch.tensor(L_values_for_this_point, device=device, dtype=b.dtype)  # shape [Nb]

# fNP tensors (keep tensors; no .item())
fnp_pdf = model_fNP(x_tensor.expand_as(b), b, flavors=[pdf_flavor])[pdf_flavor]
fnp_ff  = model_fNP(z_tensor.expand_as(b), b, flavors=[ff_flavor])[ff_flavor]

# Bessel and integrand
J0 = torch.special.j0(qT_tensor * b)  # broadcast if qT is scalar/tensor
integrand = b * J0 * fnp_pdf * fnp_ff * L_b  # shape [Nb]

# Trapezoidal integral (differentiable)
xs = torch.trapz(integrand, b)

# Remaining factors (constants)
sigma = ap.constants.ConvFact * ap.constants.FourPi * qT_value * xs / (2 * Q_value * z_value)
```

Notes:

- When batching, promote scalars to shape `[Npts, 1]` and b to `[1, Nb]`, then broadcast to `[Npts, Nb]` and call `torch.trapz(integrand, b, dim=1)`.
- If you need Ogata-like accuracy, you can still use its node placement formula to build the b-grid and weights once, but perform the final sum in Torch. That keeps the op differentiable.

---

## Concrete edits to make in `sidis_crossect_torch.py`

- Replace uses of `bstar_min` in the integrand with `bstar_min_pytorch`.
- Remove `.item()` and `float(...)` on fNP outputs; keep them as tensors through the integrand and integration.
- Precompute APFEL parts into tensors outside Autograd (once per b-node and kinematic point), e.g., `L_b`.
- Replace `self.DEObj.transform(...)` with a Torch-based integration over a fixed b-grid (trapz/Simpson). Optionally keep a switch to compare against Ogata for validation.
- Ensure all kinematic scalars used with fNP are tensors on the right device/dtype.
- Consider setting `self.dtype = torch.float64` for the integration path; cast APFEL arrays to the same dtype.

These changes let you run a standard Torch optimizer over fNP parameters with `loss.backward()` producing non-zero gradients.

---

## What remains non-differentiable (and why that’s okay for fNP fits)

- APFEL-provided TMD PDFs/FFs, Sudakov, hard factors, and the running couplings are used as constants in the Torch graph. Gradients are not needed for fNP-only fits.
- If you later need gradients w.r.t. PDF/FF parameters or alpha_s, you’d need PyTorch-native implementations or custom autograd wrappers with backward formulas.

---

## Quick checklist for using this in a fit

- [ ] Switch to the Torch integration path (no `.item()`, no NumPy in integrand).
- [ ] Precompute and cache APFEL luminosity on the chosen b-grid per kinematic point.
- [ ] Keep all fNP-dependent quantities as tensors up to the loss.
- [ ] Validate numerical agreement vs. Ogata on a few points; tune b-grid as needed.
- [ ] Use double precision if gradients are noisy.

---

## File features summary (`map/sidis_crossect_torch.py`)

- PyTorch device management (CPU/CUDA/MPS) and dtype control.
- YAML-driven configuration for PDFs, FFs, scales, grids, and fNP model.
- APFEL++ integration for DGLAP/TMD evolution, matching, Sudakov, hard factor.
- PyTorch fNP model loading with parameter summaries.
- Ogata quadrature (non-differentiable) for the b-integral.
- Kinematic data ingestion and YAML export of results.

To enable end-to-end autodiff for fNP fits, switch the b-integral to a Torch-native quadrature and keep fNP tensors alive through the integrand, as outlined above.
