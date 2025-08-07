# Ogata Integration Warnings in SIDIS Cross Section Computation

## Understanding the Warning: "Number of zero's available exceeded"

When running the SIDIS cross section computation, you may encounter warnings like:

```bash
[apfel::OgataQuadrature] Warning: Number of zero's available exceeded: the integration might not have converged.
```

This document explains what these warnings mean and how to address them.

## What is Ogata Quadrature?

The Ogata quadrature is a specialized numerical integration method designed for **Hankel transforms** (also called Bessel transforms). In TMD physics, we need to perform integrals of the form:

```math
∫₀^∞ db b J₀(qT·b) × f(b)
```

Where:

- `b` is the impact parameter
- `J₀` is the zeroth-order Bessel function
- `qT` is the transverse momentum
- `f(b)` is our integrand (containing TMD PDFs, FFs, Sudakov factors, etc.)

## Why Do These Warnings Occur?

The Ogata quadrature algorithm works by:

1. **Finding zeros** of the Bessel function J₀(qT·b)
2. **Using these zeros** as integration points for optimal accuracy
3. **Pre-calculating** a finite number of these zeros

The warning occurs when:

### 1. **High qT Values**

- Higher qT means J₀(qT·b) oscillates more rapidly
- More zeros are needed for accurate integration
- The pre-calculated zero table gets exhausted

### 2. **Rapidly Oscillating Integrands**

- TMD physics involves multiple oscillatory components
- Sudakov factors can create rapid variations
- The combination makes integration challenging

### 3. **Large Integration Range**

- b-space integration from 0 to infinity
- Function behavior changes dramatically across this range
- Standard parameters may not capture all features

## Physics Behind the Issue

In our SIDIS computation, the integrand contains:

```python
integrand = b × TMD_PDF(x,b) × TMD_FF(z,b) × Sudakov(b,μ,ζ) × NonPert(b) × J₀(qT·b)
```

Each component contributes to the integration difficulty:

1. **TMD Functions**: Decrease at large b
2. **Sudakov Factors**: Create rapid suppression at large b  
3. **Non-perturbative Functions**: Usually Gaussian-like
4. **Bessel Function**: Oscillates with frequency ∝ qT

## Impact on Results

### Are the Results Still Valid?

**Generally YES**, but with caveats:

- ✅ **Low qT region**: Usually well-converged, warnings are rare
- ⚠️ **Medium qT region**: Warnings appear but results often acceptable
- ❌ **High qT region**: Results may be unreliable

### How to Check Result Quality

1. **Compare with different integration parameters**
2. **Check for unphysical behavior** (negative cross sections, discontinuities)
3. **Verify against known limits** (collinear limits, small qT behavior)

## Solutions and Improvements

### 1. Adjust Ogata Parameters

```python
# More conservative parameters
DEObj = ap.ogata.OgataQuadrature(0, 1e-8, 0.0001)  # (order, cutoff, larger_step)

# High precision for difficult cases
DEObj = ap.ogata.OgataQuadrature(0, 1e-10, 0.000001)
```

### 2. Adaptive Integration Strategy

The improved implementation (`sidis_computation_improved.py`) uses:

```python
def adaptive_ogata_integration(self, integrand_func, qT):
    # Try standard parameters first
    # Fall back to conservative parameters
    # Use high-precision as last resort
    # Manual integration if all else fails
```

### 3. Alternative Integration Methods

For very difficult cases:

- **Trapezoidal rule** with careful sampling
- **Gaussian quadrature** with transformation
- **Monte Carlo integration** for high-dimensional cases

### 4. Physics-Based Improvements

#### Better Non-Perturbative Functions

```python
# Instead of simple Gaussian
fnp1 = np.exp(-0.1 * b**2)

# Use physics-motivated forms
fnp1 = np.exp(-g1 * b**2 - g2 * b**4)  # Collins-Soper evolution
```

#### Improved bstar Prescription

```python
# Better than simple bstar_min
def bstar_improved(b, Q):
    # Use more sophisticated prescription
    # Account for flavor dependence
    # Include radiative corrections
```

## Expected Behavior

### When Warnings Are Normal

- **Medium to high qT values** (qT/Q > 0.2)
- **Small x or z values** (where TMD evolution is significant)
- **High-precision calculations**

### When to Be Concerned

- **All points show warnings** (parameter issue)
- **Negative cross sections** (numerical instability)
- **Large discontinuities** between nearby points

## Recommended Workflow

### 1. Initial Run

```bash
python3.10 sidis_computation.py config.yaml data.yaml results.yaml
```

### 2. Check for Warnings

- Count how many points have warnings
- Look at the qT/Q distribution of warnings

### 3. Use Improved Version if Needed

```bash
python3.10 sidis_computation_improved.py config.yaml data.yaml improved_results.yaml
```

### 4. Compare Results

- Standard vs improved integration
- Look for systematic differences
- Check physics consistency

## Technical Details

### Ogata Algorithm Internals

The Ogata method transforms the Hankel integral:

```math
∫₀^∞ db b J₀(qT·b) f(b) = (π/qT²) Σₙ w_n f(x_n/qT) J₁(x_n)
```

Where:

- `x_n` are zeros of J₀
- `w_n` are quadrature weights
- The sum is over available zeros

### Parameter Meanings

```python
ap.ogata.OgataQuadrature(order, cutoff, step)
```

- **order**: Bessel function order (0 for J₀)
- **cutoff**: Minimum function value to include
- **step**: Step size in the zero-finding algorithm

## Best Practices

### 1. **Start Conservative**

- Use standard parameters first
- Only increase precision if needed

### 2. **Monitor Physics**

- Check positivity of cross sections
- Verify smooth qT dependence
- Compare with experimental trends

### 3. **Document Warnings**

- Keep track of which points have warnings
- Note the integration method used
- Include this info in results metadata

### 4. **Validate Results**

- Compare with collinear limits
- Check momentum sum rules
- Verify against other calculations

## Conclusion

Ogata integration warnings are **common and often acceptable** in TMD calculations. They indicate numerical challenges but don't necessarily invalidate results. The key is to:

1. **Understand the physics** causing the difficulty
2. **Use appropriate integration strategies**
3. **Validate results** through multiple checks
4. **Document the methodology** used

The improved implementation provides better robustness for challenging cases while maintaining the efficiency of the Ogata method where it works well.
