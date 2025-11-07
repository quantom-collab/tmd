# fNP Combo Implementations

This document describes each available fNP combo implementation. A "combo" is a specific combination of TMD PDF and TMD FF non-perturbative parameterizations (and non-perturbative evolution)

- [fNP Combo Implementations](#fnp-combo-implementations)
  - [Available Implementations](#available-implementations)
    - [1. Base Flavor-Dependent Combo - MAP22 inspired](#1-base-flavor-dependent-combo---map22-inspired)
      - [Description](#description)
      - [Components](#components)
      - [Parameter Count](#parameter-count)
    - [2. Flavor-Blind Combo - MAP22 inspired](#2-flavor-blind-combo---map22-inspired)
      - [Description](#description-1)
      - [Components](#components-1)
      - [Parameter Count](#parameter-count-1)
      - [Physical Interpretation](#physical-interpretation)
  - [Parameterization Details - MAP22 Parameterization](#parameterization-details---map22-parameterization)
    - [PDF Parameters (MAP22)](#pdf-parameters-map22)
    - [FF Parameterization (MAP22)](#ff-parameterization-map22)
      - [Reference Points and Default Parameters](#reference-points-and-default-parameters)
  - [Creating Custom Combos](#creating-custom-combos)

## Available Implementations

### 1. Base Flavor-Dependent Combo - MAP22 inspired

**File**: `fnp_base_flavor_dep.py`  
**Registry Name**: `"flavor_dep"`  
**Manager**: `fNPManager`

#### Description

The standard combo implements a flavor-dependent system where each quark flavor has its own independent set of parameters. This provides maximum flexibility but requires more parameters.

#### Components

- **Evolution**: `fNP_evolution` - Shared evolution factor (1 parameter: g₂)
- **PDF**: `TMDPDFBase` - MAP22 parameterization (11 parameters per flavor)
- **FF**: `TMDFFBase` - MAP22 parameterization (9 parameters per flavor)

#### Parameter Count

- **Per flavor**: 11 (PDF) + 9 (FF) = 20 parameters
- **Total flavors**: 8 (u, d, s, ubar, dbar, sbar, c, cbar)
- **Total**: 8 × 20 + 1 (evolution) = **161 parameters**

---

### 2. Flavor-Blind Combo - MAP22 inspired

**File**: `fnp_base_flavor_blind.py`  
**Registry Name**: `"flavor_blind"`  
**Manager**: `fNPManagerFlavorBlind`

#### Description

The flavor-blind combo implements a system where ALL flavors share identical parameters. This dramatically reduces the parameter count while maintaining the full TMD structure. All flavors evolve together when parameters are updated during optimization.

#### Components

- **Evolution**: `fNP_evolution` - Shared evolution factor (1 parameter: g₂)
- **PDF**: `TMDPDFFlavorBlind` - MAP22 parameterization (11 parameters, shared)
- **FF**: `TMDFFFlavorBlind` - MAP22 parameterization (9 parameters, shared)

#### Parameter Count

- **PDF**: 11 parameters (shared across all 8 flavors) -- {N₁, α₁, σ₁, λ, N₁ᵦ, N₁ᶜ, λ₂, α₂, α₃, σ₂, σ₃}
- **FF**: 9 parameters (shared across all 8 flavors) -- {N₃, β₁, δ₁, γ₁, λ_F, N₃ᵦ, β₂, δ₂, γ₂}
- **Evolution**: 1 parameter
- **Total**: **21 parameters**

Parameter Reduction compared to standard combo:

- **Reduction**: 87% (from 161 to 21 parameters)
- **Benefits**: Faster fitting, reduced overfitting risk, cleaner interpretation

#### Physical Interpretation

- All quark flavors have identical TMD shapes in x and b_T
- Only the collinear PDFs/FFs (from APFEL++) distinguish flavors
- The non-perturbative contributions are universal across flavors
- This is a strong assumption but may be reasonable for phenomenology

---

## Parameterization Details - MAP22 Parameterization

Both combos use the MAP22 parameterization from NangaParbat, based on `MAP22g52.h` from the C++ implementation.

### PDF Parameters (MAP22)

$$
        f_{\rm NP}(x, b_T) = S_{\rm NP}(\zeta, b_T) \cdot \frac{
            g_1(x) \exp\left(-g_1(x) \frac{b_T^2}{4}\right) +
            \lambda^2 g_{1B}^2(x) \left(1 - g_{1B}(x) \frac{b_T^2}{4}\right) \exp\left(-g_{1B}(x) \frac{b_T^2}{4}\right) +
            \lambda_2^2 g_{1C}(x) \exp\left(-g_{1C}(x) \frac{b_T^2}{4}\right)
        }{
            g_1(x) + \lambda^2 g_{1B}^2(x) + \lambda_2^2 g_{1C}(x)
        }
$$
where:
$$
        g_{1,1B,1C}(x) = N_{1,1B,1C} \frac{x^{\sigma_{1,2,3}}(1-x)^{\alpha^2_{1,2,3}}}{\hat{x}^{\sigma_{1,2,3}}(1-\hat{x})^{\alpha^2_{1,2,3}}}, \quad \hat{x} = 0.1
$$

1. N₁ - Normalization
2. α₁ - Alpha parameter
3. σ₁ - Sigma parameter
4. λ - Lambda parameter
5. N₁ᵦ - Beta normalization
6. N₁ᶜ - C normalization
7. λ₂ - Lambda 2
8. α₂ - Alpha 2
9. α₃ - Alpha 3
10. σ₂ - Sigma 2
11. σ₃ - Sigma 3

### FF Parameterization (MAP22)

$$
        D_{\rm NP}(z, b_T) = S_{\rm NP}(\zeta, b_T) \cdot \frac{
            g_3(z) \exp\left(-g_3(z) \frac{b_T^2}{4z^2}\right) +
            \frac{\lambda_F}{z^2} g_{3B}^2(z) \left(1 - g_{3B}(z) \frac{b_T^2}{4z^2}\right) \exp\left(-g_{3B}(z) \frac{b_T^2}{4z^2}\right)
        }{
            g_3(z) + \frac{\lambda_F}{z^2} g_{3B}^2(z)
        }
$$
        where:
        $$
        g_{3,3B}(z) = N_{3,3B} \frac{(z^{\beta_{1,2}}+\delta^2_{1,2})(1-z)^{\gamma^2_{1,2}}}{(\hat{z}^{\beta_{1,2}}+\delta^2_{1,2})(1-\hat{z})^{\gamma^2_{1,2}}}, \quad \hat{z} = 0.5
        $$

1. N₃ - Normalization
2. β₁ - Beta 1
3. δ₁ - Delta 1
4. γ₁ - Gamma 1
5. λ_F - Lambda F
6. N₃ᵦ - Beta normalization
7. β₂ - Beta 2
8. δ₂ - Delta 2
9. γ₂ - Gamma 2

#### Reference Points and Default Parameters

- **PDF**: x̂ = 0.1 (MAP22 standard)
- **FF**: ẑ = 0.5 (MAP22 standard)
- **Evolution**: Q₀² = 1.0 GeV² (MAP22 standard)

Both combos provide default parameter sets based on MAP22:

- **Evolution**: g₂ = 0.12840
- **PDF defaults**: See `MAP22_DEFAULT_PDF_PARAMS` or `MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND`
- **FF defaults**: See `MAP22_DEFAULT_FF_PARAMS` or `MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND`

## Creating Custom Combos

To create a new combo:

1. **Create new python module**: Create a new Python module, for example following the structure of `fnp_base_flavor_dep.py` or `fnp_base_flavor_blind.py`.

2. **Implement required classes**:
   - `fNP_evolution`: Evolution factor class
   - `TMDPDF<name>`: PDF parameterization class
   - `TMDFF<name>`: FF parameterization class
   - Default parameter dictionaries (e.g. `MAP22_DEFAULT_EVOLUTION`, `MAP22_DEFAULT_PDF_PARAMS`, `MAP22_DEFAULT_FF_PARAMS`)
   - `fNPManager<name>` class: Manager class that orchestrates the combo components

3. **Update factory**: Add your new combo to the factory mapping in `fnp_factory.py`.

4. **Create config file in `cards/` directory**:
   - File name: `fNPconfig_<name>.yaml` (e.g. `fNPconfig_custom.yaml`)
   - Content: See example in `cards/fNPconfig_base_flavor_dep.yaml` or `cards/fNPconfig_base_flavor_blind.yaml`
   - Users can now select your combo via `combo: <name>` in config.

The framework is designed to be extensible - you can implement any combination of PDF and FF parameterizations as long as they follow the interface contract.
