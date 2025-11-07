# `fnp_base_<name>.py` File Structure Documentation

This document describes the structure and organization of fNP (non-perturbative) combo files. Each combo file is a self-contained module that provides a complete implementation of TMD PDF and TMD FF non-perturbative parameterizations.

- [`fnp_base_<name>.py` File Structure Documentation](#fnp_base_namepy-file-structure-documentation)
  - [General File Structure](#general-file-structure)
    - [1. Evolution Factor Module (`fNP_evolution`)](#1-evolution-factor-module-fnp_evolution)
    - [2. TMD PDF Class](#2-tmd-pdf-class)
    - [3. TMD FF Class](#3-tmd-ff-class)
    - [4. Default Parameter Dictionaries](#4-default-parameter-dictionaries)
    - [5. Manager Class (`fNPManager`)](#5-manager-class-fnpmanager)
  - [Example: `fnp_base.py` Structure](#example-fnp_basepy-structure)
  - [The `n_flavors` Parameter](#the-n_flavors-parameter)
    - [What is `n_flavors`?](#what-is-n_flavors)
    - [Standard Usage: `n_flavors=1`](#standard-usage-n_flavors1)
    - [Alternative Usage: `n_flavors>1`](#alternative-usage-n_flavors1)
    - [Technical Details](#technical-details)
    - [Why Always 1 in Standard System?](#why-always-1-in-standard-system)
    - [When Would You Use `n_flavors>1`?](#when-would-you-use-n_flavors1)
  - [Creating New Combo Files](#creating-new-combo-files)
  - [File Naming Convention](#file-naming-convention)
  - [Configuration Files](#configuration-files)
  - [See Also](#see-also)

## General File Structure

Each fNP python file (e.g., `fnp_base_flavor_dep.py`, `fnp_base_flavor_blind.py`, etc.) follows a consistent structure with the following components:

### 1. Evolution Factor Module (`fNP_evolution`)

**Purpose**: Shared non-perturbative evolution factor used by both PDFs and FFs.

**Formula**: `S_NP(ζ, b_T) = exp[-g₂² b_T²/4 × ln(ζ/Q₀²)]`

**Key Features**:

- Single trainable parameter: `g₂` (evolution parameter)
- Shared across all flavors
- Computes evolution factor based on rapidity scale `zeta` and impact parameter `b`

**Location in file**: Section 1 (typically at the top)

### 2. TMD PDF Class

**Purpose**: Implements the non-perturbative TMD PDF parameterization.

**Classes**:

- `TMDPDFBase`: Flavor-dependent implementation (in `fnp_base_flavor_dep.py`)
- `TMDPDFFlavorBlind`: Flavor-blind implementation (in `fnp_base_flavor_blind.py`)

**Key Features**:

- MAP22 parameterization (11 parameters for PDF)
- Can be flavor-dependent (each flavor has own parameters) or flavor-blind (all flavors share parameters)
- Computes `f_NP(x, b_T)` based on Bjorken x and impact parameter b

**Location in file**: Section 2

### 3. TMD FF Class

**Purpose**: Implements the non-perturbative TMD FF parameterization.

**Classes**:

- `TMDFFBase`: Flavor-dependent implementation (in `fnp_base_flavor_dep.py`)
- `TMDFFFlavorBlind`: Flavor-blind implementation (in `fnp_base_flavor_blind.py`)

**Key Features**:

- MAP22 parameterization (9 parameters for FF)
- Can be flavor-dependent or flavor-blind
- Computes `D_NP(z, b_T)` based on energy fraction z and impact parameter b

**Location in file**: Section 3

### 4. Default Parameter Dictionaries

**Purpose**: Provide default initialization values for parameters.

**Dictionaries**:

- `MAP22_DEFAULT_EVOLUTION`: Default evolution parameters
- `MAP22_DEFAULT_PDF_PARAMS`: Default PDF parameters
- `MAP22_DEFAULT_FF_PARAMS`: Default FF parameters

**Usage**: Used when configuration doesn't specify parameters for a flavor.

**Location in file**: After the class definitions, before the manager

### 5. Manager Class (`fNPManager`)

**Purpose**: Orchestrates the combo components and provides the unified interface.

**Key Features**:

- Initializes evolution, PDF, and FF modules from configuration
- Provides `forward()`, `forward_pdf()`, and `forward_ff()` methods
- Computes `zeta = Q²` from hard scale Q
- Handles flavor selection and parameter management

**Location in file**: Section 4 (at the end)

All **manager classes** (e.g., `fNPManager`, `fNPManagerFlavorBlind`) must provide the same interface:

- `forward(x, z, b, Q)`: Evaluate both PDFs and FFs
- `forward_pdf(x, b, Q)`: Evaluate PDFs only
- `forward_ff(z, b, Q)`: Evaluate FFs only

In particular, **manager classes** must implement the `forward` method:

```python
def forward(
    self,
    x: torch.Tensor,      # Bjorken x
    z: torch.Tensor,       # Energy fraction
    b: torch.Tensor,       # Impact parameter
    Q: torch.Tensor,       # Hard scale (used to compute zeta = Q²)
    pdf_flavors: Optional[List[str]] = None,
    ff_flavors: Optional[List[str]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Returns:
        {
            "pdfs": {flavor: tensor, ...},
            "ffs": {flavor: tensor, ...}
        }
    """
```
  
The rapidity scale `zeta` is computed from the hard scale `Q` in the forward methods:

```python
zeta = Q²  # Standard SIDIS choice
```

The computation is performed in the `_compute_zeta(Q)` method of each manager class.
If you need a different formula (e.g., `zeta = Q² * z`), modify the `_compute_zeta()` method in the appropriate manager class.

`Q` is passed as a parameter to the forward methods of the manager class, allowing:

1. **Event-by-event computation**: Each event can have a different `Q` value
2. **Flexibility**: Easy to change the formula if needed

## Example: `fnp_base.py` Structure

```python
"""
Standard Flavor-Dependent fNP Implementation

Contents of this file:
- Evolution factor module (fNP_evolution): Shared across PDFs and FFs
- TMD PDF class (TMDPDFBase): MAP22 parameterization with flavor-specific parameters
- TMD FF class (TMDFFBase): MAP22 parameterization with flavor-specific parameters
- Default parameter dictionaries: MAP22_DEFAULT_EVOLUTION,
                                  MAP22_DEFAULT_PDF_PARAMS,
                                  MAP22_DEFAULT_FF_PARAMS
- Manager class (fNPManager): Orchestrates the standard combo implementation for PDFs and FFs
"""

# Section 1: Evolution Factor Module
class fNP_evolution(nn.Module):
    """Evolution factor S_NP(ζ, b_T)"""
    ...

# Section 2: TMD PDF Class
class TMDPDFBase(nn.Module):
    """TMD PDF with flavor-dependent parameters"""
    ...

# Section 3: TMD FF Class
class TMDFFBase(nn.Module):
    """TMD FF with flavor-dependent parameters"""
    ...

# Section 4: Default Parameters
MAP22_DEFAULT_EVOLUTION = {...}
MAP22_DEFAULT_PDF_PARAMS = {...}
MAP22_DEFAULT_FF_PARAMS = {...}

# Section 5: Manager Class
class fNPManager(nn.Module):
    """Manager for standard flavor-dependent system"""
    ...
```

## The `n_flavors` Parameter

### What is `n_flavors`?

The `n_flavors` parameter in `TMDPDFBase` and `TMDFFBase` controls how many flavor instances share the same parameter set within a single class instance.

### Standard Usage: `n_flavors=1`

In the standard flavor-dependent system, the manager creates **separate instances** for each flavor:

```python
# Manager creates one instance per flavor
pdf_modules["u"] = TMDPDFBase(n_flavors=1, init_params=..., free_mask=...)
pdf_modules["d"] = TMDPDFBase(n_flavors=1, init_params=..., free_mask=...)
pdf_modules["s"] = TMDPDFBase(n_flavors=1, init_params=..., free_mask=...)
# ... etc for each flavor
```

**Result**: Each flavor (u, d, s, etc.) has completely independent parameters that can evolve separately during training.

### Alternative Usage: `n_flavors>1`

If you set `n_flavors=2`, you create a single instance that serves two flavors:

```python
# Single instance serving two flavors
pdf_group = TMDPDFBase(n_flavors=2, init_params=..., free_mask=...)
# This instance has shape (2, 11) parameter tensor
# Both flavors share the same parameters and evolve together
```

**Result**: The two flavors share the same parameter set and evolve together during optimization. This is useful for:

- Grouping flavors that should be parameterized identically (e.g., u and d quarks)
- Creating a "mini flavor-blind" system for specific flavor groups
- Reducing parameter count for certain flavor combinations

### Technical Details

When `n_flavors=1`:

- Parameter tensor shape: `(1, P)` where P is number of parameters
- Each flavor gets its own independent instance
- Parameters can diverge during training

When `n_flavors=2`:

- Parameter tensor shape: `(2, P)`
- Two flavors share the same instance
- Parameters start identical and evolve together
- Access via `flavor_idx=0` or `flavor_idx=1` in forward method

### Why Always 1 in Standard System?

In the standard flavor-dependent system, `n_flavors=1` is used because:

1. **Independence**: Each flavor needs completely independent parameters
2. **Flexibility**: Allows different flavors to have different parameter values
3. **Clarity**: One instance per flavor makes the code easier to understand
4. **Manager Design**: The manager creates separate instances in a dictionary, so each flavor naturally gets its own instance

### When Would You Use `n_flavors>1`?

Use `n_flavors>1` when:

- You want to group certain flavors to share parameters (e.g., all sea quarks)
- You want a hybrid approach: some flavors independent, some grouped
- You want to reduce parameters for specific flavor combinations
- You're experimenting with different parameterization strategies

**Example**: You might want u and d quarks to share parameters, but s quark to be independent:

```python
# u and d share parameters
pdf_ud = TMDPDFBase(n_flavors=2, init_params=..., free_mask=...)
# s is independent
pdf_s = TMDPDFBase(n_flavors=1, init_params=..., free_mask=...)
```

## Creating New Combo Files

To create a new combo file:

1. **Copy structure** from existing combo file (`fnp_base.py` or `fnp_base_flavor_blind.py`)
2. **Implement required classes**:
   - `fNP_evolution`: Evolution factor (can reuse from existing combos)
   - PDF class: Your PDF parameterization
   - FF class: Your FF parameterization
   - `fNPManager`: Manager that orchestrates your combo
3. **Add default parameters**: Create default dictionaries for initialization
4. **Register in factory**: Add to `COMBO_MODULES` in `fnp_factory.py`
5. **Create config file**: Add configuration file in `cards/` folder

## File Naming Convention

- `fnp_base.py`: Standard flavor-dependent combo
- `fnp_base_flavor_blind.py`: Flavor-blind combo
- `fnp_combo_<name>.py`: Custom combo implementations

## Configuration Files

Configuration files are stored in the `cards/` folder:

- `fNPconfig_flav_blind_std.yaml`: Flavor-blind configuration
- `fNPconfig_flav_dep_std.yaml`: Flavor-dependent configuration

Each config file specifies:

- `combo`: Which combo to use ("standard" or "flavor_blind")
- `hadron`: Target hadron type
- `evolution`: Evolution parameters
- `pdf` or `pdfs`: PDF parameters (single set for flavor-blind, per-flavor for standard)
- `ff` or `ffs`: FF parameters (single set for flavor-blind, per-flavor for standard)

## See Also

- [fNP System Documentation](fnp_system.md): Overall system architecture
- [fNP Combos Documentation](fnp_combos.md): Detailed documentation of each combo implementation
