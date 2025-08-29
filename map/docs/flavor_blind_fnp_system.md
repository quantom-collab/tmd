# Flavor-Blind fNP System

This document describes the flavor-blind fNP system for SIDIS cross section computation, where all quark flavors share identical non-perturbative parameters.

## Table of Contents

- [Flavor-Blind fNP System](#flavor-blind-fnp-system)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Parameter Comparison](#parameter-comparison)
  - [File Structure](#file-structure)
  - [Implementation Details](#implementation-details)
    - [Core Classes](#core-classes)
  - [`fNPManagerFlavorBlind`](#fnpmanagerflavorblind)
  - [Testing](#testing)
  - [Usage](#usage)
    - [1. Basic Usage (No APFEL++ Required)](#1-basic-usage-no-apfel-required)
    - [2. SIDIS Cross Section Computation](#2-sidis-cross-section-computation)
    - [3. Parameter Fitting](#3-parameter-fitting)
  - [Configuration](#configuration)
  - [Assumptions](#assumptions)
  - [Future Extensions](#future-extensions)
  - [Contact](#contact)
  
## Overview

The flavor-blind fNP system is a simplified version of the standard TMD parametrization where **all flavors (u, d, s, ubar, dbar, sbar, c, cbar) share exactly the same non-perturbative parameters**. When parameters are updated during fitting, **all flavors change simultaneously**.

### Parameter Comparison

FF are intended for a single hadron in the final state.

| System | Evolution | PDF | FF | Total |
|--------|-----------|-----|----| ------|
| Standard | 1 | 8×11 = 88 | 8×9 = 72 | 161 |
| Flavor-Blind | 1 | 11 (shared) | 9 (shared) | 21 |

## File Structure

The flavor-blind system consists of the following new files:

```bash
map/
├── modules/
│   ├── fnp_base_flavor_blind.py      # Base classes for flavor-blind PDFs/FFs
│   └── fnp_manager_flavor_blind.py   # Unified manager for flavor-blind system
├── inputs/
│   └── fNPconfig_flavor_blind.yaml   # Configuration for flavor-blind parameters
└── tests/
    ├── fit_fnp_synthetic_flavor_blind.py  # Fitting script (requires APFEL++)
    └── test_flavor_blind_simple.py        # Simple test (no APFEL++ required)
```

## Implementation Details

### Core Classes

1. **`TMDPDFFlavorBlind`**: Single parameter set for all PDF flavors
2. **`TMDFFFlavorBlind`**: Single parameter set for all FF flavors  
3. **`fNPManagerFlavorBlind`**: Unified manager for both PDFs and FFs

## `fNPManagerFlavorBlind`

This class manages the flavor-blind fNP system, ensuring that all flavors share the same parameters.

## Testing

Run the comprehensive test suite:

```bash
cd map/tests
python test_flavor_blind_simple.py
```

This test verifies:

- ✅ All flavors return identical results
- ✅ Parameter reduction works correctly (21 vs 161 parameters)
- ✅ Gradient-based fitting converges
- ✅ Parameter changes affect all flavors simultaneously

## Usage

### 1. Basic Usage (No APFEL++ Required)

```python
import torch
from modules.fnp import fNPManagerFlavorBlind, load_flavor_blind_config

# Load configuration
config = load_flavor_blind_config('inputs/fNPconfig_flavor_blind.yaml')

# Initialize flavor-blind model
model = fNPManagerFlavorBlind(config)

# Evaluate PDFs (all flavors identical)
x = torch.tensor([0.1, 0.3, 0.5])
b = torch.tensor([0.5, 1.0, 2.0])
pdf_results = model.forward_pdf(x, b, ['u', 'd', 's'])

# All flavors return identical results
assert torch.allclose(pdf_results['u'], pdf_results['d'])
assert torch.allclose(pdf_results['u'], pdf_results['s'])

# Same for fragmentation functions
z = torch.tensor([0.2, 0.5, 0.8])
ff_results = model.forward_ff(z, b, ['u', 'd', 's'])
assert torch.allclose(ff_results['u'], ff_results['d'])
```

### 2. SIDIS Cross Section Computation

For full SIDIS cross section computation with APFEL++:

```bash
# Requires Python 3.10 and APFEL++/LHAPDF
cd map/tests
python3.10 fit_fnp_synthetic_flavor_blind.py --points 20 --epochs 10
```

### 3. Parameter Fitting

```python
# Set model to training mode
model.train()

# Set up optimizer (only 21 parameters to optimize!)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Fitting loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Compute predictions using flavor-blind fNPs
    predictions = compute_cross_sections(model, kinematics)
    
    # Compute loss
    loss = loss_function(predictions, targets)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
```

## Configuration

The flavor-blind system uses a simplified configuration file:

```yaml
# fNPconfig_flavor_blind.yaml
hadron: proton
zeta: 1.0

# Shared evolution (same as standard system)
evolution:
  init_g2: 0.12840
  free_mask: [true]

# Single PDF parameter set (applies to ALL flavors)
pdf:
  init_params: [0.28516, 2.9755, 0.17293, ...]  # 11 parameters
  free_mask: [true, true, true, ...]             # All trainable

# Single FF parameter set (applies to ALL flavors)  
ff:
  init_params: [0.21012, 2.12062, 0.093554, ...]  # 9 parameters
  free_mask: [true, true, true, ...]               # All trainable
```

## Assumptions

The flavor-blind system makes the following assumptions:

1. **Universal TMD Shape**: All quark flavors have identical x and bT dependence
2. **Flavor Universality**: Non-perturbative effects are flavor-independent
3. **Collinear Distinction**: Flavor differences come only from collinear PDFs/FFs

## Future Extensions

The flavor-blind system can be extended in several ways:

1. **Partial Flavor Blindness**: Group flavors (quarks vs antiquarks)
2. **Hierarchical Sharing**: Different sharing patterns for different parameters
3. **Flavor Scaling**: Add simple scaling factors while sharing shapes
4. **Adaptive Blindness**: Learn when to share vs separate parameters

## Contact

For questions about the flavor-blind fNP system, contact:

- Chiara Bissolotti (<cbissolotti@anl.gov>)
