# Modular fNP System Documentation

**Author:** Chiara Bissolotti (<cbissolotti@anl.gov>)  
**Version:** 2.0.0  
**Purpose:** Unified TMD PDF and FF non-perturbative parameterization with MAP22 implementation

---

## Table of Contents

- [Modular fNP System Documentation](#modular-fnp-system-documentation)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. New Modular Architecture](#2-new-modular-architecture)
  - [3. File Structure](#3-file-structure)
  - [4. Core Components](#4-core-components)
    - [4.1. Evolution Module (`fNP_evolution`)](#41-evolution-module-fnp_evolution)
    - [4.2. TMD PDF Base (`TMDPDFBase`)](#42-tmd-pdf-base-tmdpdfbase)
    - [4.3. TMD FF Base (`TMDFFBase`)](#43-tmd-ff-base-tmdffbase)
    - [4.4. Unified Manager (`fNPManager`)](#44-unified-manager-fnpmanager)
  - [5. MAP22 Implementation](#5-map22-implementation)
  - [6. Configuration Structure](#6-configuration-structure)
    - [Complete Example (`fNPconfig_unified.yaml`)](#complete-example-fnpconfig_unifiedyaml)
  - [7. Usage Examples](#7-usage-examples)
    - [7.1. Basic Usage](#71-basic-usage)
    - [7.2. Optimization Example](#72-optimization-example)
    - [7.3. Parameter Analysis](#73-parameter-analysis)
  - [8. Migration Guide](#8-migration-guide)
  - [9. Testing and Validation](#9-testing-and-validation)

---

## 1. Overview

The modular fNP system represents a complete reorganization of the TMD non-perturbative function framework. Key improvements include:

- **Unified PDF/FF Management**: Simultaneous optimization of both TMD PDFs and FFs
- **MAP22 Implementation**: Exact implementation of the MAP22 parameterization from NangaParbat
- **Shared Evolution**: Common evolution factor across PDFs and FFs
- **Modular Design**: Clean separation of concerns with reusable components
- **Enhanced Configuration**: Comprehensive YAML-based parameter management

## 2. New Modular Architecture

```mermaid
graph TD
    Config[fNPconfig_unified.yaml] --> Manager[fNPManager]
    Manager --> Evolution[fNP_evolution<br/>Shared Evolution]
    Manager --> PDFModules[TMD PDF Modules]
    Manager --> FFModules[TMD FF Modules]
    
    PDFModules --> |"u, d, s, etc."| PDFBase[TMDPDFBase<br/>MAP22 Implementation]
    FFModules --> |"u, d, s, etc."| FFBase[TMDFFBase<br/>MAP22 Implementation]
    
    Evolution --> |S_NP factor| PDFBase
    Evolution --> |S_NP factor| FFBase
    
    Manager --> Optimizer[PyTorch Optimizer<br/>Unified Parameters]
    
    classDef config fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef manager fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef module fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef optimizer fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class Config config
    class Manager manager
    class Evolution,PDFModules,FFModules,PDFBase,FFBase module
    class Optimizer optimizer
```

## 3. File Structure

```bash
map/modules/
├── fnp_base.py          # Base classes (Evolution, PDF, FF)
├── fnp_manager.py       # Unified manager for PDFs and FFs
├── fNP_new.py          # Main interface and exports
└── fNP.py              # Legacy file (to be replaced)

map/inputs/
├── fNPconfig_unified.yaml  # New unified configuration
└── fNPconfig.yaml         # Legacy configuration

map/tests/
├── test_fnp_modular.py    # Comprehensive test suite
└── other test files...

map/docs/
├── fNP_modular.md         # This documentation
└── fNP.md                 # Legacy documentation
```

## 4. Core Components

### 4.1. Evolution Module (`fNP_evolution`)

**Purpose:** Shared evolution factor for both TMD PDFs and FFs.

**Implementation:**

```python
S_NP(ζ, b_T) = exp[-g₂² b_T²/4 × ln(ζ/Q₀²)]
```

**Key Features:**

- Single trainable parameter `g₂`
- Shared across all PDF and FF flavors
- Parameter masking support
- Reference scale Q₀² = 1 GeV²

### 4.2. TMD PDF Base (`TMDPDFBase`)

**Purpose:** Exact MAP22 TMD PDF parameterization.

**Parameters (11 total):**

- `N₁, α₁, σ₁, λ`: Primary component
- `N₁ᵦ, α₂, σ₂`: Secondary component  
- `N₁ᶜ, α₃, σ₃`: Tertiary component
- `λ₂`: Additional coupling

**Implementation:**

```python
f_NP(x, b_T) = S_NP × [numerator] / [denominator]

numerator = g₁×exp(-g₁×(b/2)²) + λ²×g₁ᵦ²×(1-g₁ᵦ×(b/2)²)×exp(-g₁ᵦ×(b/2)²) + g₁ᶜ×λ₂²×exp(-g₁ᶜ×(b/2)²)
denominator = g₁ + λ²×g₁ᵦ² + g₁ᶜ×λ₂²

where:
g₁ = N₁ × (x/0.1)^σ₁ × ((1-x)/0.9)^α₁²
g₁ᵦ = N₁ᵦ × (x/0.1)^σ₂ × ((1-x)/0.9)^α₂²
g₁ᶜ = N₁ᶜ × (x/0.1)^σ₃ × ((1-x)/0.9)^α₃²
```

### 4.3. TMD FF Base (`TMDFFBase`)

**Purpose:** Exact MAP22 TMD FF parameterization.

**Parameters (9 total):**

- `N₃, β₁, δ₁, γ₁`: Primary component
- `N₃ᵦ, β₂, δ₂, γ₂`: Secondary component
- `λ_F`: Coupling parameter

**Implementation:**

```python
D_NP(z, b_T) = S_NP × [numerator] / [denominator]

numerator = g₃×exp(-g₃×(b/2)²/z²) + (λ_F/z²)×g₃ᵦ²×(1-g₃ᵦ×(b/2)²/z²)×exp(-g₃ᵦ×(b/2)²/z²)
denominator = g₃ + (λ_F/z²)×g₃ᵦ²

where:
g₃ = N₃ × [(z^β₁ + δ₁²)/(0.5^β₁ + δ₁²)] × ((1-z)/0.5)^γ₁²
g₃ᵦ = N₃ᵦ × [(z^β₂ + δ₂²)/(0.5^β₂ + δ₂²)] × ((1-z)/0.5)^γ₂²
```

### 4.4. Unified Manager (`fNPManager`)

**Purpose:** Single interface for managing both TMD PDFs and FFs.

**Key Features:**

- Unified parameter optimization
- Shared evolution factor management
- Per-flavor configuration
- Comprehensive parameter analysis
- PyTorch optimizer compatibility

## 5. MAP22 Implementation

The implementation exactly matches the C++ MAP22g52.h from NangaParbat:

**Parameter Mapping:**

```cpp
// C++ MAP22g52.h parameters (21 total)
this->_pars[0]  = g2        // Evolution
this->_pars[1]  = N1        // PDF primary
this->_pars[2]  = alpha1
this->_pars[3]  = sigma1
this->_pars[4]  = lambda
this->_pars[5]  = N3        // FF primary
this->_pars[6]  = beta1
this->_pars[7]  = delta1
this->_pars[8]  = gamma1
this->_pars[9]  = lambdaF
this->_pars[10] = N3B       // FF secondary
this->_pars[11] = N1B       // PDF secondary
this->_pars[12] = N1C       // PDF tertiary
this->_pars[13] = lambda2
...
```

**Python Modular Mapping:**

```python
# Evolution (1 parameter)
g2

# PDF per flavor (11 parameters)
[N₁, α₁, σ₁, λ, N₁ᵦ, N₁ᶜ, λ₂, α₂, α₃, σ₂, σ₃]

# FF per flavor (9 parameters)  
[N₃, β₁, δ₁, γ₁, λ_F, N₃ᵦ, β₂, δ₂, γ₂]
```

## 6. Configuration Structure

### Complete Example (`fNPconfig_unified.yaml`)

```yaml
# Global settings
hadron: proton
zeta: 1.0

# Shared evolution factor
evolution:
  init_g2: 0.12840
  free_mask: [true]

# TMD PDFs (11 parameters each)
pdfs:
  u:
    init_params: [0.28516, 2.9755, 0.17293, 0.39432, 0.28516, 0.28516, 0.39432, 2.9755, 2.9755, 0.17293, 0.17293]
    free_mask: [true, true, true, true, true, true, true, true, true, true, true]
  d:
    init_params: [0.25000, 2.8000, 0.16000, 0.35000, 0.25000, 0.25000, 0.35000, 2.8000, 2.8000, 0.16000, 0.16000]
    free_mask: [true, true, true, true, true, true, true, true, true, true, true]
  # ... other flavors with reduced complexity

# TMD FFs (9 parameters each)
ffs:
  u:
    init_params: [0.21012, 2.12062, 0.093554, 0.25246, 5.2915, 0.033798, 2.1012, 0.093554, 0.25246]
    free_mask: [true, true, true, true, true, true, true, true, true]
  d:
    init_params: [0.19000, 2.00000, 0.090000, 0.24000, 5.0000, 0.030000, 2.0000, 0.090000, 0.24000]
    free_mask: [true, true, true, true, true, true, true, true, true]
  # ... other flavors with reduced complexity
```

**Parameter Summary:**

- **Total**: 161 parameters (1 evolution + 8×11 PDFs + 8×9 FFs)
- **Trainable**: ~70 parameters (varies based on masking strategy)
- **Strategy**: Full complexity for u,d; reduced for sea quarks; minimal for heavy quarks

## 7. Usage Examples

### 7.1. Basic Usage

```python
import torch
import yaml
from modules.fnp_manager import fNPManager

# Load configuration
with open('map/inputs/fNPconfig_unified.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize manager
fnp_manager = fNPManager(config)

# Evaluate TMD PDFs
x = torch.tensor([0.1, 0.3, 0.5])
b = torch.tensor([0.5, 1.0, 1.5])

pdf_results = fnp_manager.forward_pdf(x, b, flavors=['u', 'd'])
print(f"PDF u: {pdf_results['u']}")
print(f"PDF d: {pdf_results['d']}")

# Evaluate TMD FFs
z = torch.tensor([0.2, 0.4, 0.7])
ff_results = fnp_manager.forward_ff(z, b, flavors=['u', 'd'])
print(f"FF u: {ff_results['u']}")
print(f"FF d: {ff_results['d']}")
```

### 7.2. Optimization Example

```python
# Set up optimizer
optimizer = torch.optim.Adam(fnp_manager.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass
    pdf_results = fnp_manager.forward_pdf(x_data, b_data, flavors=['u', 'd'])
    ff_results = fnp_manager.forward_ff(z_data, b_data, flavors=['u', 'd'])
    
    # Compute loss (example)
    pdf_loss = torch.nn.functional.mse_loss(pdf_results['u'], target_pdf_u)
    ff_loss = torch.nn.functional.mse_loss(ff_results['u'], target_ff_u)
    total_loss = pdf_loss + ff_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
```

### 7.3. Parameter Analysis

```python
# Get parameter information
param_info = fnp_manager.get_parameter_info()
print(f"Total parameters: {param_info['total_parameters']}")
print(f"Trainable parameters: {param_info['truly_trainable_parameters']}")

# Print detailed summary
fnp_manager.print_parameter_summary()

# Extract trainable parameters for saving
trainable_params = fnp_manager.get_trainable_parameters_dict()
torch.save(trainable_params, 'fitted_parameters.pth')

# Load parameters
loaded_params = torch.load('fitted_parameters.pth')
fnp_manager.set_trainable_parameters_dict(loaded_params)
```

## 8. Migration Guide

**From Legacy fNP.py to Modular System:**

1. **Update Imports:**

   ```python
   # Old
   from modules.fNP import fNP
   
   # New
   from modules.fnp_manager import fNPManager as fNP
   ```

2. **Update Configuration:**
   - Use `fNPconfig_unified.yaml` instead of `fNPconfig.yaml`
   - Separate PDF and FF configurations
   - Define masking strategy for each flavor

3. **Update Usage:**

   ```python
   # Old
   outputs = model(x, b, flavors=['u', 'd'])
   
   # New - specify PDF or FF
   pdf_outputs = model.forward_pdf(x, b, flavors=['u', 'd'])
   ff_outputs = model.forward_ff(z, b, flavors=['u', 'd'])
   ```

4. **Update SIDIS Integration:**
   - Modify `compute_flavor_sum_pytorch` to use separate PDF/FF calls
   - Update parameter extraction methods

## 9. Testing and Validation

**Test Suite:** `map/tests/test_fnp_modular.py`

**Test Coverage:**

- ✅ Evolution module functionality
- ✅ TMD PDF MAP22 implementation
- ✅ TMD FF MAP22 implementation  
- ✅ Unified manager operations
- ✅ Parameter masking and optimization
- ✅ Configuration loading
- ✅ PyTorch optimizer compatibility

**Run Tests:**

```bash
cd /path/to/tmd
python3.10 map/tests/test_fnp_modular.py
```

**Expected Output:**

```bash
🧪 Testing Modular fNP System
✅ Evolution module working
✅ TMD PDF module (MAP22) working
✅ TMD FF module (MAP22) working
✅ Unified manager working
✅ Optimization compatibility confirmed
🎉 ALL TESTS PASSED SUCCESSFULLY!
```

---

**Key Advantages of the Modular System:**

1. **Physics Accuracy**: Exact MAP22 implementation matching C++ reference
2. **Unified Optimization**: Simultaneous PDF and FF parameter fitting
3. **Flexibility**: Per-flavor parameterization control with masking
4. **Maintainability**: Clean modular design with separation of concerns
5. **Extensibility**: Easy to add new parameterizations or flavors
6. **Performance**: Efficient PyTorch implementation with gradient support

The modular fNP system provides a robust, physics-accurate, and maintainable framework for TMD non-perturbative function modeling with full integration into the PyTorch optimization ecosystem.
