# SIDIS TMD Cross-Section Computation Framework

A Python framework for computing Semi-Inclusive Deep Inelastic Scattering (SIDIS) cross-sections using Transverse Momentum Dependent (TMD) parton distribution functions (PDFs) and fragmentation functions (FFs).

## Overview

This framework provides a complete implementation for computing SIDIS cross-sections with:

- **Operator Product Expansion (OPE)** for TMD PDFs and TMD FFs
- **Perturbative evolution** using QCD evolution equations
- **Non-perturbative (fNP) parameterizations** with flexible plugin architecture
- **Hankel transforms** for momentum space calculations

The non-perturbative part uses a plugin-based system that allows different parameterization combinations to be selected via configuration files.

## Quick Start

### Running the Code

The main entry point is `sidis/main.py`. You can run it from anywhere:

```bash
# From the project root
python3 sidis/main.py -c fNPconfig_base_flavor_dep.yaml

# From the sidis/ directory
python3 main.py -c fNPconfig_base_flavor_blind.yaml

# Use default (flavor-blind, shows warning)
python3 sidis/main.py
```

### Command Line Arguments

- `-c` or `-c`: Specify the fNP configuration file name (looked up in `cards/` directory)
  - Default: `fNPconfig_base_flavor_blind.yaml`
  - If not specified, a warning is displayed in yellow

### Available Configuration Files

Configuration files are stored in the `cards/` directory:

- `fNPconfig_base_flavor_blind.yaml`: Flavor-blind system (all flavors share parameters)
- `fNPconfig_base_flavor_dep.yaml`: Flavor-dependent system (each flavor has independent parameters)

To see all available config files, run:

```bash
python3 sidis/main.py --help
```

## Features

### fNP System

The non-perturbative (fNP) system provides a flexible plugin architecture:

- **Plugin-based**: Different "combo" files can be selected via configuration
- **Flavor-dependent**: Each quark flavor (u, d, s, etc.) can have independent parameters
- **Flavor-blind**: All flavors share the same parameters (reduces parameter count)
- **Q-dependent**: The rapidity scale `zeta` is computed from the hard scale `Q` as `zeta = Q²`

### Available Combos

1. **Flavor-Dependent** (`combo: flavor_dep`):
   - Each flavor has its own parameter set
   - ~161 total parameters (8 flavors × 11 PDF + 8 flavors × 9 FF + 1 evolution)
   - Maximum flexibility for flavor-specific physics

2. **Flavor-Blind** (`combo: flavor_blind`):
   - All flavors share identical parameters
   - 21 total parameters (11 PDF + 9 FF + 1 evolution)
   - Useful for reducing parameter count and testing assumptions

## File Structure

```bash
sidis/
├── main.py                 # Main entry point
├── config.yaml             # Main configuration file
├── cards/                  # fNP configuration files
│   ├── fNPconfig_base_flavor_blind.yaml
│   └── fNPconfig_base_flavor_dep.yaml
├── model/                  # Core model components
│   ├── __init__.py         # TrainableModel class
│   ├── ope.py              # Operator Product Expansion
│   ├── evolution.py         # Perturbative evolution
│   ├── ogata.py            # Hankel transforms
│   ├── fnp_factory.py      # fNP manager factory
│   ├── fnp_base_flavor_blind.py  # Flavor-blind combo
│   └── fnp_base_flavor_dep.py    # Flavor-dependent combo
├── utilities/              # Utility modules
│   ├── colors.py           # Terminal color codes
│   └── __init__.py
└── docs/                   # Documentation
    ├── README.md           # This file
    ├── fnp_system.md       # fNP system architecture
    ├── fnp_combos.md       # Combo implementations
    └── fnp_file_structure.md  # File structure details
```

## Configuration

### Main Configuration (`config.yaml`)

Contains settings for:

- OPE grid files
- Initial scale Q₀²
- Other framework parameters

### fNP Configuration Files (`cards/*.yaml`)

Each fNP config file specifies:

- `combo`: Which combo to use (`flavor_dep` or `flavor_blind`)
- `hadron`: Target hadron type
- `evolution`: Evolution parameters (shared across flavors)
- `pdf` or `pdfs`: PDF parameters
  - Flavor-blind: Single `pdf` section (applies to all flavors)
  - Flavor-dependent: `pdfs` section with per-flavor entries
- `ff` or `ffs`: FF parameters (same structure as PDFs)

See the configuration files in `cards/` for examples.

## Usage Examples

### Example 1: Flavor-Dependent Analysis

```bash
python3 sidis/main.py -c fNPconfig_base_flavor_dep.yaml
```

This uses the flavor-dependent system where each quark flavor has independent parameters.

### Example 2: Flavor-Blind Analysis

```bash
python3 sidis/main.py -c fNPconfig_base_flavor_blind.yaml
```

This uses the flavor-blind system where all flavors share parameters.

### Example 3: Programmatic Usage

```python
from sidis.model import TrainableModel
import torch

# Initialize model with specific config
model = TrainableModel(fnp_config='fNPconfig_base_flavor_dep.yaml')

# Prepare event data: [x, PhT, Q, z]
events = torch.tensor([
    [0.1, 0.5, 3.0, 0.3],  # Event 1
    [0.2, 0.6, 4.0, 0.4],  # Event 2
])

# Run forward pass
results = model(events)
```

## Documentation

For detailed documentation, see:

- **[fNP System Documentation](fnp_system.md)**: Overall architecture and plugin system
- **[fNP Combos Documentation](fnp_implemented_combos.md)**: Detailed documentation of each combo implementation
- **[File Structure Documentation](fnp_py_file_structure.md)**: Structure of combo files and components

## Troubleshooting

### Config File Not Found

If you see an error about a missing config file:

1. Check that the file exists in `cards/` directory
2. Verify the filename is correct (case-sensitive)
3. Run `python3 sidis/main.py --help` to see available files

## Contributing

When adding new fNP combos:

1. Create a new combo file following the structure in `fnp_base_flavor_blind.py` or `fnp_base_flavor_dep.py`
2. Register it in `fnp_factory.py` in the `COMBO_MODULES` dictionary
3. Create a configuration file in `cards/`
4. Update documentation

## Authors

Chiara Bissolotti (<cbissolotti@anl.gov>)
Patrick Barry (<pbarry@anl.gov>)
