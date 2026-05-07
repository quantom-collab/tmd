# TMD Framework

A comprehensive framework for computing Transverse Momentum Dependent (TMD) parton distribution functions (PDFs) and fragmentation functions (FFs), with applications to Semi-Inclusive Deep Inelastic Scattering (SIDIS), DGLAP evolution, and spin asymmetries.

## Overview

This framework includes multiple components for TMD physics calculations:

- **SIDIS Module** (`sidis/`): Semi-Inclusive Deep Inelastic Scattering cross-section computation
- **Grid Utilities** (`grids/`): C++ code for TMD grid generation and Collins-Soper kernel calculations
- **OPE Tools** (`ope/`): Operator Product Expansion and evolution implementations
- **Spin Physics** (`spin/`): Spin asymmetry calculations and parametrizations
- **Interpolation Tools** (`interpol/`): Grid interpolation utilities

### Core Features

- **Operator Product Expansion (OPE)** for TMD PDFs and TMD FFs with QCD evolution
- **Flexible Parametrizations** for non-perturbative TMD contributions
- **Perturbative QCD** evolution using DGLAP and renormalization group equations
- **Numerical Methods**: Hankel transforms for momentum-space calculations and Akima interpolation
- **PyTorch Integration** for automatic differentiation and efficient tensor operations

## Quick Start

### SIDIS Framework

The main entry point for SIDIS calculations is `sidis/main.py`:

```bash
# From the project root
python3 sidis/main.py -c fNPconfig_base_flavor_dep.yaml

# From the sidis/ directory
python3 main.py -c fNPconfig_base_flavor_blind.yaml

# Use default (flavor-blind)
python3 sidis/main.py
```

### Command Line Arguments

- `-c`, `--config`: Specify the fNP configuration file name (looked up in `sidis/cards/` directory)
  - Default: `fNPconfig_base_flavor_blind.yaml`

### Available Configuration Files

Configuration files are stored in `sidis/cards/` and include:

- `fNPconfig_base_flavor_blind.yaml`: Flavor-blind system
- `fNPconfig_base_flavor_dep.yaml`: Flavor-dependent system
- `fNPconfig_simple.yaml`: Simplified configuration
- Additional specialized configurations

## Framework Components

### 1. SIDIS Module (`sidis/`)

The core framework for Semi-Inclusive Deep Inelastic Scattering calculations. This module implements:

- **TrainableModel**: PyTorch-based model for TMD cross-section computation
- **OPE System**: Operator Product Expansion for matching perturbative QCD to TMD
- **fNP Plugin Architecture**: Flexible non-perturbative parametrizations with multiple "combo" implementations
- **QCD Evolution**: Perturbative QCD evolution for PDFs and fragmentation functions
- **Numerical Methods**: Hankel transforms (Ogata method), special functions, and precision calculations

#### fNP System Features

The non-perturbative (fNP) system provides a flexible plugin architecture:

- **Plugin-based**: Different "combo" implementations selectable via configuration
- **Flavor-dependent**: Each quark flavor (u, d, s, c, b, etc.) can have independent parameters
- **Flavor-blind**: All flavors share the same parameters (reduces parameter count)
- **Q-dependent**: Rapidity scale computed dynamically from hard scale

### 2. Grid Utilities (`grids/`)

C++ implementations for TMD grid computation:

- **CollinsSoperKernel**: Collins-Soper evolution kernel implementation
- **GridTMDbT**: TMD grid generation and manipulation
- **Multiple Parameter Sets**: MAP22, MAP24 (FD, FI, HD), PV17, PV19
- **Grid Storage**: Pre-computed grids in `grids/` subdirectory for fast access

### 3. OPE Tools (`ope/`)

Operator Product Expansion implementations:

- **PyTorch Version** (`OPE_EVO_torch.ipynb`): Modern automatic differentiation
- **NumPy Version** (`OPE_EVO_notorch.ipynb`): Pure NumPy implementation
- **Grid Generation**: Tools for generating and managing evolution grids

### 4. Spin Physics (`spin/`)

Calculations for spin-dependent observables:

- **TMD Asymmetries** (`tmds.py`): Transverse momentum dependent spin asymmetries
- **PDIS Asymmetries** (`pdis.py`): Polarized deep inelastic scattering
- **Transversity**: Spin-dependent structure functions
- **Parametrizations**: Input parameter handling and fitting utilities

### 5. Interpolation Tools (`interpol/`)

Supporting utilities for TMD grid interpolation:

- **Akima Interpolation** (`Akima.ipynb`): High-quality smooth interpolation
- **Grid Sampling** (`2Dinterp_gridSample.ipynb`): 2D interpolation methods
- **Python Tools** (`tools.py`): Reusable interpolation utilities

## Directory Structure

```bash
tmd/                          # Root TMD framework directory
├── README.md                  # This file
├── grids/                     # C++ TMD grid utilities
│   ├── CollinsSoperKernel     # Collins-Soper kernel implementation
│   ├── GridTMDbT              # TMD grid generation
│   ├── Makefile               # Build configuration
│   ├── config.yaml            # Grid configuration
│   ├── *Parameters.yaml       # Parameter sets (MAP22, MAP24, PV17, PV19, etc.)
│   └── grids/                 # Generated grid data files
│
├── interpol/                  # Interpolation tools and utilities
│   ├── Akima.ipynb            # Akima interpolation implementation
│   ├── 2Dinterp_gridSample.ipynb
│   └── tools/
│       └── tools.py           # Interpolation utility functions
│
├── ope/                       # OPE evolution implementations
│   ├── OPE_EVO_notorch.ipynb  # OPE evolution without PyTorch
│   └── OPE_EVO_torch.ipynb    # OPE evolution with PyTorch
│
├── spin/                      # Spin asymmetry calculations
│   ├── tmds.py                # TMD spin asymmetries
│   ├── pdis.py                # PDIS spin asymmetries
│   ├── fit.py                 # Fitting utilities
│   ├── input_params.py        # Input parameters
│   ├── default_params.py      # Default parameter values
│   └── Transversity.nb        # Mathematica notebook for transversity
│
└── sidis/                     # Main SIDIS framework (Python)
    ├── main.py                # Main entry point
    ├── __init__.py
    ├── requirements.txt       # Python dependencies
    ├── cards/                 # Configuration files
    │   ├── fNPconfig_base_flavor_blind.yaml
    │   ├── fNPconfig_base_flavor_dep.yaml
    │   ├── fNPconfig_simple.yaml
    │   └── ... (additional configurations)
    │
    ├── model/                 # Core physics models
    │   ├── fnp_config.py      # fNP configuration
    │   ├── fnp_manager.py     # fNP manager
    │   ├── fnp_linked_params.py
    │   ├── ope.py             # OPE implementation
    │   ├── evolution.py       # QCD evolution
    │   ├── ogata.py           # Hankel transforms (Ogata method)
    │   ├── qcf0_tmd.py        # TMD QCF calculations
    │   ├── structure_functions.py
    │   ├── tmd_builder.py     # TMD construction
    │   ├── unpolarized_wrapper.py
    │   ├── utils.py
    │   └── fnp/               # fNP implementations and plugins
    │
    ├── one_d/                 # 1D QCD implementations
    │   ├── qcd_ff_1d.py       # Fragmentation functions
    │   └── qcd_qcf_1d.py      # QCF calculations
    │
    ├── ope/                   # OPE utilities
    │   ├── generate_grids.py
    │   ├── OPE.py
    │   └── README_OPE_grids.md
    │
    ├── qcdlib/                # QCD library
    │   ├── alphaS.py          # Running coupling
    │   ├── dglap.py           # DGLAP evolution
    │   ├── evolution_precalcs.py
    │   ├── kernels.py         # Splitting kernels
    │   ├── mellin.py          # Mellin transforms
    │   ├── params.py          # Parameters
    │   ├── special.py         # Special functions
    │   ├── eweak.py           # Electroweak corrections
    │   └── tmdmodel.py        # TMD model interface
    │
    ├── tests/                 # Test suite
    │   ├── runfit.py          # Test fitting
    │   ├── generate_mock_*.py  # Mock data generation
    │   ├── analyze_fit.ipynb
    │   ├── plots_and_studies.ipynb
    │   └── ... (data and results)
    │
    ├── utilities/             # Utility modules
    │   ├── colors.py          # Terminal colors
    │   └── __init__.py
    │
    └── docs/                  # Documentation
        ├── README.md
        ├── fnp_system.md      # fNP architecture
        ├── fnp_implemented_combos.md
        └── fnp_py_file_structure.md
```

## Configuration

### SIDIS Configuration (`sidis/cards/*.yaml`)

Each fNP configuration file in `sidis/cards/` contains:

- **Combo Selection**: Which fNP implementation to use (`flavor_dep`, `flavor_blind`, etc.)
- **Hadron Type**: Target hadron specification
- **Evolution Parameters**: QCD evolution settings (shared across flavors)
- **PDF Parameters**: Perturbative and non-perturbative PDF parametrizations
  - Flavor-blind: Single `pdf` section (applies to all flavors)
  - Flavor-dependent: `pdfs` section with per-flavor entries
- **Fragmentation Function (FF) Parameters**: Same structure as PDF parameters
- **OPE Grid Files**: Path to pre-computed evolution grids
- **Scale Settings**: Initial scale Q₀² and other hard-scale parameters

The configuration files unify previous separate `config.yaml` and fNP parameter specifications into a single file.

### Grid Configuration (`grids/config.yaml`)

Contains settings for TMD grid generation:

- Grid parameter files (MAP22, MAP24, PV17, PV19)
- Output directories
- Computation settings for Collins-Soper kernel and TMD grids

## Installation & Dependencies

### Python Dependencies (SIDIS Module)

Install Python dependencies for the SIDIS framework:

```bash
cd sidis/
pip install -r requirements.txt
```

Key dependencies:

- **PyTorch**: For automatic differentiation and tensor operations
- **NumPy**: Scientific computing
- **YAML**: Configuration file parsing
- **SciPy**: Special functions and numerical methods

### C++ Dependencies (Grid Utilities)

Building the grid utilities requires:

- **LHAPDF**: Parton distribution functions
- **APFEL++**: QCD evolution library
- **NangaParbat**: TMD physics library
- **GSL**: GNU Scientific Library
- **yaml-cpp**: YAML C++ support

On macOS (with Homebrew):

```bash
brew install gsl yaml-cpp
```

On Linux (Debian/Ubuntu):

```bash
apt-get install libgsl0-dev libyaml-cpp-dev
```

External libraries (LHAPDF, APFEL++, NangaParbat) should be installed separately following their documentation.

## Usage Examples

### Example 1: SIDIS with Flavor-Dependent Parameters

```bash
cd /path/to/tmd
python3 sidis/main.py -c fNPconfig_base_flavor_dep.yaml
```

This computes SIDIS cross-sections using flavor-dependent TMD parametrizations where each quark flavor has independent parameters.

### Example 2: SIDIS with Flavor-Blind Parameters

```bash
python3 sidis/main.py -c fNPconfig_base_flavor_blind.yaml
```

Uses a simplified model where all flavors share the same parameters.

### Example 3: Programmatic Usage (Python)

```python
from sidis.model import TrainableModel
import torch

# Initialize model with specific configuration
model = TrainableModel(fnp_config='fNPconfig_base_flavor_dep.yaml')

# Prepare event data: [x, PhT, Q, z]
# where x = Bjorken-x, PhT = transverse momentum, Q = hard scale, z = fragmentation fraction
events = torch.tensor([
    [0.1, 0.5, 3.0, 0.3],  # Event 1
    [0.2, 0.6, 4.0, 0.4],  # Event 2
])

# Compute cross-sections
results = model(events)
```

### Example 4: TMD Grid Generation

```bash
cd grids/
make
# This builds CollinsSoperKernel and GridTMDbT executables
```

### Example 5: OPE Evolution Calculations

See `ope/OPE_EVO_torch.ipynb` or `ope/OPE_EVO_notorch.ipynb` for:

- OPE matching at initial scales
- QCD evolution of PDFs and fragmentation functions
- Renormalization group solutions

## Documentation

For detailed documentation, see:

- **[fNP System Documentation](fnp_system.md)**: Overall architecture and plugin system
- **[fNP Combos Documentation](fnp_implemented_combos.md)**: Detailed documentation of each combo implementation
- **[File Structure Documentation](fnp_py_file_structure.md)**: Structure of combo files and components

## Troubleshooting

### SIDIS Framework Issues

#### Config File Not Found

If you see an error about a missing config file:

1. Check that the file exists in `sidis/cards/` directory
2. Verify the filename is correct (case-sensitive)
3. Run `python3 sidis/main.py --help` to see available files

#### Import Errors

Ensure you're running from the correct directory:

```bash
# From project root
python3 sidis/main.py -c fNPconfig_base_flavor_dep.yaml

# NOT from sidis/ directory with just:
# python3 main.py  # This might have path issues
```

#### PyTorch/GPU Issues

If PyTorch isn't using GPU:

```python
import torch
print(torch.cuda.is_available())  # Should be True if GPU available
print(torch.get_default_dtype())  # Should be torch.float64
```

### C++ Grid Building Issues

#### Missing External Libraries

If building `grids/` fails with undefined references to LHAPDF, APFEL++, or NangaParbat:

1. Verify these libraries are installed and discoverable
2. Check `pkg-config` paths: `pkg-config --list-all | grep lhapdf` (for example)
3. On macOS, ensure Homebrew environment is properly sourced

#### Compilation Errors

Check that:

- C++11 or later compiler is available: `clang++ --version`
- GSL headers are in standard include paths
- yaml-cpp is properly installed

## Testing

Run the test suite in `sidis/tests/`:

```bash
python3 sidis/tests/runfit.py
python3 sidis/tests/generate_mock_events.py
python3 sidis/tests/generate_mock_kinematics.py
```

See `sidis/tests/*.ipynb` for analysis and visualization notebooks.

## Development

### Adding New fNP Implementations

1. Create a new combo implementation following `sidis/model/fnp/` patterns
2. Register in the plugin system
3. Add configuration file to `sidis/cards/`
4. Update documentation

### Extending the Framework

- **New evolution kernels**: Modify `sidis/qcdlib/kernels.py`
- **New structure functions**: Add to `sidis/model/structure_functions.py`
- **New numerical methods**: Extend `sidis/model/ogata.py` or related modules

## Documentation References

For detailed framework documentation, see:

- **SIDIS Module**:
  - [fNP System Architecture](sidis/docs/fnp_system.md)
  - [fNP Implemented Combos](sidis/docs/fnp_implemented_combos.md)
  - [Python File Structure](sidis/docs/fnp_py_file_structure.md)

- **Grid Utilities**:
  - [OPE Grid Generation README](sidis/ope/README_OPE_grids.md)

- **Notebooks**:
  - [Interpolation Tools](interpol/Akima.ipynb)
  - [OPE Evolution (PyTorch)](ope/OPE_EVO_torch.ipynb)
  - [OPE Evolution (NumPy)](ope/OPE_EVO_notorch.ipynb)

## Contributing

To contribute to the framework:

1. Create a feature branch
2. Implement changes with clear commit messages
3. Add tests for new functionality
4. Update relevant documentation
5. Submit a pull request

Guidelines:

- Follow the existing code style and structure
- Ensure backward compatibility where possible
- Document public functions and classes
- Include docstrings with parameter descriptions

## Authors

Chiara Bissolotti (<cbissolotti@anl.gov>)
Patrick Barry (<pbarry@anl.gov>)
