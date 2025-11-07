# fNP System Documentation

The fNP (non-perturbative) system provides a plugin-based architecture for managing TMD PDF and TMD FF non-perturbative parameterizations in the SIDIS cross-section computation framework.
The system is designed to be flexible, allowing different combinations of PDF and FF parameterizations to be selected via configuration files.

- [fNP System Documentation](#fnp-system-documentation)
  - [Architecture Key Components](#architecture-key-components)
    - [1. Python Module Files `fnp_<name>.py`](#1-python-module-files-fnp_namepy)
    - [2. Configuration ('combo') Files `fNPconfig_<name>.yaml`](#2-configuration-combo-files-fnpconfig_nameyaml)
    - [3. Factory (`fnp_factory.py`)](#3-factory-fnp_factorypy)
  - [Usage](#usage)
    - [Command Line Usage](#command-line-usage)
    - [Programmatic Usage](#programmatic-usage)
  - [Creating New Combos](#creating-new-combos)
  - [See Also](#see-also)

## Architecture Key Components

The fNP system uses a **plugin architecture** where different files can be selected via configuration. Each file contains a specific combination of fNP evolution, TMD PDF and TMD FF non-perturbative parameterizations.

### 1. Python Module Files `fnp_<name>.py`

Each python module file contains everything needed for that parameterization:

- **Evolution class**: `fNP_evolution` - Shared evolution factor
- **PDF class**: TMD PDF parameterization (e.g., `TMDPDFBase`, `TMDPDFFlavorBlind`)
- **FF class**: TMD FF parameterization (e.g., `TMDFFBase`, `TMDFFFlavorBlind`)
- **Default parameters**: Default initialization values
- **Manager class**: `fNPManager` - Orchestrates the combo components, i.e. manages fNP evolution, fNP for TMD PDFs and fNP for TMD FFs all together
  
For more details, see the [fNP Python Module Files Documentation](fnp_py_file_structure.md).

### 2. Configuration ('combo') Files `fNPconfig_<name>.yaml`

**All the configuration files must be in the `cards/` directory.**
Available combos:

- `"flavor_dep"`: Flavor-dependent system (each flavor has own parameters) - in `fNPconfig_base_flavor_dep.yaml`
- `"flavor_blind"`: Flavor-blind system (all flavors share parameters) - in `fNPconfig_base_flavor_blind.yaml`

fNP Configuration File Structure

```yaml
# Global settings
hadron: proton

# Combo selection
combo: flavor_blind  # or "flavor_dep"

# Evolution parameters
evolution:
  init_g2: 0.12840
  free_mask: [true]

# PDF parameters (flavor-blind: single set for all flavors)
pdf:
  init_params: [0.28516, 2.9755, ...]
  free_mask: [true, true, ...]

# FF parameters (flavor-blind: single set for all flavors)
ff:
  init_params: [0.21012, 2.12062, ...]
  free_mask: [true, true, ...]
```

See the configuration files in `cards/` for examples.

### 3. Factory (`fnp_factory.py`)

The factory creates fNP manager instances:

- **Entry point**: `create_fnp_manager(config_path)` or `create_fnp_manager(config_dict=...)`
- **Functionality**:
  - Reads configuration file (e.g. `fNPconfig_base_flavor_dep.yaml`)
  - Selects combo module based on `combo` field
  - Imports `fNPManager` class from the combo module
  - Instantiates and returns the manager from that combo module (e.g. `fNPManager` from `fnp_base_flavor_dep.py`)

## Usage

### Command Line Usage

The fNP configuration can be specified from the command line when running `main.py`:

```bash
# Use flavor-dependent config
python3 sidis/main.py -c fNPconfig_base_flavor_dep.yaml

# Use flavor-blind config (short form)
python3 sidis/main.py -c fNPconfig_base_flavor_blind.yaml

# Use default (flavor-blind, shows warning)
python3 sidis/main.py
```

The config file is automatically looked up in the `cards/` directory. If the file is not found, an error message with instructions is displayed.

### Programmatic Usage

```python
from sidis.model.fnp_factory import create_fnp_manager

# Create manager from config file
manager = create_fnp_manager("fNPconfig_base_flavor_blind.yaml")

# Or from dictionary
config = {"combo": "flavor_blind", "hadron": "proton", ...}
manager = create_fnp_manager(config_dict=config)

# Evaluate fNP
x = torch.tensor([0.1, 0.2])  # Bjorken x
z = torch.tensor([0.3, 0.4])  # Energy fraction
b = torch.tensor([0.5, 0.6])  # Impact parameter
Q = torch.tensor([2.0, 3.0])   # Hard scale

result = manager.forward(x, z, b, Q)
pdfs = result["pdfs"]  # Dict[str, torch.Tensor]
ffs = result["ffs"]    # Dict[str, torch.Tensor]
```

## Creating New Combos

To create a new combo implementation:

1. **Create new python module**: Create a new Python module, for example following the structure of `fnp_base_flavor_dep.py` or `fnp_base_flavor_blind.py`.

2. **Implement required classes**:
   - `fNP_evolution`: Evolution factor class
   - `TMDPDFBase` (or custom name): PDF parameterization class
   - `TMDFFBase` (or custom name): FF parameterization class
   - Default parameter dictionaries (e.g. `MAP22_DEFAULT_EVOLUTION`, `MAP22_DEFAULT_PDF_PARAMS`, `MAP22_DEFAULT_FF_PARAMS`)
   - `fNPManager` class: Manager class that orchestrates the combo components

3. **Update factory**: Add your new combo to the factory mapping in `fnp_factory.py`.

4. **Create config file in `cards/` directory**:
   - File name: `fNPconfig_<name>.yaml` (e.g. `fNPconfig_custom.yaml`)
   - Content: See example in `cards/fNPconfig_base_flavor_dep.yaml` or `cards/fNPconfig_base_flavor_blind.yaml`
   - Users can now select your combo via `combo: <name>` in config.

## See Also

- [fNP Combos Documentation](fnp_combos.md): Detailed documentation of each combo implementation
- [SIDIS Model Documentation](README.md): Overall model architecture
