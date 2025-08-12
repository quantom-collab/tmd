# SIDIS Cross Section Computation

This C++ subproject provides tools for computing Semi-Inclusive Deep Inelastic Scattering (SIDIS) differential cross sections using `NangaParbat`. This is the c++ counterpart to the `python` and `pytorch` code that implements computations.

## Project Structure

```md
cpp/
├── CMakeLists.txt                  # CMake build configuration (auto-detects .cc files)
├── SIDISCrossSectionKinem.cc      # Custom kinematics interface (recommended)
├── SIDISCrossSectionData.cc       # NangaParbat data interface  
├── SIDISMultiplicities.cc         # Multiplicities computation
├── build.sh                       # Quick build script
├── README.md                      # This file
└── build/                         # Build directory (created by build)
    ├── SIDISCrossSectionKinem     # Custom kinematics executable
    ├── SIDISCrossSectionData      # Data interface executable
    └── SIDISMultiplicities        # Multiplicities executable
```

## Building

### Quick Build (Recommended)

Use the provided build script that handles everything automatically:

```bash
# Navigate to the cpp directory
cd /path/to/tmd/map/cpp

# Run the build script
./build.sh
```

### Manual Build

You can also build the project manually using the standard CMake workflow:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j  # Build all .cc files automatically

# Optional: Install system-wide
make install
```

### Build Features

- 🎯 **Automatic Detection**: All `.cc` files in the directory are automatically detected and compiled
- 🔧 **No Hardcoded Paths**: CMake uses `*-config` scripts and pkg-config for dependencies  
- ⚡ **Parallel Building**: Use `-j$(nproc)` for faster compilation
- 📦 **Consistent Linking**: All executables get the same physics library dependencies

## Automatic C++ File Compilation

This is a project built with `CMake`.
**🚀 Key Feature**: The CMake build system automatically detects and compiles **all `.cc` files** found in this directory. Simply add any new `.cc` file to the folder and it will be automatically built into an executable with the same name (without the `.cc` extension).

- No need to manually edit `CMakeLists.txt` for new programs
- All executables automatically get the same physics library dependencies
- Consistent build configuration across all programs

## Available Code Files

### SIDISCrossSectionKinem

Computes SIDIS cross sections using **custom kinematics from YAML files**. Does NOT rely on NangaParbat DataHandler.

- ✅ **Input**: Custom `kinematics.yaml` file format
- ✅ **Output**: Both YAML (detailed + arrays) and TXT formats

Example usage:

```bash
cd tmds/map/cpp/build/
./SIDISCrossSectionKinem ../../inputs/config.yaml ../../inputs/kinematics.yaml ../output
```

**Output**: Creates both `predictions.yaml` (detailed format), `predictions_arrays.yaml` (plotting format), and `predictions.txt` (tabular format):

- **predictions.yaml**: Structured YAML with detailed kinematic information per point
- **predictions_arrays.yaml**: Array-based YAML format optimized for plotting
- **predictions.txt**: Tabular format with columns for easy analysis

### SIDISCrossSectionData

Computes SIDIS cross sections using **NangaParbat DataHandler** starting from a real experimental data file.

- ✅ **Input**: NangaParbat-style data files and `datasets.yaml`
- ✅ **Output**: YAML format with experimental data comparison

Example usage:

```bash
cd tmds/map/cpp/build/
./SIDISCrossSectionData ../../inputs/config.yaml ../data/ ../output
```

**Input Requirements**:

- `data_folder/datasets.yaml` - The data folder must have a file called `datasets.yaml` that contains a specifically formatted list of all the datafiles the user wants to include in the calculation.
- `data_folder/experiment_name/datafile.yaml` - Individual data files in yaml format and with a structure compatible with the `NangaParbat`  DataHandler.

Output:

- **ReportCrossSectData.yaml**: YAML format
  
### SIDISMultiplicities

Computes SIDIS multiplicities as ratios of cross sections.

- ✅ **Input**: NangaParbat-style data format
- ✅ **Output**: Multiplicities calculations

Example usage:

```bash
cd tmds/map/cpp/build/
./SIDISMultiplicities ../../inputs/config.yaml ../data/ ../output
```

Output:

- **ReportMultiplicities.yaml**: YAML format with experimental data comparison

## Dependencies

The following libraries must be installed on your system:

- **LHAPDF** (≥6.0): Parton Distribution Functions
- **APFEL++** (≥4.0): QCD evolution and TMD framework  
- **NangaParbat**: TMD phenomenology framework
- **yaml-cpp**: YAML configuration file parsing
- **GSL** (GNU Scientific Library): Mathematical functions and integration
- **CMake** (≥3.12): Build system

## Kinematics File Format (for SIDISCrossSectionKinem)

The custom kinematics YAML file should have the following structure:

```yaml
header:
  process: "SIDIS"
  observable: "cross_section"
  target_isoscalarity: 1
  hadron: "PI"
  charge: 1
  Vs: 7.2565449
  PS_reduction:
    W: 3.1622777
    ymin: 0.1
    ymax: 0.85

data:
  PhT: [0.1030419, 0.2055688, 0.3056045, ...]  # GeV
  x: [0.03758844, 0.03758844, 0.03758844, ...]
  z: [0.5334359, 0.5370032, 0.5377358, ...]
  Q2: [1.249727, 1.249727, 1.249727, ...]     # GeV^2  
  y: [0.63139491, 0.63139491, 0.63139491, ...]
```

## Configuration File Format

The YAML configuration file specifies computation parameters:

```yaml
# Perturbative order
PerturbativeOrder: 2

# PDF set
pdfset:
  name: "MMHT2014nnlo68cl"  
  member: 0

# FF set  
ffset:
  name: "MAPFF10NNLOPIp"
  member: 0

# TMD scales
TMDscales:
  Ci: 1.0
  Cf: 1.0

# Electromagnetic coupling
alphaem:
  aref: 0.007297352566417111
  Qref: 91.1876

# X-grids for PDFs and FFs
xgridpdf:
  - [100, 9e-5, 3]
  - [60, 1e-1, 3]
  # ...

xgridff:
  - [100, 1e-4, 3]  
  - [60, 1e-1, 3]
  # ...
```

## Author

- Chiara Bissolotti: <chiara.bissolotti01@gmail.com>
