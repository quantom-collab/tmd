#!/usr/bin/env python3
"""
SIDIS Cross Section Computation using PyTorch and APFEL++ - Complete Implementation

This script computes Semi-Inclusive Deep Inelastic Scattering (SIDIS) differential
cross sections using a hybrid approach that combines:
1. APFEL++ library for TMD evolution and matching
2. PyTorch for non-perturbative function modeling and gradient computation

PHYSICS OVERVIEW:
================
SIDIS is the process: e + N → e' + h + X
where an electron scatters off a nucleon N, producing a hadron h and other particles X.

The differential cross section in TMD factorization is:
...
- f₁(x,bT): TMD PDF (parton distribution with transverse momentum dependence)
- D₁(z,bT): TMD FF (fragmentation function with transverse momentum dependence)
- S(bT): Sudakov factor (soft gluon resummation)
- H(μ): Hard factor (partonic cross section)
- bT: Fourier conjugate to qT

KINEMATIC VARIABLES:
===================
- x: Bjorken scaling variable (momentum fraction of struck parton)
- Q²: Photon virtuality (hard scale of the process)
- z: Energy fraction of produced hadron
- PhT: Transverse momentum of produced hadron
- qT = PhT/z: Intrinsic transverse momentum

TMD EVOLUTION:
==============
TMDs satisfy coupled evolution equations:
- μ² d/dμ² TMD = γμ ⊗ TMD (DGLAP-like evolution)
- ζ d/dζ TMD = -D(μ,bT) TMD (Collins-Soper evolution)

The evolution kernels are computed perturbatively, while non-perturbative
contributions are modeled using the fNP functions.

PYTORCH INTEGRATION:
===================
This implementation uses PyTorch for:
- fNP function evaluation with 34 trainable parameters
- Automatic differentiation for gradient-based fitting
- GPU acceleration (CUDA/Metal support)
- Tensor operations for efficient computation

The code maintains full compatibility with APFEL++ while adding PyTorch
capabilities for machine learning applications in TMD physics.

USAGE:
======
python3.10 sidis_computation_pytorch.py <config_file> <kinematic_data_file> <fnp_config_file> <output_folder> <output_filename>

Example:
python3.10 sidis_computation_pytorch.py inputs/config.yaml inputs/kinematics.yaml inputs/fNPconfig.yaml results/ sidis_pytorch.yaml

REQUIREMENTS:
=============
- Python 3.10+
- PyTorch 2.0+ (with GPU support optional)
- APFEL++ Python bindings (apfelpy)
- LHAPDF 6.5+
- NumPy, YAML, argparse

AUTHORS: Chiara Bissolotti (cbissolotti@anl.gov)
LICENSE: Academic use
VERSION: 2.0 (PyTorch-enabled)
"""

import os
import sys
import yaml
import torch
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any, TYPE_CHECKING, Optional

# Type annotations for LHAPDF - using Any to avoid linter errors
import lhapdf as lh

if TYPE_CHECKING:
    LHAPDF_PDF = Any
else:
    LHAPDF_PDF = Any

# Import apfelpy
import apfelpy as ap

# Import custom modules
import modules.utilities as utl
from modules.fNP import fNP


class SIDISComputationPyTorch:
    """
    PyTorch-based SIDIS cross section computation

    This class implements the SIDIS cross section computation using PyTorch tensors
    throughout for efficient computation, automatic differentiation support, and
    proper integration with the fNP module.

    Key PyTorch features:
    - All kinematic variables stored as tensors
    - Vectorized operations where possible
    - GPU support
    - Integration with fNP PyTorch module
    - Gradient computation support for optimization
    """

    def __init__(
        self, config_file: str, fnp_config_file: str, device: Optional[str] = None
    ):
        """
        Initialize the PyTorch-based computation with configuration files.

        Args:
            @param config_file: Path to global configuration YAML
            @param fnp_config_file: Path to fNP model configuration YAML
            @param device: Optional explicit device string ('cpu','cuda','mps')

        Notes:
            - Auto-detects GPU (CUDA) then MPS else CPU
            - Sets float32 default dtype (upgradeable later)
        TODO:
            - Validate cross-consistency of config and fNP config (PDF flavor sets)
        """
        self.config = self._load_config(config_file)
        self.fnp_config_file = fnp_config_file

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"\033[94mUsing CUDA GPU\n\033[0m")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print(f"\033[94mUsing Apple Metal GPU\n\033[0m")
            else:
                self.device = torch.device("cpu")
                print(f"\033[94mUsing CPU\n\033[0m")
        else:
            self.device = torch.device(device)

        # Set default dtype
        self.dtype = torch.float32

        # Call setup method to initialize all components,
        # mainly the ones read from the config file
        self.setup_computation()

    def _load_config(self, config_file: str) -> Dict:
        """
        Load YAML configuration.
        Args:
            @param config_file: Path to configuration YAML

        Returns:
            dict: Parsed configuration (empty dict on failure)

        TODO:
            - Add schema validation / default injections
            - Emit warnings
        """
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Checks at runtime that config is exactly
        # (or a subclass of) dict. Returns either
        # the parsed dict or a safe fallback of {}
        return config if isinstance(config, dict) else {}

    def setup_computation(self):
        """
        Top-level construction of all APFEL & PyTorch objects.
        This method orchestrates the entire setup process, ensuring all
        components are initialized correctly for the SIDIS computation.
        Steps:
            1. Extract config parameters
            2. Setup collinear PDFs & thresholds
            3. Setup couplings (alpha_s, alpha_em)
            4. Build TMD PDF objects (evolution, matching, Sudakov, hard factor)
            5. Build FF & TMD FF objects
            6. Initialize PyTorch fNP model
            7. Initialize Ogata quadrature (b-space transform)

        TODO:
            - Timing / profiling
        """

        # Extract config parameters
        self.PerturbativeOrder = self.config["PerturbativeOrder"]
        self.Ci = self.config["TMDscales"]["Ci"]
        self.Cf = self.config["TMDscales"]["Cf"]
        self.qToQcut = self.config.get("qToQcut", 0.3)

        # Setup PDF and FF
        self._setup_pdf()
        self._setup_couplings()
        self._setup_tmd_pdf_objects()
        self._setup_ff()
        self._setup_tmd_ff_objects()

        # Setup PyTorch-based fNP model
        self._setup_fnp_pytorch()

        # Ogata parameters: (order = 0, eps = 1e-9, rel = 1e-5)
        # Revisit for precision/perf tuning
        self.DEObj = ap.ogata.OgataQuadrature(0, 1e-9, 0.00001)

        print(f"\033[92m\n --- PyTorch SIDIS computation setup successful!\n\033[0m")

    def _setup_pdf(self):
        """
        Setup collinear PDFs objects with PyTorch integration.
        The PDFs encode the probability of finding a parton with given momentum
        fraction x inside a nucleon at scale μ.

        Key components:
            - LHAPDF interface for PDF sets (e.g., MMHT2014nnlo68cl)
            - Flavor rotation from physical to QCD evolution basis [RotPDFs: Phys → QCD]
            - Quark mass thresholds for heavy flavor treatment
            - Grid setup for numerical integration

        Notes:
            - mu0: derived from q2Min (TODO: allow explicit override)
        """

        # Get PDF set from configuration
        pdf_name = self.config["pdfset"]["name"]
        pdf_member = self.config["pdfset"]["member"]

        # Initialize LHAPDF PDF set
        self.pdf: LHAPDF_PDF = lh.mkPDF(pdf_name, pdf_member)  # type: ignore

        # Function to rotate from physical basis (u,d,s,c,b,t,g) to QCD evolution basis
        # This is needed because QCD evolution equations are simpler in the evolved basis
        self.RotPDFs = lambda x, mu: ap.PhysToQCDEv(self.pdf.xfxQ(x, mu))

        # Extract quark mass thresholds from PDF set
        # These determine when heavy quarks become active in evolution
        thresholds_list = []
        for v in self.pdf.flavors():
            if v > 0 and v < 7:  # Only quarks (not antiquarks or gluons)
                thresholds_list.append(self.pdf.quarkThreshold(v))

        # Store thresholds in both list (for APFEL) and tensor (for PyTorch) format
        self.Thresholds = thresholds_list  # Keep as list for APFEL compatibility
        self.Thresholds_tensor = torch.tensor(
            thresholds_list, dtype=self.dtype, device=self.device
        )

        # Quark masses (used in evolution kernels)
        # Masses for u,d,s,c,b in GeV (u,d,s are effectively massless)
        self.Masses = [0, 0, 0, self.pdf.quarkThreshold(4), self.pdf.quarkThreshold(5)]

        # Setup interpolation grid for x-space
        # This defines the grid points for numerical integration over momentum fraction
        self.gpdf = ap.Grid(
            [ap.SubGrid(*subgrids) for subgrids in self.config["xgridpdf"]]
        )

        # Initial scale for evolution (typically \mu_0 ~ 1 GeV)
        self.mu0 = np.sqrt(self.pdf.q2Min)

        print(f"\033[92m\n --- PDF setup successful, PDF set: {pdf_name}\n\033[0m")

    def _setup_couplings(self):
        """
        Setup \alpha_s and \alpha_em tabulations.

        Notes:
            - TabulateObject: (N=100, μ-range scaled) cubic interpolation
            - LeptThresholds: e, μ taken massless; τ included. Hardcoded as in NangaParbat.
        """
        # Extract alpha strong from PDF set
        Alphas = lambda mu: self.pdf.alphasQ(mu)

        # Tabulate α_s with 100 points in the range [0.9 * sqrt(q2Min), sqrt(q2Max)]
        self.TabAlphas = ap.TabulateObject(
            Alphas,
            100,
            np.sqrt(self.pdf.q2Min) * 0.9,
            np.sqrt(self.pdf.q2Max),
            3,
            self.Thresholds,
        )

        # Alpha_em
        aref = self.config["alphaem"]["aref"]
        Qref = self.config["alphaem"]["Qref"]
        # Leptonic thresholds: e, μ treated as massless, τ included
        # They are hardcoded also in NangaParbat, l. 84 of SIDISMultiplicities.cc
        LeptThresholds = [0.0, 0.0, 1.777]

        # Store reference values as tensors
        self.aref_tensor = torch.tensor(aref, dtype=self.dtype, device=self.device)
        self.Qref_tensor = torch.tensor(Qref, dtype=self.dtype, device=self.device)

        alphaem = ap.AlphaQED(
            AlphaRef=aref,
            MuRef=Qref,
            LeptThresholds=LeptThresholds,
            QuarkThresholds=self.Thresholds,
            pt=0,
        )
        self.TabAlphaem = ap.TabulateObject(alphaem, 100, 0.9, 1001, 3)

    def _setup_tmd_pdf_objects(self):
        """
        Setup TMD PDF objects for transverse momentum dependent evolution

        This method initializes the complete TMD machinery:
        1. DGLAP evolution objects for collinear PDF evolution
        2. TMD evolution objects for transverse momentum dependence
        3. Matching conditions between collinear and TMD PDFs
        4. Sudakov factor for soft gluon resummation
        5. Hard factor for the partonic process

        TMD PDFs satisfy the evolution equation:
        μ² d/dμ² f₁(x,b_T;μ,ζ) = γ_μ ⊗ f₁(x,b_T;μ,ζ)
        ζ d/dζ f₁(x,b_T;μ,ζ) = -D(μ,b_T) f₁(x,b_T;μ,ζ)

        where γ_μ is the anomalous dimension and D is the Collins-Soper kernel.
        """
        # Initialize DGLAP evolution objects for space-like (PDF) evolution
        # DGLAP equations govern the μ dependence of collinear PDFs
        DglapObj = ap.initializers.InitializeDglapObjectsQCD(
            self.gpdf, self.Masses, self.Thresholds
        )

        # Build evolved PDFs using DGLAP evolution
        # This evolves the input PDFs from μ₀ to arbitrary scale μ
        EvolvedPDFs = ap.builders.BuildDglap(
            DglapObj,
            lambda x, mu: ap.utilities.PhysToQCDEv(self.pdf.xfxQ(x, mu)),
            self.mu0,  # Initial scale
            self.pdf.orderQCD,  # Perturbative order
            self.TabAlphas.Evaluate,  # Strong coupling
        )

        # Tabulate evolved PDFs for fast evaluation
        # This creates interpolation tables for efficient computation
        self.TabPDFs = ap.TabulateObjectSetD(
            EvolvedPDFs, 100, np.sqrt(self.pdf.q2Min) * 0.9, np.sqrt(self.pdf.q2Max), 3
        )
        self.CollPDFs = lambda mu: self.TabPDFs.Evaluate(mu)

        # Initialize TMD objects for b_T-dependent evolution
        self.TmdObjPDF = ap.tmd.InitializeTmdObjects(self.gpdf, self.Thresholds)

        # Build evolved TMD PDFs with full μ and ζ dependence
        # TMD evolution includes both DGLAP-like (μ) and Collins-Soper (ζ) evolution
        self.EvTMDPDFs = ap.tmd.BuildTmdPDFs(
            self.TmdObjPDF,
            self.CollPDFs,
            self.TabAlphas.Evaluate,
            self.PerturbativeOrder,  # Perturbative order (NLL, NNLL, etc.)
            self.Ci,  # Initial scale parameter
        )

        # Build matching coefficients between collinear and TMD PDFs
        # At b_T → 0, TMD PDFs reduce to collinear PDFs times matching coefficients
        self.MatchTMDPDFs = ap.tmd.MatchTmdPDFs(
            self.TmdObjPDF,
            self.CollPDFs,
            self.TabAlphas.Evaluate,
            self.PerturbativeOrder,
            self.Ci,
        )

        # Quark Sudakov factor: encodes soft gluon resummation
        # This is the exponential of the integrated Collins-Soper kernel
        # S(b_T;μ,ζ) = exp(-∫ dln μ' γ_K(α_s(μ')))
        self.QuarkSudakov = ap.tmd.QuarkEvolutionFactor(
            self.TmdObjPDF,
            self.TabAlphas.Evaluate,
            self.PerturbativeOrder,
            self.Ci,
            1e5,  # Upper integration limit for Sudakov
        )

        # Hard factor: partonic cross section for the hard process
        # For SIDIS: e + q → e + q (+ soft gluons)
        # Includes virtual corrections and depends on the specific process
        self.Hf = ap.tmd.HardFactor(
            "SIDIS",  # Process type
            self.TmdObjPDF,
            self.TabAlphas.Evaluate,
            self.PerturbativeOrder,
            self.Cf,  # Final scale parameter
        )

    def _setup_ff(self):
        """
        Setup Fragmentation Function (FF) objects

        Fragmentation functions D₁^h(z,μ²) describe the probability that a parton
        with energy E fragments into a hadron h carrying energy fraction z=E_h/E.

        For SIDIS, we need FFs for the produced hadron (e.g., π⁺, π⁻, K⁺, etc.)

        Key aspects:
        - Time-like evolution (opposite to space-like PDF evolution)
        - Sum rules: Σ_h ∫ dz D₁^h(z,μ²) = 1 (probability conservation)
        - Flavor dependence: u → π⁺ vs d → π⁺ have different probabilities
        """
        # Get collinear FFs configuration file
        ff_name = self.config["ffset"]["name"]
        ff_member = self.config["ffset"]["member"]

        # Initialize LHAPDF FF set
        self.distff: LHAPDF_PDF = lh.mkPDF(ff_name, ff_member)  # type: ignore

        # Rotation from physical to QCD evolution basis (same as PDFs)
        self.RotFFs = lambda x, mu: ap.PhysToQCDEv(self.distff.xfxQ(x, mu))

        # Setup z-space grid for numerical integration
        self.gff = ap.Grid(
            [ap.SubGrid(*subgrids) for subgrids in self.config["xgridff"]]
        )

        # Print
        print(f"\033[92m\n --- FF setup successful, FF set: {ff_name}\n\033[0m")

    def _setup_tmd_ff_objects(self):
        """
        Build TMD FF evolution & matching objects.

        Notes:
            - Separate DGLAP object for time-like evolution
            - Similar tabulation density to PDFs
            - Uses same thresholds as PDFs (heavy quark treatment)
        """
        DglapObjFF = ap.initializers.InitializeDglapObjectsQCD(
            self.gff, self.Masses, self.Thresholds
        )

        # Build DGLAP objects for FFs
        EvolvedFFs = ap.builders.BuildDglap(
            DglapObjFF,
            lambda x, mu: ap.utilities.PhysToQCDEv(self.distff.xfxQ(x, mu)),
            self.mu0,
            self.distff.orderQCD,
            self.TabAlphas.Evaluate,
        )

        # Tabulate collinear FFs
        self.TabFFs = ap.TabulateObjectSetD(
            EvolvedFFs,
            100,
            np.sqrt(self.distff.q2Min) * 0.9,
            np.sqrt(self.distff.q2Max),
            3,
        )
        self.CollFFs = lambda mu: self.TabFFs.Evaluate(mu)

        # Initialize TMD FF objects
        self.TmdObjFF = ap.tmd.InitializeTmdObjects(self.gff, self.Thresholds)

        # Build evolved TMD FFs
        self.EvTMDFFs = ap.tmd.BuildTmdFFs(
            self.TmdObjFF,
            self.CollFFs,
            self.TabAlphas.Evaluate,
            self.PerturbativeOrder,
            self.Ci,
        )
        self.MatchTMDFFs = ap.tmd.MatchTmdFFs(
            self.TmdObjFF,
            self.CollFFs,
            self.TabAlphas.Evaluate,
            self.PerturbativeOrder,
            self.Ci,
        )

    def _setup_fnp_pytorch(self):
        """
        Instantiate PyTorch non-perturbative (fNP) model.
        This method initializes the fNP model for non-perturbative function using the PyTorch
        module defined in modules/fNP.py.

        Notes:
            - On failure, Gaussian fallback used (exp(-a b^2))
            - Reports total & trainable parameter counts
        TODO:
            - Allow parameter freezing via config
        """
        try:
            # Use the provided fNP config file
            if os.path.exists(self.fnp_config_file):
                # Print the path of the fNP configuration file being loaded in green
                print(
                    f"\033[95m\nLoading fNP configuration from {self.fnp_config_file}\033[0m"
                )
                config_fnp = utl.load_yaml_config(self.fnp_config_file)

                # Initialize PyTorch fNP model.
                # Set self.model_fNP to the fNP instance from the fNP module
                # with the provided fNP configuration. The following line does two things:
                # 1. fNP(config_fnp) instantiates the model/class fNP with the given config
                # 2. .to(self.device) moves all its parameters and buffers to the device stored
                # in self.device (commonly something like torch.device("cuda"), "cpu", or "mps").
                # It returns the same module (after moving it), which you assign to self.model_fNP
                self.model_fNP = fNP(config_fnp).to(self.device)
                print("✅ PyTorch fNP model loaded successfully")

                # Use the enhanced parameter analysis from the fNP module
                self.model_fNP.print_parameter_summary()

                # Additional information for debugging/development
                print("📊 Additional PyTorch Information:")
                total_params = sum(p.numel() for p in self.model_fNP.parameters())
                pytorch_trainable = sum(
                    p.numel() for p in self.model_fNP.parameters() if p.requires_grad
                )
                print(f"   PyTorch total parameters: {total_params}")
                print(f"   PyTorch requires_grad=True: {pytorch_trainable}")

                # # Check
                # # Print parameter names and values
                # for name, param in self.model_fNP.named_parameters():
                #     print(f"   - {name}: {param.numel()}, {param}: {param.data}")

            else:
                print(f"Warning: fNP config not found at {self.fnp_config_file}")
                self.model_fNP = None

        except Exception as e:
            print(f"⚠️  Warning: Could not load PyTorch fNP model: {e}")
            self.model_fNP = None

    def bstar_min(self, b: float, Q: float) -> float:
        """
        bstar prescription for TMD evolution - regulates large-b behavior

        The bstar prescription provides a smooth interpolation between the perturbative
        (small-b) and non-perturbative (large-b) regions. It's essential for TMD
        factorization to work properly.

        Formula: b* = b_max * [(1 - exp(-(b/b_max)^4)) / (1 - exp(-(b/b_min)^4))]^(1/4)

        Where:
        - b_max = 2*exp(-γ_E)/μ_F (sets the boundary of perturbative region)
        - b_min = b_max/Q (ensures proper Q dependence)
        - γ_E = Euler-Mascheroni constant

        Args:
            @param b: impact parameter |b_T| (GeV^-1)
            @param Q: hard scale (GeV)
        Returns:
            bstar value in GeV^-1
        """
        # Physical constants
        muF = 1.0  # GeV (factorization scale)
        gamma_E = 0.5772156649015329  # Euler-Mascheroni constant
        power = 4.0  # Power in bstar prescription

        # Calculate boundary scales
        bmax = 2 * np.exp(-gamma_E) / muF  # GeV^-1
        bmin = bmax / Q  # GeV^-1

        # bstar prescription formula
        numerator = 1 - np.exp(-((b / bmax) ** power))
        denominator = 1 - np.exp(-((b / bmin) ** power))

        return bmax * (numerator / denominator) ** (1 / power)

    def bstar_min_pytorch(self, b: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Pytorch tensor b* prescription (autograd-capable).

        Mirrors scalar bstar_min; kept separate for clarity.

        Args:
            @param b: tensor of |b_T| values (GeV^-1)
            @param Q: tensor of hard scales (GeV)
        Returns:
            torch.Tensor: regulated b* values
        """
        # Constants
        muF = torch.tensor(1.0, dtype=self.dtype, device=self.device)  # GeV
        gamma_E = torch.tensor(0.5772156649015329, dtype=self.dtype, device=self.device)
        power = torch.tensor(4.0, dtype=self.dtype, device=self.device)

        bmax = 2 * torch.exp(-gamma_E) / muF  # GeV^-1
        bmin = bmax / Q  # GeV^-1

        num = 1 - torch.exp(-((b / bmax) ** power))
        den = 1 - torch.exp(-((b / bmin) ** power))

        return bmax * (num / den) ** (1 / power)

    def setup_isoscalar_tmds(self, Vs: float, targetiso: float):
        """
        Setup isoscalar TMD PDFs and FFs with PyTorch tensor integration

        In nuclear targets, we need to average over proton and neutron contributions.
        The isoscalar approximation assumes:
        - Target contains both protons and neutrons
        - Nuclear effects can be approximated by simple averaging
        - Isospin symmetry relates proton and neutron TMDs

        For a target with isoscalarity parameter τ:
        - τ = +1: pure proton target
        - τ = -1: pure neutron target
        - τ = 0: symmetric nuclear target (equal p and n)

        The isoscalar TMDs are constructed as:
        f₁^(iso)(x,b_T) = f_p * f₁^p(x,b_T) + f_n * f₁^n(x,b_T)

        where f_p + f_n = 1 are the proton/neutron fractions.

        Args:
            @param Vs: √s center-of-mass energy (GeV)
            @param targetiso: isoscalarity parameter τ (−1 ≤ τ ≤ +1)

        Notes:
            - frp = |τ|, frn = 1 − |τ|
            - sign determines proton↔neutron flavor rotations
            - b range tabulated: [1e-2, 2] GeV^-1 (TODO: validate coverage vs Ogata nodes)
        """
        # Convert to PyTorch tensors for potential gradient computation
        self.Vs_tensor = torch.tensor(Vs, dtype=self.dtype, device=self.device)
        targetiso_tensor = torch.tensor(targetiso, dtype=self.dtype, device=self.device)

        # Calculate isoscalarity factors using PyTorch operations
        # This allows gradients to flow through if needed for uncertainty quantification.
        # This is the Pytorch equivalent of the C++ code:
        # const int sign = (targetiso >= 0 ? 1 : -1);
        sign = torch.where(
            targetiso_tensor >= 0,
            torch.ones_like(targetiso_tensor),
            -torch.ones_like(targetiso_tensor),
        )

        # Proton and neutron fractions
        self.frp = targetiso_tensor.abs()  # Proton fraction |τ|
        self.frn = 1 - self.frp  # Neutron fraction (1 - |τ|)
        self.sign = sign  # Sign of isoscalarity

        # Setup tabulation functions for b_T integration
        # These are needed for the Ogata quadrature integration
        def TabFunc(b: float) -> float:
            """Tabulation function: maps b_T to logarithmic variable"""
            return np.log(b)

        def InvTabFunc(y: float) -> float:
            """Inverse tabulation function: maps back to b_T"""
            return np.exp(y)

        # Define isoscalar TMD PDFs combining proton and neutron contributions
        def isTMDPDFs(b):
            """
            Construct isoscalar TMD PDFs from proton and neutron TMDs

            This function implements the isospin averaging:
            f₁^u(iso) = f_p * f₁^u(p) + f_n * f₁^u(n)
            f₁^d(iso) = f_p * f₁^d(p) + f_n * f₁^d(n)

            Note: f₁^u(n) = f₁^d(p) by isospin symmetry (up in neutron = down in proton)
            """
            # Get TMD PDFs in QCD evolution basis
            xF = ap.utilities.QCDEvToPhys(self.MatchTMDPDFs(b).GetObjects())

            # Extract sign values as integers for array indexing
            s = int(sign.item())  # ±1
            s2 = int((sign * 2).item())  # ±2

            # Construct isoscalar combination for each flavor
            xFiso = {}

            # u and ubar quarks (flavor indices ±1)
            xFiso[1] = self.frp * xF[s] + self.frn * xF[s2]  # u quark
            xFiso[-1] = self.frp * xF[-s] + self.frn * xF[-s2]  # u antiquark

            # d and dbar quarks (flavor indices ±2)
            xFiso[2] = self.frp * xF[s2] + self.frn * xF[s]  # d quark
            xFiso[-2] = self.frp * xF[-s2] + self.frn * xF[-s]  # d antiquark

            # Heavy quarks and gluons (no isospin effects)
            for i in range(3, 7):
                ip = int((i * sign).item())
                xFiso[i] = xF[ip]  # Heavy quark
                xFiso[-i] = xF[-ip]  # Heavy antiquark

            return ap.SetD(xFiso)

        # Tabulate isoscalar TMD PDFs for fast evaluation
        self.TabMatchTMDPDFs = ap.TabulateObjectSetD(
            isTMDPDFs,
            200,  # Number of grid points
            1e-2,
            2,  # b_T range: 0.01 to 2 GeV⁻¹
            1,  # Interpolation degree
            [],  # No thresholds for b_T
            TabFunc,
            InvTabFunc,  # Tabulation functions
        )

        # Define isoscalar TMD FFs (simpler - usually no nuclear effects in fragmentation)
        def isTMDFFs(b):
            """
            Isoscalar TMD FFs - typically same as single nucleon

            Fragmentation happens in vacuum after the struck quark leaves the target,
            so nuclear effects are usually minimal. We simply use the fragmentation
            functions as-is without isospin averaging.
            """
            return ap.SetD(ap.utilities.QCDEvToPhys(self.MatchTMDFFs(b).GetObjects()))

        # Tabulate isoscalar TMD FFs
        self.TabMatchTMDFFs = ap.TabulateObjectSetD(
            isTMDFFs,
            200,  # Number of grid points
            1e-2,
            2,  # b_T range: 0.01 to 2 GeV⁻¹
            1,  # Interpolation degree
            [],  # No thresholds
            TabFunc,
            InvTabFunc,  # Tabulation functions
        )

    def load_kinematic_data_pytorch(self, data_file: str) -> Dict[str, Any]:
        """
        Load kinematic YAML data and convert arrays to tensors.

        Args:
            @param data_file: path to kinematic YAML file
        Returns:
            dict: keys include tensors x,Q2,z,PhT plus header/raw_data
        TODO:
            - Validate presence & lengths of all arrays
            - Unit handling (assert dimensionless / GeV^2 where expected)
        """
        with open(data_file, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return {}

        # Convert kinematic arrays to PyTorch tensors
        kinematic_data = {}
        for key in ["x", "Q2", "z", "PhT"]:
            if key in data["data"]:
                kinematic_data[key] = torch.tensor(
                    data["data"][key], dtype=self.dtype, device=self.device
                )

        # Include header information (keep as regular dict)
        kinematic_data["header"] = data["header"]
        kinematic_data["raw_data"] = data["data"]

        return kinematic_data

    def compute_fnp_pytorch(
        self, x: torch.Tensor, b: torch.Tensor, flavor: str
    ) -> torch.Tensor:
        """
        Evaluate non-perturbative factor fNP(x,b) for a flavor.

        No direct C++ block (enhanced PyTorch path replacing placeholder).

        Args:
            @param x: Bjorken x (or z for FF usage) tensor
            @param b: impact parameter tensor (GeV^-1)
            @param flavor: flavor label ('u','d','ubar',...)
        Returns:
            torch.Tensor: fNP values (dimensionless suppression factor)
        Notes:
            - Model returns dict of flavor tensors; we extract requested
            - Gaussian fallback if model absent/fails
        TODO:
            - Batch evaluate multiple flavors to reduce overhead
            - Support separate parameterizations for PDFs vs FFs
        """
        if self.model_fNP is not None:
            # Use the PyTorch fNP model
            try:
                # Ensure tensors are on the correct device and dtype
                x = x.to(self.device, dtype=self.dtype)
                b = b.to(self.device, dtype=self.dtype)

                # Evaluate fNP for the specific flavor
                # The model returns a dictionary with all flavor contributions
                fnp_outputs = self.model_fNP(x, b, flavors=[flavor])
                return fnp_outputs[flavor]

            except Exception as e:
                # Print in yellow to indicate a warning
                print(f"\033[93mWarning: Error in PyTorch fNP evaluation: {e}\033[0m")
                # Fallback to simple Gaussian if model evaluation fails
                return torch.exp(-0.1 * b**2)
        else:
            # Simple Gaussian fallback if no fNP model is loaded
            # Print in red to indicate a serious warning
            print(
                f"\033[91mWarning: No PyTorch fNP model loaded, using fallback Gaussian.\033[0m"
            )
            # Fallback to Gaussian
            return torch.exp(-0.1 * b**2)

    def compute_sidis_cross_section_pytorch(self, data_file: str, output_file: str):
        """
        Compute SIDIS differential numerator over kinematic points.

        Args:
            @param data_file: path to kinematic YAML (x,Q2,z,PhT arrays)
            @param output_file: destination YAML for results
        Returns:
            None (writes file)
        Notes:
            - Applies cut qT/Q < qToQcut prior to expensive integration
            - Integrand includes factors: b * fNP1 * fNP2 * Σ_q (Yp e_q^2 f1 D1)/(x) * Sud^2 * α_em^2 * H / (Q^3 z)
            - Ogata transform internally applies Bessel J0(qT b)
        TODO:
            - Implement denominator for multiplicities
            - Vectorize integrals across points (shared b* evaluations)
            - Refine Yp computation (explicit y variable)
        """
        # print in green to indicate the start of computation
        print(f"\033[93m\nLoading kinematic data from {data_file}\033[0m")
        kinematic_data = self.load_kinematic_data_pytorch(data_file)

        # Check that the data file contains the required keys and has the proper header.
        required = {"Vs", "target_isoscalarity"}
        header = kinematic_data.get("header", {})
        missing = required - set(header)
        if missing:
            raise ValueError(
                f"Header missing required keys: {', '.join(sorted(missing))}"
            )

        # Extract experimental setup parameters
        Vs = kinematic_data["header"]["Vs"]  # Center-of-mass energy
        targetiso = kinematic_data["header"][
            "target_isoscalarity"
        ]  # Target isoscalarity

        # Setup isoscalar TMD PDFs and FFs (averaging over proton/neutron)
        self.setup_isoscalar_tmds(Vs, targetiso)

        # Extract kinematic variables as PyTorch tensors
        x_tensor = kinematic_data["x"]  # Bjorken x (momentum fraction)
        Q2_tensor = kinematic_data["Q2"]  # Photon virtuality squared
        z_tensor = kinematic_data["z"]  # Energy fraction of produced hadron
        PhT_tensor = kinematic_data["PhT"]  # Transverse momentum of hadron

        # Compute derived kinematic quantities
        Q_tensor = torch.sqrt(Q2_tensor)  # Hard scale Q
        qT_tensor = PhT_tensor / z_tensor  # Intrinsic transverse momentum

        # Initialize results tensor for computed cross sections
        theo_xsec = torch.zeros_like(qT_tensor)

        # Print in green to indicate progress
        print(
            f"\033[92mComputing SIDIS cross sections for {len(qT_tensor)} kinematic points using PyTorch fNP...\033[0m"
        )

        # Main computation loop - process each kinematic point
        for iqT in range(len(qT_tensor)):
            # Extract scalar values for this kinematic point (convert tensors to floats)
            qTm = float(qT_tensor[iqT].item())  # Convert tensor to float for APFEL
            Qm = float(Q_tensor[iqT].item())
            xm = float(x_tensor[iqT].item())
            zm = float(z_tensor[iqT].item())

            # Apply kinematic cut: skip points where qT > qToQcut * Q
            # This avoids the non-perturbative region where TMD factorization breaks down
            if qTm > self.qToQcut * Qm:
                # Print in yellow to indicate skipping
                print(
                    f"\033[93mSkipping qT = {qTm:.3f} (above cut qT/Q = {self.qToQcut})\033[0m"
                )
                continue

            print(
                # Print in blue to indicate current point
                f"\033[94mComputing point {iqT+1}/{len(qT_tensor)}: qT = {qTm:.3f}, Q = {Qm:.3f}, x = {xm:.4f}, z = {zm:.4f}\033[0m"
            )

            # Set renormalization and factorization scales (keep as floats)
            mu = self.Cf * Qm  # Factorization scale (typically μ = Q)
            zeta = Qm * Qm  # Collins-Soper scale (ζ = Q²)

            # Yp factor: accounts for SIDIS kinematics and target mass corrections
            # Y+ = 1 + (1 - y)² where y is the inelasticity parameter
            # This appears in the SIDIS cross section structure functions
            Yp = 1 + (1 - (Qm / Vs) ** 2 / xm) ** 2

            # Determine number of active quark flavors at scale μ
            # Important for flavor summation in TMD luminosity
            nf = int(ap.utilities.NF(mu, self.Thresholds))

            def b_integrand(b_val):  # Corresponds to C++ integrand ll. 430–470
                """
                b-space integrand for Ogata quadrature.

                Args:
                    @param b_val: scalar impact parameter (GeV^-1)
                Returns:
                    float: integrand value before Bessel transform application

                """
                # bstar prescription: regularizes large-b_T behavior
                # Smoothly interpolates between small-b (perturbative) and large-b (non-perturbative)
                bs = self.bstar_min(b_val, Qm)

                # Compute TMD luminosity: sum over all active quark flavors
                # This implements the flavor sum in the SIDIS cross section:
                # Σ_q e_q² * f₁^q(x,b_T) * D₁^q(z,b_T)
                lumiq = 0.0

                for q in range(-nf, nf + 1):
                    if q == 0:  # Skip gluon (no direct contribution in SIDIS)
                        continue

                    try:
                        # Evaluate TMD PDF: f₁^q(x, b_T; μ, ζ)
                        # This includes evolution from initial scale to μ
                        tmd_pdf = self.TabMatchTMDPDFs.EvaluatexQ(q, xm, bs)

                        # Evaluate TMD FF: D₁^q(z, b_T; μ, ζ)
                        # This describes quark → hadron fragmentation
                        tmd_ff = self.TabMatchTMDFFs.EvaluatexQ(q, zm, bs)

                        # Electric charge squared for this quark flavor
                        # e_u² = 4/9, e_d² = 1/9, etc.
                        # Use safer access to avoid map::at errors
                        try:
                            qch2 = (
                                ap.constants.QCh2[abs(q) - 1]
                                if abs(q) <= len(ap.constants.QCh2)
                                else 0.0
                            )
                        except (IndexError, AttributeError, KeyError) as charge_error:
                            # Specific charge lookup failures
                            print(
                                f"\033[93m⚠️  APFEL charge lookup failed for flavor q={q}: {charge_error}\033[0m"
                            )
                            charges = {
                                1: 4 / 9,
                                2: 1 / 9,
                                3: 1 / 9,
                                4: 4 / 9,
                                5: 1 / 9,
                                6: 4 / 9,
                            }
                            qch2 = charges.get(abs(q), 0.0)
                            print(
                                f"\033[93m   → Using fallback charge: {qch2:.3f}\033[0m"
                            )

                        # Add this flavor's contribution to luminosity
                        # Factor of Yp accounts for SIDIS kinematics
                        lumiq += Yp * tmd_pdf / xm * qch2 * tmd_ff

                    except ValueError as val_error:
                        print(
                            f"\033[93m⚠️  Invalid parameters for TMD evaluation (q={q}): {val_error}\033[0m"
                        )
                        print(
                            f"\033[93m   → x={xm:.4f}, z={zm:.4f}, b*={bs:.4f}\033[0m"
                        )
                        continue
                    except RuntimeError as runtime_error:
                        print(
                            f"\033[91m🚨 APFEL runtime error for flavor q={q}: {runtime_error}\033[0m"
                        )
                        print(
                            f"\033[91m   → This may indicate APFEL interpolation failure\033[0m"
                        )
                        continue
                    except Exception as unknown_error:
                        print(
                            f"\033[91m💥 Unexpected TMD evaluation error for q={q}: {type(unknown_error).__name__}: {unknown_error}\033[0m"
                        )
                        continue

                # Evaluate non-perturbative functions using PyTorch fNP model
                if self.model_fNP is not None:

                    # Define flavors outside try block
                    pdf_flavor = (
                        "u"  # Simplified choice - can be made more sophisticated
                    )
                    ff_flavor = "u"  # For π+ production, u-quark dominated

                    try:
                        # Convert kinematics to tensors for PyTorch evaluation
                        x_torch = torch.tensor(xm, dtype=self.dtype, device=self.device)
                        z_torch = torch.tensor(zm, dtype=self.dtype, device=self.device)
                        b_torch = torch.tensor(
                            b_val, dtype=self.dtype, device=self.device
                        )

                        # # Evaluate fNP for PDF and FF. Print in green.
                        # print(
                        #     f"\033[92mEvaluating fNP for PDF flavor '{pdf_flavor}' and FF flavor '{ff_flavor}'...\033[0m"
                        # )

                        # NOTE: The model returns tensors; convert to float for integrand.
                        # fNP1 is the PDF and fNP2 is the FF.
                        # TODO: this is the moment where we use for the FF the PDF fNP model
                        # TODO: ideally we would have separate models or at least parameters
                        fnp1_val = float(
                            self.compute_fnp_pytorch(
                                x_torch, b_torch, pdf_flavor
                            ).item()
                        )
                        fnp2_val = float(
                            self.compute_fnp_pytorch(z_torch, b_torch, ff_flavor).item()
                        )

                    # Handle specific exceptions for robustness, fNP evaluation can fail in various ways.
                    except ValueError as val_error:
                        print(
                            f"\033[93m⚠️  Invalid tensor values for fNP: {val_error}\033[0m"
                        )
                        print(
                            f"\033[93m   → Check for NaN/inf in x={xm}, z={zm}, b={b_val}\033[0m"
                        )
                        fnp1_val = np.exp(-0.1 * b_val**2)
                        fnp2_val = np.exp(-0.05 * b_val**2)
                    except KeyError as key_error:
                        print(
                            f"\033[93m⚠️  fNP model missing flavor '{pdf_flavor}' or '{ff_flavor}': {key_error}\033[0m"
                        )
                        fnp1_val = np.exp(-0.1 * b_val**2)
                        fnp2_val = np.exp(-0.05 * b_val**2)
                    except Exception as unknown_error:
                        print(
                            f"\033[91m💥 Unexpected fNP error: {type(unknown_error).__name__}: {unknown_error}\033[0m"
                        )
                        fnp1_val = 1
                        fnp2_val = 1
                else:
                    # Gaussian non-perturbative functions if no fNP model
                    fnp1_val = np.exp(-0.1 * b_val**2)
                    fnp2_val = np.exp(-0.05 * b_val**2)

                # Compute Sudakov factor: S(b_T; μ, ζ)
                # This encodes soft gluon resummation.
                # The factor appears squared because we have both PDF and FF evolution
                sudakov_factor = self.QuarkSudakov(bs, mu, zeta) ** 2

                # Hard factor: H(μ)
                # Encodes the hard partonic process e + q → e + q + g
                hard_factor = self.Hf(mu)

                # Electromagnetic coupling: α_em²(Q)
                # Running coupling evaluated at the hard scale
                alphaem2 = self.TabAlphaem.Evaluate(Qm) ** 2

                # Assemble the complete integrand
                # This implements the TMD factorization formula integrand
                integrand = (
                    b_val  # Jacobian from d²b_T integration
                    * fnp1_val  # Non-perturbative factor for PDF
                    * fnp2_val  # Non-perturbative factor for FF
                    * lumiq  # TMD luminosity (flavor sum)
                    * sudakov_factor  # Sudakov resummation factor
                    * alphaem2  # Electromagnetic coupling squared
                    * hard_factor  # Hard process factor
                    / (Qm**3 * zm)  # Kinematic normalization
                )

                return integrand

            # Perform b_T-space integration using Ogata quadrature
            # This computes: ∫₀^∞ db_T * b_T * J₀(q_T * b_T) * integrand(b_T)
            # where J₀ is the Bessel function from the Fourier transform
            try:
                integral_result = self.DEObj.transform(b_integrand, qTm)

                # Compute final differential cross section
                # Includes proper kinematic factors and constants
                # Result is dσ/dxdQdzdPhT in appropriate units
                differential_xsec = (
                    ap.constants.ConvFact  # Unit conversion factor
                    * ap.constants.FourPi  # 4π factor
                    * qTm  # qT factor from integration
                    * integral_result  # b_T integral result
                    / (2 * Qm)  # Additional kinematic factor
                    / zm  # z normalization
                )

                # Store result in tensor
                theo_xsec[iqT] = differential_xsec
                print(f"  -> σ = {differential_xsec:.6e}")

            except Exception as e:
                print(f"  -> Integration failed: {e}")
                theo_xsec[iqT] = 0.0

        # Save computed results to output file
        self.save_results_pytorch(kinematic_data, theo_xsec, output_file)
        print(f"Results saved to {output_file}")

    def save_results_pytorch(
        self, kinematic_data: Dict, results: torch.Tensor, output_file: str
    ):
        """
        Save YAML output with metadata.

        No direct C++ equivalent (enhanced output schema).

        Args:
             @param kinematic_data: dict with header/raw_data + tensors
             @param results: tensor of differential numerator values
             @param output_file: destination YAML path
        """
        output_data = {
            "header": kinematic_data["header"],
            "input_data": kinematic_data["raw_data"],
            "results": {
                "cross_sections": results.cpu().numpy().tolist(),
                "units": "differential cross section numerator (no denominator)",
                "computation_info": {
                    "perturbative_order": self.PerturbativeOrder,
                    "pdf_set": self.config["pdfset"]["name"],
                    "ff_set": self.config["ffset"]["name"],
                    "qT_cut": self.qToQcut,
                    "device": str(self.device),
                    "pytorch_version": torch.__version__,
                    "fnp_model": (
                        "PyTorch fNP"
                        if self.model_fNP is not None
                        else "Gaussian fallback"
                    ),
                },
            },
        }

        with open(output_file, "w") as f:
            yaml.dump(output_data, f, default_flow_style=True)


def main():
    """
    CLI entry point.
    Args:
        (parsed from command line)

    TODO:
        - Add --profile flag
        - Add reproducibility seed management
    """
    parser = argparse.ArgumentParser(
        description="Compute SIDIS cross sections using PyTorch"
    )
    parser.add_argument("config_file", help="Configuration YAML file")
    parser.add_argument("data_file", help="Kinematic data YAML file")
    parser.add_argument("fnp_config_file", help="fNP configuration YAML file")
    parser.add_argument("output_folder", help="Output folder for results")
    parser.add_argument("output_filename", help="Output YAML filename")
    parser.add_argument(
        "--device", help="PyTorch device (cpu, cuda, mps)", default=None, type=str
    )

    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Construct full output path
    output_file = os.path.join(args.output_folder, args.output_filename)

    # Print out in purple some features
    print(f"\033[94m\nPython version: {sys.version}\033[0m")
    print(f"\033[94mPyTorch version: {torch.__version__}\033[0m")
    print(f"\033[94mOutput folder: {args.output_folder}\033[0m")
    print(f"\033[94mOutput file: {output_file}\033[0m")

    # Initialize PyTorch computation
    sidis_comp = SIDISComputationPyTorch(
        args.config_file, args.fnp_config_file, device=args.device
    )

    # Run computation
    sidis_comp.compute_sidis_cross_section_pytorch(args.data_file, output_file)

    # Print success message in green
    print("\033[92mPyTorch SIDIS computation completed successfully!\033[0m")


# This is not strictly required, but it’s the idiomatic way to make a file safe
# to import while still runnable as a script. It is the standard Python “script
# entry point” guard.
# Every Python module has a built-in variable __name__.
# If the file is run as a script (e.g., python myfile.py or python -m pkg.module),
# Python sets __name__ = "__main__".
# If the file is imported (e.g., import myfile), Python sets __name__ to the
# module’s actual name ("myfile" or "pkg.module").
if __name__ == "__main__":
    main()
