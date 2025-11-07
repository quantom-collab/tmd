#!/usr/bin/env python3
"""
Semi-Inclusive Deep Inelastic Scattering (SIDIS) Cross Section Computation
using PyTorch and APFEL++

This script computes SIDIS differential cross sections using:
1. APFEL++ library for TMD evolution and matching
2. PyTorch for non-perturbative function modeling and gradient computation

PYTORCH INTEGRATION:
===================
This implementation uses PyTorch for:
- fNP function evaluation
- Automatic differentiation for gradient-based fitting
- GPU acceleration (CUDA/Metal support)
- Tensor operations for efficient computation

USAGE:
======
This script is meant both to be imported as a module and run as a standalone script.
As a standalone script, the user may run it by doing;

python3.10 map_crossect_pytorch.py <config_file> <kinematic_data_file> <fnp_config_file> <output_folder> <output_filename>

Example:
python3.10 map_crossect_pytorch.py inputs/config.yaml inputs/kinematics.yaml inputs/fNPconfig.yaml results/ map_pytorch.yaml

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
import math
from typing import Dict, List, Tuple, Any, TYPE_CHECKING, Optional

# Type annotations for LHAPDF
# using Any to avoid linter errors
import lhapdf as lh

LHAPDF_PDF = Any

# Import apfelpy
import apfelpy as ap

# Ensure module can be executed as a standalone script by setting import paths
# Adds both the repository root and the 'map' directory to sys.path when needed
try:
    # When imported as a package this usually succeeds without changes
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _MAP_DIR = os.path.dirname(_SCRIPT_DIR)
    _REPO_ROOT = os.path.dirname(_MAP_DIR)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    if _MAP_DIR not in sys.path:
        sys.path.insert(0, _MAP_DIR)
except Exception:
    pass

# Import custom modules
try:
    # Try importing from map.modules first (when run from repo root)
    import map.modules.utilities as utl
    from map.modules.fnp_factory import create_fnp_manager
except ImportError:
    # Fallback to direct import (when run from map directory)
    import modules.utilities as utl
    from modules.fnp_factory import create_fnp_manager


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
        # Use higher precision for oscillatory b-integrals to stabilize gradients
        self.integration_dtype = torch.float64

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

        # Prepare a default b-grid for Torch-native integration
        self._setup_bgrid()

        print(f"\033[92m\n --- PyTorch SIDIS computation setup successful!\n\033[0m")

    def _setup_bgrid(self):
        """
        Create a logarithmic b-grid and its Torch version for differentiable integration.
        Configurable via optional 'bgrid' section in YAML; falls back to [1e-2, 2] with 256 nodes.
        """
        bg = self.config.get("bgrid", {}) if isinstance(self.config, dict) else {}
        b_min = float(bg.get("b_min", 1e-2))
        b_max = float(bg.get("b_max", 2.0))
        nb = int(bg.get("Nb", 256))

        # Build numpy nodes for APFEL luminosity pretabulation
        self._b_nodes_np = np.logspace(math.log10(b_min), math.log10(b_max), num=nb)

        # Torch grid for integration (prefer float64 except on MPS which lacks support)
        use_float64 = not (self.device.type == "mps")
        integ_dtype = torch.float64 if use_float64 else torch.float32
        self.integration_dtype = (
            integ_dtype  # Override initial setting if needed for MPS
        )

        # Create logspace grid: MPS doesn't support logspace, so create on CPU then move
        if self.device.type == "mps":
            self._b_nodes_torch = torch.logspace(
                math.log10(b_min),
                math.log10(b_max),
                steps=nb,
                base=10.0,
                device="cpu",
                dtype=self.integration_dtype,
            ).to(self.device)
        else:
            self._b_nodes_torch = torch.logspace(
                math.log10(b_min),
                math.log10(b_max),
                steps=nb,
                base=10.0,
                device=self.device,
                dtype=self.integration_dtype,
            )

    def _bessel_j0_torch(self, x: torch.Tensor, n_terms: int = 30) -> torch.Tensor:
        """
        Bessel J0 implemented in Torch.
        Tries torch.special.bessel_j0 if available and supported on device,
        otherwise uses a rapidly convergent power series valid for moderate |x|,
        which matches our integration range (qT*b typically O(1)).
        """
        # MPS doesn't support bessel functions, skip to series
        if x.device.type == "mps":
            pass  # Use series implementation below
        else:
            # Try native if present on other devices
            try:
                return torch.special.bessel_j0(x)
            except (AttributeError, NotImplementedError):
                pass

        # Series: J0(x) = sum_{k=0}^∞ (-1)^k (x^2/4)^k / (k!)^2
        y = (x * x) / 4.0
        term = torch.ones_like(x)
        s = term.clone()
        # Recurrence: term_{k} = term_{k-1} * (-y) / (k^2)
        for k in range(1, n_terms + 1):
            term = term * (-y) / (k * k)
            s = s + term
        return s

    def _precompute_luminosity_constants(
        self, xm: float, zm: float, Qm: float, Yp: float
    ) -> torch.Tensor:
        """
        Precompute APFEL-driven luminosity on the common b-grid as constants (no autograd).

        L(b; x, z, Q) = Yp/x * sum_q [ e_q^2 * f1_q(x, b*; mu, zeta) * D1_q(z, b*; mu, zeta) ]
                         * Sudakov(b*; mu, zeta)^2 * alpha_em(Q)^2 * H(mu) / (Q^3 * z)

        Returns a Torch tensor on the correct device/dtype, with requires_grad=False.
        """
        mu = self.Cf * Qm
        zeta = Qm * Qm
        nf = int(ap.utilities.NF(mu, self.Thresholds))

        L_vals = np.zeros_like(self._b_nodes_np)
        for i, b_val in enumerate(self._b_nodes_np):
            bs = self.bstar_min(b_val, Qm)

            # Flavor sum luminosity
            lumiq = 0.0
            for q in range(-nf, nf + 1):
                if q == 0:
                    continue
                try:
                    tmd_pdf = self.TabMatchTMDPDFs.EvaluatexQ(q, xm, bs)
                    tmd_ff = self.TabMatchTMDFFs.EvaluatexQ(q, zm, bs)
                    try:
                        qch2 = (
                            ap.constants.QCh2[abs(q) - 1]
                            if abs(q) <= len(ap.constants.QCh2)
                            else 0.0
                        )
                    except Exception:
                        # Fallback charges
                        charges = {
                            1: 4 / 9,
                            2: 1 / 9,
                            3: 1 / 9,
                            4: 4 / 9,
                            5: 1 / 9,
                            6: 4 / 9,
                        }
                        qch2 = charges.get(abs(q), 0.0)
                    lumiq += Yp * (tmd_pdf / xm) * qch2 * tmd_ff
                except Exception:
                    # Skip pathological nodes safely
                    continue

            try:
                sudakov_factor = self.QuarkSudakov(bs, mu, zeta) ** 2
                hard_factor = self.Hf(mu)
                alphaem2 = self.TabAlphaem.Evaluate(Qm) ** 2
            except Exception:
                sudakov_factor = 0.0
                hard_factor = 0.0
                alphaem2 = 0.0

            L_vals[i] = lumiq * sudakov_factor * alphaem2 * hard_factor / (Qm**3 * zm)

        # Convert to torch tensor constants (no gradients)
        L_torch = torch.tensor(L_vals, device=self.device, dtype=self.integration_dtype)
        return L_torch

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
        module defined in modules/fnp_factory.py.

        Notes:
            - On failure, Gaussian fallback used (exp(-a b^2))
            - Reports total & trainable parameter counts
        TODO:
            - Allow parameter freezing via config
        """
        try:
            # Use the provided fNP config file
            if os.path.exists(self.fnp_config_file):
                # Print the path of the fNP configuration file being loaded
                print(
                    f"\033[95m\nLoading fNP configuration from {self.fnp_config_file}\033[0m"
                )
                config_fnp = utl.load_yaml_file(self.fnp_config_file)

                # Initialize PyTorch fNP model using factory
                # The factory automatically selects the correct combo based on config
                self.model_fNP = create_fnp_manager(config_dict=config_fnp).to(
                    self.device
                )
                print("✅ PyTorch fNP model loaded successfully")

                # Additional information for debugging/development
                print("📊 Additional PyTorch Information:")
                total_params = sum(p.numel() for p in self.model_fNP.parameters())
                pytorch_trainable = sum(
                    p.numel() for p in self.model_fNP.parameters() if p.requires_grad
                )
                print(f"   PyTorch total parameters: {total_params}")
                print(f"   PyTorch requires_grad=True: {pytorch_trainable}")

            else:
                print(f"Warning: fNP config not found at {self.fnp_config_file}")
                self.model_fNP = None

        except Exception as e:
            print(f"⚠️  Warning: Could not load PyTorch fNP model: {e}")
            import traceback

            traceback.print_exc()
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
            @param targetiso: isoscalarity parameter τ (-1 ≤ τ ≤ +1)

        Notes:
            - frp = |τ|, frn = 1 - |τ|
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
        """
        with open(data_file, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return {}

        # Convert kinematic arrays to PyTorch tensors
        kinematic_data: Dict[str, Any] = {}
        for key in ["x", "Q2", "z", "PhT"]:
            if "data" in data and key in data["data"]:
                kinematic_data[key] = torch.tensor(
                    data["data"][key], dtype=self.dtype, device=self.device
                )

        # Include header information
        kinematic_data["header"] = data.get("header", {})
        kinematic_data["raw_data"] = data.get("data", {})

        return kinematic_data

    def compute_fnp_pytorch_pdf(
        self, x: torch.Tensor, b: torch.Tensor, Q: torch.Tensor, flavor: str
    ) -> torch.Tensor:
        """
        Evaluate TMD PDF non-perturbative factor fNP1(x,b) for a flavor using the PyTorch fNP model.

        Args:
            @param x: Bjorken x tensor
            @param b: impact parameter tensor (GeV^-1)
            @param Q: Hard scale Q tensor (GeV) - used to compute zeta = Q²
            @param flavor: flavor label ('u','d','ubar',...)
        Returns:
            torch.Tensor: fNP PDF values
        """
        if self.model_fNP is not None:
            # Use the PyTorch fNP model for PDFs
            try:
                # Ensure tensors are on the correct device and dtype
                x = x.to(self.device, dtype=self.dtype)
                b = b.to(self.device, dtype=self.dtype)
                Q = Q.to(self.device, dtype=self.dtype)
                pdf_outputs = self.model_fNP.forward_pdf(x, b, Q, flavors=[flavor])
                return pdf_outputs[flavor]
            except Exception as e:
                # Print in yellow to indicate a warning
                print(
                    f"\033[93mWarning: Error in PyTorch fNP PDF evaluation: {e}\033[0m"
                )
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

    def compute_fnp_pytorch_ff(
        self, z: torch.Tensor, b: torch.Tensor, Q: torch.Tensor, flavor: str
    ) -> torch.Tensor:
        """
        Evaluate TMD FF non-perturbative factor fNP2(z,b) for a flavor using the PyTorch fNP model.

        Args:
            @param z: energy fraction z tensor
            @param b: impact parameter tensor (GeV^-1)
            @param Q: Hard scale Q tensor (GeV) - used to compute zeta = Q²
            @param flavor: flavor label ('u','d','ubar',...)
        Returns:
            torch.Tensor: fNP FF values
        """
        if self.model_fNP is not None:
            # Use the PyTorch fNP model for FFs
            try:
                # Ensure tensors are on the correct device and dtype
                z = z.to(self.device, dtype=self.dtype)
                b = b.to(self.device, dtype=self.dtype)
                Q = Q.to(self.device, dtype=self.dtype)
                ff_outputs = self.model_fNP.forward_ff(z, b, Q, flavors=[flavor])
                return ff_outputs[flavor]
            except Exception as e:
                # Print in yellow to indicate a warning
                print(
                    f"\033[93mWarning: Error in PyTorch fNP FF evaluation: {e}\033[0m"
                )
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

    def compute_fnp_pytorch(
        self, x: torch.Tensor, b: torch.Tensor, Q: torch.Tensor, flavor: str
    ) -> torch.Tensor:
        """
        Legacy method for backward compatibility - defaults to PDF evaluation.

        Args:
            @param x: Bjorken x (or z for FF usage) tensor
            @param b: impact parameter tensor (GeV^-1)
            @param Q: Hard scale Q tensor (GeV) - used to compute zeta = Q²
            @param flavor: flavor label ('u','d','ubar',...)
        Returns:
            torch.Tensor: fNP values (PDF by default)
        """
        return self.compute_fnp_pytorch_pdf(x, b, Q, flavor)

    def compute_flavor_sum_pytorch(
        self, x: torch.Tensor, z: torch.Tensor, b: torch.Tensor, Q: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the full flavor sum for SIDIS cross section using PyTorch (differentiable).

        This function implements the flavor sum that was missing from the fitting routine:
        Σ_q e_q^2 * f_q(x,b) * D_q(z,b)

        Args:
            x (torch.Tensor): Bjorken x values (should be broadcastable with b)
            z (torch.Tensor): fragmentation variable values (should be broadcastable with b)
            b (torch.Tensor): impact parameter values
            Q (torch.Tensor): energy scale (scalar tensor)

        Returns:
            torch.Tensor: Sum over all active quark flavors with electric charge weighting

        Notes:
            - Simplified implementation that focuses on the main u, d, s quarks and their antiquarks
            - Maintains gradients for fNP parameter optimization
            - Uses standard electric charge values
        """
        # Define the flavors to include and their electric charges
        flavor_contributions = [
            ("u", 2.0 / 9.0),  # up quark
            ("d", 1.0 / 9.0),  # down quark
            ("s", 1.0 / 9.0),  # strange quark
            ("ubar", 2.0 / 9.0),  # up antiquark
            ("dbar", 1.0 / 9.0),  # down antiquark
            ("sbar", 1.0 / 9.0),  # strange antiquark
        ]

        # Initialize flavor sum
        flavor_sum = torch.zeros_like(
            b, device=self.device, dtype=self.integration_dtype
        )

        # Sum over defined flavors
        for flavor_str, eq2 in flavor_contributions:
            try:
                # Get non-perturbative factors (these maintain gradients!)
                # fnp1 = PDF, fnp2 = FF as requested by user
                fnp1 = self.compute_fnp_pytorch_pdf(x, b, Q, flavor_str)  # PDF
                fnp2 = self.compute_fnp_pytorch_ff(z, b, Q, flavor_str)  # FF

                # Add this flavor's contribution to the sum
                flavor_sum += eq2 * fnp1 * fnp2

            except Exception as e:
                # Skip flavors that cause errors (e.g., not defined in fNP model)
                # This is expected for flavors not in the fNP configuration
                continue

        return flavor_sum

    def compute_sidis_cross_section_pytorch(
        self, data_file: str, output_file: str, use_ogata: bool = False
    ):
        """
        Compute SIDIS differential numerator over kinematic points.

        Args:
            @param data_file: path to kinematic YAML (x,Q2,z,PhT arrays)
            @param output_file: destination YAML for results
            @param use_ogata: if True, use Ogata quadrature (non-differentiable but accurate)
                             if False, use PyTorch trapezoidal integration (differentiable)
        Returns:
            None (writes file)
        Notes:
            - Applies cut qT/Q < qToQcut prior to expensive integration
            - Ogata: more accurate, non-differentiable, callback-based
            - PyTorch: less accurate, fully differentiable, tensor-based
            - Integrand includes factors: b * fNP1 * fNP2 * Σ_q (Yp e_q^2 f1 D1)/(x) * Sud^2 * α_em^2 * H / (Q^3 z)
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
            # Scalars for APFEL-side calculations (safe to detach, no fNP gradients here)
            qTm = float(qT_tensor[iqT].detach().cpu().numpy())
            Qm = float(Q_tensor[iqT].detach().cpu().numpy())
            xm = float(x_tensor[iqT].detach().cpu().numpy())
            zm = float(z_tensor[iqT].detach().cpu().numpy())

            # Kinematic cut
            if qTm > self.qToQcut * Qm:
                print(
                    f"\033[93mSkipping qT = {qTm:.3f} (above cut qT/Q = {self.qToQcut})\033[0m"
                )
                continue

            print(
                f"\033[94mComputing point {iqT+1}/{len(qT_tensor)}: qT = {qTm:.3f}, Q = {Qm:.3f}, x = {xm:.4f}, z = {zm:.4f}\033[0m"
            )

            # SIDIS Y+ factor (float)
            Yp = 1 + (1 - (Qm / Vs) ** 2 / xm) ** 2

            if use_ogata:
                # Use Ogata quadrature (accurate but non-differentiable)
                print(f"  Using Ogata quadrature integration")
                xs = self._compute_cross_section_ogata(iqT, xm, zm, Qm, qTm, Yp)
            else:
                # Use PyTorch trapezoidal integration (differentiable)
                print(f"  Using PyTorch trapezoidal integration")
                xs = self._compute_cross_section_pytorch(
                    iqT, x_tensor, z_tensor, qT_tensor, Q_tensor, Yp
                )

            # Final kinematic factors (convert to appropriate tensor)
            if use_ogata:
                # Ogata returns float, convert to tensor
                differential_xsec = torch.tensor(
                    ap.constants.ConvFact
                    * ap.constants.FourPi
                    * qTm
                    * xs
                    / (2.0 * Qm)
                    / zm,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                # PyTorch integration preserves tensor operations
                Q_t = Q_tensor[iqT].to(self.device).to(self.integration_dtype)
                z_t = z_tensor[iqT].to(self.device).to(self.integration_dtype)
                qT_t = qT_tensor[iqT].to(self.device).to(self.integration_dtype)

                differential_xsec = (
                    torch.tensor(
                        ap.constants.ConvFact * ap.constants.FourPi,
                        device=self.device,
                        dtype=self.integration_dtype,
                    )
                    * qT_t
                    * xs
                    / (2.0 * Q_t)
                    / z_t
                )

            # Store, casting to output dtype
            theo_xsec[iqT] = differential_xsec.to(theo_xsec.dtype)
            print(f"  -> σ = {float(differential_xsec.detach().cpu().numpy()):.6e}")

        # Save results in multiple formats (matching C++ behavior)
        self.save_results_pytorch(kinematic_data, theo_xsec, output_file)

        # Generate array format for plotting (like C++ saveResultsYAMLArrays)
        base_name = output_file.rsplit(".", 1)[0]  # Remove extension
        array_output = f"{base_name}_arrays.yaml"
        self.save_results_arrays_pytorch(kinematic_data, theo_xsec, array_output)

        print(f"\033[92m\nResults saved to:\033[0m")
        print(f"  YAML (detailed): {output_file}")
        print(f"  YAML (arrays):   {array_output}")

    def _compute_cross_section_ogata(
        self, iqT: int, xm: float, zm: float, Qm: float, qTm: float, Yp: float
    ) -> float:
        """
        Compute cross section using Ogata quadrature (non-differentiable but accurate).

        Args:
            @param iqT: kinematic point index
            @param xm, zm, Qm, qTm: kinematic variables as floats
            @param Yp: SIDIS Y+ factor
        Returns:
            float: b-integral result
        """
        # TMD scales
        mu = self.Cf * Qm
        zeta = Qm * Qm
        nf = int(ap.utilities.NF(mu, self.Thresholds))

        def b_integrand(b_val: float) -> float:
            """Ogata integrand function (non-differentiable)."""
            # bstar prescription
            bs = self.bstar_min(b_val, Qm)

            # TMD luminosity sum over flavors
            lumiq = 0.0
            for q in range(-nf, nf + 1):
                if q == 0:  # Skip gluon
                    continue

                try:
                    # TMD PDF and FF evaluation (float values)
                    tmd_pdf = self.TabMatchTMDPDFs.EvaluatexQ(q, xm, bs, mu, zeta)
                    tmd_ff = self.TabMatchTMDFFs.EvaluatexQ(q, zm, bs, mu, zeta)

                    # Electric charge squared
                    eq2 = (
                        2.0 / 9.0 if abs(q) in [2, 4, 6] else 1.0 / 9.0
                    )  # u-type vs d-type

                    lumiq += eq2 * tmd_pdf * tmd_ff
                except:
                    continue

            # Non-perturbative factors (convert tensors to floats)
            # fnp1_val = PDF, fnp2_val = FF as requested by user
            Q_t = torch.tensor(Qm, device=self.device, dtype=self.dtype)
            fnp1_val = float(
                self.compute_fnp_pytorch_pdf(
                    torch.tensor(xm, device=self.device, dtype=self.dtype),
                    torch.tensor(b_val, device=self.device, dtype=self.dtype),
                    Q_t,
                    "u",  # PDF flavor
                ).item()
            )

            fnp2_val = float(
                self.compute_fnp_pytorch_ff(
                    torch.tensor(zm, device=self.device, dtype=self.dtype),
                    torch.tensor(b_val, device=self.device, dtype=self.dtype),
                    Q_t,
                    "u",  # FF flavor
                ).item()
            )

            # Additional factors
            sudakov_factor = self.QuarkSudakov(bs, mu, zeta) ** 2
            alphaem2 = self.TabAlphaem.Evaluate(Qm) ** 2
            hard_factor = self.Hf(mu)

            # Complete integrand
            integrand = (
                b_val
                * fnp1_val
                * fnp2_val
                * lumiq
                * sudakov_factor
                * alphaem2
                * hard_factor
                * Yp
                / xm
                / (Qm**3 * zm)
            )

            return integrand

        # Perform Ogata integration
        try:
            result = self.DEObj.transform(b_integrand, qTm)
            return result
        except Exception as e:
            print(f"    Warning: Ogata integration failed: {e}")
            return 0.0

    def _compute_cross_section_pytorch(
        self,
        iqT: int,
        x_tensor: torch.Tensor,
        z_tensor: torch.Tensor,
        qT_tensor: torch.Tensor,
        Q_tensor: torch.Tensor,
        Yp: float,
    ) -> torch.Tensor:
        """
        Compute cross section using PyTorch integration (differentiable).

        Args:
            @param iqT: kinematic point index
            @param x_tensor, z_tensor, qT_tensor, Q_tensor: kinematic tensors
            @param Yp: SIDIS Y+ factor
        Returns:
            torch.Tensor: b-integral result (preserves gradients)
        """
        # Extract values for this point
        xm = float(x_tensor[iqT].detach().cpu().numpy())
        zm = float(z_tensor[iqT].detach().cpu().numpy())
        Qm = float(Q_tensor[iqT].detach().cpu().numpy())

        # Precompute APFEL luminosity constants on b-grid (Torch tensor, no grads)
        L_b = self._precompute_luminosity_constants(xm, zm, Qm, Yp)

        # Build Torch integrand preserving fNP gradients
        b = self._b_nodes_torch  # [Nb], float64
        # Prepare tensors for model evaluation
        x_t = x_tensor[iqT].to(self.device).to(self.integration_dtype)
        z_t = z_tensor[iqT].to(self.device).to(self.integration_dtype)
        qT_t = qT_tensor[iqT].to(self.device).to(self.integration_dtype)
        Q_t = Q_tensor[iqT].to(self.device).to(self.integration_dtype)

        # Evaluate fNP for chosen flavors as tensors; keep gradients
        # fnp_pdf = fnp1 (PDF), fnp_ff = fnp2 (FF) as requested by user
        pdf_flavor = "u"
        ff_flavor = "u"
        fnp_pdf = self.compute_fnp_pytorch_pdf(
            x_t.expand_as(b), b.to(self.dtype), Q_t.expand_as(b), pdf_flavor
        ).to(self.integration_dtype)
        fnp_ff = self.compute_fnp_pytorch_ff(
            z_t.expand_as(b), b.to(self.dtype), Q_t.expand_as(b), ff_flavor
        ).to(self.integration_dtype)

        # Bessel J0 and integrand
        J0 = self._bessel_j0_torch(qT_t * b)
        integrand = b * J0 * fnp_pdf * fnp_ff * L_b  # [Nb]

        # Differentiable trapezoidal integration over b
        xs = torch.trapz(integrand, b)
        return xs

    def save_results_pytorch(
        self, kinematic_data: Dict, results: torch.Tensor, output_file: str
    ):
        """
        Save YAML output with human-readable formatting matching C++ version.

        Creates structured output similar to SIDISCrossSectionKinem.cc:
        - Top-level metadata (Process, Observable, etc.)
        - Detailed kinematics list with per-point data
        - Computation information

        Args:
             @param kinematic_data: dict with header/raw_data + tensors
             @param results: tensor of differential numerator values
             @param output_file: destination YAML path
        """
        # Extract header information
        header = kinematic_data.get("header", {})
        raw_data = kinematic_data.get("raw_data", {})

        # Convert results to list for serialization
        cross_sections = results.detach().cpu().numpy().tolist()

        # Calculate derived quantities
        x_vals = raw_data.get("x", [])
        z_vals = raw_data.get("z", [])
        Q2_vals = raw_data.get("Q2", [])
        PhT_vals = raw_data.get("PhT", [])
        y_vals = raw_data.get("y", [])

        Q_vals = [float(Q2**0.5) for Q2 in Q2_vals] if Q2_vals else []
        qT_vals = (
            [float(PhT / z) for PhT, z in zip(PhT_vals, z_vals)]
            if PhT_vals and z_vals
            else []
        )

        # Build structured output matching C++ format
        output_data = {
            "Process": header.get("process", "SIDIS"),
            "Observable": header.get("observable", "cross_section"),
            "Hadron": header.get("hadron", "unknown"),
            "Charge": header.get("charge", 0),
            "Target_isoscalarity": header.get("target_isoscalarity", 0.0),
            "Vs": header.get("Vs", 0.0),
            "Kinematics": [],
        }

        # Add per-point kinematics (matching C++ structure)
        n_points = len(cross_sections)
        for i in range(n_points):
            point_data = {
                "point": i + 1,
                "PhT": float(PhT_vals[i]) if i < len(PhT_vals) else 0.0,
                "x": float(x_vals[i]) if i < len(x_vals) else 0.0,
                "z": float(z_vals[i]) if i < len(z_vals) else 0.0,
                "Q": float(Q_vals[i]) if i < len(Q_vals) else 0.0,
                "y": float(y_vals[i]) if i < len(y_vals) else 0.0,
                "qT": float(qT_vals[i]) if i < len(qT_vals) else 0.0,
                "cross_section": float(cross_sections[i]),
            }
            output_data["Kinematics"].append(point_data)

        # Add computation metadata (PyTorch-specific extension)
        output_data["Computation_Info"] = {
            "method": "PyTorch SIDIS computation",
            "perturbative_order": self.PerturbativeOrder,
            "pdf_set": self.config["pdfset"]["name"],
            "ff_set": self.config["ffset"]["name"],
            "qT_cut": self.qToQcut,
            "device": str(self.device),
            "pytorch_version": str(torch.__version__),
            "fnp_model": (
                "PyTorch fNP" if self.model_fNP is not None else "Gaussian fallback"
            ),
            "integration_dtype": str(self.integration_dtype),
            "units": "differential cross section numerator (no denominator)",
        }

        # Write YAML with human-readable formatting
        with open(output_file, "w") as f:
            # Custom YAML formatting for readability
            f.write("# SIDIS Cross Section Results - PyTorch Implementation\n")
            f.write(f"# Generated on {torch.__version__} with device {self.device}\n")
            f.write(f"# Total points: {n_points}\n")
            f.write("#\n")

            # Use block style (default_flow_style=False) with proper indentation
            yaml.dump(
                output_data,
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=False,
                allow_unicode=True,
                width=120,
                default_style=None,
            )

    def save_results_arrays_pytorch(
        self, kinematic_data: Dict, results: torch.Tensor, output_file: str
    ):
        """
        Save YAML output in array format for plotting (matching C++ saveResultsYAMLArrays).

        Creates arrays of all kinematic variables and results for easy plotting/analysis.

        Args:
             @param kinematic_data: dict with header/raw_data + tensors
             @param results: tensor of differential numerator values
             @param output_file: destination YAML path for array format
        """
        # Extract header information
        header = kinematic_data.get("header", {})
        raw_data = kinematic_data.get("raw_data", {})

        # Convert results to list for serialization
        cross_sections = results.detach().cpu().numpy().tolist()

        # Extract arrays
        x_vals = raw_data.get("x", [])
        z_vals = raw_data.get("z", [])
        Q2_vals = raw_data.get("Q2", [])
        PhT_vals = raw_data.get("PhT", [])
        y_vals = raw_data.get("y", [])

        # Calculate derived quantities
        Q_vals = [float(Q2**0.5) for Q2 in Q2_vals] if Q2_vals else []
        qT_vals = (
            [float(PhT / z) for PhT, z in zip(PhT_vals, z_vals)]
            if PhT_vals and z_vals
            else []
        )

        # Build array-based output
        output_data = {
            "Name": f"{header.get('process', 'SIDIS')}_{header.get('observable', 'cross_section')}",
            "PhT": [float(val) for val in PhT_vals],
            "x_values": [float(val) for val in x_vals],
            "z_values": [float(val) for val in z_vals],
            "Q2": [float(val) for val in Q2_vals],
            "Q_values": Q_vals,
            "y": [float(val) for val in y_vals],
            "qT": qT_vals,
            "Predictions": cross_sections,
        }

        # Write YAML with flow style for arrays (more compact)
        with open(output_file, "w") as f:
            f.write("# SIDIS Cross Section Results - Array Format for Plotting\n")
            f.write(f"# PyTorch implementation on {self.device}\n")
            f.write(f"# Total data points: {len(cross_sections)}\n")
            f.write("#\n")

            yaml.dump(
                output_data,
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=False,
                allow_unicode=True,
                width=120,
            )


def main():
    """
    CLI entry point.
    Args:
        (parsed from command line)

    TODO:
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
    parser.add_argument(
        "--use-ogata",
        action="store_true",
        help="Use Ogata quadrature instead of PyTorch integration (more accurate, non-differentiable)",
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
    map_comp = SIDISComputationPyTorch(
        args.config_file, args.fnp_config_file, device=args.device
    )

    # Run computation
    map_comp.compute_sidis_cross_section_pytorch(
        args.data_file, output_file, use_ogata=args.use_ogata
    )

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
