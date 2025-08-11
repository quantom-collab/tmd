#!/usr/bin/env python3
"""
SIDIS Cross Section Computation using APFEL++ Python wrapper

This script computes SIDIS differential cross sections for given kinematic points.

Usage:
    python3.10 sidis_computation.py <config_file> <kinematic_data_file> <fnp_config_file> <output_folder> <output_filename>

Example:
    python3.10 sidis_computation.py inputs/config.yaml inputs/kinematics.yaml inputs/fNPconfig.yaml results/ sidis_results.yaml
"""

import os
import sys
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, TYPE_CHECKING

# Type annotations for LHAPDF - using Any to avoid linter errors
import lhapdf as lh

# TODO: why do we need this piece? Remove if not needed, or
# TODO: if it is needed, explain why
if TYPE_CHECKING:
    LHAPDF_PDF = Any
else:
    LHAPDF_PDF = Any

# Import apfelpy
import apfelpy as ap

# Import custom modules
import modules.utilities as utl
from modules.fNP import fNP


class SIDISComputation:
    """
    Main class for SIDIS cross section computation
    This class initializes all necessary APFEL++ objects and configurations,
    and provides methods to compute the SIDIS cross sections based on input kinematic data.

    MAPPING TO C++ CODE (SIDISMultiplicities.cc):
    =============================================

    Key differences from C++ implementation:
    - This Python version computes DIFFERENTIAL CROSS SECTIONS (numerator only)
    - C++ version computes MULTIPLICITIES = differential_cross_section / inclusive_cross_section
    - Missing: Denominator calculation (ll. 344-366 in C++)
    - TODO: Proper fNP model integration (ll. 425-426 in C++)

    Main correspondence:
    - setup_computation() ↔ ll. 41-105: Initial setup
    - _setup_pdf() ↔ ll. 49-70: PDF setup
    - _setup_couplings() ↔ ll. 61-64, 82-85: Coupling setup
    - _setup_tmd_*() ↔ ll. 87-105, 161-174: TMD object setup
    - setup_isoscalar_tmds() ↔ ll. 268-313: Isoscalar target handling
    - compute_sidis_cross_section() ↔ ll. 376-480: Main computation loop
    - b_integrand() ↔ ll. 430-470: Core b-space integration
    """

    def __init__(self, config_file: str, fnp_config_file: str):
        """
        Initialize the computation with configuration files.

        Args:
            config_file: Path to the main configuration YAML file
            fnp_config_file: Path to the fNP configuration YAML file
        """
        self.config = self._load_config(config_file)
        self.fnp_config_file = fnp_config_file
        self.setup_computation()

    def _load_config(self, config_file: str) -> Dict:
        """
        Load configuration from YAML file

        @param config_file: Path to the configuration YAML file
        @return: Configuration dictionary
        """
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Checks at runtime that config is exactly
        # (or a subclass of) dict. Returns either
        # the parsed dict or a safe fallback of {}
        return config if isinstance(config, dict) else {}

    def setup_computation(self):
        """
        Setup all APFEL++ objects and configurations
        """
        print("Setting up SIDIS computation...")

        # Extract configuration parameters from config file.
        self.PerturbativeOrder = self.config["PerturbativeOrder"]
        self.Ci = self.config["TMDscales"]["Ci"]
        self.Cf = self.config["TMDscales"]["Cf"]
        # If the key qToQcut exists in config, use its value;
        # 0.3 is the fallback value
        self.qToQcut = self.config.get("qToQcut", 0.3)

        # Setup PDF
        self._setup_pdf()

        # Setup Alpha_s and Alpha_em
        self._setup_couplings()

        # Setup TMD objects
        self._setup_tmd_pdf_objects()

        # Setup Fragmentation Functions
        self._setup_ff()

        # Setup TMD FF objects
        self._setup_tmd_ff_objects()

        # Setup non-perturbative functions
        self._setup_fnp()

        # Setup Ogata quadrature
        self.DEObj = ap.ogata.OgataQuadrature(0, 1e-9, 0.00001)

        print("Setup computation successful!")

    def _setup_pdf(self):
        """
        Setup PDF objects
        This function initializes the PDF set and rotates it into the QCD evolution basis.
        It also sets up the x-space grid for PDFs and initializes the scale.

        Corresponds to C++ code:
        - ll. 49-50: Open LHAPDF sets (LHAPDF::PDF* distpdf = LHAPDF::mkPDF(...))
        - ll. 52-53: Rotate PDF set into QCD evolution basis (const auto RotPDFs = [=] ...)
        - ll. 55-59: Get heavy-quark thresholds from PDF LHAPDF set
        - ll. 66-70: Setup APFEL++ x-space grid for PDFs
        """

        # Read the PDF set name and member from the config file
        pdf_name = self.config["pdfset"]["name"]
        pdf_member = self.config["pdfset"]["member"]

        # Initialize LHAPDF set - using type annotation to avoid linter errors
        self.pdf: LHAPDF_PDF = lh.mkPDF(pdf_name, pdf_member)  # type: ignore

        # Rotate PDF set into QCD evolution basis
        self.RotPDFs = lambda x, mu: ap.PhysToQCDEv(self.pdf.xfxQ(x, mu))

        # Get heavy-quark thresholds
        self.Thresholds = []
        for v in self.pdf.flavors():
            if v > 0 and v < 7:
                self.Thresholds.append(self.pdf.quarkThreshold(v))

        # TODO; check why do we need to define the Masses of the quarks
        # TODO: This is present in the jupyter notebook used as an example in apfelpy
        # TODO: and also in the SIDIS_apfelpy.ipynb (cell.8)
        self.Masses = [0, 0, 0, self.pdf.quarkThreshold(4), self.pdf.quarkThreshold(5)]

        # Setup x-space grid for PDFs
        self.gpdf = ap.Grid(
            [ap.SubGrid(*subgrids) for subgrids in self.config["xgridpdf"]]
        )

        # TODO: mu0 scale here. Where is it in the C++ code?
        # Initial scale. Set the same as the initial PDF scale.
        # This is the square root of the minimum Q^2 in the PDF set.
        self.mu0 = np.sqrt(self.pdf.q2Min)

        print("PDF setup successful")

    def _setup_couplings(self):
        """
        Setup coupling constants
        This function initializes the QCD coupling and the electromagnetic coupling.
        It also sets up the grid for the couplings.

        Corresponds to C++ code:
        - ll. 61-64: Alpha_s (from PDFs). Get it from the LHAPDF set and tabulate it
        - ll. 82-85: Electromagnetic coupling squared (provided by APFEL++)
        """
        # Alpha_s
        Alphas = lambda mu: self.pdf.alphasQ(mu)
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
        LeptThresholds = [0.0, 0.0, 1.777]
        pto_aem = 0

        alphaem = ap.AlphaQED(
            AlphaRef=aref,
            MuRef=Qref,
            LeptThresholds=LeptThresholds,
            QuarkThresholds=self.Thresholds,
            pt=pto_aem,
        )

        self.TabAlphaem = ap.TabulateObject(alphaem, 100, 0.9, 1001, 3)

    def _setup_tmd_pdf_objects(self):
        """
        Setup TMD PDF objects. This actually involves also some manipulations of the colinear PDFs:
        - initializing DGLAP objects, with the proper collinear PDF grids (self.gpdf), and the proper Masses and Thresholds,
          which come from the collinear PDFs setup function, _setup_pdf(self), defined above.
        - evolving and tabulating the collinear PDFs,

        Then, there is the part for the actual TMDs
        - initializing TMD PDF objects,
        - building the evolved TMD PDFs
        - matching the TMD PDFs to the collinear PDFs

        This function also takes care of initializing the Sudakov factor and the hard factor.

        ========================
        Corresponds to C++ code:
        - ll. 87-90: Construct set of distributions as a function of the scale to be tabulated
        - ll. 92-94: Tabulate collinear PDFs
        - ll. 96: Initialize TMD PDF objects
        - ll. 98-101: Build evolved TMD PDFs
        - ll. 103: QuarkSudakov evolution factor
        - ll. 105: Get hard-factor
        """

        # Initialize QCD evolution objects
        DglapObj = ap.initializers.InitializeDglapObjectsQCD(
            self.gpdf, self.Masses, self.Thresholds
        )

        # Build DGLAP objects
        EvolvedPDFs = ap.builders.BuildDglap(
            DglapObj,
            lambda x, mu: ap.utilities.PhysToQCDEv(self.pdf.xfxQ(x, mu)),
            self.mu0,
            self.pdf.orderQCD,
            self.TabAlphas.Evaluate,
        )

        # Tabulate collinear PDFs
        self.TabPDFs = ap.TabulateObjectSetD(
            EvolvedPDFs, 100, np.sqrt(self.pdf.q2Min) * 0.9, np.sqrt(self.pdf.q2Max), 3
        )
        self.CollPDFs = lambda mu: self.TabPDFs.Evaluate(mu)

        # Initialize TMD objects
        self.TmdObjPDF = ap.tmd.InitializeTmdObjects(self.gpdf, self.Thresholds)

        # Build evolved TMD PDFs
        self.EvTMDPDFs = ap.tmd.BuildTmdPDFs(
            self.TmdObjPDF,
            self.CollPDFs,
            self.TabAlphas.Evaluate,
            self.PerturbativeOrder,
            self.Ci,
        )
        self.MatchTMDPDFs = ap.tmd.MatchTmdPDFs(
            self.TmdObjPDF,
            self.CollPDFs,
            self.TabAlphas.Evaluate,
            self.PerturbativeOrder,
            self.Ci,
        )

        # Sudakov factor and hard factor
        self.QuarkSudakov = ap.tmd.QuarkEvolutionFactor(
            self.TmdObjPDF,
            self.TabAlphas.Evaluate,
            self.PerturbativeOrder,
            self.Ci,
            1e5,
        )

        # Hard factor for SIDIS, at the right perturbative order.
        self.Hf = ap.tmd.HardFactor(
            "SIDIS",
            self.TmdObjPDF,
            self.TabAlphas.Evaluate,
            self.PerturbativeOrder,
            self.Cf,
        )

    def _setup_ff(self):
        """
        Setup fragmentation function objects

        Corresponds to C++ code:
        - ll. 150-151: Open LHAPDF FFs sets (LHAPDF::PDF* distff = LHAPDF::mkPDF(...))
        - ll. 153: Rotate FF set into the QCD evolution basis
        - ll. 155-159: Setup APFEL++ x-space grid for FFs
        """
        ff_name = self.config["ffset"]["name"]
        ff_member = self.config["ffset"]["member"]

        # Initialize LHAPDF FF set - using type annotation to avoid linter errors
        self.distff: LHAPDF_PDF = lh.mkPDF(ff_name, ff_member)  # type: ignore

        # Rotate FF set into QCD evolution basis
        self.RotFFs = lambda x, mu: ap.PhysToQCDEv(self.distff.xfxQ(x, mu))

        # Setup x-space grid for FFs
        self.gff = ap.Grid(
            [ap.SubGrid(*subgrids) for subgrids in self.config["xgridff"]]
        )

    def _setup_tmd_ff_objects(self):
        """
        Setup TMD FF objects

        Corresponds to C++ code:
        - ll. 161-165: Construct set of FF distributions as a function of the scale to be tabulated
        - ll. 167-169: Tabulate collinear FFs
        - ll. 171: Initialize TMD FF objects
        - ll. 173-174: Build evolved TMD FFs
        """
        # Initialize QCD evolution objects for FFs
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

    def _setup_fnp(self):
        """
        Setup non-perturbative functions using the provided configuration file.

        Corresponds to C++ code:
        - ll. 76-77: Get non-perturbative functions (const NangaParbat::Parameterisation* fNP = ...)
        """
        try:
            if os.path.exists(self.fnp_config_file):
                print(f"Loading fNP configuration from {self.fnp_config_file}")
                config_fnp = utl.load_yaml_config(self.fnp_config_file)
                self.model_fNP = fNP(config_fnp)
                print("✅ fNP model loaded successfully")
            else:
                # Use simple Gaussian form as fallback
                self.model_fNP = None
                print(
                    f"Warning: fNP config not found at {self.fnp_config_file}, using simple Gaussian form"
                )
        except Exception as e:
            print(f"Warning: Could not load fNP model: {e}")
            self.model_fNP = None

    def bstar_min(self, b: float, Q: float) -> float:
        """
        Implement bstar prescription. In the c++ code, the bstar prescription is called
        from NangaParbat, [ll. 392: auto bs = NangaParbat::bstarmin(b, Qm);].
        Here the bstar prescription is implemented from scratch in python with the same
        formulas, in a consistent way.
        TODO: transition this in pytorch

        @ param b: impact parameter (float)
        @ param Q: hard scale (float)
        """

        # Final scale, typically 1 GeV
        muF = 1.0  # GeV

        # Euler-Mascheroni constant
        gamma_E = 0.5772156649015329

        bmax = 2 * np.exp(-gamma_E) / muF  # GeV^-1
        bmin = bmax / Q  # GeV^-1
        power = 4

        num = 1 - np.exp(-((b / bmax) ** power))
        den = 1 - np.exp(-((b / bmin) ** power))

        return bmax * (num / den) ** (1 / power)

    def setup_isoscalar_tmds(self, Vs: float, targetiso: float):
        """
        Take into account the isoscalarity of the target. This has a direct impact on how
        the TMD PDFs and TMD FFs are combined together.
        Once we have the right combinations, TMD PDFs and TMD FFs are tabulated in b-space.

        Corresponds to C++ code:
        - ll. 207-208: Get the isoscalarity of the target
        - ll. 246-268: Take into account the isoscalarity of the target
        - ll. 269-272: Tabulate initial scale TMD PDFs in b in the physical basis
        - ll. 273-279: Tabulate initial scale TMD FFs in b in the physical basis
        """
        # Convert targetiso to tensor for consistent operations
        targetiso_tensor = torch.tensor(targetiso, dtype=torch.float32)

        # Calculate isoscalarity factors
        sign = torch.where(
            targetiso_tensor >= 0,
            torch.ones_like(targetiso_tensor),
            -torch.ones_like(targetiso_tensor),
        )

        frp = targetiso_tensor.abs()
        frn = 1 - frp

        # Functions for tabulation
        def TabFunc(b: float) -> float:
            t = torch.tensor(b, dtype=torch.float64)
            return t.log().item()

        def InvTabFunc(y: float) -> float:
            t = torch.tensor(y, dtype=torch.float64)
            return t.exp().item()

        # Isoscalar TMD PDFs
        def isTMDPDFs(b):
            xF = ap.utilities.QCDEvToPhys(self.MatchTMDPDFs(b).GetObjects())

            s = int(sign.item())
            s2 = int((sign * 2).item())

            xFiso = {}
            xFiso[1] = frp * xF[s] + frn * xF[s2]
            xFiso[-1] = frp * xF[-s] + frn * xF[-s2]
            xFiso[2] = frp * xF[s2] + frn * xF[s]
            xFiso[-2] = frp * xF[-s2] + frn * xF[-s]

            for i in range(3, 7):
                ip = int((i * sign).item())
                xFiso[i] = xF[ip]
                xFiso[-i] = xF[-ip]

            return ap.SetD(xFiso)

        self.TabMatchTMDPDFs = ap.TabulateObjectSetD(
            isTMDPDFs, 200, 1e-2, 2, 1, [], TabFunc, InvTabFunc
        )

        # Isoscalar TMD FFs
        isTMDFFs = lambda b: ap.SetD(
            ap.utilities.QCDEvToPhys(self.MatchTMDFFs(b).GetObjects())
        )

        self.TabMatchTMDFFs = ap.TabulateObjectSetD(
            isTMDFFs, 200, 1e-2, 2, 1, [], TabFunc, InvTabFunc
        )

    def load_kinematic_data(self, data_file: str) -> Dict:
        """
        Load kinematic data from file
        """
        with open(data_file, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    def compute_sidis_cross_section(self, data_file: str, output_file: str):
        """
        Main computation function
        This function loads the kinematic data, extracts the necessary parameters,
        and computes the SIDIS differential cross sections for each kinematic point.
        """
        print(f"Loading kinematic data from {data_file}")
        dataset = self.load_kinematic_data(data_file)

        # Extract kinematics
        Vs = dataset["header"]["Vs"]
        targetiso = dataset["header"]["target_isoscalarity"]

        # Setup isoscalar TMDs
        self.setup_isoscalar_tmds(Vs, targetiso)

        # Get kinematic arrays
        x_data = dataset["data"]["x"]
        Q2_data = dataset["data"]["Q2"]
        z_data = dataset["data"]["z"]
        PhT_data = dataset["data"]["PhT"]

        # Convert to PyTorch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        x_tens = torch.tensor(x_data, dtype=dtype, device=device)
        Q2_tens = torch.tensor(Q2_data, dtype=dtype, device=device)
        z_tens = torch.tensor(z_data, dtype=dtype, device=device)
        PhT_tens = torch.tensor(PhT_data, dtype=dtype, device=device)

        Q_tens = torch.sqrt(Q2_tens)
        qT_tens = PhT_tens / z_tens

        # Initialize results tensor
        theo_xsec = torch.zeros_like(qT_tens)

        print(f"Computing SIDIS cross sections for {len(qT_tens)} kinematic points...")

        # Main computation loop
        for iqT in range(len(qT_tens)):
            qTm = float(qT_tens[iqT].item())
            Qm = float(Q_tens[iqT].item())
            xm = float(x_tens[iqT].item())
            zm = float(z_tens[iqT].item())

            # Skip if qT > cut * Q
            # Corresponds to C++ ll. 376-377: if (qTv[iqT] > qToQcut * Qav) continue;
            if qTm > self.qToQcut * Qm:
                print(f"Skipping qT = {qTm:.3f} (above cut)")
                continue

            print(
                f"Computing point {iqT+1}/{len(qT_tens)}: qT = {qTm:.3f}, Q = {Qm:.3f}, x = {xm:.4f}, z = {zm:.4f}"
            )

            # Scales
            # [Corresponds to C++ ll. 372-373.]
            mu = self.Cf * Qm
            zeta = Qm * Qm

            # Yp factor for SIDIS kinematics
            # [Corresponds to C++ ll. 375: const double Yp = 1 + pow(1 - pow(Qm / Vs, 2) / xm, 2);]
            Yp = 1 + (1 - (Qm / Vs) ** 2 / xm) ** 2

            # Number of active flavors
            nf = int(ap.utilities.NF(mu, self.Thresholds))

            # Define b-integrand function
            # [Corresponds to C++ code ll. 389-408: bIntProva function]
            def b_integrand(b):
                # bstar prescription
                bs = self.bstar_min(b, Qm)

                # Luminosity: sum contributions from active quark flavors
                lumiq = 0.0
                for q in range(-nf, nf + 1):
                    if q == 0:  # Skip gluon
                        continue

                    # Get TMD PDF and FF
                    # [Corresponds to C++ ll. 402: lumibsq calculation]
                    try:
                        tmd_pdf = self.TabMatchTMDPDFs.EvaluatexQ(q, xm, bs)
                        tmd_ff = self.TabMatchTMDFFs.EvaluatexQ(q, zm, bs)

                        # Electric charge squared
                        # Corresponds to C++ ll. 456: apfel::QCh2[std::abs(q)-1]
                        qch2 = (
                            ap.constants.QCh2[abs(q) - 1]
                            if abs(q) <= len(ap.constants.QCh2)
                            else 0.0
                        )

                        # Luminosity contribution
                        # [Corresponds to C++ ll. 402 AND 403, because there is the sum sign += here in python.]
                        # [Yp * TabMatchTMDPDFs.EvaluatexQ(...) * apfel::QCh2[...] * TabMatchTMDFFs.EvaluatexQ(...)]
                        lumiq += Yp * tmd_pdf / xm * qch2 * tmd_ff

                    except Exception as e:
                        print(f"Warning: Error evaluating TMDs for q={q}: {e}")
                        continue

                # Non-perturbative evolution functions
                # [Corresponds to C++ ll. 383-386: tf1NP.Evaluate(b) and tf2NP.Evaluate(b)]
                if self.model_fNP is not None:
                    # TODO: Implement fNP model evaluation
                    # Use actual fNP model
                    fnp1 = 1.0  # Placeholder - would need proper fNP evaluation
                    fnp2 = 1.0
                else:
                    # Simple Gaussian form
                    fnp1 = np.exp(-0.1 * b**2)
                    fnp2 = np.exp(-0.05 * b**2)

                # Sudakov factor
                # [Corresponds to the call to the Sudakov at C++ ll. 406: pow(QuarkSudakov(bs, mu, zeta), 2)]
                sudakov_factor = self.QuarkSudakov(bs, mu, zeta) ** 2

                # Hard factor
                # [Corresponds to the call to the hard factor at C++ ll. 402: Hf(mu)]
                hard_factor = self.Hf(mu)

                # Alpha_em^2 factor
                # Corresponds to the call at C++ ll. 462: pow(TabAlphaem.Evaluate(Qm), 2)
                alphaem2 = self.TabAlphaem.Evaluate(Qm) ** 2

                # Full integrand
                # Corresponds to C++ ll. 406: return b * tf1NP.Evaluate(b) * tf2NP.Evaluate(b) * Lumiq * pow(QuarkSudakov(bs, mu, zeta), 2) / zm * pow(TabAlphaem.Evaluate(Qm), 2) * Hf(mu) / pow(Qm, 3);
                integrand = (
                    b
                    * fnp1
                    * fnp2
                    * lumiq
                    * sudakov_factor
                    * alphaem2
                    * hard_factor
                    / (Qm**3)
                    / zm
                )

                return integrand

            # Perform b-space integration
            # [Corresponds to the call to the HERMES case at C++ l. 411-412: DEObj.transform(bIntProva, qTm)]
            try:
                # Perform the Hankel transform using the Ogata quadrature.
                # From bT space to qT space.
                integral_result = self.DEObj.transform(b_integrand, qTm)

                # Differential cross section (numerator only, no denominator)
                # HERMES case, meaning cross section differential in PhT and not PhT2
                differential_xsec = (
                    ap.constants.ConvFact
                    * ap.constants.FourPi
                    * qTm
                    * integral_result
                    / (2 * Qm)
                    / zm
                )

                # Fill the vector with the results in PhT space
                # TODO check the space, qT or PhT
                theo_xsec[iqT] = differential_xsec

                print(f"  -> σ = {differential_xsec:.6e}")

            except Exception as e:
                print(f"  -> Integration failed: {e}")
                theo_xsec[iqT] = 0.0

        # Save results
        self.save_results(dataset, theo_xsec, output_file)
        print(f"Results saved to {output_file}")

    def save_results(self, dataset: Dict, results: torch.Tensor, output_file: str):
        """Save computation results"""
        output_data = {
            "header": dataset["header"],
            "input_data": dataset["data"],
            "results": {
                "cross_sections": results.cpu().numpy().tolist(),
                "units": "differential cross section (no denominator)",
                "computation_info": {
                    "perturbative_order": self.PerturbativeOrder,
                    "pdf_set": self.config["pdfset"]["name"],
                    "ff_set": self.config["ffset"]["name"],
                    "qT_cut": self.qToQcut,
                },
            },
        }

        with open(output_file, "w") as f:
            yaml.dump(output_data, f, default_flow_style=False)


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Compute SIDIS cross sections")
    parser.add_argument("config_file", help="Configuration YAML file")
    parser.add_argument("data_file", help="Kinematic data YAML file")
    parser.add_argument("fnp_config_file", help="fNP configuration YAML file")
    parser.add_argument("output_folder", help="Output folder for results")
    parser.add_argument("output_filename", help="Output YAML filename")

    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Construct full output path
    output_file = os.path.join(args.output_folder, args.output_filename)

    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Output folder: {args.output_folder}")
    print(f"Output file: {output_file}")

    # Initialize computation
    sidis_comp = SIDISComputation(args.config_file, args.fnp_config_file)

    # Run computation
    sidis_comp.compute_sidis_cross_section(args.data_file, output_file)

    print("SIDIS computation completed successfully!")


if __name__ == "__main__":
    main()
