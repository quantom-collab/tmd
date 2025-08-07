#!/usr/bin/env python3
"""
SIDIS Cross Section Computation using APFEL++ Python wrapper

This script computes SIDIS differential cross sections for given kinematic points.

Usage:
    python3.10 sidis_computation.py <config_file> <kinematic_data_file> <output_file>
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
    """

    def __init__(self, config_file: str):
        """
        Initialize the computation with configuration file.
        @param config_file: Path to the configuration YAML file
        """
        self.config = self._load_config(config_file)
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
        # If the key qToQcut exists, getattr returns its value;
        # 0.3 is the fallback value
        self.qToQcut = getattr(self.config, "qToQcut", 0.3)

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

        It is the python implementation of ll. 49 - 59 and 66 -70 in SIDISMultiplicities.cc
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

        It is the python implementation of ll. 61 - 64 (for Alpha_s) and 82 - 85
        (for Alpha_em) in SIDISMultiplicities.cc
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
        Setup TMD PDF objects
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
        """Setup TMD FF objects"""
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
        """Setup non-perturbative functions"""
        try:
            config_file_path = "config/config.yaml"
            if os.path.exists(config_file_path):
                config_fnp = utl.load_yaml_config(config_file_path)
                self.model_fNP = fNP(config_fnp)
            else:
                # Use simple Gaussian form as fallback
                self.model_fNP = None
                print("Warning: fNP config not found, using simple Gaussian form")
        except Exception as e:
            print(f"Warning: Could not load fNP model: {e}")
            self.model_fNP = None

    def bstar_min(self, b: float, Q: float) -> float:
        """
        Simple bstar prescription (bstar_min variant)
        b: impact parameter (float)
        Q: hard scale (float)
        """
        bmax = 1.5  # GeV^-1, typical value used in TMD phenomenology
        gamma_E = 0.5772156649015329  # Euler-Mascheroni constant
        return b / np.sqrt(1 + (b / bmax) ** 2) * 2 * np.exp(-gamma_E) / Q

    def setup_isoscalar_tmds(self, Vs: float, targetiso: float):
        """Setup isoscalar TMD PDFs and FFs"""
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
        """Load kinematic data from file"""
        with open(data_file, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    def compute_sidis_cross_section(self, data_file: str, output_file: str):
        """Main computation function"""
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
            if qTm > self.qToQcut * Qm:
                print(f"Skipping qT = {qTm:.3f} (above cut)")
                continue

            print(
                f"Computing point {iqT+1}/{len(qT_tens)}: qT = {qTm:.3f}, Q = {Qm:.3f}, x = {xm:.4f}, z = {zm:.4f}"
            )

            # Scales
            mu = self.Cf * Qm
            zeta = Qm * Qm

            # Yp factor for SIDIS kinematics
            Yp = 1 + (1 - (Qm / Vs) ** 2 / xm) ** 2

            # Number of active flavors
            nf = int(ap.utilities.NF(mu, self.Thresholds))

            # Define b-integrand function
            def b_integrand(b):
                # bstar prescription
                bs = self.bstar_min(b, Qm)

                # TMD luminosity: sum over active quark flavors
                lumiq = 0.0
                for q in range(-nf, nf + 1):
                    if q == 0:  # Skip gluon
                        continue

                    # Get TMD PDF and FF
                    try:
                        tmd_pdf = self.TabMatchTMDPDFs.EvaluatexQ(q, xm, bs)
                        tmd_ff = self.TabMatchTMDFFs.EvaluatexQ(q, zm, bs)

                        # Electric charge squared
                        qch2 = (
                            ap.constants.QCh2[abs(q) - 1]
                            if abs(q) <= len(ap.constants.QCh2)
                            else 0.0
                        )

                        # Luminosity contribution
                        lumiq += Yp * tmd_pdf / xm * qch2 * tmd_ff
                    except Exception as e:
                        print(f"Warning: Error evaluating TMDs for q={q}: {e}")
                        continue

                # Non-perturbative evolution factors
                if self.model_fNP is not None:
                    # Use actual fNP model
                    fnp1 = 1.0  # Placeholder - would need proper fNP evaluation
                    fnp2 = 1.0
                else:
                    # Simple Gaussian form
                    fnp1 = np.exp(-0.1 * b**2)
                    fnp2 = np.exp(-0.05 * b**2)

                # Sudakov factor
                sudakov_factor = self.QuarkSudakov(bs, mu, zeta) ** 2

                # Hard factor
                hard_factor = self.Hf(mu)

                # Alpha_em^2 factor
                alphaem2 = self.TabAlphaem.Evaluate(Qm) ** 2

                # Full integrand
                integrand = (
                    b
                    * fnp1
                    * fnp2
                    * lumiq
                    * sudakov_factor
                    * alphaem2
                    * hard_factor
                    / (Qm**3 * zm)
                )

                return integrand

            # Perform b-space integration
            try:
                integral_result = self.DEObj.transform(b_integrand, qTm)

                # Differential cross section (numerator only, no denominator)
                differential_xsec = (
                    ap.constants.ConvFact
                    * ap.constants.FourPi
                    * qTm
                    * integral_result
                    / (2 * Qm)
                    / zm
                )

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
    """Main function"""
    parser = argparse.ArgumentParser(description="Compute SIDIS cross sections")
    parser.add_argument("config_file", help="Configuration YAML file")
    parser.add_argument("data_file", help="Kinematic data YAML file")
    parser.add_argument("output_file", help="Output YAML file")

    args = parser.parse_args()

    # Check Python version
    print(f"Python version: {sys.version}")

    # Initialize computation
    sidis_comp = SIDISComputation(args.config_file)

    # Run computation
    sidis_comp.compute_sidis_cross_section(args.data_file, args.output_file)

    print("SIDIS computation completed successfully!")


if __name__ == "__main__":
    main()
