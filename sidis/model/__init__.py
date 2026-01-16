"""
This is the main module for the trainable model for SIDIS and TMDs.
"""

import torch
import pathlib
from omegaconf import OmegaConf
from typing import List, Dict, Any

from .ope import OPE
from .evolution import PERTURBATIVE_EVOLUTION
from .ogata import OGATA
from .fnp_factory import create_fnp_manager
from .qcf0_tmd import TruefNP, TrainablefNP
from .tmd_builder import TMDBuilder
from .structure_functions import FUUT, FUT_SinPhihMinusPhis
from qcdlib.eweak import EWEAK
from qcdlib import params

class TruthModel(torch.nn.Module):
    """Non-trainable ground truth TMD model"""
    def __init__(
        self,
        fnp_config: str = None,
        experimental_target_fragmented_hadron: List[List[str]] = [["p", "pi_plus"]],
    ):
        super().__init__()

        rootdir = pathlib.Path(__file__).resolve().parent
        self.conf = OmegaConf.load(rootdir.joinpath("../config.yaml"))
        
        # Set PyTorch default dtype from config
        dtype_str = self.conf.get('default_dtype', 'float64')
        dtype_map = {'float32': torch.float32, 'float64': torch.float64}
        torch.set_default_dtype(dtype_map[dtype_str])

        if fnp_config is None:
            fnp_config = "fNPconfig_base_flavor_blind.yaml"
        fnp_config_path = rootdir.joinpath("../cards", fnp_config)
        if not fnp_config_path.exists():
            raise FileNotFoundError(f"fNP configuration file not found: {fnp_config_path}")
        self.fnpconf = OmegaConf.load(fnp_config_path)

        self.flavs = ['u', 'd', 's', 'c', 'cb', 'sb', 'db', 'ub']
        self.quark_charges_squared = {'u': 4/9, 'd': 1/9, 's': 1/9, 'c': 4/9, 'cb': 4/9, 'sb': 1/9, 'db': 1/9, 'ub': 4/9}

        self.ope = {}
        self.ope["pdf"] = {} #--this is the unpolarized PDF OPE
        self.ope["ff"] = {} #--this is the unpolarized FF OPE
        # TODO: set up the proper OPE for the Sivers function, Collins function, and transversity function
        self.ope["Sivers"] = {} #--this is the Sivers function OPE
        # self.ope["Collins"] = {} #--this is the Collins function OPE
        # self.ope["h1"] = {} #--this is the transversity function OPE

        for expt_setup in experimental_target_fragmented_hadron:
            self.ope["pdf"][expt_setup[0]] = {}
            self.setup_ope(rootdir=rootdir, type="pdf", hadron=expt_setup[0])
            self.ope["Sivers"][expt_setup[0]] = {}
            print('Setting up the Sivers function OPE for the initial hadron. WARNING: THIS IS NOT SET UP PROPERLY YET. It is just a stopgap measure. Refer to the init for future development.')
            self.setup_ope(rootdir=rootdir, type="Sivers", hadron=expt_setup[0])
            self.ope["ff"][expt_setup[1]] = {}
            self.setup_ope(rootdir=rootdir, type="ff", hadron=expt_setup[1])
            #TODO: set up the OPE for the Collins function
            #self.ope["Collins"][expt_setup[0]] = {}
            #self.setup_ope(rootdir=rootdir, type="Collins", hadron=expt_setup[1])

        self.expt_setup = experimental_target_fragmented_hadron
        self.Q20 = self.conf.Q20

        self.evo = PERTURBATIVE_EVOLUTION(order=self.conf.tmd_resummation_order)
        self.qcf0 = TruefNP(fnp_config=self.fnpconf)
        self.tmd = TMDBuilder(self.ope, self.evo, self.qcf0, self.Q20, self.flavs)

        self.ogata_J0 = OGATA(nu=0)
        self.ogata_J1 = OGATA(nu=1)
        self.stf = {
            'FUUT': FUUT(self.tmd, self.ogata_J0, self.quark_charges_squared, self.flavs),
            'FUTS': FUT_SinPhihMinusPhis(self.tmd, self.ogata_J1, self.quark_charges_squared, self.flavs)
        }

        self.alpha_em = EWEAK()
        self._update_cache()
        self.eval()

    def setup_ope(self, rootdir: pathlib.Path, type: str = "pdf", hadron: str = "p"):
        if type == "Sivers":
            for flav in self.flavs:
                self.ope[type][hadron][flav] = OPE(rootdir.joinpath(self.conf.ope.grid_files["pdf"][hadron][flav]))
        else:
            for flav in self.flavs:
                self.ope[type][hadron][flav] = OPE(rootdir.joinpath(self.conf.ope.grid_files[type][hadron][flav]))

        if hadron == "n":
            ope_copy = {}
            ope_copy["u"] = self.ope[type][hadron]["d"]
            ope_copy["d"] = self.ope[type][hadron]["u"]
            ope_copy["ub"] = self.ope[type][hadron]["db"]
            ope_copy["db"] = self.ope[type][hadron]["ub"]
            self.ope[type][hadron] = ope_copy
        if hadron == "pi_minus":
            ope_copy = {}
            ope_copy["u"] = self.ope[type][hadron]["d"]
            ope_copy["d"] = self.ope[type][hadron]["u"]
            ope_copy["ub"] = self.ope[type][hadron]["db"]
            ope_copy["db"] = self.ope[type][hadron]["ub"]
            self.ope[type][hadron] = ope_copy

    def _update_cache(self):
        # Simplified cache update
        self.cached_grad_mode = torch.is_grad_enabled()

    def forward(self, events_tensor: torch.Tensor, expt_setup: List[str] = ["p", "pi_plus"], rs: float = 140.0) -> torch.Tensor:
        x = events_tensor[:, 0]
        PhT = events_tensor[:, 1]
        Q = events_tensor[:, 2]
        z = events_tensor[:, 3]

        # Compute qT. This is the qT for which the Fourier transform is defined.
        qT = PhT / z
        Q2 = Q**2

        initial_hadron = expt_setup[0]
        fragmented_hadron = expt_setup[1]

        # Always compute unpolarized structure function
        FUUT = self.stf['FUUT'](x, Q2, z, qT, initial_hadron, fragmented_hadron)

        # Compute kinematic factors
        alpha_em = torch.tensor([self.alpha_em.get_alpha(_) for _ in Q2.numpy()])
        gamma = 2 * params.M2 * x / Q
        y = Q2 / x / (rs**2 - params.M2)
        epsilon = (1 - y - 1/4 * gamma**2 * y**2) / (1 - y + 1/2 * y**2 + 1/4 * gamma**2 * y**2)
        sigma0 = 8 * torch.pi**2 * alpha_em**2 * z**2 * qT / x / Q**3 * y**2 / 2 / (1 - epsilon) * (1 + gamma**2 / 2 / x)

        # Only compute Sivers if angles are provided
        if events_tensor.shape[1] >= 6:
            phih = events_tensor[:, 4]
            phis = events_tensor[:, 5]
            FUT_sin_phih_minus_phis = self.stf['FUTS'](x, Q2, z, qT, initial_hadron, fragmented_hadron)
            xsec = sigma0 * (FUUT + torch.sin(phih - phis) * FUT_sin_phih_minus_phis)
        else:
            # Unpolarized only
            xsec = sigma0 * FUUT

        return xsec

class TrainableModel(TruthModel):
    """Trainable TMD model with version tracking"""
    def __init__(
        self,
        fnp_config: str = None,
        experimental_target_fragmented_hadron: List[List[str]] = [["p", "pi_plus"]],
    ):
        # Initialize TruthModel components
        super().__init__(fnp_config, experimental_target_fragmented_hadron)

        self.train()

        # Replace qcf0 with trainable version
        self.qcf0 = TrainablefNP(fnp_config=self.fnpconf)

        # Rebuild TMDBuilder and structure functions with the trainable fNP
        self.tmd = TMDBuilder(self.ope, self.evo, self.qcf0, self.Q20, self.flavs)
        self.stf = {
            'FUUT': FUUT(self.tmd, self.ogata_J0, self.quark_charges_squared, self.flavs),
            'FUTS': FUT_SinPhihMinusPhis(self.tmd, self.ogata_J1, self.quark_charges_squared, self.flavs)
        }

        self.versions: List[int] = self.qcf0.version()
        self._update_cache()

    def forward(self, events_tensor: torch.Tensor, expt_setup: List[str] = ["p", "pi_plus"], rs: float = 140.0) -> torch.Tensor:
        versions = self.qcf0.version()
        newver = any(v1 != v2 for v1, v2 in zip(self.versions, versions))
        newmode = self.cached_grad_mode != torch.is_grad_enabled()

        if newver or newmode:
            self.tmd.clear_cache()
            self._update_cache()
            self.versions = versions

        return super().forward(events_tensor, expt_setup, rs)
