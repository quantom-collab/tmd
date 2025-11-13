"""
This is the main module for the trainable model for SIDIS and TMDs.
"""

import torch
import pathlib
from omegaconf import OmegaConf
from typing import List
from .ope import OPE
from .evolution import PERTURBATIVE_EVOLUTION
from .ogata import OGATA
from .fnp_factory import create_fnp_manager
from qcdlib.eweak import EWEAK
from qcdlib import params


class TrainableModel(torch.nn.Module):
    def __init__(
        self, 
        fnp_config: str = None,
        experimental_target_fragmented_hadron: List[str] = [["p","pi_plus"]],
    ):
        """
        Initialize the trainable model for SIDIS TMD cross-section computation.

        The default experimental setup is:
            - e (electron) + p (proton) -> X + pi+
        
        Additional work will be needed to extend this to other experimental setups.

        Args:
            fnp_config (str, optional): Name of the fNP configuration file in cards/ directory.
                                       If None, defaults to 'fNPconfig_base_flavor_blind.yaml'.
        """
        super().__init__()

        # Load configuration from YAML file
        # rootdir is the folder named model
        rootdir = pathlib.Path(__file__).resolve().parent

        # rootdir.joinpath puts everything relative to the model folder
        self.conf = OmegaConf.load(rootdir.joinpath("../config.yaml"))

        # Load fNP config from cards folder
        # If fnp_config is not provided, use default
        if fnp_config is None:
            fnp_config = "fNPconfig_base_flavor_blind.yaml"

        # Construct full path to config file in cards/ directory
        fnp_config_path = rootdir.joinpath("../cards", fnp_config)

        # Validate that the config file exists
        if not fnp_config_path.exists():
            raise FileNotFoundError(
                f"fNP configuration file not found: {fnp_config_path}\n"
                f"Please ensure the file exists in the cards/ directory."
            )

        self.fnpconf = OmegaConf.load(fnp_config_path)
 
        flavs = ['u','d','s','c','cb','sb','db','ub'] #--have to get this from the config file; need to update this later
        self.flavs = flavs

        self.quark_charges_squared = {'u': 4/9, 'd': 1/9, 's': 1/9, 'c': 4/9, 'cb': 4/9, 'sb': 1/9, 'db': 1/9, 'ub': 4/9}

        self.ope = {}
        self.ope["pdf"] = {}
        self.ope["ff"] = {}
        for expt_setup in experimental_target_fragmented_hadron:
            self.ope["pdf"][expt_setup[0]] = {}
            self.ope["ff"][expt_setup[1]] = {}
            self.setup_ope(rootdir=rootdir, type="pdf", hadron=expt_setup[0])
            self.setup_ope(rootdir=rootdir, type="ff", hadron=expt_setup[1])

        self.expt_setup = experimental_target_fragmented_hadron

        self.evo = PERTURBATIVE_EVOLUTION(order=self.conf.tmd_resummation_order)

        self.ogata = OGATA()

        # Create fNP manager based on its config
        self.nonperturbative = create_fnp_manager(config_dict=self.fnpconf)

        #--set up alpha_EM
        self.alpha_em = EWEAK()

    def setup_ope(self, rootdir: pathlib.Path, type: str = "pdf", hadron: str = "p"):
        """
        Setup the OPE for the given type and hadron.
        We set up the instance of the class for each flavor of the hadron.
        Here we also perform flavor mixing,
            - for neutron, we take the proton u -> d, d -> u, ub -> db, db -> ub
            - for pi-, we take the pi+ u -> d, d -> u, ub -> db, db -> ub
        Args:
            type: "pdf" or "ff"
            hadron: "p" or "n" or "pi_plus" or "pi_minus"
        Returns:
            None
        """

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

    def forward(self, events_tensor: torch.Tensor, expt_setup: List[str] = ["p","pi_plus"], s: float = 140.0) -> torch.Tensor:
        """
        Forward pass for batch of events.
        events_tensor: shape (n_events, 4) containing [x, PhT, Q, z] for each event
        """
        # Unpack the variables from the events tensor
        x = events_tensor[:, 0]  # Bjorken x
        PhT = events_tensor[:, 1]  # Transverse momentum of detected hadron
        Q = events_tensor[:, 2]  # Hard scale
        z = events_tensor[:, 3]  # Energy fraction of hadron relative to struck quark

        # Compute qT
        qT = PhT / z

        # Setting the bT values for ogata, given qT
        bT = self.ogata.get_bTs(qT)

        # Setting the initial scale
        Q20 = self.conf.Q20
        Q2 = Q**2

        # OPE for TMD PDFs
        # ope = self.opepdf(x, bT)
        initial_hadron = expt_setup[0]
        fragmented_hadron = expt_setup[1]
        opepdf = {}
        for flav in self.flavs:
            opepdf[flav] = self.ope["pdf"][initial_hadron][flav](x, bT)
        opeff = {}
        for flav in self.flavs:
            opeff[flav] = self.ope["ff"][fragmented_hadron][flav](z, bT)

        # Perturbative evolution
        evolution = self.evo(bT, Q20, Q2)

        # Check
        # print('ope.shape',ope.shape, 'evolution.shape', evolution.shape)

        # Compute non-perturbative fNP (depends on x, z, bT, and Q)
        # Zeta is computed internally as zeta = Q² (standard SIDIS)
        fNP = self.nonperturbative(x, z, bT, Q)
        
        # Set up the integrand for the FUUT structure function
        FUUT_integrand = torch.zeros_like(bT)
        for flav in self.flavs:
            if 'b' in flav and len(flav) > 1:
                npflav = flav[0]+'bar'
            else:
                npflav = flav
            FUUT_integrand += self.quark_charges_squared[flav] * opepdf[flav] * opeff[flav] * evolution**2 * fNP["pdfs"][npflav] * fNP["ffs"][npflav]

        # Hankel transform for FUUT structure function
        FUUT = self.ogata.eval_ogata_func_var_h(FUUT_integrand, bT, qT)

        # Compute the prefactors for the SIDIS cross-section
        alpha_em = torch.tensor([self.alpha_em.get_alpha(_) for _ in Q2.numpy()])
        # Factors taken from Bacchetta, et al. JHEP 02 (2007) 093
        gamma = 2 * params.M2 * x / Q
        y = Q2 / x / (s - params.M2)
        epsilon = (1 - y - 1/4 * gamma**2 * y**2) / (1 - y + 1/2 * y**2 + 1/4 * gamma**2 * y**2)
        prefactor = 8 * torch.pi**2 * alpha_em**2 * z**2 * qT / x / Q**3 * y**2 / 2 / (1 - epsilon) * (1 + gamma**2 / 2 / x)

        # Compute the SIDIS cross-section according to the prefactors and the structure functions
        xsec = prefactor * FUUT

        return xsec
