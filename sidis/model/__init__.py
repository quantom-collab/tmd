"""
This is the main module for the trainable model for SIDIS and TMDs.
"""

import torch
import pathlib
from omegaconf import OmegaConf
from .ope import OPE
from .evolution import PERTURBATIVE_EVOLUTION
from .ogata import OGATA
from .fnp_factory import create_fnp_manager


class TrainableModel(torch.nn.Module):
    def __init__(self):  # TODO: add configuration in the init
        super().__init__()

        # Load configuration from YAML file
        # rootdir is the folder named model
        rootdir = pathlib.Path(__file__).resolve().parent

        # rootdir.joinpath puts everything relative to the model folder
        self.conf = OmegaConf.load(rootdir.joinpath("../config.yaml"))

        # Load fNP config from cards folder
        # Change this path to switch between available configurations.
        # fnp_config_path = rootdir.joinpath("../cards/fNPconfig_flav_dep_std.yaml")
        fnp_config_path = rootdir.joinpath("../cards/fNPconfig_flav_blind_std.yaml")

        self.fnpconf = OmegaConf.load(fnp_config_path)

        self.opepdf = OPE(rootdir.joinpath(self.conf.ope.grid_file))
        # self.opeff = OPE() # NOTE: not implemented yet

        self.evo = PERTURBATIVE_EVOLUTION()

        self.ogata = OGATA()

        # Create fNP manager based on its config
        self.nonperturbative = create_fnp_manager(config_dict=self.fnpconf)

    def forward(self, events_tensor: torch.Tensor) -> torch.Tensor:
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
        ope = self.opepdf(x, bT)

        # Perturbative evolution
        evolution = self.evo(bT, Q20, Q2)

        # Check
        # print('ope.shape',ope.shape, 'evolution.shape', evolution.shape)

        # Compute non-perturbative fNP (depends on x, z, bT, and Q)
        # Zeta is computed internally as zeta = Q² (standard SIDIS)
        fNP = self.nonperturbative(x, z, bT, Q)

        ftilde = ope * evolution * fNP["pdfs"]["u"]  # Using up quark as example

        # #--for fragmentation functions (need to build)
        # ope = self.opeff(x, bT)

        # Check
        # print('ope.shape',ope.shape, 'evolution.shape', evolution.shape)

        # Dtilde = ope * evolution * nonperturbative
        # #ftilde = evolution

        # Build integrand, sum over quark flavors
        integrand = ftilde
        # integrand = Dtilde * ftilde

        # Check
        # print('ope.shape',ope.shape, 'evolution.shape', evolution.shape)

        # Hankel transform
        ogata = self.ogata.eval_ogata_func_var_h(integrand, bT, qT)

        # Check
        # print('ogata.shape', ogata.shape)

        return ogata
