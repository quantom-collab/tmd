"""
Model subpackage for the SIDIS project.

This file exposes the main `EICModel` class so that it can be imported as:

    from sidis.model import EICModel
"""

from .EIC import EICModel

__all__ = ["EICModel"]

# """
# This is the main module for the trainable model for SIDIS and TMDs.
# """

# import torch
# import pathlib
# from omegaconf import OmegaConf
# from typing import List
# from .ope import OPE
# from .evolution import PERTURBATIVE_EVOLUTION
# from .ogata import OGATA, OGATA1
# from .fnp_factory import create_fnp_manager
# from qcdlib.eweak import EWEAK
# from qcdlib import params


# class TrainableModel(torch.nn.Module):
#     def __init__(
#         self, 
#         fnp_config: str = None,
#         experimental_target_fragmented_hadron: List[str] = [["p","pi_plus"]],
#     ):
#         """
#         Initialize the trainable model for SIDIS TMD cross-section computation.

#         The default experimental setup is:
#             - e (electron) + p (proton) -> X + pi+
        
#         Additional work will be needed to extend this to other experimental setups.

#         Args:
#             fnp_config (str, optional): Name of the fNP configuration file in cards/ directory.
#                                        If None, defaults to 'fNPconfig_base_flavor_blind.yaml'.
#         """
#         super().__init__()

#         # Load configuration from YAML file
#         # rootdir is the folder named model
#         rootdir = pathlib.Path(__file__).resolve().parent

#         # rootdir.joinpath puts everything relative to the model folder
#         self.conf = OmegaConf.load(rootdir.joinpath("../config.yaml"))

#         # Load fNP config from cards folder
#         # If fnp_config is not provided, use default
#         if fnp_config is None:
#             fnp_config = "fNPconfig_base_flavor_blind.yaml"

#         # Construct full path to config file in cards/ directory
#         fnp_config_path = rootdir.joinpath("../cards", fnp_config)

#         # Validate that the config file exists
#         if not fnp_config_path.exists():
#             raise FileNotFoundError(
#                 f"fNP configuration file not found: {fnp_config_path}\n"
#                 f"Please ensure the file exists in the cards/ directory."
#             )

#         self.fnpconf = OmegaConf.load(fnp_config_path)
 
#         flavs = ['u','d','s','c','cb','sb','db','ub'] #--have to get this from the config file; need to update this later
#         self.flavs = flavs

#         self.quark_charges_squared = {'u': 4/9, 'd': 1/9, 's': 1/9, 'c': 4/9, 'cb': 4/9, 'sb': 1/9, 'db': 1/9, 'ub': 4/9}

#         self.ope = {}
#         self.ope["pdf"] = {}
#         self.ope["ff"] = {}
#         for expt_setup in experimental_target_fragmented_hadron:
#             self.ope["pdf"][expt_setup[0]] = {}
#             self.ope["ff"][expt_setup[1]] = {}
#             self.setup_ope(rootdir=rootdir, type="pdf", hadron=expt_setup[0])
#             self.setup_ope(rootdir=rootdir, type="ff", hadron=expt_setup[1])

#         self.expt_setup = experimental_target_fragmented_hadron

#         self.evo = PERTURBATIVE_EVOLUTION(order=self.conf.tmd_resummation_order)

#         self.ogata0 = OGATA()      # J0 Hankel transform for FUUT
#         self.ogata1 = OGATA1()    # J1 Hankel transform for FUTS

#         # Create fNP manager based on its config (includes Sivers function)
#         self.nonperturbative = create_fnp_manager(config_dict=self.fnpconf)

#         #--set up alpha_EM
#         self.alpha_em = EWEAK()

#     def setup_ope(self, rootdir: pathlib.Path, type: str = "pdf", hadron: str = "p"):
#         """
#         Setup the OPE for the given type and hadron.
#         We set up the instance of the class for each flavor of the hadron.
#         Here we also perform flavor mixing,
#             - for neutron, we take the proton u -> d, d -> u, ub -> db, db -> ub
#             - for pi-, we take the pi+ u -> d, d -> u, ub -> db, db -> ub
#         Args:
#             type: "pdf" or "ff"
#             hadron: "p" or "n" or "pi_plus" or "pi_minus"
#         Returns:
#             None
#         """

#         for flav in self.flavs:
#             self.ope[type][hadron][flav] = OPE(rootdir.joinpath(self.conf.ope.grid_files[type][hadron][flav]))

#         if hadron == "n":
#             ope_copy = {}
#             ope_copy["u"] = self.ope[type][hadron]["d"]
#             ope_copy["d"] = self.ope[type][hadron]["u"]
#             ope_copy["ub"] = self.ope[type][hadron]["db"]
#             ope_copy["db"] = self.ope[type][hadron]["ub"]
#             self.ope[type][hadron] = ope_copy
#         if hadron == "pi_minus":
#             ope_copy = {}
#             ope_copy["u"] = self.ope[type][hadron]["d"]
#             ope_copy["d"] = self.ope[type][hadron]["u"]
#             ope_copy["ub"] = self.ope[type][hadron]["db"]
#             ope_copy["db"] = self.ope[type][hadron]["ub"]
#             self.ope[type][hadron] = ope_copy

#     # def get_TMDPDF(self, x: torch.Tensor, bT: torch.Tensor, Q: torch.Tensor, initial_hadron: str = "p") -> Dict[str, torch.Tensor]:
#     #     """
#     #     This function is a work in progress.
#     #     Compute the TMD PDF for a given x, bT, and Q.
#     #     Args:
#     #         x: Bjorken x
#     #         bT: Transverse distance in GeV^-1
#     #         Q: Hard scale in GeV
#     #     Returns:
#     #         TMD PDF for all flavors
#     #     """
#     #     evolution = self.evo(bT, self.conf.Q20, Q**2)
#     #     fNP = self.nonperturbative(x, bT, Q) # This needs to be updated! We should have a function to compute only the fNP for the TMDPDF.

#     #     TMD_PDF = {}
#     #     for flav in self.flavs:
#     #         if 'b' in flav and len(flav) > 1:
#     #             npflav = flav[0]+'bar'
#     #         else:
#     #             npflav = flav
#     #         TMD_PDF[flav] = self.ope["pdf"][initial_hadron][flav](x, bT) * evolution * fNP["pdfs"][npflav]
#     #     return TMD_PDF

#     def forward(self, events_tensor: torch.Tensor, expt_setup: List[str] = ["p","pi_plus"], s: float = 140.0) -> torch.Tensor:
#         """
#         Forward pass for batch of events.
#         events_tensor: shape (n_events, 6) containing [x, PhT, Q, z, phih, phis] for each event
#         """
#         # Unpack the variables from the events tensor
#         x = events_tensor[:, 0]  # Bjorken x
#         PhT = events_tensor[:, 1]  # Transverse momentum of detected hadron
#         Q = events_tensor[:, 2]  # Hard scale
#         z = events_tensor[:, 3]  # Energy fraction of hadron relative to struck quark
#         if len(events_tensor.shape) == 6:
#             phih = events_tensor[:, 4]  # Azimuthal angle of hadron (radians)
#         else:
#             phih = torch.zeros_like(x)
#         if len(events_tensor.shape) == 6:
#             phis = events_tensor[:, 5]  # Azimuthal angle of target spin (radians)
#         else:
#             phis = torch.zeros_like(x)

#         # Compute qT
#         qT = PhT / z

#         # Setting the bT values for ogata quadrature (different grids for J0 and J1)
#         bT0 = self.ogata0.get_bTs(qT)  # bT grid for J0 (FUUT)
#         bT1 = self.ogata1.get_bTs(qT)  # bT grid for J1 (FUTS)

#         # Setting the initial scale
#         Q20 = self.conf.Q20
#         Q2 = Q**2

#         # OPE for TMD PDFs and FFs on bT0 grid (for FUUT)
#         initial_hadron = expt_setup[0]
#         fragmented_hadron = expt_setup[1]
#         opepdf0 = {}
#         opeff0 = {}
#         for flav in self.flavs:
#             opepdf0[flav] = self.ope["pdf"][initial_hadron][flav](x, bT0)
#             opeff0[flav] = self.ope["ff"][fragmented_hadron][flav](z, bT0)

#         # OPE for TMD PDFs and FFs on bT1 grid (for FUTS)
#         opepdf1 = {}
#         opeff1 = {}
#         for flav in self.flavs:
#             opepdf1[flav] = self.ope["pdf"][initial_hadron][flav](x, bT1)
#             opeff1[flav] = self.ope["ff"][fragmented_hadron][flav](z, bT1)

#         # Perturbative evolution on both grids
#         evolution0 = self.evo(bT0, Q20, Q2)
#         evolution1 = self.evo(bT1, Q20, Q2)

#         # Compute non-perturbative fNP on both grids
#         # Zeta is computed internally as zeta = Q² (standard SIDIS)
#         fNP0 = self.nonperturbative(x, z, bT0, Q)
#         fNP1 = self.nonperturbative(x, z, bT1, Q)

#         # Compute Sivers function on bT1 grid (for FUTS)
#         sivers1 = self.nonperturbative.forward_sivers(x, bT1, Q)
        
#         # Set up the integrand for the FUUT structure function (on bT0 grid)
#         FUUT_integrand = torch.zeros_like(bT0)
#         for flav in self.flavs:
#             if 'b' in flav and len(flav) > 1:
#                 npflav = flav[0]+'bar'
#             else:
#                 npflav = flav
#             FUUT_integrand += self.quark_charges_squared[flav] * opepdf0[flav] * opeff0[flav] * evolution0**2 * fNP0["pdfs"][npflav] * fNP0["ffs"][npflav]

#         # Set up the integrand for the FUTS structure function (on bT1 grid)
#         # Uses Sivers function instead of regular PDF fNP
#         FUTS_integrand = torch.zeros_like(bT1)
#         for flav in self.flavs:
#             if 'b' in flav and len(flav) > 1:
#                 npflav = flav[0]+'bar'
#             else:
#                 npflav = flav
#             FUTS_integrand += bT1/2 * self.quark_charges_squared[flav] * opepdf1[flav] * opeff1[flav] * evolution1**2 * sivers1 * fNP1["ffs"][npflav]

#         # Hankel transform for FUUT structure function (J0)
#         FUUT = self.ogata0.eval_ogata_func_var_h(FUUT_integrand, bT0, qT)
#         # Hankel transform for FUTS structure function (J1)
#         FUTS = self.ogata1.eval_ogata_func_var_h(FUTS_integrand, bT1, qT)

#         # Compute the sigma0s for the SIDIS cross-section
#         alpha_em = torch.tensor([self.alpha_em.get_alpha(_) for _ in Q2.numpy()])
#         # Factors taken from Bacchetta, et al. JHEP 02 (2007) 093
#         gamma = 2 * params.M2 * x / Q
#         y = Q2 / x / (s - params.M2)
#         epsilon = (1 - y - 1/4 * gamma**2 * y**2) / (1 - y + 1/2 * y**2 + 1/4 * gamma**2 * y**2)
#         sigma0 = 8 * torch.pi**2 * alpha_em**2 * z**2 * qT / x / Q**3 * y**2 / 2 / (1 - epsilon) * (1 + gamma**2 / 2 / x)

#         # Compute the SIDIS cross-section according to the sigma0s and the structure functions
#         xsec = sigma0 * (FUUT + torch.sin(phih - phis) * FUTS)

#         return xsec
