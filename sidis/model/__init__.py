"""
This is the main module for the trainable model for SIDIS and TMDs.
"""

import torch
from .ope import OPE
from .evolution import PERTURBATIVE_EVOLUTION
from .evolution_events import PERTURBATIVE_EVOLUTION_EVENT
from .ogata import OGATA


class TrainableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.opepdf = OPE()
        # self.evo = PERTURBATIVE_EVOLUTION()
        self.evo = PERTURBATIVE_EVOLUTION_EVENT()

        self.ogata = OGATA()

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

        qT = PhT / z

        bT = self.ogata.get_bTs(qT)

        # bT = torch.linspace(0.001,10,100)
        Q20 = torch.tensor([1])  # --we should get this from config
        Q2 = Q**2

        ope = self.opepdf(x, bT)
        evolution = self.evo(bT, Q20, Q2)
        # print('ope.shape',ope.shape, 'evolution.shape', evolution.shape)
        # nonperturbative = self.nonperturbative(x, bT, Q2, Q20)

        ftilde = ope * evolution  # * nonperturbative

        # #--for fragmentation functions (need to build)
        # ope = self.opepdf(x, bT)
        # evolution = self.evo(bT, Q20, Q2)
        # print('ope.shape',ope.shape, 'evolution.shape', evolution.shape)
        # nonperturbative = self.nonperturbative(x, bT, Q2, Q20)

        # Dtilde = ope * evolution * nonperturbative
        # #ftilde = evolution
        # integrand = Dtilde * ftilde #--sum over quark flavors
        # #print('ope.shape',ope.shape, 'evolution.shape', evolution.shape)
        # print('evolution.shape', evolution.shape)

        # ogata = self.ogata.eval_ogata_func_var_h(integrand, bT, qT)
        ogata = self.ogata.eval_ogata_func_var_h(ftilde, bT, qT)
        # print('ogata.shape', ogata.shape)
        return ogata

        # --call real Ogata quadrature here
        # return ope @ evolution
