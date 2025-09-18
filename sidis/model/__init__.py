"""
This is the main module for the trainable model for SIDS and TMDs.
"""

import torch
from .ope import OPE
from .evolution import PERTURBATIVE_EVOLUTION

class TrainableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.opepdf = OPE()
        self.evo = PERTURBATIVE_EVOLUTION()

    def forward(self, x:torch.Tensor, pT:torch.Tensor, Q:torch.Tensor, z:torch.Tensor) -> torch.Tensor:

        bT = torch.linspace(0.001,10,100)
        Q20 = torch.tensor([1])
        Q2 = Q**2

        ope = self.opepdf(x, bT)
        evolution = self.evo(bT, Q20, Q2)
        print('ope.shape',ope.shape, 'evolution.shape', evolution.shape)
        #--call real Ogata quadrature here
        return ope @ evolution