# Here we will implement the Sivers function fNP

import torch
import torch.nn as nn

class Sivers(nn.Module):
    def __init__(self, w: float):
        super().__init__()
        self.w = w

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.w * b**2)