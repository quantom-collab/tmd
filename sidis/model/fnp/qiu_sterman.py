# Here we will implement the Qiu-Sterman function fNP. This is a collinear function.

import torch
import torch.nn as nn

class QiuSterman(nn.Module):
    def __init__(self, N: float, a: float, b: float):
        super().__init__()
        self.N = N
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.N * x**self.a * (1-x)**self.b