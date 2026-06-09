"""Kernel and shift-operator tests."""

from __future__ import annotations

import tempfile

import pytest
import torch

from Spin2 import alphaS, params
from Spin2.dglap import NonSingletDGLAP
from Spin2.kernels import NonSingletKernels


@pytest.fixture
def small_grid():
    return dict(
        dev="cpu",
        dtype=torch.float64,
        order=0,
        nx=8,
        xmin=1e-3,
        xmax=0.99,
        nQ2=5,
        Q20=params.mc2,
        LQ2max=3.0,
        loadgrid=False,
        steps=3,
        ng=12,
    )


def _jamx_convolution(dglap: NonSingletDGLAP, mu2: float, q: torch.Tensor) -> torch.Tensor:
    K = dglap.kernels.K
    Kxi = K[0](dglap.xg_2D, dglap.xig_2D, mu2)
    Kx = K[1](dglap.xg_2D, dglap.xig_2D, mu2)
    qg = torch.einsum("gji,i->gj", dglap.Lg, q)
    out = torch.einsum("g,gj,gj,gj->j", dglap.wg, dglap.jac, Kxi, qg)
    out += torch.einsum("g,gj,gj,j->j", dglap.wg, dglap.jac, Kx, q)
    return out + K[2](dglap.x_2D, dglap.xi_2D, mu2) * q


@pytest.mark.parametrize("kernel_type", ("unpolarized", "transversity", "qiu_sterman"))
def test_get_shift_matches_jamx_convolution(small_grid, kernel_type):
    x = torch.linspace(0.05, 0.9, small_grid["nx"], dtype=torch.float64)
    q = x * (1.0 - x) ** 2
    mu2 = float(small_grid["Q20"])
    with tempfile.TemporaryDirectory() as grid_dir:
        gkw = {**small_grid, "kernel_type": kernel_type, "grid_dir": grid_dir}
        if kernel_type == "qiu_sterman":
            gkw["eta"] = params.NC
        d = NonSingletDGLAP(**gkw)
        torch.testing.assert_close(d.get_shift(mu2, d.kernels.K) @ q, _jamx_convolution(d, mu2, q))


def test_transversity_first_moment_negative(small_grid):
    grid = {**small_grid, "nx": 16, "ng": 16}
    with tempfile.TemporaryDirectory() as grid_dir:
        d = NonSingletDGLAP(kernel_type="transversity", grid_dir=grid_dir, **grid)
    x = d.x
    f = x * (1.0 - x) ** 2
    f = f / torch.trapz(f, x)
    mu2 = float(d.Q20)
    assert torch.trapz(_jamx_convolution(d, mu2, f), x).item() < 0.0


def test_unpolarized_first_moment_near_zero(small_grid):
    grid = {**small_grid, "nx": 16, "ng": 16}
    with tempfile.TemporaryDirectory() as grid_dir:
        d = NonSingletDGLAP(kernel_type="unpolarized", grid_dir=grid_dir, **grid)
    x = d.x
    f = x * (1.0 - x) ** 2
    f = f / torch.trapz(f, x)
    mu2 = float(d.Q20)
    assert abs(torch.trapz(_jamx_convolution(d, mu2, f), x).item()) < 0.05


def test_eta_zero_matches_unpolarized(small_grid):
    with tempfile.TemporaryDirectory() as grid_dir:
        d0 = NonSingletDGLAP(eta=0.0, grid_dir=grid_dir, **small_grid)
        dnc = NonSingletDGLAP(eta=params.NC, grid_dir=grid_dir, **small_grid)
    mu2 = float(d0.Q20)
    diff = dnc.get_shift(mu2, dnc.kernels.K) - d0.get_shift(mu2, d0.kernels.K)
    off = diff.clone()
    off[range(diff.shape[0]), range(diff.shape[1])] = 0.0
    assert (dnc.get_shift(mu2, dnc.kernels.K).diagonal() - d0.get_shift(mu2, d0.kernels.K).diagonal()).abs().max() > 0
    assert off.abs().max() < 1e-12


def test_nc_delta_in_kc():
    k0 = NonSingletKernels("cpu", eta=0.0, order=0)
    knc = NonSingletKernels("cpu", eta=params.NC, order=0)
    x = torch.tensor([[0.5]], dtype=torch.float64)
    q2 = float(params.mc2)
    a = alphaS.get_alphaS(q2) / (2.0 * 3.141592653589793)
    got = (knc.K[2](x, x, q2) - k0.K[2](x, x, q2)).item()
    assert got == pytest.approx(-a * params.NC)
