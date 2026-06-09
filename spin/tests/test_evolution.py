"""Evolution matrix application and channel tests."""

from __future__ import annotations

import tempfile

import pytest
import torch

from spin import params
from spin.collins import COLLINS_CHANNELS, build_collins_hhat_initial
from spin.dglap import NonSingletDGLAP
from spin.evolution import (
    CollinsHhatEvolution,
    QiuStermanEvolution,
    TransversityEvolution,
    evolve_collins_hhat,
    evolve_flavors,
    evolve_transversity,
)
from spin.flavors import pack_TF_by_pdg, pdg_to_slot
from spin.qiu_sterman import FLAVORS, QiuStermanParams, build_TF_initial
from spin.transversity import TransversityParams, build_h1_initial
from spin.validation.inputs import load_pion_ff, load_transversity_inputs


@pytest.fixture
def small_grid():
    return dict(
        dev="cpu",
        dtype=torch.float64,
        order=0,
        kernel_type="transversity",
        nx=8,
        xmin=0.02,
        xmax=0.8,
        nQ2=5,
        Q20=2.4,
        LQ2max=3.0,
        loadgrid=False,
        steps=3,
        ng=12,
    )


def test_evolve_applies_M_not_MT(small_grid):
    x = torch.linspace(0.05, 0.75, small_grid["nx"], dtype=torch.float64)
    f0 = x * (1.0 - x) ** 2
    with tempfile.TemporaryDirectory() as grid_dir:
        d = NonSingletDGLAP(grid_dir=grid_dir, **small_grid)
    out = d.evolve(f0)
    for k in range(d.nQ2):
        torch.testing.assert_close(out[:, k], d.M_TF[:, :, k] @ f0, rtol=1e-12, atol=1e-14)


def test_first_q2_slice_is_ic(small_grid):
    x = torch.linspace(0.05, 0.75, small_grid["nx"], dtype=torch.float64)
    f1, g1 = load_transversity_inputs(x, 2.4, force_toy=True)
    h1_0 = build_h1_initial(x, f1, g1, TransversityParams())
    with tempfile.TemporaryDirectory() as grid_dir:
        d = NonSingletDGLAP(grid_dir=grid_dir, **small_grid)
    h1 = evolve_transversity(TransversityEvolution.from_dglap(d), h1_0)
    packed = pack_TF_by_pdg(h1_0)
    packed[pdg_to_slot(0)] = 0.0
    torch.testing.assert_close(h1.data[:, :, 0], packed, rtol=1e-10, atol=1e-12)


def test_transversity_and_collins_share_kernel(small_grid):
    x = torch.linspace(0.05, 0.75, small_grid["nx"], dtype=torch.float64)
    f0 = x**0.6 * (1.0 - x) ** 2
    with tempfile.TemporaryDirectory() as grid_dir:
        dt = NonSingletDGLAP(grid_dir=grid_dir, **small_grid)
        dc = NonSingletDGLAP(grid_dir=grid_dir, **small_grid)
    torch.testing.assert_close(dt.M_TF, dc.M_TF, rtol=1e-12, atol=1e-14)


def test_collins_channels_do_not_mix(small_grid):
    z = torch.linspace(0.05, 0.75, small_grid["nx"], dtype=torch.float64)
    D = load_pion_ff(z, 2.4, force_toy=True)
    _, Hhat0 = build_collins_hhat_initial(z, D)
    H0 = torch.zeros_like(Hhat0)
    H0[1] = Hhat0[1]
    with tempfile.TemporaryDirectory() as grid_dir:
        d = NonSingletDGLAP(grid_dir=grid_dir, **small_grid)
    H = evolve_collins_hhat(CollinsHhatEvolution.from_dglap(d), H0, COLLINS_CHANNELS)
    assert H.values[0].abs().max() == 0.0
    assert H.values[2].abs().max() == 0.0


def test_qs_gluon_zero_trans_gluon_zero(small_grid):
    gqs = {**small_grid, "kernel_type": "qiu_sterman", "eta": params.NC, "Q20": params.mc2}
    x = torch.linspace(0.05, 0.75, gqs["nx"], dtype=torch.float64)
    fq = {fl: x**0.5 * (1.0 - x) ** 3 for fl in FLAVORS}
    TF0 = build_TF_initial(x, fq, QiuStermanParams())
    f1, g1 = load_transversity_inputs(x, 2.4, force_toy=True)
    h1_0 = build_h1_initial(x, f1, g1)
    with tempfile.TemporaryDirectory() as grid_dir:
        dqs = NonSingletDGLAP(grid_dir=grid_dir, **gqs)
        dtr = NonSingletDGLAP(grid_dir=grid_dir, **small_grid)
    TF = evolve_flavors(QiuStermanEvolution.from_dglap(dqs), TF0)
    packed = pack_TF_by_pdg(TF0)
    assert packed[pdg_to_slot(0)].abs().max() == 0.0
    h1 = evolve_transversity(TransversityEvolution.from_dglap(dtr), h1_0)
    assert h1[0].abs().max() == 0.0


def test_eta_nc_suppresses_more_than_eta_zero(small_grid):
    x = torch.linspace(0.05, 0.9, 8, dtype=torch.float64)
    TF0 = x * (1.0 - x) ** 2
    g = {**small_grid, "kernel_type": "qiu_sterman", "Q20": params.mc2}
    with tempfile.TemporaryDirectory() as grid_dir:
        d0 = NonSingletDGLAP(eta=0.0, grid_dir=grid_dir, **g)
        d1 = NonSingletDGLAP(eta=params.NC, grid_dir=grid_dir, **g)
    assert d1.evolve(TF0)[..., -1].norm() < d0.evolve(TF0)[..., -1].norm()
