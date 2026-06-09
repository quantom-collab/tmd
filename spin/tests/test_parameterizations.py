"""Initial-condition parameterization tests."""

from __future__ import annotations

import torch

from Spin2.collins import (
    COLLINS_CHANNELS,
    build_collins_hhat_initial,
    hhat_from_trento_moment,
    trento_moment_from_hhat,
)
from Spin2.qiu_sterman import FLAVORS, QiuStermanParams, build_TF_initial
from Spin2.transversity import TransversityParams, build_h1_initial
from Spin2.validation.inputs import load_pion_ff, load_transversity_inputs
from Spin2.validation.validate_transversity_collins_physics import soffer_ratio_at_ic


def test_qs_ic_signs():
    x = torch.linspace(0.05, 0.5, 24, dtype=torch.float64)
    fq = {fl: x**0.6 * (1.0 - x) ** 3.5 for fl in FLAVORS}
    TF0 = build_TF_initial(x, fq, QiuStermanParams())
    iu, id_ = FLAVORS.index("u"), FLAVORS.index("d")
    assert TF0[iu].max() > 0
    assert TF0[id_].max() < 0


def test_transversity_ud_signs_and_soffer():
    x = torch.linspace(0.02, 0.8, 24, dtype=torch.float64)
    f1, g1 = load_transversity_inputs(x, 2.4, force_toy=True)
    h1_0 = build_h1_initial(x, f1, g1, TransversityParams())
    support = (x > 0.02) & (x < 0.8)
    assert (h1_0[0][support] > 0).all()
    assert (h1_0[1][support] < 0).all()
    soffer = soffer_ratio_at_ic(h1_0, f1, g1)
    assert soffer["u"] <= 1.0 + 1e-10
    assert soffer["d"] <= 1.0 + 1e-10


def test_collins_hhat_signs():
    z = torch.linspace(0.05, 0.95, 24, dtype=torch.float64)
    D = load_pion_ff(z, 2.4, force_toy=True)
    channels, Hhat0 = build_collins_hhat_initial(z, D)
    assert channels == COLLINS_CHANNELS
    support = (z >= 0.08) & (z <= 0.85)
    assert (Hhat0[0][support] < 0).all()
    assert (Hhat0[1][support] > 0).all()
    assert (Hhat0[2][support] > 0).all()


def test_trento_roundtrip():
    z = torch.linspace(0.1, 0.9, 20, dtype=torch.float64)
    Mpi = 0.13957
    H1 = 0.02 * z * (1.0 - z)
    back = trento_moment_from_hhat(
        z, hhat_from_trento_moment(z, H1, Mpi), Mpi
    )
    torch.testing.assert_close(back, H1, rtol=1e-12, atol=1e-14)
