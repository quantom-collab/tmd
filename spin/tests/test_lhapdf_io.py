"""LHAPDF loader normalization tests."""

from __future__ import annotations

import pytest
import torch

from Spin2.flavors import NAME_TO_PDG
from Spin2.validation.inputs import (
    TC_FF_SET_PION,
    TC_PDF_SET_F1,
    TC_PDF_SET_G1,
    load_helicity_g1,
    load_pion_ff,
    load_proton_pdf_fq,
    load_unpolarized_f1,
)
from Spin2.validation.lhapdf_io import load_flavor_densities, sample_xfxQ, try_import_lhapdf

Q2 = 2.4
Q = Q2**0.5


class _FakePDF:
    _density = {2: 3.0, 1: 1.5, 3: 0.4, -2: 0.2, -1: 0.25, -3: 0.1}

    def xfxQ(self, pid: int, point: float, mu: float) -> float:
        del mu
        return self._density[pid] * point


class _FakeLHAPDF:
    @staticmethod
    def mkPDF(pdf_set: str, member: int) -> _FakePDF:
        del pdf_set, member
        return _FakePDF()


@pytest.fixture
def x_grid():
    return torch.tensor([0.05, 0.2, 0.5, 0.8], dtype=torch.float64)


@pytest.fixture
def z_grid():
    return torch.tensor([0.08, 0.3, 0.6, 0.9], dtype=torch.float64)


def test_density_helper_divides_by_point(x_grid):
    fake = _FakePDF()
    got = load_flavor_densities(fake, ("u", "d"), x_grid, Q2)
    pts = x_grid.numpy()
    for fl, vec in got.items():
        xfx = sample_xfxQ(fake, NAME_TO_PDG[fl], pts, Q)
        torch.testing.assert_close(vec, torch.tensor(xfx / pts, dtype=torch.float64))


def test_unpolarized_loader_mock(monkeypatch, x_grid):
    monkeypatch.setattr("Spin2.validation.inputs.try_import_lhapdf", lambda: _FakeLHAPDF)
    fake = _FakePDF()
    f1 = load_unpolarized_f1(x_grid, Q2, force_toy=False)
    pts = x_grid.numpy()
    for fl in ("u", "d"):
        xfx = sample_xfxQ(fake, NAME_TO_PDG[fl], pts, Q)
        torch.testing.assert_close(f1[fl], torch.tensor(xfx / pts, dtype=torch.float64))


def test_proton_pdf_mock(monkeypatch, x_grid):
    monkeypatch.setattr("Spin2.validation.inputs.try_import_lhapdf", lambda: _FakeLHAPDF)
    fake = _FakePDF()
    fq = load_proton_pdf_fq(x_grid, 1.9, backend="lhapdf")
    pts = x_grid.numpy()
    for fl, vec in fq.items():
        xfx = sample_xfxQ(fake, NAME_TO_PDG[fl], pts, 1.9**0.5)
        torch.testing.assert_close(vec, torch.tensor(xfx / pts, dtype=torch.float64))


def test_pion_ff_mock(monkeypatch, z_grid):
    monkeypatch.setattr("Spin2.validation.inputs.try_import_lhapdf", lambda: _FakeLHAPDF)
    fake = _FakePDF()
    D = load_pion_ff(z_grid, Q2, force_toy=False)
    pts = z_grid.numpy()
    for fl in ("u", "d"):
        zd = sample_xfxQ(fake, NAME_TO_PDG[fl], pts, Q)
        torch.testing.assert_close(D[fl], torch.tensor(zd / pts, dtype=torch.float64))


@pytest.mark.skipif(try_import_lhapdf() is None, reason="lhapdf not installed")
def test_live_f1_roundtrip(x_grid):
    lhapdf = try_import_lhapdf()
    pdf = lhapdf.mkPDF(TC_PDF_SET_F1, 0)
    f1 = load_unpolarized_f1(x_grid, Q2, force_toy=False)
    pts = x_grid.numpy()
    xfx = sample_xfxQ(pdf, NAME_TO_PDG["u"], pts, Q)
    torch.testing.assert_close(f1["u"], torch.tensor(xfx / pts, dtype=torch.float64))
    torch.testing.assert_close(f1["u"] * x_grid, torch.tensor(xfx, dtype=torch.float64))


@pytest.mark.skipif(try_import_lhapdf() is None, reason="lhapdf not installed")
def test_live_g1_roundtrip(x_grid):
    lhapdf = try_import_lhapdf()
    pdf = lhapdf.mkPDF(TC_PDF_SET_G1, 0)
    g1 = load_helicity_g1(x_grid, Q2, force_toy=False)
    pts = x_grid.numpy()
    xfx = sample_xfxQ(pdf, NAME_TO_PDG["u"], pts, Q)
    torch.testing.assert_close(g1["u"], torch.tensor(xfx / pts, dtype=torch.float64))


@pytest.mark.skipif(try_import_lhapdf() is None, reason="lhapdf not installed")
def test_live_ff_roundtrip(z_grid):
    lhapdf = try_import_lhapdf()
    ff = lhapdf.mkPDF(TC_FF_SET_PION, 0)
    D = load_pion_ff(z_grid, Q2, force_toy=False)
    pts = z_grid.numpy()
    zd = sample_xfxQ(ff, NAME_TO_PDG["u"], pts, Q)
    torch.testing.assert_close(D["u"], torch.tensor(zd / pts, dtype=torch.float64))
