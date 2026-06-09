"""Strong coupling alpha_s(Q^2) for Spin evolution."""

from __future__ import annotations

import numpy as np
import torch

from spin import params as par

_storage: dict[float, float] = {}
_order: int = par.alphaS_order
_mu20: float = par.alphaS_mu20
_aZ: float = 0.0
_ab: float = 0.0
_ac: float = 0.0
_a0: float = 0.0


def get_Nf(Q2: float) -> int:
    nf = 3
    if Q2 >= par.mc2:
        nf += 1
    if Q2 >= par.mb2:
        nf += 1
    return nf


def beta_func(a: float, nf: int, order: int) -> float:
    beta0 = 11.0 - 2.0 / 3.0 * nf
    beta1 = 102.0 - 38.0 / 3.0 * nf
    beta2 = 2857.0 / 2.0 - 5033.0 / 18.0 * nf + 325.0 / 54.0 * nf**2
    betaf = -beta0
    if order >= 1:
        betaf += -a * beta1
    if order >= 2:
        betaf += -(a**2) * beta2
    return betaf * a**2


def evolve_a(mu20: float, a: float, q2: float, nf: int, order: int) -> float:
    lr = np.log(q2 / mu20) / 20.0
    for _ in range(20):
        xk0 = lr * beta_func(a, nf, order)
        xk1 = lr * beta_func(a + 0.5 * xk0, nf, order)
        xk2 = lr * beta_func(a + 0.5 * xk1, nf, order)
        xk3 = lr * beta_func(a + xk2, nf, order)
        a += (xk0 + 2.0 * xk1 + 2.0 * xk2 + xk3) / 6.0
    return a


def setup(order: int | None = None, mu20: float | None = None) -> None:
    """Initialize boundary couplings and clear the Q2 cache."""
    global _order, _mu20, _aZ, _ab, _ac, _a0, _storage
    _order = par.alphaS_order if order is None else order
    _mu20 = par.alphaS_mu20 if mu20 is None else mu20
    _storage = {}
    _aZ = par.alphaSMZ / (4.0 * np.pi)
    _ab = evolve_a(par.mZ2, _aZ, par.mb2, 5, _order)
    _ac = evolve_a(par.mb2, _ab, par.mc2, 4, _order)
    _a0 = evolve_a(par.mc2, _ac, _mu20, 3, _order)


setup()


def get_a(q2: float, order: int | None = None) -> float:
    order = _order if order is None else order
    key = float(q2)
    if key not in _storage:
        if par.mb2 <= q2:
            _storage[key] = evolve_a(par.mb2, _ab, q2, 5, order)
        elif par.mc2 <= q2 < par.mb2:
            _storage[key] = evolve_a(par.mc2, _ac, q2, 4, order)
        else:
            _storage[key] = evolve_a(_mu20, _a0, q2, 3, order)
    return _storage[key]


def get_alphaS(q2: float) -> float:
    return get_a(q2) * 4.0 * np.pi
