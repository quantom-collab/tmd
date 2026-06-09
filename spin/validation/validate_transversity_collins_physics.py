"""
Physics initial-condition validation for transversity and homogeneous Collins Hhat.

Run from repository root::

    python -m Spin2.validation.validate_transversity_collins_physics --toy --no-cache
    python -m Spin2.validation.validate_transversity_collins_physics

Uses paper-style IC ansätze with LHAPDF inputs when available. Toy mode is for
structural/sign/convention checks only — not paper comparison.
"""

from __future__ import annotations

import argparse
import logging
import math
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Spin2.collins import (
    COLLINS_CHANNELS,
    build_collins_hhat_initial,
    trento_moment_from_hhat,
)
from Spin2.dglap import NonSingletDGLAP
from Spin2.evolution import (
    CollinsHhatEvolution,
    TransversityEvolution,
    evolve_collins_hhat,
    evolve_transversity,
)
from Spin2.flavors import NAME_TO_PDG, pack_TF_by_pdg, pdg_to_slot
from Spin2.qiu_sterman import FLAVORS
from Spin2.transversity import TransversityParams, build_h1_initial
from Spin2.validation.inputs import (
    TC_FF_SET_PION as FF_SET_PION,
    TC_MU0_SQ as Q0_SQ_DEFAULT,
    TC_PDF_SET_F1 as PDF_SET_F1,
    TC_PDF_SET_G1 as HELICITY_SET_G1,
    load_pion_ff as load_collins_inputs,
    load_transversity_inputs,
)

logger = logging.getLogger(__name__)

_EPS = 1e-12
M_W_GEV = 80.379
M_PI_PLOT_GEV = 0.13957

_TRANS_XMIN = 0.01
_TRANS_XMAX = 0.9
_COLLINS_ZMIN = 0.05
_COLLINS_ZMAX = 0.95

_SUPPORT_X_LO = 0.02
_SUPPORT_X_HI = 0.8
_SUPPORT_Z_LO = 0.08
_SUPPORT_Z_HI = 0.85

_MU_SCALES_GEV = (2.0, 5.0, 10.0)

_DEFAULTS = dict(
    nx=80,
    nQ2=48,
    Q20=Q0_SQ_DEFAULT,
    LQ2max=4.0,
    loadgrid=True,
    steps=5,
    ng=32,
    dev="cpu",
    dtype=torch.float64,
    order=0,
)


def _flavor_index(flavor: str) -> int:
    return FLAVORS.index(flavor)


def _trapz(y: torch.Tensor, x: torch.Tensor) -> float:
    if hasattr(torch, "trapz"):
        return float(torch.trapz(y, x))
    return float(np.trapz(y.detach().cpu().numpy(), x.detach().cpu().numpy()))


def _masked_ratio(num: torch.Tensor, den: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    mask = den.abs() > eps
    ratio = torch.full_like(num, float("nan"))
    ratio[mask] = num[mask] / den[mask]
    return ratio


def soffer_ratio_at_ic(
    h1_0: torch.Tensor,
    f1: dict[str, torch.Tensor],
    g1: dict[str, torch.Tensor],
    flavors: tuple[str, ...] = ("u", "d"),
) -> dict[str, float]:
    """|h_1| / (0.5 * (f_1 + g_1)) for selected flavors at Q0."""
    out: dict[str, float] = {}
    for fl in flavors:
        i = _flavor_index(fl)
        den = 0.5 * (f1[fl] + g1[fl])
        ratio = _masked_ratio(h1_0[i].abs(), den)
        finite = ratio[torch.isfinite(ratio)]
        out[fl] = float(finite.max().item()) if finite.numel() else float("nan")
    return out


def _mu_scale_list(Q2_max: float, Q20: float) -> list[tuple[str, float]]:
    mu0 = math.sqrt(Q20)
    scales: list[tuple[str, float]] = [("Q0", mu0)]
    scales.extend((f"{mu:g} GeV", mu) for mu in _MU_SCALES_GEV if mu**2 <= Q2_max * (1.0 + 1e-9))
    if M_W_GEV**2 <= Q2_max * (1.0 + 1e-9):
        scales.append(("M_W", M_W_GEV))
    return scales


def _dglap_kwargs(
    *,
    nx: int,
    nQ2: int,
    Q20: float,
    LQ2max: float,
    loadgrid: bool,
    xmin: float,
    xmax: float,
    grid_dir: str,
) -> dict[str, Any]:
    return {
        **_DEFAULTS,
        "nx": nx,
        "nQ2": nQ2,
        "Q20": Q20,
        "LQ2max": LQ2max,
        "loadgrid": loadgrid,
        "xmin": xmin,
        "xmax": xmax,
        "kernel_type": "transversity",
        "grid_dir": grid_dir,
    }


def run_transversity_physics(
    *,
    toy: bool,
    loadgrid: bool,
    nx: int,
    nQ2: int,
    Q20: float,
    LQ2max: float,
    grid_dir: str,
    make_plots: bool,
    output_dir: Path,
) -> dict[str, Any]:
    gkw = _dglap_kwargs(
        nx=nx,
        nQ2=nQ2,
        Q20=Q20,
        LQ2max=LQ2max,
        loadgrid=loadgrid,
        xmin=_TRANS_XMIN,
        xmax=_TRANS_XMAX,
        grid_dir=grid_dir,
    )
    dglap = NonSingletDGLAP(**gkw)
    x = dglap.x
    mu0 = math.sqrt(Q20)

    f1, g1 = load_transversity_inputs(x, Q20, force_toy=toy)
    h1_0 = build_h1_initial(x, f1, g1, TransversityParams())
    evo = TransversityEvolution.from_dglap(dglap)
    h1 = evolve_transversity(evo, h1_0)

    h1_0_pdg = pack_TF_by_pdg(h1_0)
    h1_0_pdg[pdg_to_slot(0)] = 0.0
    h1_Q0_evo = h1.data[:, :, 0]
    denom = h1_0_pdg.abs().clamp(min=_EPS)
    max_rel_diff_Q0 = ((h1_Q0_evo - h1_0_pdg).abs() / denom).max().item()

    support = (x >= _SUPPORT_X_LO) & (x <= _SUPPORT_X_HI)
    iu = pdg_to_slot(NAME_TO_PDG["u"])
    id_ = pdg_to_slot(NAME_TO_PDG["d"])
    hu0 = h1.data[iu, :, 0][support]
    hd0 = h1.data[id_, :, 0][support]
    n_u_pos = int((hu0 > 0).sum().item())
    n_d_neg = int((hd0 < 0).sum().item())
    n_support = int(support.sum().item())

    soffer = soffer_ratio_at_ic(h1_0, f1, g1)
    soffer_max = max(soffer.values())

    Q2_max = float(dglap.Q2[-1].item())
    mu_scales = _mu_scale_list(Q2_max, Q20)
    mu_list = [mu for _, mu in mu_scales]
    charge_u: list[float] = []
    charge_d: list[float] = []
    for mu in mu_list:
        hu = h1.pdg_at_mu(2, mu)
        hd = h1.pdg_at_mu(1, mu)
        charge_u.append(_trapz(hu, x))
        charge_d.append(_trapz(hd, x))

    result = {
        "dglap": dglap,
        "x": x,
        "f1": f1,
        "g1": g1,
        "h1_0": h1_0,
        "h1": h1,
        "Q20": Q20,
        "mu0": mu0,
        "max_rel_diff_Q0": max_rel_diff_Q0,
        "soffer_max": soffer_max,
        "soffer": soffer,
        "n_u_pos": n_u_pos,
        "n_d_neg": n_d_neg,
        "n_support": n_support,
        "mu_scales": mu_scales,
        "charge_u": charge_u,
        "charge_d": charge_d,
        "toy": toy,
        "pdf_set_f1": "toy" if toy else PDF_SET_F1,
        "helicity_set_g1": "toy" if toy else HELICITY_SET_G1,
    }

    print("--- Transversity physics IC / evolution ---")
    print(
        f"Q0^2 = {Q20} GeV^2,  f1: {result['pdf_set_f1']},  "
        f"g1: {result['helicity_set_g1']}"
    )
    if toy:
        print("NOTE: toy inputs — structural validation only, not paper comparison.")
    else:
        print(
            "NOTE: paper used CT10 NLO (f1), DSSV (g1), alpha_s(Q0)=0.327; "
            "plots are not directly comparable unless the same sets are used."
        )
    print(f"max relative |h1(Q0) - h1_0| / max(|h1_0|, eps): {max_rel_diff_Q0:.3e}")
    print(
        f"sign counts in {_SUPPORT_X_LO} < x < {_SUPPORT_X_HI}: "
        f"u>0: {n_u_pos}/{n_support}, d<0: {n_d_neg}/{n_support}"
    )
    print(f"Soffer ratio max (u,d) at Q0: {soffer_max:.6f} (expect <= 1)")
    print("Tensor-charge-like integral int dx h1^q(x, mu) vs mu [GeV]:")
    for (_, mu), cu, cd in zip(mu_scales, charge_u, charge_d):
        print(f"  mu={mu:6.3f}  int h1^u = {cu:+.6e}  int h1^d = {cd:+.6e}")

    if make_plots:
        _plot_transversity(result, output_dir)

    return result


def run_collins_physics(
    *,
    toy: bool,
    loadgrid: bool,
    nx: int,
    nQ2: int,
    Q20: float,
    LQ2max: float,
    grid_dir: str,
    make_plots: bool,
    output_dir: Path,
) -> dict[str, Any]:
    gkw = _dglap_kwargs(
        nx=nx,
        nQ2=nQ2,
        Q20=Q20,
        LQ2max=LQ2max,
        loadgrid=loadgrid,
        xmin=_COLLINS_ZMIN,
        xmax=_COLLINS_ZMAX,
        grid_dir=grid_dir,
    )
    collins_dglap = NonSingletDGLAP(**gkw)
    z = collins_dglap.x
    mu0 = math.sqrt(Q20)

    D_piplus = load_collins_inputs(z, Q20, force_toy=toy)
    channels, Hhat0 = build_collins_hhat_initial(z, D_piplus)
    evo = CollinsHhatEvolution.from_dglap(collins_dglap)
    Hhat = evolve_collins_hhat(evo, Hhat0, channels)

    max_rel_diff_Q0 = (
        (Hhat.values[:, :, 0] - Hhat0).abs()
        / Hhat0.abs().clamp(min=_EPS)
    ).max().item()

    support = (z >= _SUPPORT_Z_LO) & (z <= _SUPPORT_Z_HI)
    fav0 = Hhat0[0][support]
    unf0 = Hhat0[1][support]
    unf_s0 = Hhat0[2][support]
    n_fav_neg = int((fav0 < 0).sum().item())
    n_unf_pos = int((unf0 > 0).sum().item())
    n_unf_s_pos = int((unf_s0 > 0).sum().item())
    n_support = int(support.sum().item())

    Mpi = M_PI_PLOT_GEV
    H1_fav = trento_moment_from_hhat(z, Hhat0[0], Mpi)
    H1_unf = trento_moment_from_hhat(z, Hhat0[1], Mpi)
    H1_unf_s = trento_moment_from_hhat(z, Hhat0[2], Mpi)
    n_h1_fav_pos = int((H1_fav[support] > 0).sum().item())
    n_h1_unf_neg = int((H1_unf[support] < 0).sum().item())
    n_h1_unf_s_neg = int((H1_unf_s[support] < 0).sum().item())

    Q2_max = float(collins_dglap.Q2[-1].item())
    mu_scales = _mu_scale_list(Q2_max, Q20)
    mu_list = [mu for _, mu in mu_scales]
    moment0: dict[str, list[float]] = {ch: [] for ch in COLLINS_CHANNELS}
    moment1: dict[str, list[float]] = {ch: [] for ch in COLLINS_CHANNELS}
    for mu in mu_list:
        H_mu = Hhat.at_mu(mu)
        for i, ch in enumerate(COLLINS_CHANNELS):
            h = H_mu[i]
            moment0[ch].append(_trapz(h, z))
            moment1[ch].append(_trapz(h * z, z))

    result = {
        "dglap": collins_dglap,
        "z": z,
        "D_piplus": D_piplus,
        "channels": channels,
        "Hhat0": Hhat0,
        "Hhat": Hhat,
        "Q20": Q20,
        "mu0": mu0,
        "max_rel_diff_Q0": max_rel_diff_Q0,
        "n_fav_neg": n_fav_neg,
        "n_unf_pos": n_unf_pos,
        "n_unf_s_pos": n_unf_s_pos,
        "n_h1_fav_pos": n_h1_fav_pos,
        "n_h1_unf_neg": n_h1_unf_neg,
        "n_h1_unf_s_neg": n_h1_unf_s_neg,
        "n_support": n_support,
        "mu_scales": mu_scales,
        "moment0": moment0,
        "moment1": moment1,
        "Mpi": Mpi,
        "toy": toy,
        "ff_set": "toy" if toy else FF_SET_PION,
        "H1_ic": {"fav": H1_fav, "unf": H1_unf, "unf_s": H1_unf_s},
    }

    print("--- Homogeneous Collins Hhat^{(3)} physics IC / evolution ---")
    print(
        f"Q0^2 = {Q20} GeV^2,  pi+ FF set: {result['ff_set']},  "
        f"M_pi (Trento plots) = {Mpi} GeV"
    )
    print(
        "NOTE: evolved object is Hhat^{(3)} (homogeneous twist-3 FF), "
        "not the full Collins function or Trento H_1^{perp(1)} directly."
    )
    if toy:
        print("NOTE: toy FFs — structural validation only.")
    else:
        print("NOTE: paper used DSS pi+ FFs; JAM20 is a proxy, not a paper match.")
    print(f"max relative |Hhat(Q0) - Hhat0| / max(|Hhat0|, eps): {max_rel_diff_Q0:.3e}")
    print(
        f"Hhat sign counts in {_SUPPORT_Z_LO} < z < {_SUPPORT_Z_HI}: "
        f"fav<0: {n_fav_neg}/{n_support}, unf>0: {n_unf_pos}/{n_support}, "
        f"unf_s>0: {n_unf_s_pos}/{n_support}"
    )
    print(
        f"H_1^{{perp(1)}}|Trento sign counts (from Hhat): "
        f"fav>0: {n_h1_fav_pos}/{n_support}, unf<0: {n_h1_unf_neg}/{n_support}, "
        f"unf_s<0: {n_h1_unf_s_neg}/{n_support}"
    )
    print("Moments int dz Hhat and int dz z Hhat vs mu [GeV]:")
    for (_, mu), i in zip(mu_scales, range(len(mu_list))):
        parts0 = "  ".join(f"{ch}={moment0[ch][i]:+.4e}" for ch in COLLINS_CHANNELS)
        parts1 = "  ".join(f"{ch}={moment1[ch][i]:+.4e}" for ch in COLLINS_CHANNELS)
        print(f"  mu={mu:6.3f}  int Hhat: {parts0}")
        print(f"           int z Hhat: {parts1}")

    if make_plots:
        _plot_collins(result, output_dir)

    return result


def _plot_transversity(res: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    x_np = res["x"].detach().cpu().numpy()
    h1 = res["h1"]
    iu = pdg_to_slot(NAME_TO_PDG["u"])
    id_ = pdg_to_slot(NAME_TO_PDG["d"])
    mu0 = res["mu0"]

    hu0 = h1.data[iu, :, 0].detach().cpu().numpy()
    hd0 = h1.data[id_, :, 0].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(x_np, x_np * hu0, label=r"$x\,h_1^u(x,Q_0)$")
    ax.plot(x_np, x_np * hd0, label=r"$x\,h_1^d(x,Q_0)$")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$x\,h_1^q$")
    ax.set_title(rf"Transversity IC at $\mu_0={mu0:.3f}$ GeV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "transversity_ic_ud.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, mu in res["mu_scales"]:
        hu = h1.pdg_at_mu(2, mu).detach().cpu().numpy()
        hd = h1.pdg_at_mu(1, mu).detach().cpu().numpy()
        ax.plot(x_np, x_np * hu, label=rf"$x\,h_1^u$, {label}")
        ax.plot(x_np, x_np * hd, "--", label=rf"$x\,h_1^d$, {label}")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$x\,h_1^q$")
    ax.set_title("Transversity evolution (u solid, d dashed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "transversity_evolution_ud.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    mu_np = np.array([mu for _, mu in res["mu_scales"]])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(mu_np, res["charge_u"], "o-", label=r"$\int dx\, h_1^u$")
    ax.plot(mu_np, res["charge_d"], "s-", label=r"$\int dx\, h_1^d$")
    ax.set_xlabel(r"$\mu$ [GeV]")
    ax.set_ylabel(r"$\int dx\, h_1^q(x,\mu)$")
    ax.set_title("Tensor-charge-like integrals (not conserved)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / "transversity_tensor_charge_integral.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_collins(res: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    z_np = res["z"].detach().cpu().numpy()
    Hhat0 = res["Hhat0"].detach().cpu().numpy()
    Hhat = res["Hhat"]
    mu0 = res["mu0"]
    ch_labels = (r"\mathrm{fav}", r"\mathrm{unf}", r"\mathrm{unf}_s")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, lab in enumerate(ch_labels):
        ax.plot(z_np, z_np * Hhat0[i], label=rf"$z\,\hat H^{{(3)}}_{{{lab}}}(z,Q_0)$")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$z\,\hat H^{(3)}$")
    ax.set_title(rf"Collins $\hat H^{{(3)}}$ IC at $\mu_0={mu0:.3f}$ GeV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "collins_hhat_ic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    styles = ["-", "--", ":"]
    for label, mu in res["mu_scales"]:
        H_mu = Hhat.at_mu(mu).detach().cpu().numpy()
        for i, (lab, sty) in enumerate(zip(ch_labels, styles)):
            ax.plot(
                z_np,
                z_np * H_mu[i],
                sty,
                label=rf"$z\,\hat H^{{(3)}}_{{{lab}}}$, {label}",
            )
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$z\,\hat H^{(3)}$")
    ax.set_title(r"Homogeneous $\hat H^{(3)}$ evolution")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "collins_hhat_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    H1 = res["H1_ic"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    name_keys = ("fav", "unf", "unf_s")
    for key, lab in zip(name_keys, ch_labels):
        h1 = H1[key].detach().cpu().numpy()
        ax.plot(
            z_np,
            z_np * h1,
            label=rf"$z\, {{H_1^{{\perp(1)}}}}_{{{lab}}}$",
        )
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$z\,H_1^{\perp(1)}$ (Trento)")
    ax.set_title(
        rf"Trento moment from IC ($M_\pi={res['Mpi']}$ GeV)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "collins_trento_moment_ic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    mu_np = np.array([mu for _, mu in res["mu_scales"]])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ch, lab in zip(COLLINS_CHANNELS, ch_labels):
        ax.plot(mu_np, res["moment1"][ch], "o-", label=rf"$\int dz\, z\,\hat H^{{(3)}}_{{{lab}}}$")
    ax.set_xlabel(r"$\mu$ [GeV]")
    ax.set_ylabel(r"$\int dz\, z\,\hat H^{(3)}(z,\mu)$")
    ax.set_title(r"First $z$-moment of $\hat H^{(3)}$ vs scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "collins_moments_vs_mu.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_physics_validation(
    *,
    toy: bool = False,
    loadgrid: bool = True,
    nx: int = 80,
    nQ2: int = 48,
    Q20: float = Q0_SQ_DEFAULT,
    LQ2max: float = 4.0,
    make_plots: bool = True,
    output_dir: Path | None = None,
    grid_dir: str | None = None,
) -> dict[str, Any]:
    """Run transversity and Collins physics IC validation."""
    out_dir = output_dir or (Path(__file__).resolve().parent / "outputs")
    if grid_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="spin_tc_phys_")
        grid_dir = tmp_ctx.name
    else:
        tmp_ctx = None

    try:
        trans = run_transversity_physics(
            toy=toy,
            loadgrid=loadgrid,
            nx=nx,
            nQ2=nQ2,
            Q20=Q20,
            LQ2max=LQ2max,
            grid_dir=grid_dir,
            make_plots=make_plots,
            output_dir=out_dir,
        )
        collins = run_collins_physics(
            toy=toy,
            loadgrid=loadgrid,
            nx=nx,
            nQ2=nQ2,
            Q20=Q20,
            LQ2max=LQ2max,
            grid_dir=grid_dir,
            make_plots=make_plots,
            output_dir=out_dir,
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    print("OK: transversity and homogeneous Collins Hhat physics validation complete.")
    return {"transversity": trans, "collins": collins}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate transversity and Collins Hhat physics ICs and evolution."
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Force toy f1/g1 and pi+ FFs even if LHAPDF is available.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build evolution matrices fresh (loadgrid=False).",
    )
    parser.add_argument("--nx", type=int, default=_DEFAULTS["nx"])
    parser.add_argument("--nQ2", type=int, default=_DEFAULTS["nQ2"])
    parser.add_argument("--Q20", type=float, default=_DEFAULTS["Q20"])
    parser.add_argument("--LQ2max", type=float, default=_DEFAULTS["LQ2max"])
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing PNG plots.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter("always", UserWarning)

    run_physics_validation(
        toy=args.toy,
        loadgrid=not args.no_cache,
        nx=args.nx,
        nQ2=args.nQ2,
        Q20=args.Q20,
        LQ2max=args.LQ2max,
        make_plots=not args.no_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
