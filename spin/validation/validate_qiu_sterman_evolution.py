"""
Numerical validation and plots for LO Qiu-Sterman DGLAP evolution (eta = N_C).

Run from the repository root::

    python -m spin.validation.validate_qiu_sterman_evolution
    python -m spin.validation.validate_qiu_sterman_evolution --with-evolution

Uses NNPDF40_nnlo_pch_as_01180 at mu0 = sqrt(1.9) GeV for the Fig. 7 initial condition.
Evolution uses :class:`spin.dglap.NonSingletDGLAP` with ``order=0``, ``eta=params.N_C``.
"""

from __future__ import annotations

import argparse
import logging
import math
import tempfile
from pathlib import Path
from typing import Any

import torch

from spin import params
from spin.dglap import NonSingletDGLAP
from spin.evolution import EvolvedT_F, QiuStermanEvolution, evolve_TF
from spin.flavors import NAME_TO_PDG, pack_TF_by_pdg, pdg_to_slot
from spin.qiu_sterman import FLAVORS, QiuStermanParams, build_TF_initial
from spin.validation.inputs import (
    QS_MU0_GEV as MU0_GEV,
    QS_MU0_SQ as MU0_SQ,
    QS_PDF_SET as PDF_SET,
    collinear_sivers_M,
    load_proton_pdf_fq,
)

logger = logging.getLogger(__name__)

_EPS = 1e-12
M_W_GEV = 80.379

# IC grid (wider x for Fig. 7)
_IC_XMIN = 1e-3
_IC_XMAX = 0.55

# Evolution validation: 0.02 < x < 0.6
_EVO_XMIN = 0.02
_EVO_XMAX = 0.6

# Fixed mu values for evolution plots (GeV); M_W added when Q2_max allows
_MU_SCALES_GEV = (2.0, 5.0, 10.0)

_DEFAULT_EVO_GRID = dict(
    dev="cpu",
    dtype=torch.float64,
    order=0,
    nx=40,
    xmin=_EVO_XMIN,
    xmax=_EVO_XMAX,
    nQ2=32,
    Q20=MU0_SQ,
    LQ2max=3.85,
    loadgrid=False,
    steps=5,
    ng=28,
)


def _flavor_index(flavor: str) -> int:
    return FLAVORS.index(flavor)


def minus_2M_TF(TF: torch.Tensor, M: float | None = None) -> torch.Tensor:
    """
    Collinear Sivers observable -2 M T_F^q(x, x).

    Default M = 0.9389 GeV (validation convention for comparison to paper figures).
    """
    mass = collinear_sivers_M() if M is None else M
    return -2.0 * mass * TF


def minus_x_over_2M_TF(
    TF: torch.Tensor, x: torch.Tensor, M: float | None = None
) -> torch.Tensor:
    """Collinear plot quantity -x/(2 M) T_F^q(x, x)."""
    mass = collinear_sivers_M() if M is None else M
    return -x * TF / (2.0 * mass)


def xf1T_perp1(
    TF: torch.Tensor, x: torch.Tensor, M: float | None = None
) -> torch.Tensor:
    """x f_{1T}^{perp(1)}(x, mu) = -x T_F^q(x, x; mu) / (2 M)."""
    mass = collinear_sivers_M() if M is None else M
    return -x * TF / (2.0 * mass)


def _masked_ratio(
    num: torch.Tensor, den: torch.Tensor, ref: torch.Tensor, eps: float = _EPS
) -> torch.Tensor:
    mask = ref.abs() > eps
    ratio = torch.full_like(num, float("nan"))
    ratio[mask] = num[mask] / den[mask]
    return ratio


def _mu_from_label(label: str) -> float:
    if label == "Q0":
        return MU0_GEV
    if label == "M_W":
        return M_W_GEV
    return float(label.split()[0])


def _mu_scale_list(Q2_max: float) -> list[tuple[str, float]]:
    """(label, mu [GeV]) including Q0 and optional M_W."""
    scales: list[tuple[str, float]] = [("Q0", MU0_GEV)]
    scales.extend((f"{mu:g} GeV", mu) for mu in _MU_SCALES_GEV)
    if M_W_GEV**2 <= Q2_max * (1.0 + 1e-9):
        scales.append(("M_W", M_W_GEV))
    return scales


def _print_ic_diagnostics(ic: dict[str, Any]) -> None:
    print("--- Qiu-Sterman IC (Fig. 7 style) ---")
    print(f"PDF: {ic['pdf_set']}  at mu0 = {ic['mu0']:.4f} GeV (Q^2 = {ic['mu0_sq']})")
    print(f"TF0 shape: {tuple(ic['TF0'].shape)}")
    print("-x/(2M) T_F^q at x ~ 0.2 (expect d>0, u<0, sea << valence):")
    obs = ic["xf1T_IC"]
    for i, fl in enumerate(FLAVORS):
        print(
            f"  {fl:5s}  max|xf1T| = {obs[i].abs().max().item():.4e}  "
            f"sign@0.2 = {ic['sign_check'][fl]}"
        )


def _print_evolution_diagnostics(ev: dict[str, Any]) -> None:
    print("--- Qiu-Sterman evolution (eta = N_C, order = 0) ---")
    print(f"TF0 shape:     {tuple(ev['TF0'].shape)}")
    print(f"T_F.data shape (PDG): {tuple(ev['T_F'].data.shape)}")
    print(
        f"Q20 = {ev['Q20']:.6g} GeV^2, mu0 = {MU0_GEV:.4f} GeV, "
        f"Q2_max = {ev['Q2_max']:.6g} GeV^2 (mu_max = {ev['mu_max']:.4f} GeV)"
    )
    print(f"max relative |TF(Q0) - TF0| / max(|TF0|, eps): {ev['max_rel_diff_Q0']:.3e}")
    if ev["max_rel_diff_Q0"] > 1e-8:
        print("WARNING: evolved first Q2 slice differs from TF0.")
    else:
        print("OK: first Q2 slice reproduces TF0.")

    print("||TF|| and ||TF(mu)|| / ||TF(Q0)||:")
    for label, norm, ratio in zip(
        ev["mu_labels"], ev["norms"], ev["norm_ratios"]
    ):
        print(f"  {label:8s}  ||TF|| = {norm:.6e}  ratio = {ratio:.6f}")

    print("Sign of x f_{1T}^{perp(1)} at x ~ 0.2 (expect u negative, d positive):")
    for label, signs in ev["sign_at_02"].items():
        print(f"  {label:8s}  u={signs['u']}  d={signs['d']}")

    if ev.get("sign_flip_warning"):
        print("WARNING: sign flip for u or d in 0.02 < x < 0.6 vs Q0.")


def _build_ic(
    x: torch.Tensor,
    qs_params: QiuStermanParams,
    pdf_backend: str = "lhapdf",
) -> dict[str, Any]:
    fq = load_proton_pdf_fq(x, MU0_SQ, backend=pdf_backend)
    TF0 = build_TF_initial(x, {fl: fq[fl] for fl in FLAVORS}, qs_params=qs_params)
    xf1T = xf1T_perp1(TF0, x)
    ix = int(torch.argmin((x - 0.2).abs()))
    sign_check = {
        fl: "+" if xf1T[_flavor_index(fl), ix].item() > 0 else "−" for fl in FLAVORS
    }
    return {
        "x": x,
        "mu0": MU0_GEV,
        "mu0_sq": MU0_SQ,
        "pdf_set": PDF_SET,
        "TF0": TF0,
        "fq": fq,
        "xf1T_IC": xf1T,
        "sign_check": sign_check,
        "M_plot": collinear_sivers_M(),
        "qs_params": qs_params,
    }


def _run_evolution(
    qs_params: QiuStermanParams,
    pdf_backend: str,
    grid_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    gkw = {**_DEFAULT_EVO_GRID, **(grid_kwargs or {})}
    with tempfile.TemporaryDirectory(prefix="spin_val_evo_") as tmp:
        dglap = NonSingletDGLAP(eta=params.NC, grid_dir=tmp, **gkw)
    fq = load_proton_pdf_fq(dglap.x, MU0_SQ, backend=pdf_backend)
    TF0 = build_TF_initial(
        dglap.x, {fl: fq[fl] for fl in FLAVORS}, qs_params=qs_params
    )
    evo = QiuStermanEvolution.from_dglap(dglap)
    T_F = evolve_TF(evo, TF0)

    Q2_max = float(dglap.Q2[-1].item())
    mu_max = math.sqrt(Q2_max)
    mu_scales = _mu_scale_list(Q2_max)

    TF_at_mu: dict[str, torch.Tensor] = {}
    xf1T_at_mu: dict[str, torch.Tensor] = {}
    norms: list[float] = []
    norm_ratios: list[float] = []
    mu_labels: list[str] = []

    TF0_pdg = pack_TF_by_pdg(TF0)
    norm_Q0 = TF0_pdg.norm().item()
    for label, mu in mu_scales:
        TF_mu_pdg = T_F.at_mu(mu)
        TF_at_mu[label] = TF_mu_pdg
        xf1T_at_mu[label] = xf1T_perp1(TF_mu_pdg, dglap.x)
        n = TF_mu_pdg.norm().item()
        norms.append(n)
        norm_ratios.append(n / norm_Q0 if norm_Q0 > 0 else float("nan"))
        mu_labels.append(label)

    TF_Q0_evo = T_F.data[:, :, 0]
    denom = TF0_pdg.abs().clamp(min=_EPS)
    max_rel_diff_Q0 = ((TF_Q0_evo - TF0_pdg).abs() / denom).max().item()

    ix = int(torch.argmin((dglap.x - 0.2).abs()))
    sign_at_02: dict[str, dict[str, str]] = {}
    support = (dglap.x >= _EVO_XMIN) & (dglap.x <= _EVO_XMAX)
    sign_flip_warning = False
    iu = pdg_to_slot(NAME_TO_PDG["u"])
    id_ = pdg_to_slot(NAME_TO_PDG["d"])
    obs_Q0_u = xf1T_at_mu["Q0"][iu, support]
    obs_Q0_d = xf1T_at_mu["Q0"][id_, support]
    for label in mu_labels:
        ou = xf1T_at_mu[label][iu, ix].item()
        od = xf1T_at_mu[label][id_, ix].item()
        sign_at_02[label] = {
            "u": "+" if ou > 0 else "−",
            "d": "+" if od > 0 else "−",
        }
        if label != "Q0":
            ou_x = xf1T_at_mu[label][iu, support]
            od_x = xf1T_at_mu[label][id_, support]
            if (
                (ou_x * obs_Q0_u < 0).any()
                or (od_x * obs_Q0_d < 0).any()
            ):
                sign_flip_warning = True

    return {
        "x": dglap.x,
        "Q2": dglap.Q2,
        "Q20": dglap.Q20,
        "Q2_max": Q2_max,
        "mu_max": mu_max,
        "TF0": TF0,
        "T_F": T_F,
        "TF_at_mu": TF_at_mu,
        "xf1T_at_mu": xf1T_at_mu,
        "mu_labels": mu_labels,
        "mu_scales": mu_scales,
        "norms": norms,
        "norm_ratios": norm_ratios,
        "max_rel_diff_Q0": max_rel_diff_Q0,
        "sign_at_02": sign_at_02,
        "sign_flip_warning": sign_flip_warning,
        "M_plot": collinear_sivers_M(),
        "eta": params.NC,
        "order": 0,
    }


def _make_plots_ic(ic: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    x = ic["x"].detach().cpu().numpy()
    obs0 = ic["xf1T_IC"].detach().cpu().numpy()
    M = ic["M_plot"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True)
    panels = [
        ("Valence", ("d", "u")),
        (r"Sea ($\bar u$, $\bar d$)", ("ubar", "dbar")),
        (r"Strange ($s$, $\bar s$)", ("sbar", "s")),
    ]
    colors = {
        "u": "tab:red",
        "d": "tab:blue",
        "s": "tab:red",
        "ubar": "tab:blue",
        "dbar": "tab:red",
        "sbar": "tab:blue",
    }
    for ax, (title, fls) in zip(axes, panels):
        for fl in fls:
            ax.plot(x, obs0[_flavor_index(fl)], color=colors[fl], label=fl)
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel(r"$x$")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    fig.supylabel(rf"$x\, f_{{1T}}^{{\perp(1)}} = -x\,T_F/(2M)$, $M={M}$ GeV")
    fig.suptitle(
        rf"Fig. 7 IC: {ic['pdf_set']}, $\mu_0={ic['mu0']:.3f}$ GeV",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "ic_xf1T_six_flavors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved IC plot to %s", output_dir)


def _make_plots_evolution(ev: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    x = ev["x"].detach().cpu().numpy()
    M = ev["M_plot"]
    mu_labels = ev["mu_labels"]
    xf1T = ev["xf1T_at_mu"]

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(mu_labels) - 1, 1)) for i in range(len(mu_labels))]

    def _plot_panel(
        flavors: tuple[str, ...],
        title: str,
        filename: str,
        *,
        ratio: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        T_F: EvolvedT_F = ev["T_F"]
        for ci, label in enumerate(mu_labels):
            if ratio and label == "Q0":
                continue
            for fl in flavors:
                pdg = NAME_TO_PDG[fl]
                if ratio:
                    y_mu = T_F.pdg_at_mu(pdg, _mu_from_label(label))
                    y = _masked_ratio(
                        y_mu,
                        T_F.pdg_at_mu(pdg, MU0_GEV),
                        T_F.pdg_at_mu(pdg, MU0_GEV),
                    ).detach().cpu().numpy()
                    ls = "-" if fl == "u" else "--"
                    ax.plot(
                        x,
                        y,
                        color=colors[ci],
                        ls=ls,
                        lw=1.4,
                        label=rf"{fl}, $\mu={label}$",
                    )
                else:
                    y = xf1T[label][pdg_to_slot(pdg)].detach().cpu().numpy()
                    ax.plot(
                        x,
                        y,
                        color=colors[ci],
                        lw=1.6 if len(flavors) == 1 else 1.4,
                        label=rf"{fl}, $\mu={label}$" if len(flavors) > 1 else label,
                    )
        ax.set_xlim(_EVO_XMIN, _EVO_XMAX)
        ax.set_xlabel(r"$x$")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        if ratio:
            ax.set_ylabel(r"$T_F^q(x,\mu)\,/\,T_F^q(x,Q_0)$")
            ax.set_title(title + r" ($\eta=N_C$)")
        else:
            ax.set_ylabel(rf"$x\, f_{{1T}}^{{\perp(1)}}$ [GeV], $M={M}$")
            ax.set_title(title + r" ($\eta=N_C$, LO)")
        ax.legend(fontsize=8, ncol=2 if not ratio else 1)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _plot_panel(("u", "d"), r"$u$, $d$ evolution", "evo_ud_xf1T.png")
    _plot_panel(("ubar", "dbar"), r"Sea evolution", "evo_sea_xf1T.png")
    _plot_panel(("s", "sbar"), r"Strange evolution", "evo_strange_xf1T.png")
    _plot_panel(("u", "d"), r"$u$, $d$ ratio to $Q_0$", "evo_ud_TF_ratio.png", ratio=True)

    logger.info("Saved evolution plots to %s", output_dir)


def run_validation(
    make_plots: bool = True,
    output_dir: str | Path | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    with_evolution: bool = False,
    pdf_backend: str = "lhapdf",
    nx_ic: int = 48,
) -> dict[str, Any]:
    """
    Build paper IC and optionally run LO DGLAP evolution (eta = N_C only).

    Parameters
    ----------
    with_evolution : bool
        If True, evolve on 0.02 < x < 0.6 and produce mu-scan plots/diagnostics.
    pdf_backend : str
        ``'lhapdf'`` (default) or ``'toy'``.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir = Path(output_dir)

    x_ic = torch.logspace(
        torch.log10(torch.tensor(_IC_XMIN)),
        torch.log10(torch.tensor(_IC_XMAX)),
        nx_ic,
        dtype=torch.float64,
    )
    qs_params = QiuStermanParams()
    ic = _build_ic(x_ic, qs_params, pdf_backend=pdf_backend)
    result: dict[str, Any] = dict(ic)

    _print_ic_diagnostics(ic)

    if with_evolution:
        ev = _run_evolution(qs_params, pdf_backend, grid_kwargs)
        result.update(evolution=ev)
        _print_evolution_diagnostics(ev)

    if make_plots:
        _make_plots_ic(ic, output_dir)
        if with_evolution:
            _make_plots_evolution(result["evolution"], output_dir)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Qiu-Sterman IC (NNPDF40 PDF) and LO DGLAP evolution (eta=N_C)."
    )
    parser.add_argument(
        "--with-evolution",
        action="store_true",
        help="Run eta=N_C evolution, diagnostics, and mu-scan plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib figures",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output directory (default: spin/validation/outputs)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_validation(
        make_plots=not args.no_plots,
        output_dir=args.output_dir,
        with_evolution=args.with_evolution,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
