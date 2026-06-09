"""
Numerical diagnostics for LO non-singlet kernel shift operators.

Compares discrete ``get_shift(mu2, K)`` matrices for unpolarized, Qiu-Sterman,
and transversity kernels on a common grid.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from spin import alphaS, params
from spin.dglap import NonSingletDGLAP

_DEFAULT_GRID = dict(
    dev="cpu",
    dtype=torch.float64,
    order=0,
    nx=16,
    xmin=0.02,
    xmax=0.8,
    nQ2=8,
    Q20=2.4,
    LQ2max=3.5,
    loadgrid=False,
    steps=3,
    ng=16,
)


@dataclass
class KernelShiftComparison:
    """Discrete shift operators at one Q^2 for three kernel types."""

    Q2: float
    shift_unpol: torch.Tensor
    shift_qs: torch.Tensor
    shift_trans: torch.Tensor
    shift_k1_subtraction: torch.Tensor
    alpha_over_2pi: float

    @property
    def shift_qs_minus_unpol(self) -> torch.Tensor:
        return self.shift_qs - self.shift_unpol

    @property
    def shift_unpol_minus_trans(self) -> torch.Tensor:
        return self.shift_unpol - self.shift_trans


def shift_xi_component(dglap: NonSingletDGLAP, mu2: float) -> torch.Tensor:
    """Convolution part from kernel slot ``K[0]`` only."""
    Kxi = dglap.kernels.K[0](dglap.xg_2D, dglap.xig_2D, mu2)
    return torch.einsum("g,gj,gj,gji->ij", dglap.wg, dglap.jac, Kxi, dglap.Lg)


def shift_k1_subtraction_only(dglap: NonSingletDGLAP, mu2: float) -> torch.Tensor:
    """
    Discretized diagonal piece from the unpolarized ``K[1]`` subtraction slot.

    Uses the same ``Ac[xi != x] = 0`` mask as :meth:`NonSingletDGLAP.get_shift`.
    """
    Kx = dglap.kernels.K[1](dglap.xg_2D, dglap.xig_2D, mu2)
    Ac = torch.zeros((dglap.nx, dglap.nx), dtype=dglap.dtype, device=dglap.dev)
    Ac[:] = torch.einsum("g,gj,gj->j", dglap.wg, dglap.jac, Kx)
    Ac[dglap.xi_2D != dglap.x_2D] = 0.0
    return Ac


def shift_cf_1minusx_regular_only(
    d_unpol: NonSingletDGLAP,
    d_trans: NonSingletDGLAP,
    mu2: float,
) -> torch.Tensor:
    """Convolution-slot piece of ``P_qq - P_h1 = C_F(1-x)`` (``K[0]`` difference)."""
    axi_u = shift_xi_component(d_unpol, mu2)
    axi_t = shift_xi_component(d_trans, mu2)
    return axi_u.T - axi_t.T


def discretized_cf_1minusx_shift(
    d_unpol: NonSingletDGLAP,
    d_trans: NonSingletDGLAP,
    mu2: float,
) -> torch.Tensor:
    """
    Full discrete ``C_F(1-x)`` contribution: ``shift_unpol - shift_trans``.

    Unpolarized and transversity share the LO ``K[1]`` subtraction slot, so the
    difference is only ``(Axi_unpol - Axi_trans).T``.
    """
    return shift_cf_1minusx_regular_only(d_unpol, d_trans, mu2)


def build_kernel_triplet(
    grid_kwargs: dict[str, Any] | None = None,
    grid_dir: str | None = None,
) -> tuple[NonSingletDGLAP, NonSingletDGLAP, NonSingletDGLAP]:
    """Three DGLAP builders on the same grid parameters."""
    gkw = {**_DEFAULT_GRID, **(grid_kwargs or {})}
    if grid_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="spin_kern_")
        grid_dir = tmp.name
        # caller should keep tmp alive if needed; tests use TemporaryDirectory context
    d_unpol = NonSingletDGLAP(kernel_type="unpolarized", grid_dir=grid_dir, **gkw)
    d_qs = NonSingletDGLAP(
        kernel_type="qiu_sterman", eta=params.NC, grid_dir=grid_dir, **gkw
    )
    d_trans = NonSingletDGLAP(kernel_type="transversity", grid_dir=grid_dir, **gkw)
    return d_unpol, d_qs, d_trans


def compare_shifts_at_Q2(
    d_unpol: NonSingletDGLAP,
    d_qs: NonSingletDGLAP,
    d_trans: NonSingletDGLAP,
    Q2: float | None = None,
) -> KernelShiftComparison:
    """Build shift operators at ``Q2`` (default ``Q20``)."""
    mu2 = float(d_unpol.Q20 if Q2 is None else Q2)
    shift_u = d_unpol.get_shift(mu2, d_unpol.kernels.K)
    shift_q = d_qs.get_shift(mu2, d_qs.kernels.K)
    shift_t = d_trans.get_shift(mu2, d_trans.kernels.K)
    shift_k1 = shift_k1_subtraction_only(d_unpol, mu2)
    a = alphaS.get_alphaS(mu2) / (2.0 * 3.141592653589793)
    return KernelShiftComparison(
        Q2=mu2,
        shift_unpol=shift_u,
        shift_qs=shift_q,
        shift_trans=shift_t,
        shift_k1_subtraction=shift_k1,
        alpha_over_2pi=a,
    )


def verify_transversity_vs_unpolarized(
    cmp: KernelShiftComparison,
    d_unpol: NonSingletDGLAP,
    d_trans: NonSingletDGLAP,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    """``shift_trans = shift_unpol - discretized C_F(1-x) contribution``."""
    shift_1mx = discretized_cf_1minusx_shift(d_unpol, d_trans, cmp.Q2)
    torch.testing.assert_close(
        cmp.shift_trans,
        cmp.shift_unpol - shift_1mx,
        rtol=rtol,
        atol=atol,
        msg="transversity shift should equal unpolarized minus C_F(1-x) piece",
    )
    # Independent construction matches total matrix difference
    torch.testing.assert_close(
        shift_1mx,
        cmp.shift_unpol_minus_trans,
        rtol=rtol,
        atol=atol,
    )


def verify_qiu_sterman_eta_diagonal(
    cmp: KernelShiftComparison,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    """Qiu-Sterman vs unpolarized differs only by ``-alpha_s/(2pi)*N_C`` on diagonal."""
    diff = cmp.shift_qs_minus_unpol
    off = diff.clone()
    off.fill_diagonal_(0.0)
    assert off.abs().max().item() < atol, "off-diagonal difference should vanish"
    diag = diff.diagonal()
    expected = torch.full_like(diag, -cmp.alpha_over_2pi * params.NC)
    torch.testing.assert_close(diag, expected, rtol=rtol, atol=atol)


def verify_eta_zero_matches_unpolarized(
    d_unpol: NonSingletDGLAP,
    grid_dir: str,
    grid_kwargs: dict[str, Any],
    Q2: float | None = None,
    *,
    atol: float = 1e-12,
) -> None:
    """``kernel_type='qiu_sterman', eta=0`` shift equals unpolarized."""
    gkw = {**grid_kwargs}
    d_qs0 = NonSingletDGLAP(
        kernel_type="qiu_sterman", eta=0.0, grid_dir=grid_dir, **gkw
    )
    mu2 = float(d_unpol.Q20 if Q2 is None else Q2)
    s_u = d_unpol.get_shift(mu2, d_unpol.kernels.K)
    s_q0 = d_qs0.get_shift(mu2, d_qs0.kernels.K)
    torch.testing.assert_close(s_q0, s_u, rtol=1e-12, atol=atol)


def run_shift_diagnostics(
    grid_kwargs: dict[str, Any] | None = None,
    Q2: float | None = None,
) -> KernelShiftComparison:
    """Build triplet, compare shifts, run checks, print summary."""
    with tempfile.TemporaryDirectory(prefix="spin_kern_diag_") as tmp:
        d_u, d_q, d_t = build_kernel_triplet(grid_kwargs, grid_dir=tmp)
        cmp = compare_shifts_at_Q2(d_u, d_q, d_t, Q2=Q2)
        gkw = {**_DEFAULT_GRID, **(grid_kwargs or {})}
        verify_transversity_vs_unpolarized(cmp, d_u, d_t)
        verify_qiu_sterman_eta_diagonal(cmp)
        verify_eta_zero_matches_unpolarized(d_u, tmp, gkw, Q2=Q2)
        shift_1mx = discretized_cf_1minusx_shift(d_u, d_t, cmp.Q2)

    print("--- Kernel shift diagnostics ---")
    print(f"Q2 = {cmp.Q2:.6g} GeV^2,  alpha_s/(2pi) = {cmp.alpha_over_2pi:.6g}")
    print(
        f"|shift_trans - (shift_unpol - shift_1mx)|_max = "
        f"{(cmp.shift_trans - (cmp.shift_unpol - shift_1mx)).abs().max():.3e}"
    )
    print(
        f"|shift_1mx - (shift_unpol - shift_trans)|_max = "
        f"{(shift_1mx - cmp.shift_unpol_minus_trans).abs().max():.3e}"
    )
    diff = cmp.shift_qs_minus_unpol
    off = diff.clone()
    off.fill_diagonal_(0.0)
    print(f"|off-diagonal(shift_qs - shift_unpol)|_max = {off.abs().max():.3e}")
    print(
        f"diagonal(shift_qs - shift_unpol) vs -alpha_s/(2pi)*N_C: "
        f"{diff.diagonal().mean():.6g} (expect {-cmp.alpha_over_2pi * params.NC:.6g})"
    )
    print("OK: transversity/unpolarized/K[1] relation and Qiu-Sterman eta diagonal.")
    return cmp


def make_evolution_comparison_plot(
    output_dir: str | Path,
    grid_kwargs: dict[str, Any] | None = None,
) -> Path:
    """
    Evolve the same toy input with unpolarized, transversity, and eta=N_C Qiu-Sterman.

    Saves ``kernel_evolution_comparison.png``.
    """
    import matplotlib.pyplot as plt

    from spin.evolution import NonSingletEvolution

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spin_kern_plot_") as tmp:
        gkw = {**_DEFAULT_GRID, **(grid_kwargs or {})}
        d_u = NonSingletDGLAP(kernel_type="unpolarized", grid_dir=tmp, **gkw)
        d_q0 = NonSingletDGLAP(
            kernel_type="qiu_sterman", eta=0.0, grid_dir=tmp, **gkw
        )
        d_t = NonSingletDGLAP(kernel_type="transversity", grid_dir=tmp, **gkw)
        d_q = NonSingletDGLAP(
            kernel_type="qiu_sterman", eta=params.NC, grid_dir=tmp, **gkw
        )

        x = d_u.x
        f0 = x * (1.0 - x) ** 2
        evo_u = NonSingletEvolution.from_dglap(d_u)
        evo_q0 = NonSingletEvolution.from_dglap(d_q0)
        evo_t = NonSingletEvolution.from_dglap(d_t)
        evo_q = NonSingletEvolution.from_dglap(d_q)

        out_u = evo_u(f0)[..., -1].detach().cpu().numpy()
        out_q0 = evo_q0(f0)[..., -1].detach().cpu().numpy()
        out_t = evo_t(f0)[..., -1].detach().cpu().numpy()
        out_q = evo_q(f0)[..., -1].detach().cpu().numpy()
        mu_max = float(d_u.Q2[-1].sqrt())
    f0_np = f0.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(x_np, f0_np, "k--", label=r"$f_0(x)$", lw=1.2)
    axes[0].plot(x_np, out_u, label="unpolarized")
    axes[0].plot(x_np, out_q0, "--", label=r"Qiu-Sterman $\eta=0$", alpha=0.85)
    axes[0].plot(x_np, out_t, label="transversity")
    axes[0].plot(x_np, out_q, label=rf"Qiu-Sterman $\eta=N_C$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$f(x)$ at $Q_{\max}$")
    axes[0].set_title(rf"Evolved toy input at $\mu \approx {mu_max:.2f}$ GeV")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    ratio_u = out_u / f0_np
    ratio_q0 = out_q0 / f0_np
    ratio_t = out_t / f0_np
    ratio_q = out_q / f0_np
    axes[1].plot(x_np, ratio_u, label="unpolarized / f0")
    axes[1].plot(x_np, ratio_q0, "--", label=r"QS $\eta=0$ / f0", alpha=0.85)
    axes[1].plot(x_np, ratio_t, label="transversity / f0")
    axes[1].plot(x_np, ratio_q, label=r"QS $\eta=N_C$ / f0")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$f(Q_{\max}) / f_0$")
    axes[1].set_title("Evolution ratio to IC")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "kernel_evolution_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    run_shift_diagnostics()
    out = Path(__file__).resolve().parent / "outputs"
    p = make_evolution_comparison_plot(out)
    print(f"Saved evolution comparison plot to {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
