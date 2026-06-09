"""
Paper-style qualitative evolution validation for transversity and Collins Hhat.

Compare evolved shapes, magnitudes, and moments at Q² = 2.4, 10, and 1000 GeV²
against the qualitative expectations of the published figure.  Not a point-by-point
reproduction (inputs are NNPDF40 / NNPDFpol20 / JAM20, not CT10 / DSSV / DSS).

Run from repository root::

    mamba run -n base python -m spin.validation.compare_to_transversity_collins_paper
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from spin.collins import COLLINS_CHANNELS, build_collins_hhat_initial
from spin.dglap import NonSingletDGLAP
from spin.evolution import (
    CollinsHhatEvolution,
    TransversityEvolution,
    evolve_collins_hhat,
    evolve_transversity,
)
from spin.flavors import NAME_TO_PDG, pdg_to_slot
from spin.transversity import TransversityParams, build_h1_initial
from spin.validation.inputs import (
    TC_FF_SET_PION as FF_SET_PION,
    TC_PDF_SET_F1 as PDF_SET_F1,
    TC_PDF_SET_G1 as HELICITY_SET_G1,
    load_pion_ff as load_collins_inputs,
    load_transversity_inputs,
)
from spin.validation.lhapdf_io import try_import_lhapdf

Q0_SQ = 2.4
PAPER_Q2_SCALES: tuple[tuple[str, float], ...] = (
    ("Q0", 2.4),
    ("10", 10.0),
    ("1000", 1000.0),
)

_XMIN = 0.01
_XMAX = 0.95
_ZMIN = 0.05
_ZMAX = 0.95

_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

_GRID = dict(
    nx=80,
    nQ2=56,
    Q20=Q0_SQ,
    LQ2max=4.0,
    loadgrid=False,
    steps=5,
    ng=32,
    dev="cpu",
    dtype=torch.float64,
    order=0,
)


def _trapz(y: torch.Tensor, coord: torch.Tensor) -> float:
    if hasattr(torch, "trapz"):
        return float(torch.trapz(y, coord))
    return float(np.trapz(y.detach().cpu().numpy(), coord.detach().cpu().numpy()))


def _peak_location_value(
    coord: torch.Tensor, values: torch.Tensor, *, use_abs: bool = False
) -> tuple[float, float]:
    arr = values.abs() if use_abs else values
    idx = int(torch.argmax(arr).item())
    return float(coord[idx].item()), float(values[idx].item())


def _normalized_shape(values: torch.Tensor) -> torch.Tensor:
    peak = values.abs().max()
    if float(peak) < 1e-30:
        return torch.zeros_like(values)
    return values / peak


def _shape_l2_distance(
    coord: torch.Tensor, ref: torch.Tensor, other: torch.Tensor
) -> float:
    s_ref = _normalized_shape(ref)
    s_other = _normalized_shape(other)
    diff = s_ref - s_other
    return math.sqrt(max(_trapz(diff**2, coord), 0.0))


def _print_baselines() -> None:
    print("=" * 60)
    print("PAPER-STYLE EVOLUTION COMPARISON")
    print("=" * 60)
    print()
    print("PDF baseline:")
    print(f"  {PDF_SET_F1}")
    print("Helicity baseline:")
    print(f"  {HELICITY_SET_G1}")
    print("Collins FF baseline:")
    print(f"  {FF_SET_PION}")
    print()
    print(
        "This is not an exact reproduction of the paper,\n"
        "which used CT10 + DSSV + DSS pion FFs."
    )
    print()


def _require_lhapdf() -> None:
    if try_import_lhapdf() is None:
        raise ImportError(
            "LHAPDF is required for paper comparison. "
            "Run with: mamba run -n base python -m spin.validation.compare_to_transversity_collins_paper"
        )


def _build_transversity(grid_dir: str):
    dglap = NonSingletDGLAP(
        **_GRID,
        kernel_type="transversity",
        xmin=_XMIN,
        xmax=_XMAX,
        grid_dir=grid_dir,
    )
    x = dglap.x
    f1, g1 = load_transversity_inputs(x, Q0_SQ, force_toy=False)
    h1_0 = build_h1_initial(x, f1, g1, TransversityParams())
    h1 = evolve_transversity(TransversityEvolution.from_dglap(dglap), h1_0)
    return x, h1, dglap


def _build_collins(grid_dir: str):
    dglap = NonSingletDGLAP(
        **_GRID,
        kernel_type="transversity",
        xmin=_ZMIN,
        xmax=_ZMAX,
        grid_dir=grid_dir,
    )
    z = dglap.x
    D = load_collins_inputs(z, Q0_SQ, force_toy=False)
    channels, Hhat0 = build_collins_hhat_initial(z, D)
    Hhat = evolve_collins_hhat(CollinsHhatEvolution.from_dglap(dglap), Hhat0, channels)
    return z, Hhat, channels, Hhat0


def _h1_at_q2(h1, pdg: int, q2: float, x: torch.Tensor) -> torch.Tensor:
    return h1.pdg_at_mu(pdg, math.sqrt(q2))


def _plot_transversity_overlays(
    x: torch.Tensor,
    h1,
    output_dir: Path,
) -> dict[str, dict[str, tuple[float, float]]]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    x_np = x.detach().cpu().numpy()
    iu = pdg_to_slot(NAME_TO_PDG["u"])
    id_ = pdg_to_slot(NAME_TO_PDG["d"])
    peaks: dict[str, dict[str, tuple[float, float]]] = {"u": {}, "d": {}}

    for flavor, slot, fname in (
        ("u", iu, "paper_compare_transversity_u.png"),
        ("d", id_, "paper_compare_transversity_d.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        print(f"\nTransversity {flavor} (x h1^{flavor}):")
        for label, q2 in PAPER_Q2_SCALES:
            hq = _h1_at_q2(h1, NAME_TO_PDG[flavor], q2, x)
            xh = (x * hq).detach().cpu().numpy()
            ax.plot(x_np, xh, label=rf"$Q^2={q2:g}$ GeV$^2$")
            loc, val = _peak_location_value(x, x * hq, use_abs=(flavor == "d"))
            peaks[flavor][label] = (loc, val)
            print(f"  Q²={q2:g}: peak at x={loc:.4f}, value={val:+.5f}")
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(rf"$x\,h_1^{flavor}$")
        ax.set_title(rf"Transversity $h_1^{flavor}$ evolution (paper scales)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return peaks


def _evolution_strength_table(h1, x: torch.Tensor) -> dict[str, dict[str, dict[str, float]]]:
    print()
    print("Evolution strength R = max_x |h1(x,Q²)| / max_x |h1(x,Q0²)|")
    print(f"{'Q²':>10}  {'R_u':>10}  {'R_d':>10}")
    ratios: dict[str, dict[str, dict[str, float]]] = {
        "h1": {"u": {}, "d": {}},
        "xh": {"u": {}, "d": {}},
    }
    hu0 = _h1_at_q2(h1, 2, Q0_SQ, x)
    hd0 = _h1_at_q2(h1, 1, Q0_SQ, x)
    max_u0 = float(hu0.abs().max())
    max_d0 = float(hd0.abs().max())
    max_xhu0 = float((x * hu0).abs().max())
    max_xhd0 = float((x * hd0).abs().max())

    for label, q2 in PAPER_Q2_SCALES:
        hu = _h1_at_q2(h1, 2, q2, x)
        hd = _h1_at_q2(h1, 1, q2, x)
        ru = float(hu.abs().max()) / max_u0
        rd = float(hd.abs().max()) / max_d0
        ratios["h1"]["u"][label] = ru
        ratios["h1"]["d"][label] = rd
        print(f"{q2:10g}  {ru:10.4f}  {rd:10.4f}")

    print()
    print("Paper-plot strength R_xh = max_x |x h1(x,Q²)| / max_x |x h1(x,Q0²)|")
    print(f"{'Q²':>10}  {'R_xh_u':>10}  {'R_xh_d':>10}")
    for label, q2 in PAPER_Q2_SCALES:
        hu = _h1_at_q2(h1, 2, q2, x)
        hd = _h1_at_q2(h1, 1, q2, x)
        ru = float((x * hu).abs().max()) / max_xhu0
        rd = float((x * hd).abs().max()) / max_xhd0
        ratios["xh"]["u"][label] = ru
        ratios["xh"]["d"][label] = rd
        print(f"{q2:10g}  {ru:10.4f}  {rd:10.4f}")
    return ratios


def _tensor_charge_analysis(h1, x: torch.Tensor, output_dir: Path) -> dict[str, list[float]]:
    print()
    print("Tensor charge ratios ∫dx h1^q(x,Q²) relative to Q0²")
    iu = NAME_TO_PDG["u"]
    id_ = NAME_TO_PDG["d"]
    delta_u0 = _trapz(_h1_at_q2(h1, iu, Q0_SQ, x), x)
    delta_d0 = _trapz(_h1_at_q2(h1, id_, Q0_SQ, x), x)
    q2_list: list[float] = []
    ru_list: list[float] = []
    rd_list: list[float] = []

    for label, q2 in PAPER_Q2_SCALES:
        du = _trapz(_h1_at_q2(h1, iu, q2, x), x)
        dd = _trapz(_h1_at_q2(h1, id_, q2, x), x)
        ru = du / delta_u0 if abs(delta_u0) > 1e-30 else float("nan")
        rd = dd / delta_d0 if abs(delta_d0) > 1e-30 else float("nan")
        q2_list.append(q2)
        ru_list.append(ru)
        rd_list.append(rd)
        print(f"  Q²={q2:g}:  delta_u ratio={ru:.4f}   delta_d ratio={rd:.4f}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogx(q2_list, ru_list, "o-", label=r"$\delta_u/\delta_u(Q_0^2)$")
    ax.semilogx(q2_list, rd_list, "s-", label=r"$\delta_d/\delta_d(Q_0^2)$")
    ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$Q^2$ [GeV$^2$]")
    ax.set_ylabel("tensor-charge ratio")
    ax.set_title("Tensor charge evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "tensor_charge_ratios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"q2": q2_list, "u": ru_list, "d": rd_list}


def _plot_collins_overlays(
    z: torch.Tensor,
    Hhat,
    output_dir: Path,
) -> dict[str, dict[str, tuple[float, float]]]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    z_np = z.detach().cpu().numpy()
    ch_map = {
        "fav": (0, "paper_compare_collins_fav.png", r"-z\,\hat H^{(3)}_{\mathrm{fav}}", True),
        "unf": (1, "paper_compare_collins_unf.png", r"z\,\hat H^{(3)}_{\mathrm{unf}}", False),
        "unf_s": (2, "paper_compare_collins_unfs.png", r"z\,\hat H^{(3)}_{\mathrm{unf}_s}", False),
    }
    peaks: dict[str, dict[str, tuple[float, float]]] = {k: {} for k in ch_map}

    for ch_name, (idx, fname, ylab, negate) in ch_map.items():
        conv = "z Hhat (fav: multiply by -1 for display)" if negate else "z Hhat"
        print(f"\nCollins {ch_name} ({conv}):")
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for label, q2 in PAPER_Q2_SCALES:
            H_mu = Hhat.at_mu(math.sqrt(q2))
            zh = z * H_mu[idx]
            plot_vec = -zh if negate else zh
            plot_y = plot_vec.detach().cpu().numpy()
            ax.plot(z_np, plot_y, label=rf"$Q^2={q2:g}$ GeV$^2$")
            loc, val = _peak_location_value(z, plot_vec)
            peaks[ch_name][label] = (loc, val)
            print(f"  Q²={q2:g}: peak at z={loc:.4f}, value={val:+.5f}")
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(rf"${ylab}$")
        ax.set_title(rf"Collins $\hat H^{{(3)}}_{{{ch_name}}}$ evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return peaks


def _collins_moment_analysis(
    z: torch.Tensor, Hhat, output_dir: Path
) -> dict[str, dict[str, float]]:
    print()
    print("Collins z-moments M_ch = ∫ dz z Hhat_ch(z,Q²), ratios to Q0²")
    moments: dict[str, dict[str, float]] = {ch: {} for ch in COLLINS_CHANNELS}
    m0: dict[str, float] = {}

    for i, ch in enumerate(COLLINS_CHANNELS):
        m0[ch] = _trapz(Hhat.at_mu(math.sqrt(Q0_SQ))[i] * z, z)

    for label, q2 in PAPER_Q2_SCALES:
        for i, ch in enumerate(COLLINS_CHANNELS):
            m = _trapz(Hhat.at_mu(math.sqrt(q2))[i] * z, z)
            ratio = m / m0[ch] if abs(m0[ch]) > 1e-30 else float("nan")
            moments[ch][label] = ratio
        parts = "  ".join(f"{ch}={moments[ch][label]:.4f}" for ch in COLLINS_CHANNELS)
        print(f"  Q²={q2:g}: {parts}")

    import matplotlib.pyplot as plt

    q2_vals = [q2 for _, q2 in PAPER_Q2_SCALES]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ch in COLLINS_CHANNELS:
        ax.semilogx(q2_vals, [moments[ch][lab] for lab, _ in PAPER_Q2_SCALES], "o-", label=ch)
    ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$Q^2$ [GeV$^2$]")
    ax.set_ylabel(r"$M_{\mathrm{ch}}(Q^2) / M_{\mathrm{ch}}(Q_0^2)$")
    ax.set_title(r"Collins $\int dz\, z\,\hat H^{(3)}$ moment ratios")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "collins_moment_ratios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return moments


def _shape_metrics(
    x: torch.Tensor, h1, z: torch.Tensor, Hhat
) -> dict[str, dict[str, float]]:
    print()
    print("Normalized-shape L2 distance to Q0²=2.4 (0 = identical shape)")
    ref_q2 = Q0_SQ
    distances: dict[str, dict[str, float]] = {}

    print("Transversity:")
    for flavor, pdg in (("u", 2), ("d", 1)):
        ref = _h1_at_q2(h1, pdg, ref_q2, x)
        distances[flavor] = {}
        for label, q2 in PAPER_Q2_SCALES:
            if q2 == ref_q2:
                distances[flavor][label] = 0.0
                continue
            other = _h1_at_q2(h1, pdg, q2, x)
            distances[flavor][label] = _shape_l2_distance(x, ref, other)
        d10 = distances[flavor]["10"]
        d1000 = distances[flavor]["1000"]
        print(f"  {flavor}:  Q²=10 dist={d10:.4f}   Q²=1000 dist={d1000:.4f}")

    print("Collins:")
    ch_names = ("fav", "unf", "unf_s")
    for i, ch in enumerate(ch_names):
        ref = Hhat.at_mu(math.sqrt(ref_q2))[i]
        distances[ch] = {}
        for label, q2 in PAPER_Q2_SCALES:
            if q2 == ref_q2:
                distances[ch][label] = 0.0
                continue
            other = Hhat.at_mu(math.sqrt(q2))[i]
            distances[ch][label] = _shape_l2_distance(z, ref, other)
        d10 = distances[ch]["10"]
        d1000 = distances[ch]["1000"]
        print(f"  {ch}:  Q²=10 dist={d10:.4f}   Q²=1000 dist={d1000:.4f}")

    return distances


def _check_signs(x: torch.Tensor, h1, z: torch.Tensor, Hhat) -> dict[str, bool]:
    support_x = (x >= 0.02) & (x <= 0.8)
    support_z = (z >= 0.08) & (z <= 0.85)

    hu = _h1_at_q2(h1, 2, Q0_SQ, x)[support_x]
    hd = _h1_at_q2(h1, 1, Q0_SQ, x)[support_x]
    trans_u_pos = bool((hu > 0).all())
    trans_d_neg = bool((hd < 0).all())

    H0 = Hhat.at_mu(math.sqrt(Q0_SQ))
    coll_fav_neg = bool((H0[0][support_z] < 0).all())
    coll_unf_pos = bool((H0[1][support_z] > 0).all())
    coll_unfs_pos = bool((H0[2][support_z] > 0).all())

    return {
        "trans_u_pos": trans_u_pos,
        "trans_d_neg": trans_d_neg,
        "coll_fav_neg": coll_fav_neg,
        "coll_unf_pos": coll_unf_pos,
        "coll_unfs_pos": coll_unfs_pos,
    }


def _monotone_decreasing(ratios: Iterable[float], tol: float = 1e-6) -> bool:
    vals = list(ratios)
    return all(vals[i + 1] <= vals[i] + tol for i in range(len(vals) - 1))


def _final_assessment(
    signs: dict[str, bool],
    strength: dict[str, dict[str, dict[str, float]]],
    tensor: dict[str, list[float]],
    moments: dict[str, dict[str, float]],
    shape_dist: dict[str, dict[str, float]],
) -> str:
    print()
    print("=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)

    trans_signs_ok = signs["trans_u_pos"] and signs["trans_d_neg"]
    coll_signs_ok = (
        signs["coll_fav_neg"] and signs["coll_unf_pos"] and signs["coll_unfs_pos"]
    )
    tensor_ok = _monotone_decreasing(tensor["u"]) and _monotone_decreasing(tensor["d"])
    moments_ok = all(
        _monotone_decreasing([moments[ch][lab] for lab, _ in PAPER_Q2_SCALES])
        for ch in COLLINS_CHANNELS
    )

    ru10 = strength["xh"]["u"]["10"]
    ru1000 = strength["xh"]["u"]["1000"]
    rd10 = strength["xh"]["d"]["10"]
    rd1000 = strength["xh"]["d"]["1000"]
    strength_ok = (
        0.7 < ru10 < 1.0
        and 0.4 < ru1000 < 0.95
        and 0.7 < rd10 < 1.0
        and 0.4 < rd1000 < 0.95
    )
    h1_grows = (
        strength["h1"]["u"]["1000"] > 1.05 or strength["h1"]["d"]["1000"] > 1.05
    )

    shape_ok = all(
        shape_dist[ch]["1000"] < 0.5 for ch in ("u", "d", "fav", "unf", "unf_s")
    )

    norm_ok = ru1000 > 0.05 and moments["fav"]["1000"] > 0.05

    qualitative_ok = trans_signs_ok and coll_signs_ok and tensor_ok and moments_ok and strength_ok

    checks = [
        ("1. Transversity signs correct (u>0, d<0)?", trans_signs_ok),
        ("2. Collins signs correct (fav<0, unf>0)?", coll_signs_ok),
        ("3. Tensor charges monotonically decreasing?", tensor_ok),
        ("4. Collins z-moments monotonically decreasing?", moments_ok),
        (
            "5. |x h1| suppression matches paper (modest @10, strong @1000)?",
            strength_ok,
        ),
        (
            "5b. Note: max|h1| can grow at low x while |x h1| suppresses?",
            h1_grows,
        ),
        ("6. Normalization within ~factor 2 (not collapsed to ~0)?", norm_ok),
        ("7. Shape changes moderate (L2 dist < 0.5 at Q²=1000)?", shape_ok),
    ]
    for question, ok in checks:
        print(f"  {question}  {'YES' if ok else 'NO'}")

    verdict = "PASS" if qualitative_ok and norm_ok else "NEEDS INVESTIGATION"
    print()
    print(f"VERDICT: {verdict}")
    return verdict


def run_comparison(*, output_dir: Path | None = None) -> str:
    _require_lhapdf()
    _print_baselines()

    out_dir = output_dir or _OUTPUT_DIR

    with tempfile.TemporaryDirectory(prefix="spin_paper_cmp_") as grid_dir:
        x, h1, _ = _build_transversity(grid_dir)
        z, Hhat, _, _ = _build_collins(grid_dir)

        signs = _check_signs(x, h1, z, Hhat)

        print("--- STEP 1: Transversity overlays ---")
        _plot_transversity_overlays(x, h1, out_dir)

        print()
        print("--- STEP 2: Evolution strength ---")
        strength = _evolution_strength_table(h1, x)

        print()
        print("--- STEP 3: Tensor charge evolution ---")
        tensor = _tensor_charge_analysis(h1, x, out_dir)

        print()
        print("--- STEP 4: Collins Hhat overlays ---")
        _plot_collins_overlays(z, Hhat, out_dir)

        print()
        print("--- STEP 5: Collins moment evolution ---")
        moments = _collins_moment_analysis(z, Hhat, out_dir)

        print()
        print("--- STEP 6: Paper-shape metrics ---")
        shape_dist = _shape_metrics(x, h1, z, Hhat)

        verdict = _final_assessment(signs, strength, tensor, moments, shape_dist)

    print()
    print(f"Plots written to {out_dir}/")
    for name in (
        "paper_compare_transversity_u.png",
        "paper_compare_transversity_d.png",
        "tensor_charge_ratios.png",
        "paper_compare_collins_fav.png",
        "paper_compare_collins_unf.png",
        "paper_compare_collins_unfs.png",
        "collins_moment_ratios.png",
    ):
        print(f"  {name}")

    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_OUTPUT_DIR,
        help="Directory for comparison plots.",
    )
    args = parser.parse_args()
    run_comparison(output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
