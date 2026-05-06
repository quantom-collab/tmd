"""
Non-perturbative TMD evolution factor shared by PDF and FF f_NP pieces.

Implements the Collins-Soper style factor
  S_NP(ζ, b) ∝ exp(-g2^2 * b^2/4 * ln(ζ/Q₀²))
with a single width parameter ``g2`` that can be fixed or trainable per card.

Configuration mirrors PDF/FF blocks. Trainable ``g2`` with bounds is stored in
*logit space* and mapped with the same sigmoid reparametrization as bounded
TMD parameters (``g2 = lo + (hi - lo) σ(θ)``), so the optimizer never leaves the
open interval.

``TMDBuilder`` calls
``forward_evolution`` on the manager, which delegates here.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from ...utilities.colors import tcolors
except ImportError:
    try:
        from utilities.colors import tcolors
    except ImportError:
        from sidis.utilities.colors import tcolors

from ..fnp_config import ParameterLinkParser


class fNP_evolution(nn.Module):
    """Shared non-perturbative evolution factor (trainable or fixed ``g2``)."""

    # ``free_g2`` is the raw trainable tensor: physical ``g2`` when unbounded,
    # logit ``θ`` when bounded (sigmoid map). ``None`` when ``g2`` is fixed (buffer).
    free_g2: Optional[nn.Parameter]

    def __init__(
        self,
        init_params: List[float],
        free_mask: List[Any],
        bounds_list: List[Optional[Tuple[float, float]]],
    ) -> None:

        # Initialize the parent class (nn.Module)constructor
        super().__init__()

        # Validate the input parameters
        if len(init_params) != 1 or len(free_mask) != 1:
            raise ValueError(
                f"[fnp/fnp_evolution.py] {tcolors.FAIL}Evolution expects exactly "
                f"one init_param and one free_mask entry; got "
                f"len(init_params)={len(init_params)}, len(free_mask)={len(free_mask)}"
                f"{tcolors.ENDC}"
            )
        bounds_list = list(bounds_list)
        while len(bounds_list) < 1:
            bounds_list.append(None)

        parser = ParameterLinkParser()
        parsed = parser.parse_entry(free_mask[0], "evolution", "g2")
        if parsed["type"] != "boolean":
            raise ValueError(
                f"[fnp/fnp_evolution.py] {tcolors.FAIL}evolution free_mask must be "
                f"a boolean only (no references/expressions); got type={parsed['type']!r}"
                f"{tcolors.ENDC}"
            )

        init_val = float(init_params[0])
        bound = bounds_list[0]
        # Q₀² for ln(ζ/Q₀²); kept as a buffer so `.to(device)` moves it with the module.
        self.register_buffer("Q0_squared", torch.tensor(1.0, dtype=torch.float32))
        self.free_g2 = None
        # When True, ``free_g2`` holds logit θ and physical g₂ uses sigmoid.
        self._logit_reparam = False
        # Buffers used only in the bounded-trainable branch (lo/hi on same device as module).
        self.register_buffer(
            "_g2_lo",
            torch.zeros(1, dtype=torch.float32),
        )
        self.register_buffer(
            "_g2_hi",
            torch.ones(1, dtype=torch.float32),
        )
        # Non-zero size only for fixed g₂; kept as length-1 tensor for broadcasting like before.
        self.register_buffer(
            "_g2_fixed",
            torch.tensor([init_val], dtype=torch.float32),
        )

        if parsed["is_fixed"]:
            # Fixed nominal value; no trainable parameter, ``free_g2`` stays ``None``.
            self._g2_fixed.copy_(torch.tensor([init_val], dtype=torch.float32))
            return

        if bound is not None:
            lo, hi = bound
            assert hi > lo, f"evolution param_bounds require hi > lo, got {(lo, hi)}"
            self._logit_reparam = True
            self._g2_lo.copy_(torch.tensor([lo], dtype=torch.float32))
            self._g2_hi.copy_(torch.tensor([hi], dtype=torch.float32))

            # Match PDF/FF: map initial physical value to interior-uniform in u then logit( u ).
            u = (init_val - lo) / (hi - lo)
            u_t = torch.tensor(u, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
            theta = torch.logit(u_t).unsqueeze(0)
            self.free_g2 = nn.Parameter(theta)
            return

        # Trainable and unbounded: optimize ``g₂`` directly in physical space.
        self._logit_reparam = False
        self.free_g2 = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))

    def uses_logit_reparam(self) -> bool:
        """True iff the trainable tensor is logit θ with sigmoid mapping to ``g₂``."""
        return self._logit_reparam

    @property
    def g2(self) -> torch.Tensor:
        """Physical ``g₂`` (shape ``[1]``), whether fixed, direct, or sigmoid-mapped."""
        if self.free_g2 is None:
            return self._g2_fixed
        if self._logit_reparam:
            lo = self._g2_lo
            hi = self._g2_hi
            return lo + (hi - lo) * torch.sigmoid(self.free_g2)
        return self.free_g2

    def forward(self, b: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        if b.dim() > zeta.dim():
            zeta = zeta.unsqueeze(-1)
        g2 = self.g2
        return torch.exp(-(g2**2) * (b**2) * torch.log(zeta / self.Q0_squared) / 4.0)
