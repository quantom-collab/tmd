"""Smoke tests for validation entry points."""

from __future__ import annotations

import tempfile
from pathlib import Path

from spin.validation.validate_qiu_sterman_evolution import run_validation
from spin.validation.validate_transversity_collins_physics import run_physics_validation


def test_qs_validation_toy_smoke():
    result = run_validation(
        make_plots=False,
        with_evolution=True,
        pdf_backend="toy",
        nx_ic=16,
        grid_kwargs=dict(nx=10, nQ2=6, loadgrid=False, steps=2, ng=10, LQ2max=3.0),
    )
    assert result["TF0"].shape[0] == 6
    if "evolution" in result:
        assert result["evolution"]["norm_ratios"][0] == 1.0


def test_transversity_collins_toy_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        out = run_physics_validation(
            toy=True,
            loadgrid=False,
            nx=16,
            nQ2=6,
            Q20=2.4,
            LQ2max=2.0,
            make_plots=False,
            output_dir=Path(tmp),
        )
    assert out["transversity"]["soffer_max"] <= 1.0 + 1e-10
    assert out["collins"]["max_rel_diff_Q0"] < 1e-8
