"""
Path resolution for unified SIDIS YAML cards (perturbative + fNP in one file).

Callers pass either ``None`` (default card name), a basename looked up under
``sidis_dir/cards/``, or a filesystem path. Resolution order matches what CLIs
and ``TruthModel`` expect so the same string works from project root or notebooks.
"""

import pathlib
from typing import Optional


def resolve_card_path(
    fnp_config: Optional[str],
    sidis_dir: pathlib.Path,
    default_name: str = "fNPconfig_simple.yaml",
) -> pathlib.Path:
    """
    Resolve a configuration card path under ``sidis_dir/cards/`` or an explicit filesystem path.

    Used by ``TruthModel`` and by CLIs (``main.py``, ``runfit.py``).
    """
    # Cards live in a single package-relative directory so CLIs can pass only a basename.
    cards_dir = sidis_dir / "cards"
    if fnp_config is None:
        path = cards_dir / default_name
    else:
        p = pathlib.Path(fnp_config)
        # Prefer an absolute path that already exists (typical for notebooks or scripted runs).
        if p.is_absolute() and p.is_file():
            path = p.resolve()
        # Next: basename or relative fragment under ``cards/`` (CLI default).
        elif (cards_dir / fnp_config).is_file():
            path = (cards_dir / fnp_config).resolve()
        # CWD-relative files: ``./my.yaml`` without making them absolute first in the caller.
        elif p.is_file():
            path = p.resolve()
        else:
            # Defer ``FileNotFoundError`` to the unified check below (clear error for typos).
            path = (cards_dir / fnp_config).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration card not found: {path}")
    return path
