"""
Produce cross-section outputs from an events file using a chosen fNP model.

Reads kinematic points from a PyTorch-saved events file,
runs the SIDIS model with the specified configuration card, and writes:
  - PyTorch format: events tensor + cross-section tensor
  - YAML format: human-readable list of kinematic point + cross_section

Usage (from repo root):
  python sidis/tests/run_cross_section_from_mock.py
  python sidis/tests/run_cross_section_from_mock.py -c fNPconfig_base_flexible.yaml
  python sidis/tests/run_cross_section_from_mock.py -e path/to/events.dat -c fNPconfig_base.yaml
"""

import argparse
import pathlib
import sys

import torch

# Ensure sidis can be imported from repo root
_repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Set default dtype before importing model
torch.set_default_dtype(torch.float64)

from omegaconf import OmegaConf
from sidis.model import TrainableModel


def main():
    parser = argparse.ArgumentParser(
        description="""Produce cross-section outputs from an events file using a chosen fNP model.

Reads kinematic points from a PyTorch-saved events file, runs the SIDIS model
with the specified configuration card, and writes outputs to tests/outs/:
  - PyTorch format: events tensor + cross-section tensor
  - YAML format: human-readable list of kinematic point + cross_section""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Usage examples:
  From repo root:
    python sidis/tests/run_cross_section_from_mock.py
    python sidis/tests/run_cross_section_from_mock.py -c fNPconfig_base_flexible.yaml
    python sidis/tests/run_cross_section_from_mock.py -e path/to/events.dat -c fNPconfig_base.yaml
    python sidis/tests/run_cross_section_from_mock.py -e mock_events_100.dat -o my_outs/

  From sidis/tests/:
    python run_cross_section_from_mock.py
    python run_cross_section_from_mock.py -c fNPconfig_base_flexible.yaml -e mock_events_100.dat""",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="fNPconfig_base.yaml",
        help="fNP configuration card (in sidis/cards/). Default: fNPconfig_base.yaml",
    )
    parser.add_argument(
        "-e",
        "--events",
        type=str,
        default=None,
        help="Path to events file (PyTorch .pt/.dat). Default: mock_events_100.dat in tests/",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for .pt and .yaml files. Default: sidis/tests/outs/",
    )
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    sidis_dir = script_dir.parent

    # Resolve events file path
    if args.events is None:
        events_file = script_dir / "mock_events_100.dat"
    else:
        events_file = pathlib.Path(args.events)
        if not events_file.is_absolute():
            if events_file.exists():
                events_file = events_file.resolve()
            else:
                events_file = script_dir / events_file
    outs_dir = pathlib.Path(args.output_dir) if args.output_dir else script_dir / "outs"

    if not events_file.exists():
        print(f"Error: events file not found: {events_file}")
        print(
            "Generate it first (e.g. run the mock data cell in test_model_flexible.ipynb)."
        )
        sys.exit(1)

    events_tensor = torch.load(events_file)
    n_events = events_tensor.shape[0]
    n_cols = events_tensor.shape[1]
    print(f"Loaded {n_events} kinematic points from {events_file}")
    print(
        f"Shape: {events_tensor.shape} (columns: x, PhT, Q, z"
        + (", phih, phis" if n_cols >= 6 else "")
        + ")"
    )

    # Resolve config path (must be in sidis/cards/)
    config_name = args.config
    cards_dir = sidis_dir / "cards"
    config_path = cards_dir / config_name
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        print(f"Available configs in {cards_dir}:")
        if cards_dir.exists():
            for f in sorted(cards_dir.glob("*.yaml")):
                print(f"  - {f.name}")
        sys.exit(1)

    print(f"Loading model with config: {config_name}")
    model = TrainableModel(fnp_config=config_name)
    model.eval()

    with torch.no_grad():
        cross_section = model(events_tensor)

    cross_section = cross_section.cpu()
    events_cpu = events_tensor.cpu()

    outs_dir.mkdir(parents=True, exist_ok=True)
    base_name = "cross_section_output"

    # PyTorch format: same content
    pt_path = outs_dir / f"{base_name}.pt"
    torch.save(
        {
            "events": events_cpu,
            "cross_section": cross_section,
        },
        pt_path,
    )
    print(f"Written PyTorch: {pt_path}")

    # YAML format: human-readable, same content
    yaml_path = outs_dir / f"{base_name}.yaml"
    column_names = ["x", "PhT", "Q", "z"]
    if n_cols >= 6:
        column_names.extend(["phih", "phis"])

    rows = []
    for i in range(n_events):
        row = {}
        for j, name in enumerate(
            ["x", "PhT", "Q", "z"] + (["phih", "phis"] if n_cols >= 6 else [])
        ):
            if j < events_cpu.shape[1]:
                row[name] = float(events_cpu[i, j].item())
        row["cross_section"] = float(cross_section[i].item())
        rows.append(row)

    out_data = {
        "description": f"Cross-sections for kinematic points from {events_file.name} (config: {config_name})",
        "n_events": n_events,
        "kinematic_columns": column_names,
        "data": rows,
    }
    with open(yaml_path, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(out_data)))
    print(f"Written YAML:   {yaml_path}")

    print("Done.")


if __name__ == "__main__":
    main()
