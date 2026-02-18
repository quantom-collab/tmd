"""
Produce cross-section outputs from an events file using a chosen fNP model.

Reads kinematic points from a PyTorch-saved events file,
runs the SIDIS model with the specified configuration card, and writes:
  - PyTorch format: events tensor + cross-section tensor
  - YAML format: human-readable list of kinematic point + cross_section

Usage (from repo root):
  python sidis/tests/run_cross_section.py
  python sidis/tests/run_cross_section.py -c fNPconfig_base_flexible.yaml
  python sidis/tests/run_cross_section.py -e inputs/mock_events_1000.dat -c fNPconfig_simple.yaml
"""

import argparse
import pathlib
import torch
import sys

# Ensure sidis can be imported from repo root
_repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from sidis.utilities.colors import tcolors

# Set default dtype before importing model
torch.set_default_dtype(torch.float64)

from omegaconf import OmegaConf
from sidis.model import TrainableModel

# Default values for command line arguments
DEFAULT_EVENTS_FILE = "mock_events_1000.dat"
DEFAULT_CONFIG_FILE = "fNPconfig_simple.yaml"
DEFAULT_OUTPUT_DIR = "outs/"


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    # Help text
    parser = argparse.ArgumentParser(
        description=f"""{tcolors.BOLDWHITE}Produce cross-section outputs from an events file using a chosen fNP model. {tcolors.ENDC}

Reads kinematic points from a PyTorch-saved events file, runs the SIDIS model
with the specified configuration card, and writes outputs to tests/outs/:
  - PyTorch format: events tensor + cross-section tensor
  - YAML format: human-readable list of kinematic point + cross_section""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Usage examples:
  
  From repo root:
    python sidis/tests/run_cross_section.py
    python sidis/tests/run_cross_section.py -c fNPconfig_base_flexible.yaml
    python sidis/tests/run_cross_section.py -e path/to/events.dat -c fNPconfig_simple.yaml
    python sidis/tests/run_cross_section.py -e mock_events_100.dat -o my_outs/

  From sidis/tests/:
    python run_cross_section.py
    python run_cross_section.py -c fNPconfig_simple.yaml -e inputs/mock_events_100.dat""",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="fNPconfig_simple.yaml",
        help="fNP configuration card (in sidis/cards/). Default: fNPconfig_simple.yaml",
    )
    parser.add_argument(
        "-e",
        "--events",
        type=str,
        default=None,
        help=f"Path to events file (PyTorch .pt/.dat). Default: mock_events_1000.dat in tests/{tcolors.ENDC}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for .pt and .yaml files. Default: sidis/tests/outs/",
    )

    # Parse arguments
    args = parser.parse_args()

    # Resolve paths
    script_dir = pathlib.Path(__file__).resolve().parent
    sidis_dir = script_dir.parent

    # Resolve events file path
    if args.events is None:
        events_file = script_dir / "inputs" / DEFAULT_EVENTS_FILE
    else:
        events_file = pathlib.Path(args.events)
        if not events_file.is_absolute():
            if events_file.exists():
                events_file = events_file.resolve()
            else:
                events_file = script_dir / events_file

    # Resolve output directory path
    outs_dir = (
        pathlib.Path(args.output_dir)
        if args.output_dir
        else script_dir / DEFAULT_OUTPUT_DIR
    )

    if not events_file.exists():
        print(
            f"{tcolors.FAIL}Error: events file not found: {events_file}{tcolors.ENDC}"
        )
        print(
            f"{tcolors.WARNING}Generate it first (e.g. run the mock data cell in test_model_flexible.ipynb).{tcolors.ENDC}"
        )
        sys.exit(1)

    # Load events tensor
    events_tensor = torch.load(events_file)

    # Get number of events and columns
    n_events = events_tensor.shape[0]
    n_cols = events_tensor.shape[1]

    print(
        f"{tcolors.GREEN}\nLoaded {n_events} kinematic points from {events_file}{tcolors.ENDC}"
    )
    print(
        f"{tcolors.BOLDWHITE}Shape: {events_tensor.shape} (columns: x, PhT, Q, z"
        + (", phih, phis" if n_cols >= 6 else "")
        + f"){tcolors.ENDC}"
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

    # Load model
    print(f"{tcolors.GREEN}Loading model with config: {config_name}{tcolors.ENDC}")
    model = TrainableModel(fnp_config=config_name)
    model.eval()

    # Run model forward pass
    with torch.no_grad():
        cross_section = model(events_tensor)

    # Ensure cross section and events are on the CPU
    cross_section = cross_section.cpu()
    events_cpu = events_tensor.cpu()

    # Create output directory if it doesn't exist
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
