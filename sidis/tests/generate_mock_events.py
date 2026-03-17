"""
Produce cross-section outputs from a kinematics file using a chosen fNP model.

Reads kinematic points from a PyTorch-saved kinematics file,
runs the SIDIS model with the specified configuration card, and writes:
  - PyTorch format: events tensor + cross-section tensor
  - YAML format: human-readable list of kinematic point + cross_section

Usage (from repo root):
  python sidis/tests/generate_mock_events.py
  python sidis/tests/generate_mock_events.py -c fNPconfig_base_flexible.yaml
  python sidis/tests/generate_mock_events.py -e kin/mock_kinematics.dat -c fNPconfig_simple.yaml
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
DEFAULT_KINEMATICS_FILE = "mock_kinematics_1000.dat"
DEFAULT_CONFIG_FILE = "fNPconfig_simple.yaml"
DEFAULT_OUTPUT_DIR = "events"


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    # Help text
    parser = argparse.ArgumentParser(
        description=f"""{tcolors.BOLDGREEN}Produce cross-section outputs from a kinematics file using a chosen fNP model. {tcolors.ENDC}

    Reads kinematic points from a PyTorch-saved kinematics file, runs the SIDIS model
    with the specified configuration card, and writes outputs to tests/events/:
    - PyTorch format: events tensor + cross-section tensor
    - YAML format: human-readable list of kinematic point + cross_section""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Usage examples:
    
    From repo root:
        python sidis/tests/generate_mock_events.py

    The default is equivalent to running:
        python sidis/tests/generate_mock_events.py -e kin/mock_kinematics_1000.dat -c fNPconfig_simple.yaml -o events/cross_section_events_1000

    From sidis/tests/:
        python generate_mock_events.py -c fNPconfig_simple.yaml -e kin/mock_kinematics_10000.pt -o events/cross_section_events_10000""",
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
        help=f"Path to kinematics file (PyTorch .pt/.dat). Default: mock_kinematics.dat in tests/kin/{tcolors.ENDC}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (without extension). Writes <path>.pt and <path>.yaml. Default: events/cross_section_events_<n_events>",
    )

    # Parse arguments
    args = parser.parse_args()

    # Resolve paths
    script_dir = pathlib.Path(__file__).resolve().parent
    sidis_dir = script_dir.parent

    # Resolve kinematics file path, in the kin/ directory.
    # If no kinematics file is provided, use the default one.
    if args.events is None:
        events_file = script_dir / "kin" / DEFAULT_KINEMATICS_FILE
    else:
        # Get the path provided by the user.
        events_file = pathlib.Path(args.events)
        # If the path provided is not absolute, if it exists, make it absolute.
        # If it doesn't exist, treat is as a path under the kin/ directory.
        if not events_file.is_absolute():
            if events_file.exists():
                events_file = events_file.resolve()
            else:
                events_file = script_dir / "kin" / events_file

    if not events_file.exists():
        print(
            f"{tcolors.FAIL}Error: kinematics file not found: {events_file}{tcolors.ENDC}"
        )
        print(
            f"{tcolors.WARNING}Generate it first with: python sidis/tests/generate_mock_kinematics.py{tcolors.ENDC}"
        )
        sys.exit(1)

    # Load events tensor
    events_tensor = torch.load(events_file)

    # Get number of events and columns
    n_events = events_tensor.shape[0]
    n_cols = events_tensor.shape[1]

    # Print events file and shape
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

    # Resolve output file path (base without extension)
    if args.output is not None:
        output_base = pathlib.Path(args.output).with_suffix("")
        if not output_base.is_absolute():
            output_base = output_base.resolve()
    else:
        output_base = (
            script_dir / DEFAULT_OUTPUT_DIR / f"cross_section_events_{n_events}"
        )

    # Create output directory if it doesn't exist
    output_base.parent.mkdir(parents=True, exist_ok=True)

    # PyTorch format: same content + metadata.
    # kinematic_columns documents the column order of events (used by runfit.py).
    kinematic_columns = ["x", "PhT", "Q", "z"]
    if n_cols >= 6:
        kinematic_columns.extend(["phih", "phis"])
    pt_path = output_base.with_suffix(".pt")
    torch.save(
        {
            "events": events_cpu,
            "cross_section": cross_section,
            "config": config_name,
            "n_events": n_events,
            "kinematic_columns": kinematic_columns,
            "description": f"Cross-sections for kinematic points from {events_file.name}",
        },
        pt_path,
    )
    print(f"Written PyTorch: {pt_path}")

    # YAML format: human-readable, same content.
    yaml_path = output_base.with_suffix(".yaml")
    rows = []
    for i in range(n_events):
        row = {}
        for j, name in enumerate(kinematic_columns):
            if j < events_cpu.shape[1]:
                row[name] = float(events_cpu[i, j].item())
        row["cross_section"] = float(cross_section[i].item())
        rows.append(row)

    out_data = {
        "description": f"Cross-sections for kinematic points from {events_file.name}",
        "config": config_name,
        "n_events": n_events,
        "data": rows,
    }
    with open(yaml_path, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(out_data)))
    print(f"Written YAML:   {yaml_path}")


if __name__ == "__main__":
    main()
