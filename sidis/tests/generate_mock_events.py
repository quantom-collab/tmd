"""
Generate synthetic unpolarized SIDIS event data.

Creates mock events with 4 columns [x, PhT, Q, z] in realistic kinematic ranges.
PhT respects the TMD boundary PhT/(z·Q) < 0.2 per event.

Saves both .dat and .pt files (same content, readable and PyTorch format).
Creates inputs/ folder if it does not exist.
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

# Add repo root to path so sidis can be imported when script is run directly
_repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from sidis.utilities.colors import tcolors


def generate_mock_events(n_events: int = 1000, seed: int = 42) -> torch.Tensor:
    """
    Generate synthetic unpolarized SIDIS event data.

    Each event has 4 columns: [x, PhT, Q, z]. Values are drawn uniformly
    in realistic kinematic ranges (see below). PhT respects the boundary
    PhT/(z*Q) < 0.2 for each event.

    Parameters
    ----------
    n_events : int
        Number of events to generate
    seed : int
        Random seed for reproducibility

    Returns
    -------
    torch.Tensor
        Event tensor of shape (n_events, 4)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Realistic kinematic ranges
    x_min, x_max = 0.001, 0.9  # Bjorken x
    PhT_min, PhT_max = 0.1, 5.0  # Transverse momentum (GeV)
    Q_min, Q_max = 2.0, 1000.0  # Hard scale (GeV)
    z_min, z_max = 0.2, 0.8  # Fragmentation fraction

    # Generate x, Q, z in a naive way, uniformly in their ranges
    # x sampled log-uniform in [x_min, x_max]
    x = np.exp(np.random.uniform(np.log(x_min), np.log(x_max), n_events))
    Q = np.random.uniform(Q_min, Q_max, n_events)
    z = np.random.uniform(z_min, z_max, n_events)

    # PhT must satisfy PhT/(z*Q) < 0.2 for each event, i.e. PhT < 0.2 * z * Q
    # This is to be sure to stay in the TMD region.
    # PhT_upper is a numpy array (shape: [n_events]), giving the upper bound on PhT for each event.
    # For each event, PhT_upper[i] = min(PhT_max, 0.2 * z[i] * Q[i]).
    # This ensures the sampled PhT never exceeds the physical TMD bound PhT/(z*Q)<0.2, but never above PhT_max.
    PhT_upper = np.minimum(PhT_max, 0.2 * z * Q)

    # Resample Q, z for events where upper bound is below PhT_min.
    # For some (Q, z), the TMD bound 0.2 * z * Q is smaller than PhT_min.
    # So those (Q, z) pairs are kinematically invalid for the chosen PhT range,
    # and we need to resample them until we get a valid combination (Q, z)
    # where the PhT range is not null (i.e. PhT_upper > PhT_min).
    mask = PhT_upper < PhT_min
    while np.any(mask):
        # Get the number of events where the TMD bound is below PhT_min
        n_bad = np.sum(mask)
        # Resample Q, z for those events
        Q[mask] = np.random.uniform(Q_min, Q_max, n_bad)
        z[mask] = np.random.uniform(z_min, z_max, n_bad)
        # Recalculate the upper bound on PhT for those events
        PhT_upper[mask] = np.minimum(PhT_max, 0.2 * z[mask] * Q[mask])
        # Check again if there are any events where the TMD bound is below PhT_min
        mask = PhT_upper < PhT_min

    # Sample PhT for all events
    PhT = np.random.uniform(PhT_min, PhT_upper, n_events)

    # Stack the events into a tensor
    events = torch.tensor(np.column_stack([x, PhT, Q, z]), dtype=torch.float64)

    # Print some information about the generated events
    print(
        f"{tcolors.GREEN}Generated {n_events} events: shape {events.shape}{tcolors.ENDC}"
    )
    print(f"{tcolors.GREEN}Columns: [x, PhT, Q, z]{tcolors.ENDC}")
    print(f"{tcolors.GREEN}Ranges:")
    print(f"{tcolors.GREEN}    x in [{x_min}, {x_max}]{tcolors.ENDC}")
    print(f"{tcolors.GREEN}    Q in [{Q_min}, {Q_max}]{tcolors.ENDC}")
    print(f"{tcolors.GREEN}    z in [{z_min}, {z_max}]{tcolors.ENDC}")
    print(
        f"{tcolors.GREEN}    PhT in [{PhT_min}, {PhT_max}] (respecting TMD boundary PhT/(z*Q) < 0.2){tcolors.ENDC}"
    )

    return events


EPILOG = """
Examples (run from repo root):
  # Default: 1000 events, seed 42, save to sidis/tests/inputs/
  python sidis/tests/generate_mock_events.py

  # 100 events with custom seed
  python sidis/tests/generate_mock_events.py -n 100 -s 123

  # Custom output path (creates parent dirs; saves both .dat and .pt)
  python sidis/tests/generate_mock_events.py -n 500 -o sidis/tests/inputs/my_events.dat

  # From sidis/tests/ directory
  python generate_mock_events.py -n 1000 -o inputs/mock_events.dat
"""


def main():
    script_dir = pathlib.Path(__file__).resolve().parent
    inputs_dir = script_dir / "inputs"

    parser = argparse.ArgumentParser(
        description=f"{tcolors.BOLDWHITE}Generates synthetic unpolarized SIDIS event data [x, PhT, Q, z]. {tcolors.ENDC}"
        f"{tcolors.BOLDWHITE}\nSaves both .dat and .pt (same content). \nCreates inputs/ if missing.{tcolors.ENDC}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument(
        "-n",
        "--n_events",
        type=int,
        default=1000,
        help="Number of events to generate (default: 1000)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output file path without extension, or with .dat/.pt (default: inputs/mock_events_<n>)",
    )
    args = parser.parse_args()

    # Default output: inputs/mock_events_<n> (saved as .dat and .pt)
    if args.output is None:
        args.output = inputs_dir / f"mock_events_{args.n_events}"
        # args.output = inputs_dir / f"mock_events_{args.n_events}_seed_{args.seed}"
    else:
        # Strip .dat or .pt if user provided it
        stem = args.output.stem
        parent = args.output.parent
        args.output = parent / stem

    # Create inputs/ (or output parent dir) if it does not exist
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Generate events
    events = generate_mock_events(n_events=args.n_events, seed=args.seed)

    # Save events
    path_dat = args.output.with_suffix(".dat")
    path_pt = args.output.with_suffix(".pt")
    torch.save(events, path_dat)
    torch.save(events, path_pt)

    print(f"{tcolors.BLUE}Saved to: {path_dat}{tcolors.ENDC}")
    print(f"{tcolors.BLUE}Saved to: {path_pt}{tcolors.ENDC}")


if __name__ == "__main__":
    main()
