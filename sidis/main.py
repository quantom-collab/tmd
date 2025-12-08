import torch
import time
import argparse
import pathlib
import sys

# This ensures the code only runs when the script is executed directly,
# not when imported as a module. This is a Python best practice that
# prevents unintended code execution during imports.
if __name__ == "__main__":
    # Add parent directory to path so sidis can be imported as a package
    # This allows running as: python3 sidis/main.py from tmd/ directory
    parent_dir = pathlib.Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    from omegaconf import OmegaConf
    from sidis.model import EICModel

    # Set default tensor dtype to float64 for high precision calculations
    # (important for QCD calculations that require numerical stability)
    torch.set_default_dtype(torch.float64)

    # Get the directory containing this script (sidis/)
    rootdir = pathlib.Path(__file__).resolve().parent

    # Initialize the EIC model for SIDIS TMD cross-section computation
    model = EICModel()

    # Test with some toy events
    events_tensor = torch.tensor([[0.1, 1.0, 5.0, 0.2, 0.4, 0.0],
                                  [0.2, 1.5, 7.5, 0.3, 0.8, 0.0],
                                  [0.3, 2.0, 10.0, 0.4, 1.2, 0.0],
                                  [0.4, 2.5, 12.5, 0.5, 1.6, 0.0],
                                  [0.5, 3.0, 15.0, 0.6, 2.0, 0.0],
                                  [0.6, 3.5, 17.5, 0.7, 2.4, 0.0]])

    print(model(events_tensor, expt_setup=["p","pi_plus"], rs=140.0))
