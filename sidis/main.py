import torch
import time
import argparse
import pathlib

# This ensures the code only runs when the script is executed directly,
# not when imported as a module. This is a Python best practice that
# prevents unintended code execution during imports.
if __name__ == "__main__":

    from omegaconf import OmegaConf
    from model import TrainableModel
    from utilities.colors import tcolors

    # Set default tensor dtype to float64 for high precision calculations
    # (important for QCD calculations that require numerical stability)
    torch.set_default_dtype(torch.float64)

    # Get the directory containing this script (sidis/)
    rootdir = pathlib.Path(__file__).resolve().parent
    cards_dir = rootdir.joinpath("cards")

    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="SIDIS TMD Cross-Section Computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
        Examples:
        python3 sidis/main.py -c fNPconfig_base_flavor_dep.yaml
        python3 sidis/main.py -c fNPconfig_base_flavor_blind.yaml
        python3 sidis/main.py  # Uses default: fNPconfig_base_flavor_blind.yaml

        Available config files in cards/:
        {chr(10).join(f'  - {f.name}' for f in sorted(cards_dir.glob('*.yaml')) if cards_dir.exists())}
        """,
    )

    parser.add_argument(
        "--fnp_config",
        "-c",
        type=str,
        default="fNPconfig_base_flavor_blind.yaml",
        help="fNP configuration file name (looked up in cards/ directory). "
        "Default: fNPconfig_base_flavor_blind.yaml",
    )

    args = parser.parse_args()

    # Check if config file exists in cards/ directory
    config_path = cards_dir.joinpath(args.fnp_config)

    if not config_path.exists():
        print(
            f"{tcolors.FAIL}Error: Configuration file not found: {config_path}{tcolors.ENDC}"
        )
        print(
            f"\n{tcolors.WARNING}Please ensure the config file exists in the cards/ directory.{tcolors.ENDC}"
        )
        print(f"Available config files in {cards_dir}:")
        if cards_dir.exists():
            for f in sorted(cards_dir.glob("*.yaml")):
                print(f"  - {f.name}")
        else:
            print(f"  {tcolors.FAIL}cards/ directory not found!{tcolors.ENDC}")
        print(f"\nUsage: python3 sidis/main.py -c <config_file>\n")
        exit(1)

    # Show warning if using default
    if args.fnp_config == "fNPconfig_base_flavor_blind.yaml":
        print(
            f"{tcolors.WARNING}Warning: Using default fNP config: {args.fnp_config}{tcolors.ENDC}"
        )
        print(
            f"{tcolors.WARNING}Specify -c <config_file> to use a different configuration.{tcolors.ENDC}\n"
        )
    else:
        print(f"{tcolors.GREEN}Using fNP config: {args.fnp_config}{tcolors.ENDC}\n")

    # Initialize the trainable model for TMD
    # parton distribution functions and fragmentation functions
    model = TrainableModel(fnp_config=args.fnp_config)

    # Load event data from file as a tensor
    events_file = rootdir.joinpath("toy_events.dat")
    events_tensor = torch.load(events_file)

    # Print out some information about the events
    print(f"Loaded events from {events_file}")
    print(f"Event data shape: {events_tensor.shape}")
    print(f"Events tensor:\n{events_tensor}\n")

    # Run the model forward pass with the full tensor
    print(f"{tcolors.GREEN}Results from model forward pass:{tcolors.ENDC}")
    print(model(events_tensor))

    """
    NOTE FROM CHIARA: I'm not sure if this is needed, or why the following is here
    but I'm keeping it here for now.
    Isn't the perturbative evolution part of the model, and therefore already called
    in the model = TrainableModel() instantiation?
    """
    # from model.evolution import PERTURBATIVE_EVOLUTION
    # import qcdlib.params as params

    # t0 = time.time()
    # pert_evo = PERTURBATIVE_EVOLUTION(order=3)
    # bT = torch.linspace(0.01,10,100)
    # Q20 = torch.tensor([params.mc2])
    # Q2 = torch.linspace(params.mc2,100,10000)
    # sudakov = pert_evo.forward(bT, Q20, Q2)
    # t1 = time.time()
    # print(f"Time taken for evolution of shape {sudakov.shape}: {t1-t0} seconds")
