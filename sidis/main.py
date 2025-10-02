import torch
import time

# This ensures the code only runs when the script is executed directly,
# not when imported as a module. This is a Python best practice that
# prevents unintended code execution during imports.
if __name__ == "__main__":
    import pathlib
    from omegaconf import OmegaConf
    from model import TrainableModel

    # Set default tensor dtype to float64 for high precision calculations
    # (important for QCD calculations that require numerical stability)
    torch.set_default_dtype(torch.float64)

    # Get the directory containing this script
    rootdir = pathlib.Path(__file__).resolve().parent

    # Load configuration from YAML file
    conf = OmegaConf.load(rootdir.joinpath("config.yaml"))

    # Uncomment to set output directory relative to script location:
    # conf.outdir = rootdir.joinpath(conf.outdir)

    # Print the full configuration:
    # print(conf)

    # Initialize the trainable model for TMD
    # parton distribution functions and fragmentation functions
    model = TrainableModel()

    # Load event data from file as a tensor
    events_file = rootdir.joinpath("toy_events.dat")

    events_tensor = torch.load(events_file)

    print(f"Loaded events from {events_file}")
    print(f"Event data shape: {events_tensor.shape}")
    print(f"Events tensor:\n{events_tensor}")

    # Run the model forward pass with the full tensor
    print(model(events_tensor))

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
