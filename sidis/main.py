import torch
import time

if __name__ == "__main__":
    import pathlib
    from omegaconf import OmegaConf
    from model import TrainableModel

    torch.set_default_dtype(torch.float64)

    rootdir = pathlib.Path(__file__).resolve().parent
    conf = OmegaConf.load(rootdir.joinpath("config.yaml"))
    # conf.outdir = rootdir.joinpath(conf.outdir)
    print(conf)

    model = TrainableModel()
    x = torch.tensor([0.1,0.2,0.3])
    pT = torch.tensor([0.1,0.2,0.3])
    Q = torch.tensor([1,2,3])
    z = torch.tensor([0.1,0.2,0.3])
    print(model(x, pT, Q, z))

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