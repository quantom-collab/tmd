# sidis/model/ogata1test.py

import time
import torch

from sidis.model.ogata import OGATA, OGATA1  # J0 and J1 Ogata classes

# ----------------------------------------------------------------------
# Parameters and test functions
# ----------------------------------------------------------------------
a = 0.5  # Gaussian width (a > 0)

def test_func_J0(bT: torch.Tensor) -> torch.Tensor:
    """
    f0(b) = b * exp(-a b^2)
    Benchmark integral:
      I0(q) = ∫_0^∞ b e^{-a b^2} J0(b q) db = (1/(2a)) exp(-q^2/(4a))
    """
    return bT * torch.exp(-a * bT**2)

def test_func_J1(bT: torch.Tensor) -> torch.Tensor:
    """
    f1(b) = b^2 * exp(-a b^2)
    Benchmark integral:
      I1(q) = ∫_0^∞ b^2 e^{-a b^2} J1(b q) db = (q / (4 a^2)) exp(-q^2/(4a))
    """
    return bT**2 * torch.exp(-a * bT**2)

def analytic_J0(qT: torch.Tensor) -> torch.Tensor:
    """
    Analytic result for:
      ∫_0^∞ b e^{-a b^2} J0(b q) db
      = (1/(2a)) * exp(-q^2/(4a))
    """
    return (1.0 / (2.0 * a)) * torch.exp(-qT**2 / (4.0 * a))

def analytic_J1(qT: torch.Tensor) -> torch.Tensor:
    """
    Analytic result for:
      ∫_0^∞ b^2 e^{-a b^2} J1(b q) db
      = (q / (4 a^2)) * exp(-q^2/(4a))
    """
    return (qT / (4.0 * a**2)) * torch.exp(-qT**2 / (4.0 * a))


# ----------------------------------------------------------------------
# Benchmark helper
# ----------------------------------------------------------------------
def benchmark(transformer, test_func, analytic_fn, name="J0"):

    print(f"\n==================== Benchmarking {name} Hankel Transform ====================")

    # qT grid
    Nq = 40
    qT_values = torch.linspace(0.1, 5.0, Nq, dtype=torch.get_default_dtype())

    # bT grid for each qT
    bTs = transformer.get_bTs(qT_values)        # (Nq, Nb)
    integrand = test_func(bTs)                  # (Nq, Nb)

    # Numerical Ogata result
    t0 = time.time()
    numeric = transformer.eval_ogata_func_var_h(integrand, bTs, qT_values)
    t1 = time.time()

    # Analytic reference
    exact = analytic_fn(qT_values)

    abs_err = torch.abs(numeric - exact)
    rel_err = abs_err / torch.clamp(torch.abs(exact), min=1e-15)

    print(f"Runtime: {t1 - t0:.5f} seconds")
    print(f"Max abs error: {abs_err.max().item():.3e}")
    print(f"Max rel error: {rel_err.max().item():.3e}")

    # Some sample points
    indices = [0, Nq//4, Nq//2, 3*Nq//4, Nq-1]
    print("\nSample results:")
    for i in indices:
        print(
            f"qT={qT_values[i]:.2f}  "
            f"numeric={numeric[i]:.6e}  "
            f"exact={exact[i]:.6e}  "
            f"rel.err={rel_err[i]:.3e}"
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Instantiate J0 and J1 transformers
    og0 = OGATA()   # J0
    og1 = OGATA1()  # J1

    benchmark(og0, test_func_J0, analytic_J0, name="J0 (b e^{-a b^2})")
    benchmark(og1, test_func_J1, analytic_J1, name="J1 (b^2 e^{-a b^2})")
