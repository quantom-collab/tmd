# spin

LO x-space non-singlet DGLAP evolution for spin-dependent collinear quantities (Qiu–Sterman, transversity, homogeneous Collins \(\hat H^{(3)}\)).

Run tests and validation from the `tmd/` repository root (so `import spin` resolves).

- Qiu–Sterman \(T_F^q(x,x;\mu)\) with \(P_{qq}^T = P_{qq} - N_C\,\delta(1-x)\)
- transversity \(h_1^q(x;\mu)\) with \(P_{h1}\)
- homogeneous Collins \(\hat H^{(3)}_{h/q}(z;\mu)\) with the same \(P_{h1}\)

## Quick start

```python
from spin.dglap import NonSingletDGLAP
from spin.evolution import TransversityEvolution, evolve_transversity
from spin.transversity import TransversityParams, build_h1_initial
from spin.validation.inputs import load_transversity_inputs

dglap = NonSingletDGLAP(kernel_type="transversity", Q20=2.4, loadgrid=False)
x = dglap.x
f1, g1 = load_transversity_inputs(x, 2.4)
h1_0 = build_h1_initial(x, f1, g1, TransversityParams())
h1 = evolve_transversity(TransversityEvolution.from_dglap(dglap), h1_0)
```

## Validation

```bash
cd tmd
PYTHONPATH=. python3 -m pytest spin/tests/ -v

PYTHONPATH=. python3 -m spin.validation.validate_qiu_sterman_evolution --with-evolution
PYTHONPATH=. python3 -m spin.validation.validate_transversity_collins_physics --toy --no-cache --no-plots

mamba run -n base env PYTHONPATH=. python -m spin.validation.validate_transversity_collins_physics --no-cache
mamba run -n base env PYTHONPATH=. python -m spin.validation.compare_to_transversity_collins_paper
```

## Design notes

- One `NonSingletDGLAP` builder; kernel selected by `kernel_type`.
- Shift operator: `get_shift` returns `Axi.T + Ac` (jamx convolution convention).
- Evolution applies `M @ f` via `einsum("jik,...i->...jk", M, f0)`.
- Cached grids use prefix `spin_nonsinglet_kernel_v3_...`.

See `spin/docs/` for physics conventions and caveats.
