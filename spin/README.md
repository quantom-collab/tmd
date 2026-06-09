# Spin2

Spin2 is the cleaned production version of the Spin evolution package. The original `Spin/` directory is retained only as development/debug history.

LO x-space non-singlet DGLAP evolution for spin-dependent collinear quantities:

- Qiu–Sterman \(T_F^q(x,x;\mu)\) with \(P_{qq}^T = P_{qq} - N_C\,\delta(1-x)\)
- transversity \(h_1^q(x;\mu)\) with \(P_{h1}\)
- homogeneous Collins \(\hat H^{(3)}_{h/q}(z;\mu)\) with the same \(P_{h1}\)

## Quick start

```python
from Spin2.dglap import NonSingletDGLAP
from Spin2.evolution import TransversityEvolution, evolve_transversity
from Spin2.transversity import TransversityParams, build_h1_initial
from Spin2.validation.inputs import load_transversity_inputs

dglap = NonSingletDGLAP(kernel_type="transversity", Q20=2.4, loadgrid=False)
x = dglap.x
f1, g1 = load_transversity_inputs(x, 2.4)
h1_0 = build_h1_initial(x, f1, g1, TransversityParams())
h1 = evolve_transversity(TransversityEvolution.from_dglap(dglap), h1_0)
```

## Validation

```bash
python3 -m pytest Spin2/tests/ -v

python3 -m Spin2.validation.validate_qiu_sterman_evolution --with-evolution
python3 -m Spin2.validation.validate_transversity_collins_physics --toy --no-cache --no-plots

mamba run -n base python -m Spin2.validation.validate_transversity_collins_physics --no-cache
mamba run -n base python -m Spin2.validation.compare_to_transversity_collins_paper
```

## Design notes

- One `NonSingletDGLAP` builder; kernel selected by `kernel_type`.
- Shift operator: `get_shift` returns `Axi.T + Ac` (jamx convolution convention).
- Evolution applies `M @ f` via `einsum("jik,...i->...jk", M, f0)`.
- Cached grids use prefix `spin2_nonsinglet_kernel_v3_...`.

See `Spin2/docs/` for physics conventions and caveats.
