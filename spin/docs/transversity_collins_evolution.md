# Transversity and Collins \(\hat H^{(3)}\) evolution

## Evolved objects

| Object | Variable | Kernel |
|--------|----------|--------|
| Transversity PDF | \(h_1^q(x;\mu)\) | \(P_{h1}\) |
| Homogeneous Collins FF | \(\hat H^{(3)}_{h/q}(z;\mu)\) | \(P_{h1}\) |

Trento Collins moment (not evolved directly):

\[
\hat H^{(3)} = -2 z M_h\, H_1^{\perp(1)}|_{\mathrm{Trento}}.
\]

## Initial conditions

At \(Q_0^2 = 2.4\,\mathrm{GeV}^2\):

**Transversity**

\[
h_1^q(x,Q_0) = \tfrac12\,N_q^h\,x^{a_q}(1-x)^{b_q}\,[f_1^q(x)+g_1^q(x)],
\]

with `TransversityParams`; \(f_1\) from NNPDF40, \(g_1\) from NNPDFpol20.

**Collins \(\hat H^{(3)}\)**

Paper ansatz using pi\(^+\) FFs \(D_{q}(z,Q_0)\) from `JAM20-SIDIS_FF_pion_nlo`.

## LHAPDF convention

All loaders divide `xfxQ` by \(x\) or \(z\) to return \(f\) or \(D\).

## Validation

```bash
python3 -m spin.validation.validate_transversity_collins_physics --toy --no-cache
mamba run -n base python -m spin.validation.compare_to_transversity_collins_paper
```

## Caveats

- Homogeneous LO evolution; not full twist-3 Collins NLO.
- Tensor charge \(\int dx\,h_1^q\) decreases with \(Q\); plot \(x h_1\) for paper-style figures.
- Inputs are NNPDF/JAM20 proxies, not CT10/DSSV/DSS.
