# Qiu–Sterman evolution

## Evolved object

\(T_F^q(x,x;\mu)\) — collinear Qiu–Sterman function. Observable:

\[
x f_{1T}^{\perp(1)}(x,\mu) = -\frac{x}{2M}\,T_F^q(x,x;\mu).
\]

## Initial condition

At \(\mu_0 = \sqrt{1.9}\,\mathrm{GeV}\):

\[
T_F^q(x,\mu_0) = N_q(x)\,f_q(x,\mu_0),
\]

with paper \(N_q(x)\) from `QiuStermanParams` and unpolarized \(f_q\) from LHAPDF (`NNPDF40_nnlo_pch_as_01180`).

## Kernel

LO non-singlet DGLAP with \(P_{qq}^T = P_{qq} - N_C\,\delta(1-x)\) (`kernel_type="qiu_sterman"`, `eta=N_C`).

## Validation

```bash
python3 -m spin.validation.validate_qiu_sterman_evolution --with-evolution
```

## Caveats

- Evolves \(T_F\), not \(N_q\) alone.
- `g1T` in `QiuStermanParams` is for b-space/TMD use only.
- PDF inputs use `xfxQ/x`; not a CT10 reproduction.
