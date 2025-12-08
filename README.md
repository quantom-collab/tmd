## MT branch: SIDIS TMD cross section from structure functions

This branch implements a simple, parametric calculation of the SIDIS TMD cross section starting from fitted structure functions.

- **Event kinematics**  
  The main entry point is `sidis/main.py`, which builds an `events_tensor` with rows of  
  \([x, P_{hT}, Q, z, \phi_h, \phi_s]\). This tensor is passed to `sidis.model.EICModel`, which interprets:
  - \(x\): Bjorken \(x\)  
  - \(P_{hT}\): transverse momentum of the detected hadron  
  - \(Q\): hard scale (so \(Q^2\) is the virtuality)  
  - \(z\): hadron energy fraction  
  - \(\phi_h\): azimuthal angle of the hadron  
  - \(\phi_s\): azimuthal angle of the target spin

- **Structure functions**  
  Inside `sidis/model/EIC.py`, the model constructs three structure functions using fixed parametric fits:
  - `get_FUU`: unpolarized structure function \(F_{UU}\)  
  - `get_Siv`: Sivers structure function \(F_{UT}^{\sin(\phi_h-\phi_s)}\)  
  - `get_Col`: Collins structure function \(F_{UT}^{\sin(\phi_h+\phi_s)}\)  
  Each of these is a function of \((x, Q^2, z, P_{hT})\), with a log–log type evolution in \(Q^2\) and Gaussian \(P_{hT}\) dependence.

- **Prefactor and cross section**  
  The forward pass first computes:
  - \(q_T = P_{hT} / z\)  
  - \(Q^2 = Q^2\)  
  - the electroweak coupling \(\alpha_{\text{em}}(Q^2)\) via `qcdlib.eweak.EWEAK`  
  - the usual SIDIS kinematic factors \(\gamma, y, \varepsilon\) following Bacchetta et al., JHEP 02 (2007) 093  
  - a common prefactor \(\sigma_0(x, Q^2, z, q_T)\) for the SIDIS cross section

  The differential cross section for each event is then built as
  \[
    \sigma = \sigma_0 \,\Big( F_{UU}
    + \sin(\phi_h - \phi_s)\, F_{UT}^{\sin(\phi_h-\phi_s)}
    + \varepsilon\, \sin(\phi_h + \phi_s)\, F_{UT}^{\sin(\phi_h+\phi_s)} \Big),
  \]
  where:
  - the first term gives the unpolarized contribution,
  - the second term encodes the Sivers modulation,
  - the third term encodes the Collins modulation with the longitudinal–transverse interference factor \(\varepsilon\).

For unpolarized calculations, \(\phi_h\) and \(\phi_s\) can be set to zero, so only the \(F_{UU}\) term contributes; for transversely polarized targets, nonzero \(\phi_h\) and \(\phi_s\) activate the Sivers and Collins terms.


