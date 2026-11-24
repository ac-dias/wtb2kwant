# MoS2 monolayer with spin-orbit coupling (YAeHMOP, extended Hückel)

- `huckel-SOC.dat`: non-orthogonal TB Hamiltonian (`.ham` format, SOC, 34 spin-orbitals) written by
  `yaehmop2wtb.py` from a `yaehmop --soc` run on an 11x11x1 mesh. **Reduced copy**: only the entries
  with |H| > 1e-8 eV or |S| > 1e-9 are kept (the reader fills the rest with zeros). Bands differ from
  the full file by < 1e-7 eV within VBM-6 .. VBM+4 eV (5e-6 eV over all bands).
- `basis_set-SOC`: orbital positions, l, m and spin (+1 up, -1 down).
- `MoS2.bind`, `MoS2_soc.dat`: YAeHMOP input and SOC parameters.
- `MoS2.bind.band`: YAeHMOP band structure (reference), 121 k points along Γ-M-K-Γ.
