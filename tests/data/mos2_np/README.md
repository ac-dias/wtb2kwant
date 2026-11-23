# MoS2 monolayer test data (non-polarized, Wannier90 + VASP)

Used by `tests/test_mos2_np.py` and `tests/validate_mos2_np.py`.

| file | content |
|---|---|
| `tb_hr.dat` | WanTiBEXOS Hamiltonian (from `param_gen_vasp`): header + `wannier90_hr.dat` |
| `wannier90.wout` | Wannier90 output (lattice, atoms, final WF centres) |
| `wannier90.win` | Wannier90 input (Mo: d, S: p projections; 11 WFs) |
| `wannier90_centres.xyz` | WF centres (`write_xyz = true`), same as the `.wout` final state |
| `wannier90_band.{dat,kpt,labelinfo.dat}` | Wannier90 band structure along Γ–K–K'–Γ (reference) |

Lattice constant 3.1655 Å, 16 Å vacuum, E_F = −0.9676 eV, scissors 0.5 eV (not used).
