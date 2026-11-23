# wtb2kwant

**wtb2kwant** converts tight-binding Hamiltonians in the
[WanTiBEXOS](https://github.com/ac-dias/wantibexos) format into a
self-contained **[Kwant](https://kwant-project.org) Python script**. You can then use
that script to compute band structures and quantum transport in nanoribbons and other
nanostructures cut from the original crystal.

It accepts both kinds of WanTiBEXOS Hamiltonians:

- **orthogonal** Wannier90-type Hamiltonians (`tb_hr.dat`);
- **non-orthogonal** Hamiltonians with an overlap matrix S (`.ham`), written by the
  SIESTA, DFTB+, FHI-aims, CRYSTAL and YAeHMOP interfaces. Transport with these
  Hamiltonians is exact: the overlap is kept, not approximated.

```
tb_hr.dat + wannier90.wout        ─┐
                                   ├─►  wtb2kwant  ──►  kwant_model.py  ──►  ribbons, leads, G(E), ...
tb-*.ham  + basis or structure    ─┘
```

> **New in v0.2:** support for non-orthogonal Hamiltonians (`.ham` + overlap S), with
> exact transport and generalized band structures. See
> [Non-orthogonal Hamiltonians](#non-orthogonal-hamiltonians) below and Section 4.3 of the
> [manual](manual/wtb2kwant_manual.pdf).

The generated script embeds the converted parameters (lattice, orbital positions,
on-site blocks and hopping blocks). It needs only `numpy`, `kwant` and `matplotlib`,
so you can copy it, share it and modify it freely.

## Features

- Reads the WanTiBEXOS **Wannier90-type** Hamiltonian (`tb_hr.dat`, produced by
  `param_gen_vasp`). The spin types are NP, SP and SOC.
- Reads the WanTiBEXOS **non-orthogonal** Hamiltonians (`.ham` with overlap matrix S) from
  the SIESTA, DFTB+, FHI-aims, CRYSTAL and YAeHMOP interfaces.
  - Orbital positions come from the `.basis` file (SIESTA, YAeHMOP) or from the crystal
    structure (POSCAR, `geometry.in`, DFTB+ `.gen`, CRYSTAL output, or ASE formats) plus
    `--norb`.
  - Transport is **exact** through the energy-dependent substitution H′ = H − E(S − 1).
  - Bands come from the generalized problem H(k)c = E S(k)c.
  - Löwdin orthogonalization is available as an experimental option.
- Reads the **Wannier centres, atoms and lattice** directly from `wannier90.wout`, so you
  don't have to rerun Wannier90. For spin-polarized runs it takes two `.wout` files
  (up and down).
- Assigns each Wannier function to its nearest atom. Kwant sites are atoms with
  `norbs` orbitals each. Alternatively, `--sites centres` makes one site per Wannier
  function.
- **Folds Wannier centres back into the home cell and corrects the R vectors
  accordingly**, so cuts in real space (ribbon edges) are physically correct.
- Divides by the Wannier90 degeneracy weights, symmetrizes, and applies an energy cutoff
  for small matrix elements.
- Handles **long-range Wannier hoppings**:
  - ribbon bands are computed from a Bloch Hamiltonian built with the primitive period,
    so they are not folded;
  - leads automatically use the smallest supercell that Kwant can finalize.
- The generated script provides `make_ribbon`, `ribbon_bands`, `make_two_terminal`,
  `conductance` (parallel), `channel_count`, `bulk_hk` and more, plus a ready-to-run
  example: zigzag and armchair ribbon bands with the Kwant transmission.

## Installation

The easiest way is the installer. It uses a conda environment named `kwant` and creates
it from conda-forge if it does not exist:

```bash
git clone https://github.com/ac-dias/wtb2kwant.git
cd wtb2kwant
./install.sh            # options: --env NAME, --no-editable, --dev, --python X.Y
conda activate kwant
```

The installer also writes launchers into `bin/` that call the environment's Python
directly. If you add that folder to your `PATH`, you can use the tool without activating
conda:

```bash
export PATH="$PWD/bin:$PATH"                      # e.g. in ~/.bashrc
wtb2kwant tb_hr.dat --wout wannier90.wout -o kwant_model.py
wtb2kwant-python kwant_model.py --help            # runs scripts with the env's Python
```

The launchers are machine-specific, so they are not versioned: `bin/` contains only its
own `.gitignore`.

Manual installation, in any environment that has Kwant:

```bash
conda create -n kwant -c conda-forge python=3.12 kwant numpy matplotlib
conda activate kwant
pip install -e .
```

## Quick start

```bash
# 1. Convert (in the folder of your Wannier90/WanTiBEXOS calculation)
wtb2kwant tb_hr.dat --wout wannier90.wout -o kwant_model.py

# 2. Built-in example: bands + Kwant transmission of zigzag and armchair
#    ribbons (20 Å wide) -> bands_ribbons.png + bands_ribbons_transmission.dat
python kwant_model.py
python kwant_model.py --no-transport                 # bands only (seconds)
python kwant_model.py --iv                           # Landauer I-V, 0-3 V (cheap Hamiltonian)
python kwant_model.py --iv --vrange -1 2 --nv 31     # I-V in a chosen bias range
python kwant_model.py --width-zz 30 --width-ac 25 --emin -2 --emax 2 --ne 81
python kwant_model.py --help
```

The transmission is parallelized over energies. By default it uses about one process
per physical core; set it with `--nproc`. For the MoS₂ example (61 energies,
11 orbitals per cell), each ribbon takes about 4 minutes on a 12-core workstation.
For quick scans, `--hop-cutoff 1e-3` drops small hoppings: it is about 4× faster but
less accurate.

![MoS2 ribbons](manual/figures/bands_ribbons.png)

### Current–voltage curves

`python kwant_model.py --iv` computes the transmission on a fine energy grid. It uses a
cheaper Hamiltonian, with hoppings below 10⁻³ eV dropped by default; change this with
`--hop-cutoff`. It then computes the Landauer current

I(V) = g_s (e/h) ∫ T(E) [f(E − E_F − V/2) − f(E − E_F + V/2)] dE

in the rigid-band approximation (T(E) computed at zero bias, symmetric bias, 300 K by
default). The default bias range is 0 to 3 V; choose another one with `--vrange VMIN VMAX`
(negative biases allowed). The outputs are `iv_ribbons.png`, `iv_ribbons_T.dat` and
`iv_ribbons_IV.dat`.

![MoS2 I-V](manual/figures/iv_ribbons.png)

Transport, using the generated script as a module:

```python
from kwant_model import make_two_terminal, conductance

syst = make_two_terminal("zigzag", width=20.0, length=30.0).finalized()
T = conductance(syst, [-0.5, 0.0, 0.5])   # energies relative to E_F (eV)
```

Spin-polarized calculations:

```bash
wtb2kwant tb_hr.dat --wout wannier90.up.wout wannier90.dn.wout -o kwant_model.py
```

### Non-orthogonal Hamiltonians

The format is detected automatically. Kwant needs to know on which atom each orbital
sits, so give the orbital positions with `--basis` (SIESTA, YAeHMOP) or with
`--structure` plus the number of orbitals per spin of each species (`--norb`):

```bash
wtb2kwant tb-NP.ham --basis basis_set-NP.basis                 # SIESTA, YAeHMOP
wtb2kwant fhi_tb.ham --structure geometry.in --efermi -5.0096  # FHI-aims, one species
wtb2kwant hbn_sp_tb.ham --structure hbn_sp.out --norb B=14,N=14  # CRYSTAL
wtb2kwant dftb.ham --structure geo.gen --norb Mo=9,S=4          # DFTB+
```

The generated script and its options are the same as in the orthogonal case. In the
script, the on-site terms and hoppings are functions of the energy, and `conductance()`
passes it automatically. Some interfaces write E_F = 0 in the header, so give the Fermi
level with `--efermi`.

### Command-line options

| option | default | description |
|---|---|---|
| `input` | – | WanTiBEXOS Hamiltonian file (`tb_hr.dat` or `.ham`) |
| `--wout F [F]` | required | `wannier90.wout` (two files, up then down, for SP) |
| `--sites {atoms,centres}` | `atoms` | group Wannier functions by nearest atom, or one site per WF |
| `--basis F` | – | `basis_set-*.basis` (non-orthogonal: SIESTA, YAeHMOP) |
| `--structure F` | – | crystal structure (non-orthogonal: DFTB+, FHI-aims, CRYSTAL, ...) |
| `--norb Sp=n,...` | – | orbitals per spin of each species (with `--structure`) |
| `--s-cutoff X` | `1e-6` | drop overlap elements with \|S\| < X |
| `--orthogonalize {none,lowdin}` | `none` | keep S (exact), or Löwdin (experimental, see below) |
| `--cutoff X` | `1e-5` | drop matrix elements with \|H\| < X (eV) |
| `--efermi X` | from file | override the Fermi energy (eV) |
| `-o FILE` | `kwant_model.py` | generated script |

## Validation

`tests/validate_mos2_np.py` runs a full validation on monolayer MoS₂ (Wannier90/VASP, 11
orbitals). The input data ship with the repository in `tests/data/mos2_np/`. The script
writes `tests/results/mos2_np/REPORT.md` and the figures. Summary:

- **Bulk bands vs `wannier90_band.dat`:** 0.02 meV error without cutoff, 0.13 meV with the
  default cutoff.
- **Hopping orientation inside Kwant:** 660 of 660 blocks are correct; the `wraparound`
  bands agree to 10⁻¹⁴ eV.
- **Ribbon bands:** the primitive-period Bloch Hamiltonian matches `kwant.physics.Bands`
  on the supercell to 10⁻¹⁴ eV.
- **Transmission of perfect ribbons:** an integer equal to the number of open channels.
- **Landauer current:** reproduces the analytic case T = 1, where I/V = 2e²/h.

`tests/validate_mos2_yaehmop_np.py` validates the **non-orthogonal** path with an
extended-Hückel MoS₂ model from YAeHMOP (17 orbitals, overlaps up to 0.44). It runs in
about 3 min and writes `tests/results/mos2_yaehmop_np/REPORT.md`:

- **Generalized bulk bands vs YAeHMOP:** 0.008 meV error in the window VBM−6…+4 eV; the gap
  is reproduced to 6 decimals.
- **Transmission with H − E(S − 1):** equal to the number of channels of the generalized
  ribbon bands at all energies, for both ribbons (difference exactly 0).
- **Löwdin (negative result, kept as a warning):**
  - exact in the bulk (it converges with the cutoff);
  - wrong by ~2 eV in nanoribbons. The Löwdin on-site energies absorb the bulk
    environment (Mo 5s: 0.2 → 24 eV), and at an edge the missing neighbours no longer
    compensate this.
  - For strongly non-orthogonal bases, keep S (the default).

```bash
python -m pytest -q                        # quick regression tests (~20 s)
python tests/validate_mos2_np.py           # Wannier90 report (~13 min on 12 cores; --quick ~5 min)
python tests/validate_mos2_yaehmop_np.py   # non-orthogonal report (~3 min)
```

## Documentation

The full manual (in English) is in [`manual/`](manual/): `manual/wtb2kwant_manual.pdf`.
It covers the input formats, the conventions, how the folding and supercells work, the
API of the generated script, and the validation.

## Current limitations and roadmap

- Supported: Wannier90-type and non-orthogonal WanTiBEXOS input; 2D crystals (ribbons).
- SP and SOC are implemented, but have not yet been validated against reference data.
- The scissors operator is recorded in the generated script but not applied.
- Non-orthogonal bases: Kwant operators (local DOS, currents) are not yet adapted to S.
- Planned: nanotubes; 3D bulk crystals and slabs.

### Pending tests

- **SP and SOC:** validate against reference band structures.
  - Wannier90 SP with two `.wout` files: no data set yet.
  - SOC: the YAeHMOP MoS₂ SOC data already exist.
  - SP: the CRYSTAL h-BN and FHI-aims graphene data already exist.
- **SIESTA and DFTB+ inputs:** validate with 2D test cases. The available SIESTA example
  (Sn₃O₄) is 3D, and the DFTB+ example has no geometry file.
- **FHI-aims and CRYSTAL:** reading and positions were checked by hand (graphene Dirac
  point, h-BN spin degeneracy). Add reduced versions of these data sets to the automated
  tests, together with reference bands.
- **Wannier90 validation report:** rerun `tests/validate_mos2_np.py` with v0.2.
  `tests/results/mos2_np/` was generated with v0.1; the code paths are unchanged, and the
  pytest suite already covers v0.2.

## License

wtb2kwant is released under the [MIT License](LICENSE).
