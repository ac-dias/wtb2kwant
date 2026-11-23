"""Leitores dos formatos de entrada.

- ``read_wtb_wannier``: arquivo ``tb_hr.dat`` do WanTiBEXOS (gerado por
  ``param_gen_vasp``), ou seja, cabeçalho + ``wannier90_hr.dat`` (dois blocos no
  caso spin-polarizado).
- ``read_wout``: centros finais das funções de Wannier, átomos e rede a partir
  do ``wannier90.wout``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HrBlock:
    """Um bloco ``wannier90_hr.dat``: H_mn(R) = <m,0|H|n,R> (eV)."""
    num_wann: int
    rvecs: np.ndarray        # (nrpts, 3) inteiros
    ndegen: np.ndarray       # (nrpts,)
    ham: np.ndarray          # (nrpts, num_wann, num_wann) complexo, JÁ dividido por ndegen


@dataclass
class WtbWannier:
    systype: str             # NP | SP | SOC
    scissors: float
    efermi: float
    lattice: np.ndarray      # (3, 3), linhas = vetores (Å)
    blocks: list = field(default_factory=list)  # 1 bloco (NP/SOC) ou 2 (SP: up, dn)


def _parse_hr(lines: list[str], pos: int) -> tuple[HrBlock, int]:
    pos += 1  # linha de comentário ("written on ...")
    num_wann = int(lines[pos].split()[0]); pos += 1
    nrpts = int(lines[pos].split()[0]); pos += 1
    ndegen: list[int] = []
    while len(ndegen) < nrpts:
        ndegen += [int(x) for x in lines[pos].split()]
        pos += 1
    ndegen_a = np.array(ndegen[:nrpts], dtype=float)

    nlines = nrpts * num_wann * num_wann
    data = np.array([lines[pos + k].split()[:7] for k in range(nlines)], dtype=float)
    pos += nlines
    data = data.reshape(nrpts, num_wann * num_wann, 7)

    rvecs = data[:, 0, 0:3].astype(int)
    m = data[:, :, 3].astype(int) - 1
    n = data[:, :, 4].astype(int) - 1
    ham = np.zeros((nrpts, num_wann, num_wann), dtype=complex)
    for ir in range(nrpts):
        ham[ir, m[ir], n[ir]] = data[ir, :, 5] + 1j * data[ir, :, 6]
        ham[ir] /= ndegen_a[ir]
    return HrBlock(num_wann, rvecs, ndegen_a, ham), pos


def read_wtb_wannier(path: str) -> WtbWannier:
    """Lê o ``tb_hr.dat`` (formato Wannier90) do WanTiBEXOS."""
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines()]
    systype = lines[0].split()[0].upper()
    if systype not in ("NP", "SP", "SOC"):
        raise ValueError(f"{path}: tipo de sistema desconhecido '{systype}'")
    scissors = float(lines[1].split()[0])
    efermi = float(lines[2].split()[0])
    lattice = np.array([[float(x) for x in lines[3 + i].split()[:3]] for i in range(3)])

    out = WtbWannier(systype, scissors, efermi, lattice)
    pos = 6
    nblocks = 2 if systype == "SP" else 1
    for _ in range(nblocks):
        blk, pos = _parse_hr(lines, pos)
        out.blocks.append(blk)
    return out


@dataclass
class WoutData:
    lattice: np.ndarray      # (3, 3) Å
    species: list            # símbolos dos átomos
    atoms: np.ndarray        # (natoms, 3) cartesiano Å
    centres: np.ndarray      # (num_wann, 3) cartesiano Å (estado final)
    spreads: np.ndarray      # (num_wann,) Å^2


_FLOAT = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?"
_RE_ATOM = re.compile(
    rf"^\s*\|\s*([A-Za-z][A-Za-z0-9_]*)\s+\d+\s+{_FLOAT}\s+{_FLOAT}\s+{_FLOAT}\s+\|"
    rf"\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+\|")
_RE_CENTRE = re.compile(
    rf"WF centre and spread\s+(\d+)\s+\(\s*({_FLOAT}),\s*({_FLOAT}),\s*({_FLOAT})\s*\)\s+({_FLOAT})")
_RE_LAT = re.compile(rf"^\s*a_([123])\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$")


def read_wout(path: str) -> WoutData:
    """Lê rede, átomos e os centros do bloco ``Final State`` do ``.wout``."""
    with open(path) as f:
        lines = f.read().splitlines()

    lattice = np.zeros((3, 3))
    got_lat = set()
    species, atoms = [], []
    final_idx = None
    in_sites = False
    for k, ln in enumerate(lines):
        if len(got_lat) < 3:
            m = _RE_LAT.match(ln)
            if m:
                i = int(m.group(1)) - 1
                lattice[i] = [float(m.group(j)) for j in (2, 3, 4)]
                got_lat.add(i)
                continue
        if "Site       Fractional Coordinate" in ln and not atoms:
            in_sites = True
            continue
        if in_sites:
            m = _RE_ATOM.match(ln)
            if m:
                species.append(m.group(1))
                atoms.append([float(m.group(j)) for j in (2, 3, 4)])
            elif ln.strip().startswith("*"):
                in_sites = False
        if "Final State" in ln:
            final_idx = k  # fica com o último (caso haja restart)

    if final_idx is None:
        raise ValueError(f"{path}: bloco 'Final State' não encontrado "
                         "(a wannierização terminou?)")
    centres, spreads = {}, {}
    for ln in lines[final_idx + 1:]:
        m = _RE_CENTRE.search(ln)
        if m:
            w = int(m.group(1)) - 1
            centres[w] = [float(m.group(j)) for j in (2, 3, 4)]
            spreads[w] = float(m.group(5))
        elif "Sum of centres" in ln:
            break
    nw = len(centres)
    if len(got_lat) < 3 or not atoms or nw == 0:
        raise ValueError(f"{path}: não consegui ler rede/átomos/centros")
    return WoutData(lattice, species, np.array(atoms),
                    np.array([centres[i] for i in range(nw)]),
                    np.array([spreads[i] for i in range(nw)]))


# =====================================================================
# Formato não ortogonal (base localizada): SIESTA, DFTB+, FHI-aims, CRYSTAL, YAeHMOP
# =====================================================================
@dataclass
class WtbHam:
    """Arquivo .ham do WanTiBEXOS (base não ortogonal).

    Cabeçalho: tipo, tesoura, E_F, 3 vetores de rede, nbasis, ncell, comentário.
    Linhas: Rx Ry Rz (cartesiano, Å)  i  j  ReH  ImH  S, com H_ij(R) = <i,0|H|j,R>.
    """
    systype: str
    scissors: float
    efermi: float
    lattice: np.ndarray            # (3, 3) Å
    nbasis: int                    # inclui o fator 2 de SP/SOC
    rvecs: np.ndarray              # (nR, 3) inteiros
    ham: np.ndarray                # (nR, nbasis, nbasis) complexo (eV)
    ovl: np.ndarray                # (nR, nbasis, nbasis) real/complexo
    max_r_residual: float = 0.0    # maior desvio de R cartesiano -> inteiro (em unidades de rede)


def read_wtb_ham(path: str) -> WtbHam:
    """Lê o .ham não ortogonal (mesmo formato para todas as interfaces do WanTiBEXOS)."""
    with open(path) as f:
        head = [f.readline() for _ in range(9)]
        systype = head[0].split()[0].upper()
        if systype not in ("NP", "SP", "SOC"):
            raise ValueError(f"{path}: tipo de sistema desconhecido '{systype}'")
        scissors = float(head[1].split()[0])
        efermi = float(head[2].split()[0])
        lattice = np.array([[float(x) for x in head[3 + i].split()[:3]] for i in range(3)])
        nbasis = int(head[6].split()[0])
        if not head[8].lstrip().startswith("#"):
            raise ValueError(f"{path}: linha 9 deveria ser o comentário '#rcell ...'")
        data = np.loadtxt(f, comments="#", ndmin=2)
    if data.shape[1] < 8:
        raise ValueError(f"{path}: esperado 8 colunas (Rx Ry Rz i j ReH ImH S)")

    rfrac = data[:, 0:3] @ np.linalg.inv(lattice)
    rint = np.rint(rfrac).astype(int)
    resid = float(np.abs(rfrac - rint).max()) if len(rint) else 0.0
    if resid > 1e-3:
        raise ValueError(f"{path}: vetores R não são combinações inteiras da rede "
                         f"(desvio {resid:.2e}); confira os vetores de rede do cabeçalho")
    uniq, inv = np.unique(rint, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    i = data[:, 3].astype(int) - 1
    j = data[:, 4].astype(int) - 1
    if i.max() >= nbasis or j.max() >= nbasis:
        raise ValueError(f"{path}: índice de orbital maior que nbasis = {nbasis}")
    ham = np.zeros((len(uniq), nbasis, nbasis), dtype=complex)
    ovl = np.zeros((len(uniq), nbasis, nbasis), dtype=complex)
    ham[inv, i, j] = data[:, 5] + 1j * data[:, 6]
    ovl[inv, i, j] = data[:, 7]
    return WtbHam(systype, scissors, efermi, lattice, nbasis, uniq, ham, ovl, resid)


@dataclass
class BasisOrbital:
    index: int          # 1-based, como no arquivo
    species: str
    position: np.ndarray
    l: int
    m: int
    spin: int


def read_basis(path: str) -> list:
    """Lê o basis_set-*.basis (SIESTA, YAeHMOP): bindex espécie x y z l m spin."""
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.split("#")[0].strip()
            if not ln:
                continue
            t = ln.split()
            out.append(BasisOrbital(int(t[0]), t[1], np.array([float(x) for x in t[2:5]]),
                                    int(t[5]), int(t[6]), int(t[7])))
    out.sort(key=lambda o: o.index)
    return out


@dataclass
class Structure:
    lattice: np.ndarray     # (3, 3) Å (pode ser None se o formato não trouxer)
    species: list
    positions: np.ndarray   # (natoms, 3) cartesiano Å
    source: str = ""


def _read_poscar(path):
    L = [ln.rstrip("\n") for ln in open(path)]
    scale = float(L[1].split()[0])
    lat = np.array([[float(x) for x in L[2 + i].split()[:3]] for i in range(3)])
    lat = lat * scale if scale > 0 else lat * (abs(scale) / abs(np.linalg.det(lat))) ** (1 / 3)
    k = 5
    tok = L[5].split()
    if all(t.lstrip("-").isdigit() for t in tok):
        raise ValueError(f"{path}: POSCAR sem a linha de espécies (formato VASP 4) não suportado")
    names, counts = tok, [int(x) for x in L[6].split()]
    k = 7
    if L[k].strip()[0] in "sS":     # Selective dynamics
        k += 1
    direct = L[k].strip()[0] in "dD"
    k += 1
    nat = sum(counts)
    xyz = np.array([[float(x) for x in L[k + a].split()[:3]] for a in range(nat)])
    xyz = xyz @ lat if direct else xyz * (scale if scale > 0 else 1.0)
    species = [n for n, c in zip(names, counts) for _ in range(c)]
    return Structure(lat, species, xyz, "POSCAR")


def _read_aims(path):
    lat, species, xyz, frac = [], [], [], []
    for ln in open(path):
        t = ln.split("#")[0].split()
        if not t:
            continue
        if t[0] == "lattice_vector":
            lat.append([float(x) for x in t[1:4]])
        elif t[0] in ("atom", "atom_frac"):
            species.append(t[4])
            xyz.append([float(x) for x in t[1:4]])
            frac.append(t[0] == "atom_frac")
    lat = np.array(lat) if len(lat) == 3 else None
    xyz = np.array(xyz)
    for a, isf in enumerate(frac):
        if isf:
            xyz[a] = xyz[a] @ lat
    return Structure(lat, species, xyz, "FHI-aims geometry.in")


def _read_gen(path):
    L = [ln.split("#")[0].strip() for ln in open(path)]
    L = [ln for ln in L if ln]
    nat, kind = int(L[0].split()[0]), L[0].split()[1].upper()
    names = L[1].split()
    species, xyz = [], []
    for a in range(nat):
        t = L[2 + a].split()
        species.append(names[int(t[1]) - 1])
        xyz.append([float(x) for x in t[2:5]])
    xyz = np.array(xyz)
    lat = None
    if kind in ("S", "F"):
        lat = np.array([[float(x) for x in L[3 + nat + i].split()[:3]] for i in range(3)])
        if kind == "F":
            xyz = xyz @ lat
    return Structure(lat, species, xyz, "DFTB+ .gen")


_Z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
      "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
      "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
      "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
      "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44,
      "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52,
      "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "Hf": 72, "Ta": 73, "W": 74, "Re": 75,
      "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83}


def _read_crystal_out(path):
    """Saída do CRYSTAL (SCF ou PROPERTIES): vetores de rede + bloco de átomos.

    Usa o último bloco 'DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)' e o bloco
    'ATOM N.AT. SHELL X(A) Y(A) Z(A)' que o segue (3 casas decimais), ou o bloco
    'CARTESIAN COORDINATES - PRIMITIVE CELL' (mais preciso), se existir.
    """
    L = open(path).read().splitlines()
    lat = None
    for k, ln in enumerate(L):
        if "DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)" in ln:
            lat = np.array([[float(x) for x in L[k + 1 + i].split()[:3]] for i in range(3)])
    species, xyz = [], []
    for k, ln in enumerate(L):
        if "CARTESIAN COORDINATES - PRIMITIVE CELL" in ln:
            species, xyz = [], []
            for ln2 in L[k + 4:]:
                t = ln2.split()
                if len(t) < 6 or not t[0].isdigit():
                    break
                species.append(t[2].capitalize())
                xyz.append([float(x) for x in t[3:6]])
    if not xyz:
        for k, ln in enumerate(L):
            if ln.split()[:3] == ["ATOM", "N.AT.", "SHELL"]:
                species, xyz = [], []
                for ln2 in L[k + 2:]:
                    t = ln2.split()
                    if not t or t[0].startswith("*") or not t[0].isdigit():
                        break
                    species.append(t[2].capitalize())
                    xyz.append([float(x) for x in t[4:7]])
    if lat is None or not xyz:
        raise ValueError(f"{path}: não encontrei rede/átomos na saída do CRYSTAL")
    return Structure(lat, species, np.array(xyz), "CRYSTAL output")


def read_structure(path: str, fmt: str = "auto") -> Structure:
    """Lê a estrutura cristalina.

    Formatos: 'poscar' (VASP POSCAR/CONTCAR), 'aims' (geometry.in), 'gen' (DFTB+),
    'crystal' (saída .out do CRYSTAL) e, se o ASE estiver instalado, qualquer
    formato que ele leia ('ase:<formato>' ou detecção automática).
    """
    import os
    name = os.path.basename(path).lower()
    if fmt == "auto":
        if name.startswith(("poscar", "contcar")) or name.endswith(".vasp"):
            fmt = "poscar"
        elif name.startswith("geometry.in") or name.endswith(".in"):
            fmt = "aims"
        elif name.endswith(".gen"):
            fmt = "gen"
        elif name.endswith((".out", ".cry")) and "CRYSTAL" in open(path).read(200000):
            fmt = "crystal"
        else:
            fmt = "ase"
    if fmt == "poscar":
        return _read_poscar(path)
    if fmt == "aims":
        return _read_aims(path)
    if fmt == "gen":
        return _read_gen(path)
    if fmt == "crystal":
        return _read_crystal_out(path)
    if fmt.startswith("ase"):
        try:
            import ase.io
        except ImportError as err:
            raise ValueError(f"{path}: formato não reconhecido e ASE não instalado") from err
        atoms = ase.io.read(path, format=fmt.split(":", 1)[1] if ":" in fmt else None)
        lat = np.array(atoms.cell) if atoms.cell.rank == 3 else None
        return Structure(lat, atoms.get_chemical_symbols(), atoms.get_positions(), "ASE")
    raise ValueError(f"formato de estrutura desconhecido: {fmt}")
