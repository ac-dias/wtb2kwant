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
