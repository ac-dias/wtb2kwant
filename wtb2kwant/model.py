"""Modelo tight-binding intermediário, independente do formato de entrada.

Convenção: ``hoppings[R][i, j] = <i, 0 | H | j, R>`` (eV), com R inteiro (índices
de rede) e i, j índices de orbital na ordenação do modelo (orbitais agrupados por
site). Isso é o mesmo que ``H_mn(R)`` do Wannier90.

Folding dos centros
-------------------
Cada função de Wannier (WF) w tem centro c_w. Ela é associada a um site (átomo
mais próximo, ou o próprio centro) cuja posição na célula de origem é p_s, com
c_w ≈ p_s + n_w·A (n_w inteiro). O orbital antigo (w, R) passa a ser o orbital
novo (w, R + n_w), portanto::

    H_novo_ij(R + n_j - n_i) = H_antigo_ij(R)

Isso não muda autovalores (é uma mudança de gauge diagonal), mas é essencial
para o Kwant: as posições reais dos orbitais definem os cortes de fitas/tubos.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .readers import WoutData, WtbWannier


@dataclass
class Site:
    name: str
    species: str
    position: np.ndarray        # cartesiano (Å), na célula de origem
    orbitals: list              # rótulos, ex.: "wf3" ou "up:wf3"


@dataclass
class TBModel:
    systype: str
    efermi: float
    scissors: float
    lattice: np.ndarray                     # (3, 3)
    periodic_dims: list
    sites: list = field(default_factory=list)
    hoppings: dict = field(default_factory=dict)   # tuple(R) -> (norb, norb)
    notes: list = field(default_factory=list)

    @property
    def norb(self) -> int:
        return sum(len(s.orbitals) for s in self.sites)

    def site_slices(self) -> list:
        out, i0 = [], 0
        for s in self.sites:
            out.append(slice(i0, i0 + len(s.orbitals)))
            i0 += len(s.orbitals)
        return out

    def hk(self, kfrac) -> np.ndarray:
        """H(k) com k em coordenadas fracionárias (unidades de b_i)."""
        kfrac = np.asarray(kfrac, dtype=float)
        H = np.zeros((self.norb, self.norb), dtype=complex)
        for R, M in self.hoppings.items():
            H += M * np.exp(2j * np.pi * np.dot(kfrac, R))
        return H

    def bands(self, kfrac_list) -> np.ndarray:
        return np.array([np.linalg.eigvalsh(self.hk(k)) for k in kfrac_list])

    def site_blocks(self):
        """Retorna (onsite, hops).

        onsite[s] = bloco <s,0|H|s,0>
        hops = lista de (R, s, t, bloco <s,0|H|t,R>) contendo só metade dos
        pares (o outro é o hermitiano conjugado, que o Kwant gera sozinho).
        """
        sl = self.site_slices()
        nsite = len(self.sites)
        zero = (0, 0, 0)
        onsite = [self.hoppings[zero][sl[s], sl[s]] for s in range(nsite)]
        hops = []
        for R, M in sorted(self.hoppings.items()):
            for s in range(nsite):
                for t in range(nsite):
                    if R == zero and s == t:
                        continue
                    key = (R, s, t)
                    partner = (tuple(-r for r in R), t, s)
                    if partner < key:   # guarda só um representante do par
                        continue
                    blk = M[sl[s], sl[t]]
                    if np.any(blk != 0):
                        hops.append((R, s, t, blk))
        return onsite, hops


def _frac(lattice, xyz):
    return np.asarray(xyz) @ np.linalg.inv(lattice)


def _hermitize(hops: dict) -> dict:
    out = {}
    for R, M in hops.items():
        mR = tuple(-r for r in R)
        if mR in hops:
            out[R] = 0.5 * (M + hops[mR].conj().T)
        else:  # não deveria acontecer em um hr.dat completo
            out[R] = M
            out[mR] = M.conj().T
    return out


def from_wannier(wtb: WtbWannier, wouts: list, sites_mode: str = "atoms",
                 cutoff: float = 1e-4, max_dist_warn: float = 1.0,
                 efermi: float | None = None) -> TBModel:
    """Monta o modelo a partir do tb_hr.dat + .wout(s).

    wouts: um .wout para NP/SOC, dois (up, dn) para SP.
    sites_mode: "atoms" (orbitais agrupados no átomo mais próximo) ou
                "centres" (cada WF é um site na posição do seu centro).
    cutoff: elementos |H| < cutoff (eV) são descartados.
    """
    nch = len(wtb.blocks)
    if len(wouts) != nch:
        raise ValueError(f"Sistema {wtb.systype}: esperado(s) {nch} arquivo(s) .wout, "
                         f"recebido(s) {len(wouts)}" +
                         (" (spin-polarizado: passe o .wout do up e do dn)" if nch == 2 else ""))
    for c, (blk, wo) in enumerate(zip(wtb.blocks, wouts)):
        if len(wo.centres) != blk.num_wann:
            raise ValueError(f"canal {c}: .wout tem {len(wo.centres)} centros, "
                             f"hr tem num_wann={blk.num_wann}")

    lattice = wtb.lattice
    if not np.allclose(lattice, wouts[0].lattice, atol=1e-3):
        warnings.warn("Rede do tb_hr.dat difere da do .wout; usando a do tb_hr.dat")

    # Dimensões periódicas: aquelas em que existe hopping relevante com R_d != 0
    periodic = []
    for d in range(3):
        mx = -1.0
        for blk in wtb.blocks:
            mask = blk.rvecs[:, d] != 0
            if mask.any():
                mx = max(mx, np.abs(blk.ham[mask]).max())
        if mx > cutoff:
            periodic.append(d)
    notes = [f"dimensões periódicas: {periodic}"]

    # Átomos dobrados para a célula de origem (nas direções periódicas)
    species = wouts[0].species
    afrac = _frac(lattice, wouts[0].atoms)
    for d in periodic:
        afrac[:, d] -= np.floor(afrac[:, d] + 1e-8)
    atoms = afrac @ lattice

    # Associa cada WF (canal c, índice w) a um site e a um deslocamento n
    chlabel = ["up", "dn"] if nch == 2 else [""]
    assign = []   # (c, w, chave_site, n(3,), posição_site)
    for c, wo in enumerate(wouts):
        for w, cen in enumerate(wo.centres):
            label = f"{chlabel[c]}:wf{w + 1}" if chlabel[c] else f"wf{w + 1}"
            if sites_mode == "atoms":
                best = None
                for a in range(len(atoms)):
                    df = _frac(lattice, cen - atoms[a])
                    n = np.zeros(3, dtype=int)
                    for d in periodic:
                        n[d] = int(np.round(df[d]))
                    dist = np.linalg.norm(cen - atoms[a] - n @ lattice)
                    if best is None or dist < best[0]:
                        best = (dist, a, n)
                dist, a, n = best
                if dist > max_dist_warn:
                    warnings.warn(f"{label}: centro a {dist:.2f} Å do átomo mais próximo "
                                  f"({species[a]}{a + 1}); considere --sites centres")
                assign.append((c, w, label, ("atom", a), n, atoms[a], species[a]))
            elif sites_mode == "centres":
                f = _frac(lattice, cen)
                n = np.zeros(3, dtype=int)
                for d in periodic:
                    n[d] = int(np.floor(f[d] + 1e-8))
                assign.append((c, w, label, ("wf", c, w), n, cen - n @ lattice, "X"))
            else:
                raise ValueError(f"sites_mode inválido: {sites_mode}")

    # Cria os sites (ordem: átomos na ordem do .wout; orbitais por canal e WF)
    site_keys, sites, members = [], [], []
    for entry in sorted(assign, key=lambda e: (e[3], e[0], e[1])):
        key = entry[3]
        if key not in site_keys:
            site_keys.append(key)
            if key[0] == "atom":
                a = key[1]
                same = sum(1 for k in site_keys if k[0] == "atom" and species[k[1]] == species[a])
                name = f"{species[a]}{same}"
            else:
                name = entry[2].replace(":", "_")
            sites.append(Site(name, entry[6], np.array(entry[5]), []))
            members.append([])
        si = site_keys.index(key)
        sites[si].orbitals.append(entry[2])
        members[si].append(entry)

    # Índice novo de cada (c, w) e deslocamento n
    newidx = {}
    shift = {}
    i = 0
    for si, mem in enumerate(members):
        for e in mem:
            newidx[(e[0], e[1])] = i
            shift[(e[0], e[1])] = e[4]
            i += 1
    norb = i

    # Remapeia os hoppings: R_novo = R + n_j - n_i
    hops: dict = {}
    for c, blk in enumerate(wtb.blocks):
        nw = blk.num_wann
        idx = np.array([newidx[(c, w)] for w in range(nw)])
        nsh = np.array([shift[(c, w)] for w in range(nw)])     # (nw, 3)
        for ir, R in enumerate(blk.rvecs):
            Rnew = R[None, None, :] + nsh[None, :, :] - nsh[:, None, :]   # (nw, nw, 3)
            Hr = blk.ham[ir]
            # agrupa por R_novo distinto
            flat = Rnew.reshape(-1, 3)
            uniq, inv = np.unique(flat, axis=0, return_inverse=True)
            inv = inv.reshape(nw, nw)
            for u, Ru in enumerate(uniq):
                key = tuple(int(x) for x in Ru)
                M = hops.get(key)
                if M is None:
                    M = hops[key] = np.zeros((norb, norb), dtype=complex)
                mm, nn = np.nonzero(inv == u)
                M[idx[mm], idx[nn]] += Hr[mm, nn]

    hops = _hermitize(hops)

    # Cutoff
    nkept = ntot = 0
    out = {}
    for R, M in hops.items():
        ntot += np.count_nonzero(M)
        M = M.copy()
        M[np.abs(M) < cutoff] = 0.0
        if np.any(M != 0):
            out[R] = M
            nkept += np.count_nonzero(M)
    if (0, 0, 0) not in out:
        out[(0, 0, 0)] = np.zeros((norb, norb), dtype=complex)
    notes.append(f"cutoff {cutoff:g} eV: {nkept}/{ntot} elementos mantidos, "
                 f"{len(out)} vetores R")

    nshifted = sum(1 for v in shift.values() if np.any(v != 0))
    notes.append(f"{nshifted}/{len(shift)} WFs com centro fora da célula de origem (R corrigidos)")

    return TBModel(systype=wtb.systype,
                   efermi=wtb.efermi if efermi is None else efermi,
                   scissors=wtb.scissors, lattice=lattice, periodic_dims=periodic,
                   sites=sites, hoppings=out, notes=notes)
