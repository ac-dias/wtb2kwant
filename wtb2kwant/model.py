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

from .readers import WoutData, WtbHam, WtbWannier


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
    overlaps: dict | None = None                  # base não ortogonal: tuple(R) -> S(R)

    @property
    def nonorthogonal(self) -> bool:
        return self.overlaps is not None

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

    def sk(self, kfrac) -> np.ndarray:
        """S(k) (identidade se a base for ortogonal)."""
        if self.overlaps is None:
            return np.eye(self.norb, dtype=complex)
        kfrac = np.asarray(kfrac, dtype=float)
        S = np.zeros((self.norb, self.norb), dtype=complex)
        for R, M in self.overlaps.items():
            S += M * np.exp(2j * np.pi * np.dot(kfrac, R))
        return S

    def bands(self, kfrac_list) -> np.ndarray:
        """Autovalores de H(k) c = E S(k) c (problema generalizado se houver S)."""
        if self.overlaps is None:
            return np.array([np.linalg.eigvalsh(self.hk(k)) for k in kfrac_list])
        from scipy.linalg import eigh
        return np.array([eigh(self.hk(k), self.sk(k), eigvals_only=True)
                         for k in kfrac_list])

    def site_blocks_hs(self):
        """Como site_blocks(), para base não ortogonal: retorna
        (onsite_H, onsite_S, hops) com hops = [(R, s, t, H_blk, S_blk)], sobre a
        união dos vetores R de H e S (só metade dos pares hermitianos)."""
        sl = self.site_slices()
        nsite = len(self.sites)
        zero = (0, 0, 0)
        Z = np.zeros((self.norb, self.norb), dtype=complex)
        onH = [self.hoppings.get(zero, Z)[sl[s], sl[s]] for s in range(nsite)]
        onS = [self.overlaps.get(zero, Z)[sl[s], sl[s]] for s in range(nsite)]
        hops = []
        for R in sorted(set(self.hoppings) | set(self.overlaps)):
            MH = self.hoppings.get(R, Z)
            MS = self.overlaps.get(R, Z)
            for s in range(nsite):
                for t in range(nsite):
                    if R == zero and s == t:
                        continue
                    if (tuple(-r for r in R), t, s) < (R, s, t):
                        continue
                    hb, sb = MH[sl[s], sl[t]], MS[sl[s], sl[t]]
                    if np.any(hb != 0) or np.any(sb != 0):
                        hops.append((R, s, t, hb, sb))
        return onH, onS, hops

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


# =====================================================================
# Base não ortogonal (.ham do WanTiBEXOS)
# =====================================================================
def _parse_norb(spec: str | dict | None) -> dict:
    if spec is None:
        return {}
    if isinstance(spec, dict):
        return dict(spec)
    out = {}
    for item in spec.replace(";", ",").split(","):
        if item.strip():
            k, v = item.split("=")
            out[k.strip()] = int(v)
    return out


def orbital_atoms_from_basis(basis: list, nbasis: int):
    """(espécies, posições dos átomos, átomo de cada orbital, rótulos) a partir do .basis."""
    if len(basis) == nbasis // 2 and nbasis % 2 == 0:
        nb = len(basis)        # .basis com só um bloco de spin: duplica
        basis = basis + basis
        spins = [1] * nb + [-1] * nb
    elif len(basis) == nbasis:
        spins = [b.spin for b in basis]
    else:
        raise ValueError(f".basis tem {len(basis)} orbitais, o hamiltoniano tem {nbasis}")
    atoms, species, orb_atom = [], [], []
    keys = {}
    for b in basis:
        key = (b.species, *np.round(b.position, 3))
        if key not in keys:
            keys[key] = len(atoms)
            atoms.append(b.position)
            species.append(b.species)
        orb_atom.append(keys[key])
    spinful = nbasis % 2 == 0 and any(sp != 0 for sp in spins)
    labels = []
    for k, b in enumerate(basis):
        lab = f"o{(k % (nbasis // 2) if spinful else k) + 1}(l{b.l},m{b.m})"
        if spinful:
            lab = ("up:" if spins[k] >= 0 else "dn:") + lab
        labels.append(lab)
    return species, np.array(atoms), orb_atom, labels


def orbital_atoms_from_structure(struct, nbasis: int, spinful: bool, norb=None):
    """Átomo de cada orbital supondo orbitais contíguos por átomo, na ordem da
    estrutura, e (SP/SOC) bloco spin-up seguido do bloco spin-down."""
    nspatial = nbasis // 2 if spinful else nbasis
    nat = len(struct.species)
    norb = _parse_norb(norb)
    uniq = list(dict.fromkeys(struct.species))
    if not norb:
        if len(uniq) == 1 and nspatial % nat == 0:
            norb = {uniq[0]: nspatial // nat}
        else:
            raise ValueError(
                f"{nspatial} orbitais (por spin) para {nat} átomos de {len(uniq)} espécies: "
                f"informe os orbitais por espécie, ex.: --norb " +
                ",".join(f"{u}=N" for u in uniq))
    missing = [u for u in uniq if u not in norb]
    if missing:
        raise ValueError(f"--norb não define as espécies {missing}")
    counts = [norb[sp] for sp in struct.species]
    if sum(counts) != nspatial:
        raise ValueError(f"--norb dá {sum(counts)} orbitais por spin, o hamiltoniano tem "
                         f"{nspatial}: confira os números de orbitais por espécie")
    orb_atom, labels = [], []
    for a, c in enumerate(counts):
        orb_atom += [a] * c
    labels = [f"o{k + 1}" for k in range(nspatial)]
    if spinful:
        orb_atom = orb_atom + orb_atom
        labels = ["up:" + x for x in labels] + ["dn:" + x for x in labels]
    return list(struct.species), np.array(struct.positions), orb_atom, labels


def _periodic_dims(rvecs, mats, cut):
    out = []
    for d in range(3):
        mask = rvecs[:, d] != 0
        if mask.any() and any(np.abs(M[mask]).max() > c for M, c in zip(mats, cut)):
            out.append(d)
    return out


def check_overlap_positive(model, nk=24):
    """Menor autovalor de S(k) numa malha das direções periódicas."""
    pd = model.periodic_dims
    grids = [np.arange(nk) / nk if d in pd else [0.0] for d in range(3)]
    smin = np.inf
    for k1 in grids[0]:
        for k2 in grids[1]:
            for k3 in grids[2]:
                smin = min(smin, np.linalg.eigvalsh(model.sk((k1, k2, k3)))[0])
    return smin


def from_wtb_ham(ham: WtbHam, basis=None, structure=None, norb=None,
                 cutoff: float = 1e-5, s_cutoff: float = 1e-6,
                 efermi: float | None = None, lattice_tol: float = 1e-3) -> TBModel:
    """Monta o modelo não ortogonal a partir do .ham + (.basis ou estrutura).

    As posições vêm do .basis (SIESTA, YAeHMOP) ou da estrutura cristalina
    (POSCAR, geometry.in, .gen, saída do CRYSTAL...) com --norb por espécie.
    Cada átomo vira um site com seus orbitais; átomos fora da célula de origem
    são dobrados com a mesma correção de R do caso Wannier.
    """
    spinful = ham.systype in ("SP", "SOC")
    if basis is not None:
        species, atoms, orb_atom, labels = orbital_atoms_from_basis(basis, ham.nbasis)
    elif structure is not None:
        if structure.lattice is not None and not np.allclose(structure.lattice, ham.lattice,
                                                             atol=lattice_tol):
            warnings.warn("rede da estrutura difere da rede do .ham; usando a do .ham "
                          "(as posições cartesianas são usadas como estão)")
        species, atoms, orb_atom, labels = orbital_atoms_from_structure(
            structure, ham.nbasis, spinful, norb)
    else:
        raise ValueError("base não ortogonal: forneça o .basis ou a estrutura cristalina")

    lattice = ham.lattice
    periodic = _periodic_dims(ham.rvecs, [ham.ham, ham.ovl], [cutoff, s_cutoff])
    notes = [f"base não ortogonal (S incluído); dimensões periódicas: {periodic}"]

    # dobra os átomos para a célula de origem; deslocamento inteiro por átomo
    af = _frac(lattice, atoms)
    nat_shift = np.zeros((len(atoms), 3), dtype=int)
    for d in periodic:
        nat_shift[:, d] = np.floor(af[:, d] + 1e-8).astype(int)
    atoms_home = atoms - nat_shift @ lattice
    nshifted = int(np.any(nat_shift != 0, axis=1).sum())

    # sites = átomos (na ordem em que aparecem); orbitais na ordem do arquivo
    nsite = len(atoms)
    members = [[] for _ in range(nsite)]
    for o, a in enumerate(orb_atom):
        members[a].append(o)
    sites, newidx, count = [], {}, {}
    i = 0
    for a in range(nsite):
        if not members[a]:
            continue
        count[species[a]] = count.get(species[a], 0) + 1
        sites.append(Site(f"{species[a]}{count[species[a]]}", species[a],
                          atoms_home[a], [labels[o] for o in members[a]]))
        for o in members[a]:
            newidx[o] = i
            i += 1
    norb_tot = i
    idx = np.array([newidx[o] for o in range(ham.nbasis)])
    nsh = np.array([nat_shift[orb_atom[o]] for o in range(ham.nbasis)])

    def remap(mats):
        out = {}
        for ir, R in enumerate(ham.rvecs):
            M = mats[ir]
            if not np.any(M):
                continue
            Rnew = R[None, None, :] + nsh[None, :, :] - nsh[:, None, :]
            flat = Rnew.reshape(-1, 3)
            uniq, inv = np.unique(flat, axis=0, return_inverse=True)
            inv = inv.reshape(ham.nbasis, ham.nbasis)
            for u, Ru in enumerate(uniq):
                key = tuple(int(x) for x in Ru)
                mm, nn = np.nonzero(inv == u)
                vals = M[mm, nn]
                if not np.any(vals):
                    continue
                T = out.get(key)
                if T is None:
                    T = out[key] = np.zeros((norb_tot, norb_tot), dtype=complex)
                T[idx[mm], idx[nn]] += vals
        return out

    H = _hermitize(remap(ham.ham))
    S = _hermitize(remap(ham.ovl))
    for d in range(3):      # remove termos através do vácuo
        if d not in periodic:
            H = {R: M for R, M in H.items() if R[d] == 0}
            S = {R: M for R, M in S.items() if R[d] == 0}

    def apply_cut(D, c):
        out, kept, tot = {}, 0, 0
        for R, M in D.items():
            tot += np.count_nonzero(M)
            M = M.copy()
            M[np.abs(M) < c] = 0.0
            if np.any(M):
                out[R] = M
                kept += np.count_nonzero(M)
        return out, kept, tot

    H, hk_, ht_ = apply_cut(H, cutoff)
    S, sk_, st_ = apply_cut(S, s_cutoff)
    zero = (0, 0, 0)
    H.setdefault(zero, np.zeros((norb_tot, norb_tot), dtype=complex))
    S.setdefault(zero, np.eye(norb_tot, dtype=complex))
    notes.append(f"cutoff H {cutoff:g} eV: {hk_}/{ht_} elementos; cutoff S {s_cutoff:g}: "
                 f"{sk_}/{st_} elementos; {len(set(H) | set(S))} vetores R")
    notes.append(f"{nshifted}/{nsite} átomos fora da célula de origem (R corrigidos)")

    m = TBModel(systype=ham.systype, efermi=ham.efermi if efermi is None else efermi,
                scissors=ham.scissors, lattice=lattice, periodic_dims=periodic,
                sites=sites, hoppings=H, notes=notes, overlaps=S)
    smin = check_overlap_positive(m)
    m.notes.append(f"menor autovalor de S(k) (malha 24x24): {smin:.3e}")
    if smin <= 1e-8:
        warnings.warn(f"S(k) não é positiva-definida (min = {smin:.2e}); reduza --s-cutoff")
    return m
