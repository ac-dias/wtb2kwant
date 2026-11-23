"""Ortogonalização simétrica (Löwdin) de um modelo não ortogonal.

Para cada k numa malha N x N das direções periódicas:

    H_L(k) = S(k)^(-1/2) H(k) S(k)^(-1/2)

e de volta ao espaço real, H_L(R) = (1/N^2) sum_k H_L(k) exp(-2 pi i k.R). Cada
bloco (s, t) recebe o vetor R de "imagem mínima" (o menor |R.A + p_t - p_s|
entre as N^2 imagens equivalentes); empates são divididos igualmente, como as
degenerescências do Wannier90. Os orbitais de Löwdin são os mais próximos
(em norma) dos orbitais atômicos originais e continuam associados aos mesmos
sites. O resultado é exato nos pontos da malha; fora dela o erro cai
rapidamente com N, e é estimado e registrado nas notas do modelo.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np

from .model import TBModel


def _inv_sqrt(S):
    w, U = np.linalg.eigh(S)
    if w[0] <= 0:
        raise ValueError(f"S(k) não é positiva-definida (min = {w[0]:.2e})")
    return (U * w ** -0.5) @ U.conj().T, w[0], w[-1]


def lowdin_orthogonalize(model: TBModel, mesh: int | None = None,
                         cutoff: float = 1e-5, ntest: int = 40, seed: int = 0) -> TBModel:
    if not model.nonorthogonal:
        return model
    pd = model.periodic_dims
    if len(pd) != 2:
        raise NotImplementedError("Löwdin implementado para sistemas 2D")
    rmax = max(max(abs(R[d]) for d in pd) for R in set(model.hoppings) | set(model.overlaps))
    if mesh is None:
        mesh = max(2 * rmax + 1, 15)
        mesh = 2 * mesh + 1           # folga: o alcance de S^-1/2 H S^-1/2 é maior
    N = mesh
    sl = model.site_slices()
    pos = np.array([s.position for s in model.sites])
    A = model.lattice

    ks = [(i / N, j / N) for i in range(N) for j in range(N)]
    HL = {}
    smin, smax = np.inf, 0.0
    for kk in ks:
        k3 = np.zeros(3)
        k3[pd[0]], k3[pd[1]] = kk
        X, lo, hi = _inv_sqrt(model.sk(k3))
        smin, smax = min(smin, lo), max(smax, hi)
        HL[kk] = X @ model.hk(k3) @ X

    # transformada inversa numa caixa base de R; depois imagem mínima por par (s, t)
    half = N // 2
    base = [(a, b) for a in range(-half, N - half) for b in range(-half, N - half)]
    karr = np.array(ks)
    Hstack = np.array([HL[kk] for kk in ks])          # (Nk, n, n)
    out = {}
    nsite = len(model.sites)
    for r in base:
        ph = np.exp(-2j * np.pi * (karr @ np.array(r, dtype=float)))
        M = np.tensordot(ph, Hstack, axes=(0, 0)) / len(ks)
        for s in range(nsite):
            for t in range(nsite):
                blk = M[sl[s], sl[t]]
                if not np.any(np.abs(blk) > 0.1 * cutoff):
                    continue
                cands = []
                for m1, m2 in itertools.product((-1, 0, 1), repeat=2):
                    Rc = [0, 0, 0]
                    Rc[pd[0]], Rc[pd[1]] = r[0] + N * m1, r[1] + N * m2
                    d = np.linalg.norm(np.array(Rc) @ A + pos[t] - pos[s])
                    cands.append((d, tuple(Rc)))
                dmin = min(c[0] for c in cands)
                ties = [c[1] for c in cands if c[0] < dmin + 1e-5]
                for Rc in ties:
                    T = out.get(Rc)
                    if T is None:
                        T = out[Rc] = np.zeros((model.norb, model.norb), dtype=complex)
                    T[sl[s], sl[t]] += blk / len(ties)

    # hermitiza e corta
    herm = {}
    for R, M in out.items():
        mR = tuple(-x for x in R)
        P = out.get(mR, np.zeros_like(M))
        herm[R] = 0.5 * (M + P.conj().T)
    final = {}
    for R, M in herm.items():
        M = M.copy()
        M[np.abs(M) < cutoff] = 0.0
        if np.any(M):
            final[R] = M
    final.setdefault((0, 0, 0), np.zeros((model.norb, model.norb), dtype=complex))

    new = TBModel(systype=model.systype, efermi=model.efermi, scissors=model.scissors,
                  lattice=model.lattice, periodic_dims=list(pd), sites=model.sites,
                  hoppings=final, notes=list(model.notes), overlaps=None)

    # erro fora da malha: pontos aleatórios
    rng = np.random.default_rng(seed)
    dev = 0.0
    for _ in range(ntest):
        k3 = np.zeros(3)
        k3[pd[0]], k3[pd[1]] = rng.random(2)
        dev = max(dev, np.abs(new.bands([k3])[0] - model.bands([k3])[0]).max())
    nel = sum(np.count_nonzero(M) for M in final.values())
    new.notes.append(f"ortogonalização de Löwdin: malha {N}x{N}, {len(final)} vetores R, "
                     f"{nel} elementos; cond(S) <= {smax / smin:.1e}; erro máx. das bandas "
                     f"fora da malha ({ntest} pontos k aleatórios): {dev:.2e} eV")
    if smax / smin > 1e6:
        warnings.warn(f"S mal condicionada (cond = {smax / smin:.1e}): Löwdin pode ser instável")
    if dev > 1e-2:
        warnings.warn(f"erro de Löwdin fora da malha = {dev:.2e} eV; aumente --lowdin-mesh")
    return new
