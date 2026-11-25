"""Gera o script Python/Kwant autocontido a partir de um TBModel."""
from __future__ import annotations

import datetime

import numpy as np

from . import __version__
from .model import TBModel


def _fmt(x: float) -> str:
    s = f"{x:.6f}"
    if s in ("0.000000", "-0.000000"):
        return "0.0"
    s = s.rstrip("0")
    return s + "0" if s.endswith(".") else s


def _mat(a: np.ndarray, indent: str) -> str:
    rows = ["[" + ", ".join(_fmt(v) for v in row) + "]" for row in a]
    inner = (",\n" + indent + " ").join(rows)
    return f"np.array([{inner}])"


def _block(b: np.ndarray, imag_tol: float, indent: str) -> str:
    re_s = _mat(b.real, indent)
    if np.abs(b.imag).max() <= imag_tol:
        return re_s
    return f"{re_s}\n{indent}+ 1j * {_mat(b.imag, indent)}"


def generate_script(model: TBModel, sources: list, imag_tol: float = 1e-5) -> str:
    pdims = model.periodic_dims
    if len(pdims) != 2:
        raise NotImplementedError(
            f"Por enquanto só sistemas 2D (periódicos em 2 direções); "
            f"detectado: {pdims}")
    prim = model.lattice[pdims]
    nonorth = model.nonorthogonal
    if nonorth:
        onsite, onsite_s, hops5 = model.site_blocks_hs()
        hops = [(R, a, b, hb) for R, a, b, hb, sb in hops5 if np.any(hb != 0)]
        ovls = [(R, a, b, sb) for R, a, b, hb, sb in hops5 if np.any(sb != 0)]
    else:
        onsite, hops = model.site_blocks()
        onsite_s, ovls = None, []

    L = []
    w = L.append
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    w('#!/usr/bin/env python')
    w('# COMO RODAR:  conda activate kwant  &&  python kwant_model.py   (opções: python kwant_model.py --help)')
    w('"""')
    w(f'Modelo Kwant gerado por wtb2kwant {__version__} em {now}.')
    w('')
    w('Entrada:')
    for s in sources:
        w(f'    {s}')
    w(f'Sistema: {model.systype} | E_F = {model.efermi} eV | '
      f'operador tesoura no arquivo = {model.scissors} eV (NÃO aplicado)')
    w('Sites (orbitais agrupados):')
    for s in model.sites:
        w(f'    {s.name:6s} {len(s.orbitals):2d} orb  pos = '
          f'({s.position[0]:.4f}, {s.position[1]:.4f}, {s.position[2]:.4f}) Å  '
          f'[{", ".join(s.orbitals)}]')
    w('Notas da conversão:')
    for n in model.notes:
        w(f'    - {n}')
    w('')
    w('Convenções:')
    w('    HOPPINGS = lista de (R, s, t, bloco) com bloco = <s, 0 | H | t, R> (eV),')
    w('    R em unidades de PRIM_VECS. Só metade dos pares é listada; o Kwant gera')
    w('    o hermitiano conjugado. Com SHIFT_TO_FERMI = True o zero de energia é E_F.')
    if nonorth:
        w('    Base NÃO ortogonal: OVERLAPS = (R, s, t, <s,0|t,R>) e ONSITE_S. O Kwant não tem')
        w('    matriz de sobreposição; na energia E usamos H\' = H - E (S - 1), que dá a mesma')
        w('    equação (H - E S) psi = 0 e a mesma corrente: transmissão exata. Os termos são')
        w('    funções do parâmetro E: kwant.smatrix(syst, E, params=dict(E=E)) (conductance()')
        w('    já faz isso). Bandas: problema generalizado H(k) c = E S(k) c (ribbon_bands()).')
        w('    Com SHIFT_TO_FERMI o deslocamento é H -> H - E_F S.')
    w('')
    w('Como rodar (precisa de kwant, numpy e matplotlib; ex.: conda activate kwant):')
    w('    python kwant_model.py                          # bandas + transmissão das fitas ZZ e AC (W = 20 Å)')
    w('    python kwant_model.py --no-transport           # só bandas (rápido)')
    w('    python kwant_model.py --iv                     # curva I-V (0-3 V) das duas fitas')
    w('    python kwant_model.py --iv --vrange -1 2       # I-V num intervalo de bias escolhido')
    if model.systype in ("SP", "SOC"):
        w('    (SP/SOC: T e I saem também por canal de spin; --no-spin desliga)')
    w('    python kwant_model.py --width-zz 30 --width-ac 25 --emin -2 --emax 2 -o bandas.png')
    w('    python kwant_model.py --help                   # lista todas as opções')
    w('')
    w('Uso como biblioteca (ex.: transporte):')
    w('    from kwant_model import make_two_terminal, conductance')
    w('    syst = make_two_terminal("zigzag", width=20.0, length=30.0).finalized()')
    w('    T = conductance(syst, [-0.5, 0.0, 0.5])   # energias relativas a E_F (eV), paralelo')
    if model.systype in ("SP", "SOC"):
        w('    from kwant_model import spin_conductance')
        w('    Ts = spin_conductance(syst, [0.0])         # dict: total, uu, ud, du, dd, mix')
        w('    # T[s\', s]: s = spin injetado no lead 0, s\' = spin coletado no lead 1')
    w('Funções: make_ribbon, ribbon_bands, ribbon_hsk, make_two_terminal, conductance, channel_count,')
    w('         spin_conductance, spin_transmission, spin_channel_count, orbital_spins,')
    w('         has_spin_labels,')
    w('         bulk_hk, bulk_sk, bulk_bands,')
    w('         landauer_current, run_iv,')
    w('         make_bulk, make_lattice, min_supercell, ribbon_period.')
    w('"""')
    w('import argparse')
    w('import os')
    w('')
    w('# conductance() paraleliza sobre energias: evita competição com threads BLAS')
    w('for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):')
    w('    os.environ.setdefault(_v, "1")')
    w('')
    w('import numpy as np')
    w('import kwant')
    w('')
    w('# ' + '=' * 70)
    w('# PARÂMETROS CONVERTIDOS (gerados automaticamente)')
    w('# ' + '=' * 70)
    w(f'SYSTYPE = "{model.systype}"')
    w(f'EFERMI = {model.efermi!r}          # eV')
    w(f'SCISSORS = {model.scissors!r}      # eV (informativo, não aplicado)')
    w('SHIFT_TO_FERMI = True       # subtrai E_F dos termos on-site')
    w('HOP_CUTOFF = 0.0            # eV; >0 descarta elementos pequenos (leads menores, transporte')
    w('                            # mais rápido, menos preciso). Também via --hop-cutoff.')
    w(f'NONORTHOGONAL = {nonorth!r}     # base não ortogonal (matriz de sobreposição S)')
    w('OVL_CUTOFF = 0.0            # idem para S (adimensional). Também via --ovl-cutoff.')
    w('')
    w('# Vetores de rede periódicos (Å)')
    w('PRIM_VECS = np.array([' + ",\n                      ".join(
        "[" + ", ".join(repr(float(x)) for x in row) + "]" for row in prim) + '])')
    w('')
    w('SITES = [')
    for s in model.sites:
        p = ", ".join(repr(float(round(x, 10))) for x in s.position)
        w(f'    dict(name="{s.name}", species="{s.species}", pos=({p}),')
        w(f'         orbitals={s.orbitals!r}),')
    w(']')
    w('')
    w('# Blocos on-site <s,0|H|s,0>, um por site (eV)')
    w('ONSITE = [')
    for s, b in zip(model.sites, onsite):
        w(f'    # {s.name}')
        w(f'    {_block(b, imag_tol, "    ")},')
    w(']')
    w('')
    w(f'# Hoppings: (R, s, t, <s,0|H|t,R>) -- {len(hops)} blocos')
    w('HOPPINGS = [')
    for R, s, t, b in hops:
        Rp = tuple(R[d] for d in pdims)
        w(f'    ({Rp!r}, {s}, {t}, {_block(b, imag_tol, "     ")}),')
    w(']')
    w('')
    if nonorth:
        w('# Sobreposição on-site <s,0|s,0>, um bloco por site')
        w('ONSITE_S = [')
        for st, b in zip(model.sites, onsite_s):
            w(f'    # {st.name}')
            w(f'    {_block(b, 1e-9, "    ")},')
        w(']')
        w('')
        w(f'# Sobreposições: (R, s, t, <s,0|t,R>) -- {len(ovls)} blocos')
        w('OVERLAPS = [')
        for R, a, b2, blk in ovls:
            Rp = tuple(R[d] for d in pdims)
            w(f'    ({Rp!r}, {a}, {b2}, {_block(blk, 1e-9, "     ")}),')
        w(']')
    else:
        w('ONSITE_S = None             # base ortogonal')
        w('OVERLAPS = []')
    w('')
    w(_RUNTIME)
    return "\n".join(L) + "\n"


_RUNTIME = r'''
# ======================================================================
# CONSTRUÇÃO DOS SISTEMAS
# ======================================================================
def _filtered(blocks, cut):
    if cut <= 0:
        return blocks
    out = []
    for R, s, t, blk in blocks:
        b = np.where(np.abs(blk) < cut, 0.0, blk)
        if np.any(b != 0):
            out.append((R, s, t, b))
    return out


def _hoppings():
    """HOPPINGS filtrados por HOP_CUTOFF (blocos vazios removidos)."""
    return _filtered(HOPPINGS, HOP_CUTOFF)


def _overlaps():
    """OVERLAPS filtrados por OVL_CUTOFF (vazio se a base for ortogonal)."""
    return _filtered(OVERLAPS, OVL_CUTOFF)


def _hs_blocks():
    """dict (R, s, t) -> [H, S] juntando hoppings e sobreposições (base não ortogonal)."""
    out = {}
    for R, s, t, blk in _hoppings():
        out.setdefault((tuple(R), s, t), [None, None])[0] = np.asarray(blk, dtype=complex)
    for R, s, t, blk in _overlaps():
        out.setdefault((tuple(R), s, t), [None, None])[1] = np.asarray(blk, dtype=complex)
    for key, (h, o) in out.items():
        shape = (h if h is not None else o).shape
        out[key] = [h if h is not None else np.zeros(shape, complex),
                    o if o is not None else np.zeros(shape, complex)]
    return out


def _onsite_value(H0, S0):
    """On-site para base não ortogonal: H' = H - E (S - 1), função do parâmetro E."""
    D = S0 - np.eye(len(S0))

    def onsite(site, E):
        return H0 - E * D
    return onsite


def _hopping_value(Hb, Sb):
    """Hopping para base não ortogonal: H' = H - E S, função do parâmetro E."""
    def hopping(site1, site2, E):
        return Hb - E * Sb
    return hopping


def make_lattice():
    """Rede de Kwant: um sublattice por site, com norbs = nº de orbitais."""
    return kwant.lattice.general(
        PRIM_VECS, [s["pos"] for s in SITES],
        norbs=[len(s["orbitals"]) for s in SITES],
        name=[s["name"] for s in SITES])


def make_bulk(lat=None):
    """Builder com simetria translacional 2D completa (template para fill)."""
    lat = lat or make_lattice()
    subs = lat.sublattices
    bulk = kwant.Builder(kwant.TranslationalSymmetry(*lat.prim_vecs))
    # syst[a, b] = <a|H|b>; HoppingKind(d, a, b) -> pares (a(x+d), b(x))
    # a = s na célula 0, b = t na célula R  =>  d = -R
    if not NONORTHOGONAL:
        for s, sub in enumerate(subs):
            H0 = np.array(ONSITE[s], dtype=complex)
            if SHIFT_TO_FERMI:
                H0 = H0 - EFERMI * np.eye(len(H0))
            bulk[sub(0, 0)] = H0
        for R, s, t, blk in _hoppings():
            bulk[kwant.builder.HoppingKind(tuple(-r for r in R), subs[s], subs[t])] = blk
        return bulk
    ef = EFERMI if SHIFT_TO_FERMI else 0.0
    for s, sub in enumerate(subs):
        H0 = np.array(ONSITE[s], dtype=complex)
        S0 = np.array(ONSITE_S[s], dtype=complex)
        bulk[sub(0, 0)] = _onsite_value(H0 - ef * S0, S0)
    for (R, s, t), (Hb, Sb) in _hs_blocks().items():
        bulk[kwant.builder.HoppingKind(tuple(-r for r in R), subs[s], subs[t])] = \
            _hopping_value(Hb - ef * Sb, Sb)
    return bulk


def _bloch(onsite, blocks, kfrac):
    sl, i0 = [], 0
    for s in SITES:
        sl.append(slice(i0, i0 + len(s["orbitals"])))
        i0 += len(s["orbitals"])
    M = np.zeros((i0, i0), dtype=complex)
    for s in range(len(SITES)):
        M[sl[s], sl[s]] += onsite[s]
    for R, s, t, blk in blocks:
        ph = np.exp(2j * np.pi * np.dot(kfrac, R))
        M[sl[s], sl[t]] += blk * ph
        M[sl[t], sl[s]] += np.conj(blk.T * ph)
    return M


def bulk_sk(kfrac):
    """S(k) do cristal 2D (identidade se a base for ortogonal)."""
    if not NONORTHOGONAL:
        n = sum(len(s["orbitals"]) for s in SITES)
        return np.eye(n, dtype=complex)
    return _bloch(ONSITE_S, _overlaps(), kfrac)


def bulk_hk(kfrac):
    """H(k) do cristal 2D (k fracionário), para conferir com as bandas DFT/W90.
    Com SHIFT_TO_FERMI: H - E_F (ortogonal) ou H - E_F S(k) (não ortogonal)."""
    H = _bloch(ONSITE, _hoppings(), kfrac)
    if SHIFT_TO_FERMI:
        H -= EFERMI * bulk_sk(kfrac)
    return H


def bulk_bands(kfrac):
    """Autovalores do cristal em k (problema generalizado se a base não for ortogonal)."""
    if not NONORTHOGONAL:
        return np.linalg.eigvalsh(bulk_hk(kfrac))
    from scipy.linalg import eigh
    return eigh(bulk_hk(kfrac), bulk_sk(kfrac), eigvals_only=True)


def _int_vec(period):
    return np.dot(period, PRIM_VECS)


def ribbon_period(kind):
    """Vetor de período (inteiros em PRIM_VECS) para 'zigzag' ou 'armchair'.

    Para rede hexagonal: zigzag = a1; armchair = menor combinação inteira
    perpendicular a a1. Também aceita uma tupla (m, n) qualquer.
    """
    if not isinstance(kind, str):
        return tuple(int(x) for x in kind)
    a1, a2 = PRIM_VECS
    if kind == "zigzag":
        return (1, 0)
    if kind == "armchair":
        # tolerância de 0.2 grau: redes relaxadas raramente têm 120.000 graus exatos
        best = None
        for m in range(-4, 5):
            for n in range(-4, 5):
                if (m, n) == (0, 0):
                    continue
                v = m * a1 + n * a2
                if abs(np.dot(v, a1)) < 3.5e-3 * np.linalg.norm(a1) * np.linalg.norm(v):
                    if best is None or np.linalg.norm(v) < best[0] - 1e-6:
                        best = (np.linalg.norm(v), (m, n))
        if best is None:
            raise ValueError("não achei vetor inteiro perpendicular a a1")
        m, n = best[1]
        return (m, n) if (m, n) >= (-m, -n) else (-m, -n)
    raise ValueError(f"tipo de fita desconhecido: {kind}")


def _ribbon_frame(period):
    T = _int_vec(period)
    t = T / np.linalg.norm(T)
    nz = np.cross(PRIM_VECS[0], PRIM_VECS[1])
    nz /= np.linalg.norm(nz)
    n = np.cross(nz, t)              # direção da largura (no plano)
    return T, t, n


def _ribbon_shape(period, width, lat):
    """Faixa 0 <= (r - r0)·n < width, com r0 escolhido para incluir a célula 0."""
    T, t, n = _ribbon_frame(period)
    u0 = min(np.dot(s["pos"], n) for s in SITES) - 1e-6

    def shape(site):
        u = np.dot(site.pos, n) - u0
        return 0 <= u < width

    starts = [sub(0, 0) for sub in lat.sublattices]
    return shape, starts, T, t, n, u0


def min_supercell(kind, width=20.0, nmax=40):
    """Menor N tal que a fita com período N*T seja finalizável no Kwant, isto é,
    só existam hoppings entre células vizinhas (exigência para leads e Bands).

    Começa por uma estimativa geométrica (alcance do hopping mais longo ao longo
    da fita) e aumenta N até a finalização funcionar.
    """
    period = ribbon_period(kind)
    T, t, n = _ribbon_frame(period)
    pos = [np.asarray(s["pos"]) for s in SITES]
    lmax = 0.0
    for R, s, u, blk in list(_hoppings()) + list(_overlaps()):
        d = np.dot(R, PRIM_VECS) + pos[u] - pos[s]
        lmax = max(lmax, abs(np.dot(d, t)))
    N = int(np.floor(lmax / np.linalg.norm(T) + 1e-9)) + 1
    while N <= nmax:
        try:
            make_ribbon(kind, width, supercell=N).finalized()
            return N
        except ValueError as err:
            if "Further-than-nearest-neighbor" not in str(err):
                raise
            N += 1
    raise RuntimeError(f"não achei supercélula finalizável com N <= {nmax}")


def make_ribbon(kind="zigzag", width=20.0, supercell=1):
    """Nanofita infinita (simetria 1D ao longo de supercell*período), largura em Å.

    Com supercell=1 o builder serve para ribbon_bands() (bandas sem dobramento),
    mas pode não ser finalizável no Kwant se houver hoppings longos. Para usar
    como lead/kwant.physics.Bands, use supercell=min_supercell(kind, width).
    """
    lat = make_lattice()
    period = ribbon_period(kind)
    shape, starts, T, t, n, u0 = _ribbon_shape(period, width, lat)
    vec = tuple(supercell * p for p in period)
    rib = kwant.Builder(kwant.TranslationalSymmetry(lat.vec(vec)))
    rib.fill(make_bulk(lat), shape, starts)
    return rib


def make_two_terminal(kind="zigzag", width=20.0, length=30.0, supercell=None):
    """Região de espalhamento finita (comprimento em Å) + dois leads de fita.

    Os leads usam supercélula de N períodos (N = min_supercell(kind, width) por
    padrão). A região de espalhamento precisa ser pelo menos tão longa quanto a
    célula do lead; se `length` for menor, ela é aumentada automaticamente.
    """
    lat = make_lattice()
    period = ribbon_period(kind)
    N = min_supercell(kind, width) if supercell is None else supercell
    shape_w, starts, T, t, n, u0 = _ribbon_shape(period, width, lat)
    lmin = N * np.linalg.norm(T) + 1e-6
    if length < lmin:
        print(f"[make_two_terminal] {kind}: length {length:g} Å < célula do lead "
              f"({N} x {np.linalg.norm(T):.3f} Å); usando {lmin:.2f} Å")
        length = lmin
    v0 = min(np.dot(s["pos"], t) for s in SITES) - 1e-6

    def shape(site):
        v = np.dot(site.pos, t) - v0
        return shape_w(site) and 0 <= v < length

    bulk = make_bulk(lat)
    syst = kwant.Builder()
    syst.fill(bulk, shape, starts)
    lead = kwant.Builder(kwant.TranslationalSymmetry(-lat.vec(tuple(N * p for p in period))))
    lead.fill(bulk, shape_w, starts)
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    return syst


def _ncpu():
    """Nº de processos padrão: ~núcleos físicos (metade dos lógicos, por causa do
    hyper-threading). O cálculo é limitado por memória; mais processos não ajudam."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, n // 2)


def _params(e):
    """params do Kwant na energia e: H' = H - E (S - 1) na base não ortogonal."""
    return dict(E=e) if NONORTHOGONAL else None


def conductance(syst, energies, nproc=None):
    """Transmissão lead 0 -> 1 (unidades de e^2/h, sem fator de spin).

    nproc: nº de processos (padrão: metade dos núcleos lógicos ~ núcleos físicos).
    Usa processos com fork que herdam o sistema finalizado; só índices e números
    trafegam pelas filas (nada de pickling de funções), com balanceamento
    dinâmico. Em sistemas sem fork o cálculo é serial.
    """
    fsyst = syst.finalized() if not isinstance(syst, kwant.system.System) else syst

    def one(e):
        return [kwant.smatrix(fsyst, e, params=_params(e)).transmission(1, 0)]

    return _energy_map(one, energies, nproc)[:, 0]


# ----------------------------------------------------------------------
# Transporte resolvido em spin (SP e SOC)
# ----------------------------------------------------------------------
SPIN_KEYS = ("uu", "ud", "du", "dd")
"""Componentes T[s', s]: s' = spin coletado no lead 1 (dreno), s = spin injetado
no lead 0 (fonte). "ud" = entra dn, sai up (spin-flip)."""


def orbital_spins():
    """dict nome do site -> spin de cada orbital (+1 up, -1 dn, 0 sem rótulo),
    lido dos prefixos 'up:'/'dn:' dos rótulos em SITES."""
    return {s["name"]: np.array([1 if o.startswith("up:") else -1 if o.startswith("dn:") else 0
                                 for o in s["orbitals"]]) for s in SITES}


def has_spin_labels():
    """True se o modelo tem spin explícito (SP ou SOC) e todo orbital é up ou dn."""
    return SYSTYPE in ("SP", "SOC") and all(np.all(v != 0) for v in orbital_spins().values())


def _interface_spins(fsyst, lead, spins):
    """Spin de cada orbital da interface do lead (mesma ordem da autoenergia)."""
    return np.concatenate([spins[fsyst.sites[i].family.name]
                           for i in fsyst.lead_interfaces[lead]])


def _lead_cell_overlap(lead, e):
    """S da célula do lead (identidade na base ortogonal). Na base não ortogonal a
    célula guarda H' = H - E (S - 1): S = 1 + H'(0) - H'(1)."""
    h0 = lead.cell_hamiltonian(params=_params(0.0))
    if not NONORTHOGONAL:
        return np.eye(len(h0))
    return np.eye(len(h0)) + (h0 - lead.cell_hamiltonian(params=_params(1.0)))


def _mode_spin_weights(phi, sp_up, Sh):
    """Matriz W^up (nmodos x nmodos) dos modos propagantes de um lead, medida na
    célula do lead vizinha à região de espalhamento:

        W^up = M^-1/2 A^+ P_up A M^-1/2,   A = S^1/2 phi,   M = A^+ A.

    É positiva e W^up + W^dn = 1. A matriz é coerente entre todos os modos:
    os termos fora da diagonal dão conta das superposições (um spin puro
    injetado é em geral uma combinação de modos com k diferentes, cujo spin só
    precessa numa escala 2 pi / Delta k ao longo do lead).
    """
    A = Sh @ phi
    M = A.conj().T @ A
    ev, U = np.linalg.eigh(M)
    Mi = (U / np.sqrt(ev)) @ U.conj().T
    return Mi @ (A.conj().T @ (A * sp_up[:, None])) @ Mi


def spin_transmission(fsyst, e, projection="modes", spins=None):
    """[T, T_uu, T_ud, T_du, T_dd] na energia e (lead 0 -> 1).

    projection = "modes" (padrão): spin dos modos propagantes dos leads,

        T_s's = Tr[ t^+ W_1^(s') t W_0^(s) ],

    t = bloco 1 <- 0 da matriz S (kwant.smatrix) e W^(s) = projeção de spin no
    espaço dos modos propagantes, medida na célula de cada lead vizinha à região
    de espalhamento (veja _mode_spin_weights): W_0 nos modos que entram pelo
    lead 0, W_1 nos que saem pelo lead 1. As componentes são >= 0 e somam
    exatamente T. Usa só os modos propagantes (sem o acoplamento lead-região),
    então o spin coletado em cada canal acompanha spin_channel_count. Se os
    modos têm spin puro (SP, ou SOC só com S_z) é exato.

    projection = "sym" ou "block": NEGF com Gamma projetado nos orbitais da
    interface, T_s's = Tr[Gamma_1^(s') G_10 Gamma_0^(s) G_10^+], com
    Gamma^(s) = {P_s, Gamma}/2 ("sym", soma = T) ou P_s Gamma P_s ("block",
    componentes >= 0, soma != T). Com SOC o Gamma da interface inclui o
    acoplamento lead-região, que mistura os spins: perto de cruzamentos de
    bandas as componentes ficam instáveis (até negativas com "sym", soma > 2T
    com "block"). Mantidas para comparação.

    O spin de cada orbital vem dos rótulos up:/dn: (orbital_spins()); a base
    não ortogonal entra pelo S^1/2 da célula do lead ("modes") e não atrapalha
    as projeções de orbitais porque S é diagonal em spin.
    """
    spins = orbital_spins() if spins is None else spins
    if projection == "modes":
        sm = kwant.smatrix(fsyst, e, params=_params(e))
        t = sm.submatrix(1, 0)
        out = [sm.transmission(1, 0)]
        if t.size == 0:
            return out + [0.0] * 4
        W = []
        for lead, half in ((0, 0), (1, 1)):         # 0: modos que entram; 1: que saem
            li = sm.lead_info[lead]
            n = len(li.momenta) // 2
            cols = np.arange(n) + half * n
            ld = fsyst.leads[lead]
            sp = np.concatenate([spins[ld.sites[i].family.name] for i in range(ld.cell_size)])
            ev, U = np.linalg.eigh(_lead_cell_overlap(ld, e))
            Sh = (U * np.sqrt(np.clip(ev, 0, None))) @ U.conj().T
            W.append(_mode_spin_weights(li.wave_functions[:, cols], (sp > 0).astype(float), Sh))
        W0u, W1u = W
        W0d, W1d = np.eye(len(W0u)) - W0u, np.eye(len(W1u)) - W1u
        td = t.conj().T
        for A in (W1u, W1d):
            for B in (W0u, W0d):
                out.append(np.trace(td @ A @ t @ B).real)
        return out

    gf = kwant.greens_function(fsyst, e, out_leads=[1], in_leads=[0], params=_params(e))
    G = gf.submatrix(1, 0)
    gam = [1j * (gf.lead_info[i] - gf.lead_info[i].conj().T) for i in (0, 1)]
    sp = [_interface_spins(fsyst, i, spins) for i in (0, 1)]

    def proj(i, s):
        P = (sp[i] == s).astype(float)
        if projection == "block":
            return gam[i] * np.outer(P, P)
        if projection == "sym":
            return 0.5 * (P[:, None] * gam[i] + gam[i] * P[None, :])
        raise ValueError(f"projection inválida: {projection}")

    Gd = G.conj().T
    out = [np.trace(gam[1] @ G @ gam[0] @ Gd).real]
    for so, si in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        out.append(np.trace(proj(1, so) @ G @ proj(0, si) @ Gd).real)
    return out


def spin_conductance(syst, energies, nproc=None, projection="modes"):
    """Transmissão resolvida em spin (SP/SOC), em paralelo como conductance().

    Retorna dict com arrays: "total" (= conductance), "uu", "ud", "du", "dd"
    (T[s', s], s' = spin no lead 1, s = spin injetado no lead 0; veja
    spin_transmission) e "mix" = total - (uu + ud + du + dd).
    SP: só uu e dd são não nulos (T_up = uu, T_dn = dd) e mix = 0.
    """
    if not has_spin_labels():
        raise ValueError("o modelo não tem spin explícito com rótulos up:/dn: "
                         "(NP, ou SOC do Wannier90 sem --soc-spin-order)")
    fsyst = syst.finalized() if not isinstance(syst, kwant.system.System) else syst
    spins = orbital_spins()
    arr = _energy_map(lambda e: spin_transmission(fsyst, e, projection, spins), energies, nproc)
    res = dict(total=arr[:, 0])
    for n, k in enumerate(SPIN_KEYS):
        res[k] = arr[:, n + 1]
    res["mix"] = res["total"] - sum(res[k] for k in SPIN_KEYS)
    return res


def _energy_map(one, energies, nproc=None):
    """Aplica one(e) -> lista de floats em cada energia; retorna array (nE, nout).

    Processos com fork herdam o sistema finalizado (nada de pickling de
    funções); só índices e números trafegam pelas filas. Sem fork: serial.
    """
    import multiprocessing as mp
    energies = list(energies)
    nproc = min(_ncpu() if nproc is None else nproc, len(energies))
    if nproc <= 1 or "fork" not in mp.get_all_start_methods():
        return np.array([np.atleast_1d(one(e)) for e in energies], dtype=float)

    ctx = mp.get_context("fork")
    tasks, results = ctx.Queue(), ctx.Queue()
    for i in range(len(energies)):
        tasks.put(i)
    for _ in range(nproc):
        tasks.put(None)

    def worker():
        while True:
            i = tasks.get()
            if i is None:
                break
            try:
                results.put((i, [float(x) for x in np.atleast_1d(one(energies[i]))], None))
            except Exception as err:          # devolve o erro ao processo pai
                results.put((i, None, repr(err)))

    procs = [ctx.Process(target=worker, daemon=True) for _ in range(nproc)]
    for p in procs:
        p.start()
    rows = [None] * len(energies)
    errors = []
    for _ in range(len(energies)):
        i, t, err = results.get()
        rows[i] = t
        if err:
            errors.append(f"E={energies[i]}: {err}")
    for p in procs:
        p.join()
    if errors:
        raise RuntimeError("falha no cálculo de transporte:\n" + "\n".join(errors[:5]))
    return np.array(rows, dtype=float)


def channel_count(rib, energies, nk=4001):
    """Nº de modos propagantes (dE/dk > 0) da fita em cada energia, a partir das
    bandas: é o valor esperado da transmissão numa fita perfeita."""
    ks, E = ribbon_bands(rib, nk)
    out = []
    for e in energies:
        sg = np.sign(E - e)
        out.append(int(np.sum((sg[:-1] < 0) & (sg[1:] > 0))))
    return np.array(out)


def _ribbon_sites(rib):
    """Sites do domínio fundamental na ordem usada por ribbon_hsk."""
    return sorted(rib.sites(), key=lambda s: (s.family.name, tuple(s.tag)))


def spin_channel_count(rib, energies, nk=4001):
    """Canais (modos com dE/dk > 0) por spin, das bandas da fita: o valor esperado
    de T_up e T_dn numa fita perfeita. Retorna dict "up", "dn" (arrays).

    Se H(k) e S(k) não acoplam up e dn (SP, ou SOC só com S_z) as bandas de cada
    bloco de spin são contadas separadamente: inteiros exatos. Com SOC geral cada
    canal entra com o seu peso de spin <P_s> (Mulliken: Re c^+ P_s S c, com
    c^+ S c = 1), média dos dois pontos k que cercam o cruzamento: a contagem
    pode ser fracionária, e up + dn = channel_count.
    """
    from scipy.linalg import eigh
    hk, sk = ribbon_hsk(rib)
    spins = orbital_spins()
    sp = np.concatenate([spins[s.family.name] for s in _ribbon_sites(rib)])
    up = sp > 0
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    ks = np.linspace(-np.pi, np.pi, nk)
    probe = [hk(k) for k in (0.3, 1.7)] + [sk(k) for k in (0.3, 1.7)]
    conserved = max(np.abs(M[np.ix_(up, ~up)]).max() for M in probe) < 1e-12
    out = {"up": np.zeros(len(energies)), "dn": np.zeros(len(energies))}

    def crossings(Eprev, Ecur, wprev=None, wcur=None):
        cross = ((Eprev[None, :] < energies[:, None])
                 & (Ecur[None, :] > energies[:, None]))     # (nE, nbandas)
        if wprev is None:
            return cross.sum(axis=1)
        return (cross * (0.5 * (wprev + wcur))[None, :]).sum(axis=1)

    if conserved:
        prev = {}
        for k in ks:
            Hk, Sk = hk(k), (sk(k) if NONORTHOGONAL else None)
            for name, msk in (("up", up), ("dn", ~up)):
                H = Hk[np.ix_(msk, msk)]
                E = (eigh(H, Sk[np.ix_(msk, msk)], eigvals_only=True) if NONORTHOGONAL
                     else np.linalg.eigvalsh(H))
                if name in prev:
                    out[name] += crossings(prev[name], E)
                prev[name] = E
        return out
    prev = None
    for k in ks:
        H = hk(k)
        S = sk(k) if NONORTHOGONAL else None
        E, C = eigh(H, S) if NONORTHOGONAL else np.linalg.eigh(H)
        SC = S @ C if NONORTHOGONAL else C
        wup = np.real(np.sum(np.conj(C[up]) * SC[up], axis=0))
        if prev is not None:
            out["up"] += crossings(prev[0], E, prev[1], wup)
            out["dn"] += crossings(prev[0], E, 1 - prev[1], 1 - wup)
        prev = (E, wup)
    return out


def ribbon_hsk(rib):
    """Retorna (H(k), S(k)) da fita a partir de um builder com simetria 1D.

    Monta H(k) = sum_n h_n e^{ikn} (e S(k)) diretamente dos termos do builder,
    então funciona mesmo com hoppings que ligam células distantes (que o Kwant
    não finaliza). k em radianos por período do builder. Na base não ortogonal
    os termos são H' = H - E D(E): H = H'(0) e D = H'(0) - H'(1), com
    D = S - 1 no on-site e D = S nos hoppings.
    """
    sym = rib.symmetry
    sites = _ribbon_sites(rib)
    off, n0 = {}, 0
    for s in sites:
        off[s] = n0
        n0 += s.family.norbs

    def ev(v, *args):
        if callable(v):
            h0 = np.atleast_2d(np.asarray(v(*args, 0.0), dtype=complex))
            h1 = np.atleast_2d(np.asarray(v(*args, 1.0), dtype=complex))
            return h0, h0 - h1
        h0 = np.atleast_2d(np.asarray(v, dtype=complex))
        return h0, np.zeros_like(h0)

    terms = []
    for s in sites:
        h, d = ev(rib[s], s)
        terms.append((off[s], off[s], 0, h, d + np.eye(len(d)), False))
    for a, b in rib.hoppings():
        na, nb = sym.which(a)[0], sym.which(b)[0]
        a0, b0 = sym.to_fd(a), sym.to_fd(b)
        h, d = ev(rib[a, b], a, b)
        terms.append((off[a0], off[b0], nb - na, h, d, True))

    def build(k, which):
        M = np.zeros((n0, n0), dtype=complex)
        for i, j, dn, h, sv, hc in terms:
            v = h if which == 0 else sv
            blk = v * np.exp(1j * k * dn)
            M[i:i + v.shape[0], j:j + v.shape[1]] += blk
            if hc:
                M[j:j + v.shape[1], i:i + v.shape[0]] += blk.conj().T
        return M

    return (lambda k: build(k, 0)), (lambda k: build(k, 1))


def ribbon_hk(rib):
    """H(k) da fita (veja ribbon_hsk; na base não ortogonal use também S(k))."""
    return ribbon_hsk(rib)[0]


def ribbon_bands(rib, nk=201):
    """Bandas da fita, k em [-pi, pi] (radianos por período do builder).
    Na base não ortogonal resolve o problema generalizado H(k) c = E S(k) c."""
    hk, sk = ribbon_hsk(rib)
    ks = np.linspace(-np.pi, np.pi, nk)
    if not NONORTHOGONAL:
        return ks, np.array([np.linalg.eigvalsh(hk(k)) for k in ks])
    from scipy.linalg import eigh
    return ks, np.array([eigh(hk(k), sk(k), eigvals_only=True) for k in ks])


def ribbon_info(rib):
    from collections import Counter
    c = Counter(s.family.name for s in rib.sites())
    return dict(c)


def plot_ribbon_top(ax, rib, kind, length=20.0):
    """Vista de cima de um trecho da fita (para conferir as bordas).

    Os sites do domínio fundamental do Kwant (que é oblíquo no espaço de tags)
    são trazidos para uma janela geométrica de um período e repetidos.
    """
    period = ribbon_period(kind)
    T, t, n = _ribbon_frame(period)
    a = np.linalg.norm(T)
    species = {x["name"]: x["species"] for x in SITES}
    colors = {}
    ncells = max(1, int(np.ceil(length / a)))
    for s in rib.sites():
        p0 = np.asarray(s.pos)
        p0 = p0 - np.floor(np.dot(p0, t) / a + 1e-9) * T
        sp = species[s.family.name]
        col = colors.setdefault(sp, f"C{len(colors)}")
        big = sp == SITES[0]["species"]
        for i in range(ncells):
            p = p0 + i * T
            x, y = np.dot(p, t), np.dot(p, n)
            ax.scatter(x, y, s=45 if big else 22, color=col, edgecolors="k",
                       linewidths=0.3, zorder=3 if big else 2)
    for sp, col in colors.items():
        ax.scatter([], [], color=col, label=sp)
    ax.set_aspect("equal")
    ax.set_xlabel("along the ribbon (Å)")
    ax.set_ylabel("width (Å)")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)


G0_UA_PER_EV = 38.7404585    # e^2/h em µA/V: I(µA) = 38.74 * g_s * integral T (f_L - f_R) dE(eV)


def fermi(E, mu, kT):
    """Distribuição de Fermi-Dirac (E, mu, kT em eV); kT = 0 -> degrau."""
    E = np.asarray(E, dtype=float)
    if kT <= 0:
        return np.where(E < mu, 1.0, np.where(E > mu, 0.0, 0.5))
    x = np.clip((E - mu) / kT, -700, 700)
    return 1.0 / (np.exp(x) + 1.0)


def landauer_current(energies, trans, biases, kT=0.025852, spin_degeneracy=None,
                     mu0=None):
    """Corrente de Landauer I(V) em µA, aproximação de banda rígida.

    I(V) = g_s (e/h) ∫ T(E) [f(E, mu0 + V/2) - f(E, mu0 - V/2)] dE

    energies, trans: T(E) tabelado (E em eV, malha uniforme e cobrindo
    [mu0 - Vmax/2 - 10 kT, mu0 + Vmax/2 + 10 kT]). biases em V.
    T(E) não depende do bias (sem queda de potencial na região de espalhamento).
    spin_degeneracy: 2 para NP (padrão), 1 para SP/SOC (spins já no modelo).
    mu0: potencial químico em V = 0 (padrão: E_F, ou seja, 0 se SHIFT_TO_FERMI).
    """
    if spin_degeneracy is None:
        spin_degeneracy = 2 if SYSTYPE == "NP" else 1
    if mu0 is None:
        mu0 = 0.0 if SHIFT_TO_FERMI else EFERMI
    E = np.asarray(energies, dtype=float)
    T = np.asarray(trans, dtype=float)
    trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x / 1.x
    out = []
    for V in np.atleast_1d(biases):
        window = fermi(E, mu0 + V / 2, kT) - fermi(E, mu0 - V / 2, kT)
        out.append(spin_degeneracy * G0_UA_PER_EV * trapz(T * window, E))
    return np.array(out)


SPIN_LABELS = {"uu": r"$\uparrow\to\uparrow$", "dd": r"$\downarrow\to\downarrow$",
               "ud": r"$\downarrow\to\uparrow$", "du": r"$\uparrow\to\downarrow$"}
SPIN_STYLE = {"uu": ("C3", "-"), "dd": ("C0", "--"), "ud": ("C2", "-"), "du": ("C1", "--")}
SPIN_ARROW = {"up": r"$\uparrow$", "dn": r"$\downarrow$"}


def _spin_keys_to_show():
    """SP: só as componentes que conservam spin; SOC: as quatro."""
    return ("uu", "dd") if SYSTYPE == "SP" else SPIN_KEYS


def _spin_columns(res, kinds, field):
    """Colunas extras (e cabeçalho) com as componentes de spin de cada fita
    (e, para T, os canais por spin das bandas)."""
    cols, head = [], []
    for kind in kinds:
        for k in _spin_keys_to_show():
            cols.append(res[kind][2][field][k])
            head.append(f"{field}_{kind}_{k}")
        if field == "T":
            for name in ("up", "dn"):
                cols.append(res[kind][2]["N"][name])
                head.append(f"N_{kind}_{name}")
    return cols, head


def run_iv(widths, vmax=3.0, nv=61, de=0.01, length=30.0, temperature=300.0,
           nproc=None, output="iv_ribbons.png", vmin=0.0, spin=None, projection="modes"):
    """T(E) numa malha fina + curva I-V (bias de vmin a vmax, em V) para as fitas
    zigzag e armchair. Bias negativo é permitido (a corrente troca de sinal).

    Usa HOP_CUTOFF (defina antes de chamar) para baratear o cálculo.
    Salva a figura e os arquivos <output>_T.dat e <output>_IV.dat.
    spin: resolve T e I por canal de spin (padrão: sim em SP/SOC com rótulos
    up/dn); as componentes T_s's (s = spin injetado, s' = spin coletado) vão
    para colunas extras e para uma segunda figura <output>_spin.png.
    Cada componente usa degenerescência de spin 1: I_s's = (e/h) ∫ T_s's (f_L - f_R).
    Retorna (energies, biases, res), res[fita] = (T, I, spin), com spin = None ou
    dict(T=spin_conductance(...), I={componente: I(V)}, N=spin_channel_count(...)).
    """
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spin = has_spin_labels() if spin is None else spin
    kT = 8.617333e-5 * temperature
    mu0 = 0.0 if SHIFT_TO_FERMI else EFERMI
    if vmax <= vmin:
        raise ValueError(f"intervalo de bias inválido: vmin = {vmin} >= vmax = {vmax}")
    emax = max(abs(vmin), abs(vmax)) / 2 + 10 * kT + de
    energies = mu0 + np.arange(-emax, emax + de / 2, de)
    biases = np.linspace(vmin, vmax, nv)
    res = {}
    kinds = ("zigzag", "armchair")
    for kind, width in zip(kinds, widths):
        t0 = time.time()
        syst = make_two_terminal(kind, width, length).finalized()
        if spin:
            Ts = spin_conductance(syst, energies, nproc, projection)
            T = Ts["total"]
        else:
            T = conductance(syst, energies, nproc)
        I = landauer_current(energies, T, biases, kT)
        sres = None
        if spin:
            sres = dict(T=Ts, I={k: landauer_current(energies, Ts[k], biases, kT, 1)
                                 for k in SPIN_KEYS + ("mix",)},
                        N=spin_channel_count(make_ribbon(kind, width), energies))
            print(f"{kind}: max |T - soma das componentes de spin| = "
                  f"{np.abs(Ts['mix']).max():.2e}")
        res[kind] = (T, I, sres)
        print(f"{kind}: T(E) em {len(energies)} energias + I-V ({time.time() - t0:.0f} s)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for kind, c, width in zip(kinds, ("C0", "C3"), widths):
        a1.plot(energies - mu0, res[kind][0], color=c, lw=1, label=f"{kind}, W = {width:g} Å")
        a2.plot(biases, res[kind][1], "o-", color=c, ms=3, lw=1,
                label=f"{kind}, W = {width:g} Å")
    a1.set_xlabel(r"$E - E_F$ (eV)")
    a1.set_ylabel(r"$T(E)$ ($e^2/h$ per spin)" if SYSTYPE == "NP"
                  else r"$T(E)$ ($e^2/h$, both spins)")
    a1.set_xlim(energies[0] - mu0, energies[-1] - mu0)
    a1.set_ylim(bottom=0)
    a1.axvline(0, color="k", ls=":", lw=0.6)
    a1.set_title(f"transmission (hopping cutoff {HOP_CUTOFF:g} eV"
                 + (f", overlap cutoff {OVL_CUTOFF:g})" if NONORTHOGONAL else ")"), fontsize=10)
    a1.legend(fontsize=8)
    a2.set_xlabel("bias $V$ (V)")
    a2.set_ylabel(r"current $I$ ($\mu$A)")
    a2.set_xlim(vmin, vmax)
    if vmin >= 0:
        a2.set_ylim(bottom=0)
    else:
        a2.axhline(0, color="k", lw=0.5)
        a2.axvline(0, color="k", lw=0.5)
    a2.set_title(f"Landauer I-V, rigid band, symmetric bias, T = {temperature:g} K",
                 fontsize=10)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    base = os.path.splitext(output)[0]
    plt.close(fig)

    tcols, thead, icols, ihead = [], [], [], []
    if spin:
        tcols, thead = _spin_columns(res, kinds, "T")
        icols, ihead = _spin_columns(res, kinds, "I")
        soc = SYSTYPE == "SOC"
        ncol = 3 if soc else 2
        fig, axs = plt.subplots(2, ncol, figsize=(5.5 * ncol, 8), squeeze=False)
        for row, (kind, width) in enumerate(zip(kinds, widths)):
            at, ai = axs[row][0], axs[row][-1]
            T, I, sres = res[kind]
            for name, c in (("up", "C3"), ("dn", "C0")):
                at.plot(energies - mu0, sres["N"][name], color=c, alpha=0.3, lw=2.5,
                        drawstyle="steps-mid", label=f"channels {SPIN_ARROW[name]} (bands)")
            at.plot(energies - mu0, T, color="0.6", lw=2.2, label="total")
            ai.plot(biases, I, color="0.6", lw=2.2, label="total")
            for k in _spin_keys_to_show():
                col, ls = SPIN_STYLE[k]
                if k in ("uu", "dd"):
                    at.plot(energies - mu0, sres["T"][k], color=col, ls=ls, lw=1,
                            label=SPIN_LABELS[k])
                ai.plot(biases, sres["I"][k], color=col, ls=ls, lw=1, label=SPIN_LABELS[k])
            at.set_title(f"{kind}, W = {width:g} Å: spin-conserving transmission", fontsize=10)
            ai.set_title(f"{kind}: spin-resolved current, T = {temperature:g} K", fontsize=10)
            at.set_xlabel(r"$E - E_F$ (eV)")
            at.set_ylabel(r"$T_{\sigma'\sigma}$ ($e^2/h$)")
            ai.set_xlabel("bias $V$ (V)")
            ai.set_ylabel(r"$I_{\sigma'\sigma}$ ($\mu$A)")
            panels = [at, ai]
            if soc:                  # spin-flip num painel próprio (escala própria, sem eixo duplo)
                af = axs[row][1]
                for k in ("ud", "du"):
                    col, ls = SPIN_STYLE[k]
                    af.plot(energies - mu0, sres["T"][k], color=col, ls=ls, lw=1,
                            label=SPIN_LABELS[k])
                af.set_title(f"{kind}: spin-flip transmission", fontsize=10)
                af.set_xlabel(r"$E - E_F$ (eV)")
                af.set_ylabel(r"$T_{\sigma'\sigma}$, $\sigma' \neq \sigma$ ($e^2/h$)")
                panels.append(af)
            for a in panels:
                a.axhline(0, color="k", lw=0.4)
                a.legend(fontsize=7, title=r"$\sigma\to\sigma'$ (lead 0 $\to$ 1)",
                         title_fontsize=7)
            for a in panels[::2]:
                a.axvline(0, color="k", ls=":", lw=0.6)
        fig.tight_layout()
        fig.savefig(base + "_spin.png", dpi=150)
        plt.close(fig)

    np.savetxt(base + "_T.dat",
               np.column_stack([energies - mu0, res["zigzag"][0], res["armchair"][0]]
                               + tcols),
               header="  ".join(["E-E_F(eV)  T_zigzag  T_armchair"] + thead)
                      + f"   hop_cutoff={HOP_CUTOFF} W={widths} L={length}"
                      + ("  [T_<fita>_<s'><s>: s = spin injetado (lead 0), "
                         f"s' = spin coletado (lead 1); projeção {projection}]" if spin else ""),
               fmt="%12.6f")
    np.savetxt(base + "_IV.dat",
               np.column_stack([biases, res["zigzag"][1], res["armchair"][1]] + icols),
               header="  ".join(["V(V)  I_zigzag(uA)  I_armchair(uA)"] + ihead)
                      + f"   T={temperature}K spin_degeneracy={2 if SYSTYPE == 'NP' else 1}",
               fmt="%12.6f")
    print(f"figura: {output}" + (f", {base}_spin.png" if spin else "")
          + f"; dados: {base}_T.dat, {base}_IV.dat")
    return energies, biases, res


# ======================================================================
# EXEMPLO: bandas de nanofitas zigzag e armchair
# ======================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Bandas e transmissão de nanofitas zigzag e armchair a partir do "
                    "modelo TB convertido (energias relativas a E_F).",
        epilog="exemplos:\n"
               "  python %(prog)s                      # bandas + transmissão\n"
               "  python %(prog)s --no-transport       # só bandas (rápido)\n"
               "  python %(prog)s --width-zz 30 --width-ac 25 -o fitas.png\n"
               "  python %(prog)s --emin -1.5 --emax 1.5 --ne 121 --nproc 8\n"
               "  python %(prog)s --iv                 # T(E) + curva I-V (0-3 V, cutoff 1e-3)\n"
               "  python %(prog)s --iv --vrange -1 2 --nv 31   # I-V de -1 a 2 V\n\n"
               "Saídas: figura (-o) e, com transporte, um .dat com E, T_zigzag, T_armchair.\n"
               "SP/SOC: T e I também por canal de spin (colunas extras; --no-spin desliga).\n"
               "Transmissão em unidades de e^2/h, sem fator de spin; a região de\n"
               "espalhamento é a própria fita (perfeita): edite make_two_terminal() para\n"
               "adicionar defeitos, potenciais etc.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width-zz", type=float, default=20.0, help="largura da fita zigzag em Å [20]")
    ap.add_argument("--width-ac", type=float, default=20.0, help="largura da fita armchair em Å [20]")
    ap.add_argument("--emin", type=float, default=-3.0, help="E mínima, eV rel. a E_F [-3]")
    ap.add_argument("--emax", type=float, default=3.0, help="E máxima, eV rel. a E_F [3]")
    ap.add_argument("--nk", type=int, default=201, help="número de pontos k nas bandas [201]")
    ap.add_argument("--no-transport", action="store_true", help="não calcula a transmissão")
    ap.add_argument("--ne", type=int, default=61, help="número de energias na transmissão [61]")
    ap.add_argument("--length", type=float, default=30.0,
                    help="comprimento da região de espalhamento em Å [30]")
    ap.add_argument("--nproc", type=int, default=None,
                    help="processos para a transmissão [metade dos núcleos lógicos]")
    ap.add_argument("--hop-cutoff", type=float, default=None,
                    help="descarta hoppings |H| < X eV (mais rápido, menos preciso) [0]")
    ap.add_argument("--ovl-cutoff", type=float, default=None,
                    help="base não ortogonal: descarta |S| < X [0; 1e-4 com --iv]")
    ap.add_argument("--iv", action="store_true",
                    help="em vez das bandas, calcula T(E) fino e a curva I-V das duas fitas "
                         "no intervalo --vrange (usa --hop-cutoff 1e-3 se não for dado)")
    ap.add_argument("--vrange", type=float, nargs=2, metavar=("VMIN", "VMAX"), default=None,
                    help="intervalo de bias do I-V em V, ex.: --vrange -1 2 [0 3]")
    ap.add_argument("--vmax", type=float, default=3.0,
                    help="bias máximo do I-V em V (equivale a --vrange 0 VMAX) [3]")
    ap.add_argument("--nv", type=int, default=61, help="número de pontos de bias [61]")
    ap.add_argument("--de", type=float, default=0.01, help="passo de energia de T(E) no I-V, eV [0.01]")
    ap.add_argument("--temperature", type=float, default=300.0, help="temperatura do I-V em K [300]")
    ap.add_argument("--no-spin", action="store_true",
                    help="SP/SOC: não resolve T e I por canal de spin (só o total)")
    ap.add_argument("--spin-projection", choices=["modes", "sym", "block"], default="modes",
                    help="SOC: definição das componentes de spin (modes: spin dos modos dos "
                         "leads, >= 0 e soma = T; sym/block: Gamma projetado na interface, "
                         "só para comparação) [modes]")
    ap.add_argument("-o", "--output", default=None,
                    help="figura de saída [bands_ribbons.png, ou iv_ribbons.png com --iv]")
    args = ap.parse_args()
    global HOP_CUTOFF
    global OVL_CUTOFF
    if args.hop_cutoff is not None:
        HOP_CUTOFF = args.hop_cutoff
    if args.ovl_cutoff is not None:
        OVL_CUTOFF = args.ovl_cutoff
    spin = has_spin_labels() and not args.no_spin
    if SYSTYPE != "NP" and not has_spin_labels() and not args.no_spin:
        print(f"[aviso] {SYSTYPE} sem rótulos up:/dn: nos orbitais: transporte só total")
    if args.iv:
        if args.hop_cutoff is None:
            HOP_CUTOFF = 1e-3
        if args.ovl_cutoff is None and NONORTHOGONAL:
            OVL_CUTOFF = 1e-4
        vmin, vmax = args.vrange if args.vrange else (0.0, args.vmax)
        run_iv((args.width_zz, args.width_ac), vmax, args.nv, args.de, args.length,
               args.temperature, args.nproc, args.output or "iv_ribbons.png", vmin=vmin,
               spin=spin, projection=args.spin_projection)
        return
    if args.output is None:
        args.output = "bands_ribbons.png"

    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    transport = not args.no_transport
    if transport:
        fig, axd = plt.subplot_mosaic([["b0", "t0", "b1", "t1"], ["s0", "s0", "s1", "s1"]],
                                      figsize=(13, 9),
                                      gridspec_kw=dict(width_ratios=[3, 1.1, 3, 1.1],
                                                       height_ratios=[3, 1.3]))
    else:
        fig, axd = plt.subplot_mosaic([["b0", "b1"], ["s0", "s1"]], figsize=(10, 9),
                                      gridspec_kw=dict(height_ratios=[3, 1.3]))
    ylab = r"$E - E_F$ (eV)" if SHIFT_TO_FERMI else "E (eV)"
    eref = 0.0 if SHIFT_TO_FERMI else EFERMI
    energies = np.linspace(args.emin, args.emax, args.ne) + (0 if SHIFT_TO_FERMI else EFERMI)
    tdata, sdata = {}, {}

    for col, (kind, width) in enumerate([("zigzag", args.width_zz),
                                         ("armchair", args.width_ac)]):
        rib = make_ribbon(kind, width)
        info = ribbon_info(rib)
        ks, E = ribbon_bands(rib, args.nk)
        a = np.linalg.norm(_int_vec(ribbon_period(kind)))
        ax = axd[f"b{col}"]
        ax.plot(ks / a, E, "k-", lw=0.6)
        ax.axhline(eref, color="r", ls="--", lw=0.6)
        ax.set_ylim(args.emin + eref, args.emax + eref)
        ax.set_xlim(ks[0] / a, ks[-1] / a)
        ax.set_xlabel(r"$k$ (Å$^{-1}$)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{kind}, W = {width:g} Å, period {a:.3f} Å\n"
                     + ", ".join(f"{k}:{v}" for k, v in info.items()), fontsize=9)
        plot_ribbon_top(axd[f"s{col}"], rib, kind)
        print(f"{kind}: período {ribbon_period(kind)} ({a:.4f} Å), sites/célula {info}, "
              f"{E.shape[1]} bandas")

        if transport:
            t0 = time.time()
            syst = make_two_terminal(kind, width, args.length).finalized()
            if spin:
                sdata[kind] = spin_conductance(syst, energies, args.nproc, args.spin_projection)
                T = sdata[kind]["total"]
            else:
                T = conductance(syst, energies, args.nproc)
            nch = channel_count(rib, energies)
            tdata[kind] = T
            axt = axd[f"t{col}"]
            axt.plot(nch, energies, "-", color="0.75", lw=2.5, drawstyle="steps-mid",
                     label="open channels (bands)")
            axt.plot(T, energies, "k.-", lw=0.8, ms=3, label="Kwant")
            if spin:
                nsp = spin_channel_count(rib, energies)
                sdata[kind]["n_up"], sdata[kind]["n_dn"] = nsp["up"], nsp["dn"]
                for name, c in (("up", "C3"), ("dn", "C0")):
                    axt.plot(nsp[name], energies, "-", color=c, alpha=0.3, lw=2.5,
                             drawstyle="steps-mid",
                             label=f"channels {SPIN_ARROW[name]} (bands)")
                for k in _spin_keys_to_show():
                    c, ls = SPIN_STYLE[k]
                    axt.plot(sdata[kind][k], energies, color=c, ls=ls, lw=0.9,
                             label=SPIN_LABELS[k])
                print(f"{kind}: max |T - soma das componentes de spin| = "
                      f"{np.abs(sdata[kind]['mix']).max():.2e}")
            axt.axhline(eref, color="r", ls="--", lw=0.6)
            axt.set_ylim(args.emin + eref, args.emax + eref)
            axt.set_xlim(left=0)
            axt.set_xlabel(r"$T$ ($e^2/h$)")
            axt.tick_params(labelleft=False)
            axt.set_title("transmission", fontsize=9)
            axt.legend(fontsize=6, loc="upper right")
            print(f"{kind}: transmissão em {len(energies)} energias "
                  f"({time.time() - t0:.0f} s); "
                  f"max |T - canais| = {np.abs(T - nch).max():.2e}")

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"figura salva em {args.output}")
    if transport:
        dat = os.path.splitext(args.output)[0] + "_transmission.dat"
        cols, head = [], []
        for kind in sdata:
            for k in _spin_keys_to_show():
                cols.append(sdata[kind][k])
                head.append(f"T_{kind}_{k}")
            for name in ("up", "dn"):
                cols.append(sdata[kind]["n_" + name])
                head.append(f"N_{kind}_{name}")
        np.savetxt(dat, np.column_stack([energies, tdata["zigzag"], tdata["armchair"]] + cols),
                   header=f"E{'-E_F' if SHIFT_TO_FERMI else ''}(eV)  T_zigzag(e^2/h)  "
                          + "  ".join(["T_armchair(e^2/h)"] + head)
                          + f"   W_zz={args.width_zz} W_ac={args.width_ac} L={args.length} A"
                          + (f"  [T_<fita>_<s'><s>: s = spin injetado (lead 0), s' = spin "
                             f"coletado (lead 1); projeção {args.spin_projection}]"
                             if sdata else ""), fmt="%12.6f")
        print(f"transmissão salva em {dat}")


if __name__ == "__main__":
    main()
'''
