"""Testes de SP/SOC: bandas SOC contra o YAeHMOP e transporte resolvido em spin.

Roda com:  python -m pytest tests -q
"""
import importlib.util
import os

import numpy as np
import pytest

from wtb2kwant.codegen import generate_script
from wtb2kwant.model import Site, TBModel, from_wtb_ham
from wtb2kwant.readers import read_basis, read_wtb_ham

HERE = os.path.dirname(os.path.abspath(__file__))
DSOC = os.environ.get("WTB2KWANT_MOS2_SOC_DIR", os.path.join(HERE, "data", "mos2_yaehmop_soc"))
DNP = os.environ.get("WTB2KWANT_MOS2_YAEHMOP_DIR", os.path.join(HERE, "data", "mos2_yaehmop_np"))
needs_soc = pytest.mark.skipif(not os.path.isdir(DSOC), reason="dados SOC indisponíveis")
needs_np = pytest.mark.skipif(not os.path.isdir(DNP), reason="dados YAeHMOP NP indisponíveis")


def _yaehmop_bands(path):
    ks, ev, cur = [], [], None
    for ln in open(path):
        if ln.startswith("; K point:"):
            if cur is not None:
                ev.append(cur)
            ks.append([float(x) for x in ln.split(":")[1].split()])
            cur = []
        elif ln.startswith("#END"):
            break
        elif cur is not None and ln.strip() and not ln.startswith(";"):
            cur.append(float(ln))
    ev.append(cur)
    return np.array(ks), np.array(ev)


def _load(model, path, name):
    path.write_text(generate_script(model, ["test"]))
    spec = importlib.util.spec_from_file_location(name, path)
    km = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(km)
    return km


def _soc_model(**kw):
    ham = read_wtb_ham(f"{DSOC}/huckel-SOC.dat")
    return from_wtb_ham(ham, basis=read_basis(f"{DSOC}/basis_set-SOC"), **kw)


@needs_soc
def test_soc_bands_match_yaehmop():
    m = _soc_model(cutoff=0, s_cutoff=0)
    assert [len(s.orbitals) for s in m.sites] == [18, 8, 8]
    assert m.sites[0].orbitals[0].startswith("up:") and m.sites[0].orbitals[9].startswith("dn:")
    ks, ref = _yaehmop_bands(f"{DSOC}/MoS2.bind.band")
    E = m.bands(ks)
    E -= E[:, :18].max()
    ref = ref - ref[:, :18].max()
    win = (ref > -6) & (ref < 4)
    assert np.abs(E - ref)[win].max() < 1e-4
    iK = int(np.argmin(np.linalg.norm(ks - [1 / 3, 1 / 3, 0], axis=1)))
    split = E[iK, 17] - E[iK, 16]                 # desdobramento SOC do topo da valência em K
    assert abs(split - (ref[iK, 17] - ref[iK, 16])) < 1e-5
    assert abs(split - 0.165) < 1e-3


@needs_soc
def test_soc_spin_transmission(tmp_path):
    """SOC: T = canais; "modes" (padrão): componentes >= 0, soma = T, spin-flip pequeno e
    T coletado em cada spin = canais por spin; "sym": soma = T; "block": componentes >= 0."""
    pytest.importorskip("kwant")
    km = _load(_soc_model(cutoff=3e-3, s_cutoff=3e-4, efermi=-0.05), tmp_path / "km_soc.py",
               "km_soc")
    assert km.SYSTYPE == "SOC" and km.has_spin_labels()
    syst = km.make_two_terminal("zigzag", 8.0, 12.0).finalized()
    E = [-1.5, -0.3, 1.2]
    rib = km.make_ribbon("zigzag", 8.0)
    nch = km.channel_count(rib, E)
    r = km.spin_conductance(syst, E, nproc=3)                       # padrão: "modes"
    assert np.allclose(r["total"], nch, atol=1e-6)
    assert np.allclose(r["total"], km.conductance(syst, E, nproc=3))
    for k in km.SPIN_KEYS:
        assert np.all(r[k] >= -1e-10)
    assert np.allclose(r["mix"], 0, atol=1e-8)
    assert np.all(r["ud"] + r["du"] < 0.05 * r["total"])
    nsp = km.spin_channel_count(rib, E, nk=2001)
    assert np.allclose(r["uu"] + r["ud"], nsp["up"], atol=0.05)
    assert np.allclose(r["dd"] + r["du"], nsp["dn"], atol=0.05)
    r1 = km.spin_conductance(syst, E, nproc=1)
    assert all(np.allclose(r[k], r1[k]) for k in r)
    rs = km.spin_conductance(syst, E, nproc=1, projection="sym")
    assert np.allclose(rs["mix"], 0, atol=1e-8)
    rb = km.spin_conductance(syst, E, nproc=1, projection="block")
    assert all(np.all(rb[k] >= -1e-10) for k in km.SPIN_KEYS)


def _magnetic_sp(m, delta):
    """SP sintético a partir de um modelo NP não ortogonal: H_up = H - delta S,
    H_dn = H + delta S (bandas up = NP - delta, dn = NP + delta)."""
    sl = m.site_slices()
    sites, idx_up, idx_dn, i = [], [], [], 0
    for s, sli in zip(m.sites, sl):
        n = len(s.orbitals)
        sites.append(Site(s.name, s.species, s.position,
                          ["up:" + o for o in s.orbitals] + ["dn:" + o for o in s.orbitals]))
        idx_up += list(range(i, i + n))
        idx_dn += list(range(i + n, i + 2 * n))
        i += 2 * n
    up, dn = np.array(idx_up), np.array(idx_dn)
    Z = np.zeros((m.norb, m.norb), complex)
    H, S = {}, {}
    for R in set(m.hoppings) | set(m.overlaps):
        h, o = m.hoppings.get(R, Z), m.overlaps.get(R, Z)
        Hn, Sn = np.zeros((i, i), complex), np.zeros((i, i), complex)
        Hn[np.ix_(up, up)] = h - delta * o
        Hn[np.ix_(dn, dn)] = h + delta * o
        Sn[np.ix_(up, up)] = o
        Sn[np.ix_(dn, dn)] = o
        H[R], S[R] = Hn, Sn
    return TBModel(systype="SP", efermi=m.efermi, scissors=0.0, lattice=m.lattice,
                   periodic_dims=m.periodic_dims, sites=sites, hoppings=H, overlaps=S,
                   notes=["SP sintético (teste)"])


@needs_np
def test_sp_spin_channels_magnetic(tmp_path):
    """SP com troca +-delta: T_up(E) = T_NP(E + delta), T_dn(E) = T_NP(E - delta), sem spin-flip."""
    pytest.importorskip("kwant")
    ham = read_wtb_ham(f"{DNP}/huckel-NP.dat")
    mnp = from_wtb_ham(ham, basis=read_basis(f"{DNP}/basis_set-NP"), cutoff=3e-3,
                       s_cutoff=3e-4, efermi=-0.05)
    delta = 0.4
    knp = _load(mnp, tmp_path / "km_np.py", "km_np")
    ksp = _load(_magnetic_sp(mnp, delta), tmp_path / "km_sp.py", "km_sp")
    assert ksp.has_spin_labels() and not knp.has_spin_labels()
    E = np.array([-1.2, -0.2, 1.0])
    ssp = ksp.make_two_terminal("zigzag", 8.0, 12.0).finalized()
    snp = knp.make_two_terminal("zigzag", 8.0, 12.0).finalized()
    r = ksp.spin_conductance(ssp, E, nproc=3)
    assert np.allclose(r["ud"], 0, atol=1e-10) and np.allclose(r["du"], 0, atol=1e-10)
    assert np.allclose(r["mix"], 0, atol=1e-8)
    assert np.allclose(r["uu"], knp.conductance(snp, E + delta, nproc=3), atol=1e-6)
    assert np.allclose(r["dd"], knp.conductance(snp, E - delta, nproc=3), atol=1e-6)
    assert not np.allclose(r["uu"], r["dd"])                 # os canais são de fato diferentes
    # correntes por spin: I_up + I_dn = I total (degenerescência 1 em cada componente)
    Ef = np.linspace(-1.0, 1.0, 81)
    Tf = ksp.spin_conductance(ssp, Ef, nproc=3)
    V = [0.3, 0.8]
    Iu = ksp.landauer_current(Ef, Tf["uu"], V, spin_degeneracy=1)
    Id = ksp.landauer_current(Ef, Tf["dd"], V, spin_degeneracy=1)
    It = ksp.landauer_current(Ef, Tf["total"], V)
    assert np.allclose(Iu + Id, It)


@needs_np
def test_sp_conservation_law_route(tmp_path):
    """Conferência independente dos canais SP: lei de conservação S_z do Kwant nos leads e
    smatrix.transmission((1, s'), (0, s)) ("conservation") = projeção "modes" = T_NP(E -+ delta)."""
    pytest.importorskip("kwant")
    ham = read_wtb_ham(f"{DNP}/huckel-NP.dat")
    mnp = from_wtb_ham(ham, basis=read_basis(f"{DNP}/basis_set-NP"), cutoff=3e-3,
                       s_cutoff=3e-4, efermi=-0.05)
    delta = 0.3
    knp = _load(mnp, tmp_path / "km_np2.py", "km_np2")
    ksp = _load(_magnetic_sp(mnp, delta), tmp_path / "km_sp2.py", "km_sp2")
    E = np.array([-1.3, -0.4, 0.2, 1.1])
    for kind, w in (("zigzag", 8.0), ("armchair", 9.0)):
        sc = ksp.make_two_terminal(kind, w, 12.0, spin_conservation=True).finalized()
        sm = ksp.make_two_terminal(kind, w, 12.0).finalized()
        rc = ksp.spin_conductance(sc, E, nproc=2, projection="conservation")
        rm = ksp.spin_conductance(sm, E, nproc=2)
        for k in ("total",) + ksp.SPIN_KEYS:
            assert np.allclose(rc[k], rm[k], atol=1e-8), (kind, k, rc[k], rm[k])
        assert np.allclose(rc["ud"], 0, atol=1e-12) and np.allclose(rc["du"], 0, atol=1e-12)
        snp = knp.make_two_terminal(kind, w, 12.0).finalized()
        assert np.allclose(rc["uu"], knp.conductance(snp, E + delta, nproc=2), atol=1e-6)
        assert np.allclose(rc["dd"], knp.conductance(snp, E - delta, nproc=2), atol=1e-6)
    # sem lei de conservação nos leads a rota recusa; e ela só vale para SP
    with pytest.raises(ValueError):
        ksp.spin_transmission(sm, 0.2, "conservation")
    with pytest.raises(ValueError):
        knp.make_two_terminal("zigzag", 8.0, 12.0, spin_conservation=True)


def test_wannier_soc_spin_order_labels():
    """SOC do Wannier90: --soc-spin-order dá os rótulos up:/dn: (blocos ou alternados)."""
    from wtb2kwant.model import from_wannier
    from wtb2kwant.readers import HrBlock, WoutData, WtbWannier
    lat = np.diag([3.0, 3.0, 20.0])
    nw = 4
    rv = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    ham = np.zeros((len(rv), nw, nw), complex)
    ham[0] = np.diag([0.1, 0.1, -0.1, -0.1])
    ham[1:] = -0.5 * np.eye(nw)
    wtb = WtbWannier("SOC", 0.0, 0.0, lat, [HrBlock(nw, rv, np.ones(len(rv)), ham)])
    wo = WoutData(lat, ["X"], np.zeros((1, 3)), np.zeros((nw, 3)), np.ones(nw))
    lab = from_wannier(wtb, [wo], soc_spin_order="blocks").sites[0].orbitals
    assert [x[:3] for x in lab] == ["up:", "up:", "dn:", "dn:"]
    lab = from_wannier(wtb, [wo], soc_spin_order="interleaved").sites[0].orbitals
    assert [x[:3] for x in lab] == ["up:", "dn:", "up:", "dn:"]
    assert from_wannier(wtb, [wo]).sites[0].orbitals[0] == "wf1"     # sem opção: sem spin
    with pytest.raises(ValueError):
        from_wannier(WtbWannier("NP", 0.0, 0.0, lat, wtb.blocks), [wo], soc_spin_order="blocks")


@needs_np
def test_spin_channel_count(tmp_path):
    """Canais por spin das bandas: SP exato (= T_uu, T_dd); SOC fracionário com up + dn = total."""
    pytest.importorskip("kwant")
    ham = read_wtb_ham(f"{DNP}/huckel-NP.dat")
    mnp = from_wtb_ham(ham, basis=read_basis(f"{DNP}/basis_set-NP"), cutoff=3e-3,
                       s_cutoff=3e-4, efermi=-0.05)
    ksp = _load(_magnetic_sp(mnp, 0.4), tmp_path / "km_spc.py", "km_spc")
    E = np.array([-1.2, -0.2, 1.0])
    rib = ksp.make_ribbon("zigzag", 8.0)
    n = ksp.spin_channel_count(rib, E, nk=2001)
    assert np.array_equal(n["up"], [1, 1, 1]) and np.array_equal(n["dn"], [2, 3, 1])
    assert np.array_equal(n["up"] + n["dn"], ksp.channel_count(rib, E, nk=2001))
    if os.path.isdir(DSOC):
        ks = _load(_soc_model(cutoff=3e-3, s_cutoff=3e-4, efermi=-0.05), tmp_path / "km_scc.py",
                   "km_scc")
        rib = ks.make_ribbon("zigzag", 8.0)
        E = np.array([-0.3, 1.2])
        n = ks.spin_channel_count(rib, E, nk=2001)
        assert np.allclose(n["up"] + n["dn"], ks.channel_count(rib, E, nk=2001))
        assert np.allclose(n["up"], [2, 1], atol=0.01)       # spin quase conservado ali
