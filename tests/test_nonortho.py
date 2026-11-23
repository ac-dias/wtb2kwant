"""Testes da base não ortogonal (MoS2 do YAeHMOP) e dos leitores de estrutura.

Roda com:  python -m pytest tests -q
"""
import importlib.util
import os

import numpy as np
import pytest

from wtb2kwant.codegen import generate_script
from wtb2kwant.model import from_wtb_ham, orbital_atoms_from_structure
from wtb2kwant.readers import read_basis, read_structure, read_wtb_ham

D = os.environ.get("WTB2KWANT_MOS2_YAEHMOP_DIR",
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mos2_yaehmop_np"))
needs_data = pytest.mark.skipif(not os.path.isdir(D), reason="dados YAeHMOP indisponíveis")


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


@needs_data
def test_nonortho_bulk_bands_match_yaehmop():
    ham = read_wtb_ham(f"{D}/huckel-NP.dat")
    m = from_wtb_ham(ham, basis=read_basis(f"{D}/basis_set-NP"), cutoff=0, s_cutoff=0)
    ks, ref = _yaehmop_bands(f"{D}/MoS2.bind.band")
    E = m.bands(ks)
    E -= E[:, :9].max()
    ref = ref - ref[:, :9].max()
    win = (ref > -6) & (ref < 4)
    assert np.abs(E - ref)[win].max() < 1e-4
    assert [len(s.orbitals) for s in m.sites] == [9, 4, 4]


@needs_data
def test_nonortho_script_transmission_equals_channels(tmp_path):
    """H' = H - E(S-1) no Kwant: T da fita perfeita = nº de canais das bandas generalizadas."""
    pytest.importorskip("kwant")
    ham = read_wtb_ham(f"{D}/huckel-NP.dat")
    m = from_wtb_ham(ham, basis=read_basis(f"{D}/basis_set-NP"), cutoff=3e-3, s_cutoff=3e-4,
                     efermi=-0.05)
    path = tmp_path / "km_no.py"
    path.write_text(generate_script(m, ["test"]))
    spec = importlib.util.spec_from_file_location("km_no", path)
    km = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(km)
    assert km.NONORTHOGONAL
    syst = km.make_two_terminal("zigzag", 8.0, 12.0).finalized()
    E = [-1.5, -0.3, 0.8]
    T = km.conductance(syst, E, nproc=3)
    nch = km.channel_count(km.make_ribbon("zigzag", 8.0), E)
    assert np.allclose(T, nch, atol=1e-6)


def test_structure_readers_and_orbital_map(tmp_path):
    (tmp_path / "POSCAR").write_text(
        "MoS2\n1.0\n3.16 0 0\n-1.58 2.737 0\n0 0 16\nMo S\n1 2\nDirect\n"
        "0.333333 0.666667 0.5\n0.666667 0.333333 0.40\n0.666667 0.333333 0.60\n")
    (tmp_path / "geo.gen").write_text(
        "3 S\nMo S\n1 1 0.0 1.826 8.0\n2 2 1.58 0.913 6.43\n3 2 1.58 0.913 9.57\n"
        "0 0 0\n3.16 0 0\n-1.58 2.737 0\n0 0 16\n")
    (tmp_path / "geometry.in").write_text(
        "lattice_vector 3.16 0 0\nlattice_vector -1.58 2.737 0\nlattice_vector 0 0 16\n"
        "atom 0.0 1.826 8.0 Mo\natom_frac 0.666667 0.333333 0.40 S\natom 1.58 0.913 9.57 S\n")
    for name in ("POSCAR", "geo.gen", "geometry.in"):
        st = read_structure(str(tmp_path / name))
        assert st.species == ["Mo", "S", "S"]
        assert st.lattice.shape == (3, 3)
        sp, pos, orb_atom, labels = orbital_atoms_from_structure(st, 34, True, "Mo=9,S=4")
        assert orb_atom[:9] == [0] * 9 and orb_atom[9:13] == [1] * 4 and orb_atom[17:26] == [0] * 9
        assert labels[0].startswith("up:") and labels[17].startswith("dn:")
    with pytest.raises(ValueError):
        orbital_atoms_from_structure(st, 17, False, None)       # 2 espécies sem --norb
    with pytest.raises(ValueError):
        orbital_atoms_from_structure(st, 17, False, "Mo=9,S=3")  # soma errada
