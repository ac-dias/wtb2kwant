"""Teste de regressão com o cálculo MoS2 NP (Wannier90/VASP).

Roda com:  module load kwant && python -m pytest tests -q
É pulado se os dados não estiverem disponíveis.
"""
import importlib.util
import os

import numpy as np
import pytest

from wtb2kwant.codegen import generate_script
from wtb2kwant.model import from_wannier
from wtb2kwant.readers import read_wout, read_wtb_wannier

# Dados do MoS2 no próprio repositório (sobrescreva com WTB2KWANT_MOS2_DIR)
D = os.environ.get("WTB2KWANT_MOS2_DIR",
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mos2_np"))
pytestmark = pytest.mark.skipif(not os.path.isdir(D), reason="dados MoS2 indisponíveis")


def _ref():
    kp = np.loadtxt(f"{D}/wannier90_band.kpt", skiprows=1)[:, :3]
    ref = np.loadtxt(f"{D}/wannier90_band.dat")[:, 1].reshape(-1, len(kp)).T
    return kp, ref


def _model(cutoff=1e-5):
    return from_wannier(read_wtb_wannier(f"{D}/tb_hr.dat"),
                        [read_wout(f"{D}/wannier90.wout")], cutoff=cutoff)


def test_bulk_bands_match_wannier90():
    kp, ref = _ref()
    m = _model(cutoff=0.0)
    assert m.periodic_dims == [0, 1]
    assert np.abs(m.bands(kp) - ref).max() < 1e-4


def test_generated_script(tmp_path):
    kwant = pytest.importorskip("kwant")
    m = _model()
    path = tmp_path / "km.py"
    path.write_text(generate_script(m, ["test"]))
    spec = importlib.util.spec_from_file_location("km", path)
    km = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(km)

    kp, ref = _ref()
    E = np.array([np.linalg.eigvalsh(km.bulk_hk(k[:2])) for k in kp])
    assert np.abs(E - (ref - km.EFERMI)).max() < 1e-3

    # orientação dos hoppings no Kwant: syst[s(0), t(R)] == <s,0|H|t,R>
    lat = km.make_lattice()
    subs = lat.sublattices
    fin = kwant.Builder()
    fin.fill(km.make_bulk(lat), lambda s: s.pos[0] ** 2 + s.pos[1] ** 2 < 30 ** 2, subs[0](0, 0))
    for R, s, t, blk in km.HOPPINGS[::10]:
        try:
            v = fin[subs[s](0, 0), subs[t](*R)]
        except KeyError:
            continue
        assert np.allclose(v, blk)

    # ribbon_hk (período primitivo) == kwant Bands na supercélula, desdobrado
    kind, W = "armchair", 12.0
    N = km.min_supercell(kind, W)
    hk = km.ribbon_hk(km.make_ribbon(kind, W))
    b = kwant.physics.Bands(km.make_ribbon(kind, W, supercell=N).finalized())
    K = 0.7
    Ep = np.sort(np.concatenate([np.linalg.eigvalsh(hk((K + 2 * np.pi * j) / N))
                                 for j in range(N)]))
    assert np.allclose(np.sort(b(K)), Ep, atol=1e-9)


def test_parallel_conductance_and_iv(tmp_path):
    """conductance() paralelo funciona em módulo carregado via importlib e bate com
    o serial; landauer_current reproduz o caso analítico T = 1."""
    pytest.importorskip("kwant")
    path = tmp_path / "km2.py"
    path.write_text(generate_script(_model(), ["test"]))
    spec = importlib.util.spec_from_file_location("km2", path)
    km = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(km)
    km.HOP_CUTOFF = 3e-3                      # sistema pequeno/barato
    syst = km.make_two_terminal("armchair", 12.0, 20.0).finalized()
    E = np.linspace(-1, 1, 7)
    assert np.allclose(km.conductance(syst, E, nproc=3), km.conductance(syst, E, nproc=1))
    Eg = np.linspace(-2, 2, 4001)
    I = km.landauer_current(Eg, np.ones_like(Eg), [1.0])
    assert abs(I[0] - 2 * 38.7404585) < 1e-3
    # bias simétrico: I(-V) = -I(V) para qualquer T(E) (intervalo com bias negativo)
    Tr = 1 + np.sin(3 * Eg) ** 2
    V = np.array([-1.5, -0.4, 0.4, 1.5])
    Iv = km.landauer_current(Eg, Tr, V)
    assert np.allclose(Iv[:2], -Iv[:1:-1])
