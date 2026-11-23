#!/usr/bin/env python
"""Validação completa com o MoS2 NP; gera relatório e figuras em tests/results/mos2_np/.

Uso:
    python tests/validate_mos2_np.py            # tudo (~25 min com 12 núcleos)
    python tests/validate_mos2_np.py --quick    # sem T(E) no exemplo e sem I-V (~5 min)
    python tests/validate_mos2_np.py --skip-iv  # sem a curva I-V
"""
import argparse
import datetime
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import kwant  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wtb2kwant.codegen import generate_script  # noqa: E402
from wtb2kwant.model import from_wannier  # noqa: E402
from wtb2kwant.readers import read_wout, read_wtb_wannier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="pula T(E) do exemplo e o I-V")
ap.add_argument("--skip-iv", action="store_true", help="pula a curva I-V")
cli = ap.parse_args()

# Dados do MoS2 no próprio repositório (sobrescreva com WTB2KWANT_MOS2_DIR)
D = os.environ.get("WTB2KWANT_MOS2_DIR",
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mos2_np"))
OUT = os.path.join(HERE, "results", "mos2_np")
os.makedirs(OUT, exist_ok=True)

rep = []
w = rep.append
w("# Validação wtb2kwant: MoS2 monocamada (NP, Wannier90/VASP)")
w("")
w(f"Gerado em {datetime.datetime.now():%Y-%m-%d %H:%M} por `tests/validate_mos2_np.py`.")
_Drel = os.path.relpath(D, os.path.dirname(HERE))
w(f"Dados: `{_Drel}` (`tb_hr.dat` + `wannier90.wout`). Referência: `wannier90_band.dat`.")
w("")

wtb = read_wtb_wannier(f"{D}/tb_hr.dat")
wo = read_wout(f"{D}/wannier90.wout")
kp = np.loadtxt(f"{D}/wannier90_band.kpt", skiprows=1)[:, :3]
bd = np.loadtxt(f"{D}/wannier90_band.dat")
nk = len(kp)
xk = bd[:nk, 0]
ref = bd[:, 1].reshape(-1, nk).T
labels = [ln.split() for ln in open(f"{D}/wannier90_band.labelinfo.dat")]

# ---------------------------------------------------------------- 1. leitura
w("## 1. Leitura e folding dos centros")
w("")
w(f"- Sistema {wtb.systype}, E_F = {wtb.efermi} eV, tesoura = {wtb.scissors} eV, "
  f"num_wann = {wtb.blocks[0].num_wann}, nrpts = {len(wtb.blocks[0].rvecs)}")
m0 = from_wannier(wtb, [wo], cutoff=0.0)
w("")
w("| WF | centro (.wout) (Å) | site | deslocamento n |")
w("|---|---|---|---|")
for c, cen in enumerate(wo.centres):
    # recomputa a associação para exibir
    best = None
    for a, pa in enumerate(m0.sites):
        df = (cen - pa.position) @ np.linalg.inv(wtb.lattice)
        n = np.array([round(df[0]), round(df[1]), 0])
        d = np.linalg.norm(cen - pa.position - n @ wtb.lattice)
        if best is None or d < best[0]:
            best = (d, pa.name, n)
    w(f"| wf{c + 1} | ({cen[0]:.3f}, {cen[1]:.3f}, {cen[2]:.3f}) | {best[1]} "
      f"(d = {best[0]:.3f} Å) | {tuple(int(x) for x in best[2])} |")
w("")
for n in m0.notes:
    w(f"- {n}")
w("")

# ---------------------------------------------------------------- 2. bandas bulk
w("## 2. Bandas do bulk: modelo convertido × Wannier90")
w("")
w("| cutoff (eV) | elementos mantidos | vetores R | erro máx. (meV) |")
w("|---|---|---|---|")
for cut in [0.0, 1e-5, 3e-5, 1e-4, 1e-3]:
    m = from_wannier(wtb, [wo], cutoff=cut)
    E = m.bands(kp)
    w(f"| {cut:g} | {m.notes[1].split(': ')[1].split(' elementos')[0]} | "
      f"{len(m.hoppings)} | {1e3 * np.abs(E - ref).max():.3f} |")
w("")

model = from_wannier(wtb, [wo], cutoff=1e-5)
path = os.path.join(OUT, "kwant_model_test.py")
with open(path, "w") as f:
    f.write(generate_script(model, [f"{_Drel}/tb_hr.dat", f"{_Drel}/wannier90.wout"]))
spec = importlib.util.spec_from_file_location("km", path)
km = importlib.util.module_from_spec(spec)
spec.loader.exec_module(km)

Egen = np.array([np.linalg.eigvalsh(km.bulk_hk(k[:2])) for k in kp]) + km.EFERMI
w(f"- Script gerado (cutoff 1e-5), `bulk_hk`: erro máx. = "
  f"{1e3 * np.abs(Egen - ref).max():.3f} meV")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw=dict(width_ratios=[2, 1]))
ax1.plot(xk, ref - km.EFERMI, "-", color="0.6", lw=2.5)
ax1.plot(xk, Egen - km.EFERMI, "r--", lw=0.9)
ax1.plot([], [], "-", color="0.6", lw=2.5, label="Wannier90 (wannier90_band.dat)")
ax1.plot([], [], "r--", lw=0.9, label="generated Kwant script (bulk_hk)")
for lab in labels:
    ax1.axvline(float(lab[2]), color="k", lw=0.4)
ax1.set_xticks([float(lab[2]) for lab in labels])
ax1.set_xticklabels([lab[0].replace("G", "Γ") for lab in labels])
ax1.set_xlim(xk[0], xk[-1])
ax1.axhline(0, color="b", ls=":", lw=0.6)
ax1.set_ylabel(r"$E - E_F$ (eV)")
ax1.legend(fontsize=8, loc="lower right")
ax1.set_title("MoS$_2$ bulk bands")
ax2.semilogy(xk, np.abs(Egen - ref).max(axis=1) * 1e3 + 1e-6, "k-", lw=0.8)
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(ax1.get_xticklabels())
ax2.set_xlim(xk[0], xk[-1])
ax2.set_ylabel("max. error per k-point (meV)")
ax2.set_title("deviation (cutoff 1e-5 eV)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bulk_bands_vs_w90.png"), dpi=150)
w("")
w("![bulk](bulk_bands_vs_w90.png)")
w("")

# ---------------------------------------------------------------- 3. Kwant
w("## 3. Convenções no Kwant")
w("")
lat = km.make_lattice()
subs = lat.sublattices
fin = kwant.Builder()
fin.fill(km.make_bulk(lat), lambda s: s.pos[0] ** 2 + s.pos[1] ** 2 < 40 ** 2, subs[0](0, 0))
bad = ntest = 0
for R, s, t, blk in km.HOPPINGS:
    try:
        v = fin[subs[s](0, 0), subs[t](*R)]
    except KeyError:
        continue
    ntest += 1
    bad += not np.allclose(v, blk)
w(f"- Orientação `syst[s(0), t(R)] == <s,0|H|t,R>` em um floco de raio 40 Å: "
  f"{ntest - bad}/{ntest} blocos corretos (de {len(km.HOPPINGS)} no total).")
fw = kwant.wraparound.wraparound(km.make_bulk(lat)).finalized()
dev = 0
for k in kp[::10]:
    H = fw.hamiltonian_submatrix(params=dict(k_x=2 * np.pi * k[0], k_y=2 * np.pi * k[1]))
    dev = max(dev, np.abs(np.linalg.eigvalsh(H) - np.linalg.eigvalsh(km.bulk_hk(k[:2]))).max())
w(f"- `kwant.wraparound` do builder bulk × `bulk_hk`: erro máx. = {dev:.1e} eV")
w("")

# ---------------------------------------------------------------- 4. fitas
w("## 4. Nanofitas (W = 20 Å)")
w("")
w("| tipo | período | sites/célula | supercélula p/ leads (N) | Kwant Bands(N·T) × ribbon_hk desdobrado |")
w("|---|---|---|---|---|")
for kind in ["zigzag", "armchair"]:
    rib = km.make_ribbon(kind, 20.0)
    hk = km.ribbon_hk(rib)
    N = km.min_supercell(kind, 20.0)
    b = kwant.physics.Bands(km.make_ribbon(kind, 20.0, supercell=N).finalized())
    dev = 0
    for K in [0.0, 0.9, -2.1]:
        Ep = np.sort(np.concatenate([np.linalg.eigvalsh(hk((K + 2 * np.pi * j) / N))
                                     for j in range(N)]))
        dev = max(dev, np.abs(np.sort(b(K)) - Ep).max())
    a = np.linalg.norm(km._int_vec(km.ribbon_period(kind)))
    w(f"| {kind} | {km.ribbon_period(kind)} = {a:.4f} Å | {km.ribbon_info(rib)} | {N} | {dev:.1e} eV |")
w("")
old = sys.argv
sys.argv = ["km", "-o", os.path.join(OUT, "bands_ribbons.png")] + (["--no-transport"] if cli.quick else [])
km.main()
sys.argv = old
w("![fitas](bands_ribbons.png)")
w("")

# ---------------------------------------------------------------- 5. transporte
w("## 5. Transporte (fita perfeita, W = 20 Å, região de espalhamento de 30 Å)")
w("")
w("Numa fita perfeita a transmissão deve ser inteira e igual ao número de bandas que "
  "cruzam a energia com velocidade positiva.")
w("")
energies = [-0.5, 0.0, 0.2, 1.0]
w("| tipo | " + " | ".join(f"T(E−E_F = {e:+.1f} eV)" for e in energies) + " | tempo |")
w("|---|" + "---|" * (len(energies) + 1))
for kind in ["zigzag", "armchair"]:
    t0 = time.time()
    syst = km.make_two_terminal(kind, width=20.0, length=30.0).finalized()
    T = km.conductance(syst, energies)
    hk = km.ribbon_hk(km.make_ribbon(kind, 20.0))
    # contagem de canais a partir das bandas: nº de cruzamentos com dE/dk > 0
    ks = np.linspace(-np.pi, np.pi, 2001)
    Eb = np.array([np.linalg.eigvalsh(hk(k)) for k in ks])
    nch = []
    for e in energies:
        s = np.sign(Eb - e)
        nch.append(int(np.sum((s[:-1] < 0) & (s[1:] > 0))))
    w(f"| {kind} | " + " | ".join(f"{t:.3f} (bandas: {n})" for t, n in zip(T, nch))
      + f" | {time.time() - t0:.0f} s |")
w("")

# ---------------------------------------------------------------- 6. I-V
if not (cli.quick or cli.skip_iv):
    w("## 6. Curvas I-V (Landauer, banda rígida, hamiltoniano com cutoff 1e-3 eV)")
    w("")
    w("I(V) = 2 (e/h) ∫ T(E) [f(E − E_F − V/2) − f(E − E_F + V/2)] dE, T = 300 K, "
      "T(E) com passo de 0,01 eV, calculado a bias zero.")
    w("")
    Ech = np.linspace(-2, 2, 4001)
    Ian = km.landauer_current(Ech, np.ones_like(Ech), [1.0, 2.0])
    w(f"- Verificação analítica (T = 1): I/V = {Ian[0]:.3f} µA/V "
      f"(esperado 2e²/h = 77.481 µA/V)")
    km.HOP_CUTOFF = 1e-3
    t0 = time.time()
    energies, biases, res = km.run_iv((20.0, 20.0), vmax=3.0, nv=61, de=0.01,
                                      output=os.path.join(OUT, "iv_ribbons.png"))
    km.HOP_CUTOFF = 0.0
    w(f"- Tempo total: {time.time() - t0:.0f} s")
    w("")
    sel = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    w("| V (V) | " + " | ".join(f"{v:g}" for v in sel) + " |")
    w("|---|" + "---|" * len(sel))
    for kind in ("zigzag", "armchair"):
        I = res[kind][1]
        w(f"| I {kind} (µA) | " + " | ".join(f"{np.interp(v, biases, I):.2f}" for v in sel) + " |")
    w("")
    w("![iv](iv_ribbons.png)")
    w("")

with open(os.path.join(OUT, "REPORT.md"), "w") as f:
    f.write("\n".join(rep) + "\n")
print("\n".join(rep))
print(f"\nRelatório: {OUT}/REPORT.md")
