#!/usr/bin/env python
"""Validação da base NÃO ortogonal com o MoS2 do YAeHMOP (Hückel estendido).

Gera relatório e figuras em tests/results/mos2_yaehmop_np/. Para ser rápido, usa
fitas estreitas (W = 12 Å) e um H de alcance curto (cutoff 1e-3 eV em H, 1e-4 em S).

Uso:  python tests/validate_mos2_yaehmop_np.py [--skip-iv]
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
from wtb2kwant.lowdin import lowdin_orthogonalize  # noqa: E402
from wtb2kwant.model import from_wtb_ham  # noqa: E402
from wtb2kwant.readers import read_basis, read_wtb_ham  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--skip-iv", action="store_true")
cli = ap.parse_args()

D = os.environ.get("WTB2KWANT_MOS2_YAEHMOP_DIR", os.path.join(HERE, "data", "mos2_yaehmop_np"))
OUT = os.path.join(HERE, "results", "mos2_yaehmop_np")
os.makedirs(OUT, exist_ok=True)
DREL = os.path.relpath(D, os.path.dirname(HERE))
W = 12.0                    # largura das fitas (Å)
CUT_H, CUT_S = 1e-3, 1e-4   # alcance curto (validação rápida)
NOCC = 9                    # MoS2: 18 elétrons, sem spin -> 9 bandas ocupadas

rep = []
w = rep.append
t_start = time.time()


def yaehmop_bands(path):
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


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w("# Validação wtb2kwant: base não ortogonal, MoS2 (YAeHMOP, Hückel estendido)")
w("")
w(f"Gerado em {datetime.datetime.now():%Y-%m-%d %H:%M} por `tests/validate_mos2_yaehmop_np.py`.")
w(f"Dados: `{DREL}` (`huckel-NP.dat` + `basis_set-NP`). Referência: `MoS2.bind.band` (YAeHMOP).")
w("Base: Mo 5s5p4d (9 orbitais), S 3s3p (4 orbitais); 17 orbitais, S ≠ 1.")
w("")

ham = read_wtb_ham(f"{D}/huckel-NP.dat")
basis = read_basis(f"{D}/basis_set-NP")
ks, ref = yaehmop_bands(f"{D}/MoS2.bind.band")
ref_rel = ref - ref[:, :NOCC].max()
win = (ref_rel > -6) & (ref_rel < 4)

# ---------------------------------------------------------------- 1. bandas bulk
w("## 1. Bandas do bulk (problema generalizado) × YAeHMOP")
w("")
w("Energias relativas ao VBM, como no script de comparação da interface. A interface do")
w("YAeHMOP desloca o zero de energia por uma constante (7,83 eV), o que não altera as bandas")
w("relativas.")
w("")
w("| cutoff H (eV) | cutoff S | vetores R | erro máx. janela VBM−6…+4 eV (meV) | erro máx. todas (meV) | gap (eV) |")
w("|---|---|---|---|---|---|")
for ch, cs in [(0, 0), (1e-5, 1e-6), (1e-4, 1e-5), (CUT_H, CUT_S)]:
    m = from_wtb_ham(ham, basis=basis, cutoff=ch, s_cutoff=cs)
    E = m.bands(ks)
    E -= E[:, :NOCC].max()
    w(f"| {ch:g} | {cs:g} | {len(set(m.hoppings) | set(m.overlaps))} | "
      f"{1e3 * np.abs(E - ref_rel)[win].max():.3f} | {1e3 * np.abs(E - ref_rel).max():.2f} | "
      f"{E[:, NOCC:].min():.6f} |")
w(f"\nReferência YAeHMOP: gap = {ref_rel[:, NOCC:].min():.6f} eV.")
w("")

# modelo de validação (alcance curto) e E_F no meio do gap
model = from_wtb_ham(ham, basis=basis, cutoff=CUT_H, s_cutoff=CUT_S)
Eabs = model.bands(ks)
vbm, cbm = Eabs[:, :NOCC].max(), Eabs[:, NOCC:].min()
model.efermi = 0.5 * (vbm + cbm)
w(f"Modelo usado daqui em diante: cutoff H {CUT_H:g} eV, S {CUT_S:g}; E_F no meio do gap "
  f"({model.efermi:.4f} eV no zero de energia do arquivo).")
for n in model.notes:
    w(f"- {n}")
w("")

path = os.path.join(OUT, "kwant_model_test.py")
with open(path, "w") as f:
    f.write(generate_script(model, [f"{DREL}/huckel-NP.dat", f"{DREL}/basis_set-NP"]))
km = load(path, "km_yae")
Egen = np.array([km.bulk_bands(k[:2]) for k in ks])
Egen_rel = Egen - Egen[:, :NOCC].max()
w(f"- Script gerado, `bulk_bands` × YAeHMOP: janela {1e3 * np.abs(Egen_rel - ref_rel)[win].max():.2f} meV, "
  f"todas {1e3 * np.abs(Egen_rel - ref_rel).max():.1f} meV")

xk = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ks @ (2 * np.pi * np.linalg.inv(ham.lattice).T), axis=0), axis=1))])
fig, ax = plt.subplots(figsize=(7, 5))
off = Egen[:, :NOCC].max()
ax.plot(xk, ref_rel + off, "-", color="0.6", lw=2.5)
ax.plot(xk, Egen, "r--", lw=0.9)
ax.plot([], [], "-", color="0.6", lw=2.5, label="YAeHMOP (MoS2.bind.band)")
ax.plot([], [], "r--", lw=0.9, label=f"generated Kwant script (cutoff H {CUT_H:g} eV, S {CUT_S:g})")
for i, lab in zip((0, 40, 80, 120), ("Γ", "M", "K", "Γ")):
    ax.axvline(xk[i], color="k", lw=0.4)
ax.set_xticks([xk[i] for i in (0, 40, 80, 120)])
ax.set_xticklabels(["Γ", "M", "K", "Γ"])
ax.set_xlim(xk[0], xk[-1])
ax.set_ylim(-7, 6)
ax.axhline(0, color="b", ls=":", lw=0.6)
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("MoS$_2$ bulk, non-orthogonal basis (extended Hückel)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bulk_bands_vs_yaehmop.png"), dpi=150)
w("")
w("![bulk](bulk_bands_vs_yaehmop.png)")
w("")

# ---------------------------------------------------------------- 2. Kwant
w("## 2. Termos dependentes da energia no Kwant")
w("")
lat = km.make_lattice()
subs = lat.sublattices
fin = kwant.Builder()
fin.fill(km.make_bulk(lat), lambda s: s.pos[0] ** 2 + s.pos[1] ** 2 < 15 ** 2, subs[0](0, 0))
hs = km._hs_blocks()
bad = ntest = 0
for (R, s, t), (Hb, Sb) in list(hs.items()):
    try:
        f = fin[subs[s](0, 0), subs[t](*R)]
    except KeyError:
        continue
    for E in (0.0, 0.7, -1.3):
        ntest += 1
        bad += not np.allclose(f(subs[s](0, 0), subs[t](*R), E),
                               Hb - km.EFERMI * Sb - E * Sb)
w(f"- `syst[s(0), t(R)](E) == (H - E_F S) - E S` para 3 energias: {ntest - bad}/{ntest} corretos.")
w("")

# ---------------------------------------------------------------- 3. fitas
w(f"## 3. Nanofitas (W = {W:g} Å): bandas generalizadas e transmissão")
w("")
w("A transmissão usa H' = H − E(S − 1) (exata). Numa fita perfeita ela deve ser igual ao")
w("número de bandas do problema generalizado que cruzam E com velocidade positiva.")
w("")
old = sys.argv
sys.argv = ["km", "--width-zz", str(W), "--width-ac", str(W), "--length", "20", "--ne", "41",
            "--emin", "-2", "--emax", "2", "-o", os.path.join(OUT, "bands_ribbons.png")]
t0 = time.time()
km.main()
sys.argv = old
dat = np.loadtxt(os.path.join(OUT, "bands_ribbons_transmission.dat"))
w("| tipo | N (supercélula do lead) | sites/célula | max \\|T − canais\\| (41 energias) |")
w("|---|---|---|---|")
for col, kind in ((1, "zigzag"), (2, "armchair")):
    rib = km.make_ribbon(kind, W)
    nch = km.channel_count(rib, dat[:, 0])
    w(f"| {kind} | {km.min_supercell(kind, W)} | {km.ribbon_info(rib)} | "
      f"{np.abs(dat[:, col] - nch).max():.1e} |")
w(f"\nTempo (bandas + transmissão): {time.time() - t0:.0f} s")
w("")
w("![fitas](bands_ribbons.png)")
w("")

# ---------------------------------------------------------------- 4. I-V
if not cli.skip_iv:
    w("## 4. Curvas I-V (Landauer, banda rígida, 300 K)")
    w("")
    t0 = time.time()
    energies, biases, res = km.run_iv((W, W), vmax=3.0, nv=31, de=0.02, length=20.0,
                                      output=os.path.join(OUT, "iv_ribbons.png"))
    w(f"T(E) com passo de 0,02 eV, modelo já com o alcance curto (sem cutoff extra); "
      f"tempo {time.time() - t0:.0f} s.")
    w("")
    sel = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    w("| V (V) | " + " | ".join(f"{v:g}" for v in sel) + " |")
    w("|---|" + "---|" * len(sel))
    for kind in ("zigzag", "armchair"):
        w(f"| I {kind} (µA) | " + " | ".join(f"{np.interp(v, biases, res[kind][1]):.2f}"
                                             for v in sel) + " |")
    w("")
    w("![iv](iv_ribbons.png)")
    w("")

# ---------------------------------------------------------------- 5. Löwdin
w("## 5. Ortogonalização de Löwdin × base não ortogonal")
w("")
t0 = time.time()
mL = lowdin_orthogonalize(model, cutoff=CUT_H)
w(f"- {mL.notes[-1]}")
nel_no = sum(np.count_nonzero(M) for M in model.hoppings.values()) + \
    sum(np.count_nonzero(M) for M in model.overlaps.values())
nel_L = sum(np.count_nonzero(M) for M in mL.hoppings.values())
w(f"- elementos: {nel_no} (H + S, não ortogonal) → {nel_L} (H de Löwdin), mesmo cutoff de {CUT_H:g} eV")
pathL = os.path.join(OUT, "kwant_model_lowdin_test.py")
with open(pathL, "w") as f:
    f.write(generate_script(mL, [f"{DREL}/huckel-NP.dat", f"{DREL}/basis_set-NP", "(Löwdin)"]))
kmL = load(pathL, "km_yae_L")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
w("")
w("| fita | N (lead), não ortogonal | N (lead), Löwdin | desvio máx. das bandas Löwdin, \\|E − E_F\\| < 2 eV (meV) |")
w("|---|---|---|---|")
for ax, kind in zip(axes, ("zigzag", "armchair")):
    ks1, Eno = km.ribbon_bands(km.make_ribbon(kind, W), 101)
    _, EL = kmL.ribbon_bands(kmL.make_ribbon(kind, W), 101)
    mask = np.abs(Eno) < 2
    dev = np.abs(EL - Eno)[mask].max()
    w(f"| {kind} | {km.min_supercell(kind, W)} | {kmL.min_supercell(kind, W)} | {1e3 * dev:.1f} |")
    ax.plot(ks1, Eno, "-", color="0.6", lw=2)
    ax.plot(ks1, EL, "r--", lw=0.8)
    ax.plot([], [], "-", color="0.6", lw=2, label="non-orthogonal, H − E(S − 1)")
    ax.plot([], [], "r--", lw=0.8, label="Löwdin (bulk-orthogonalized)")
    ax.set_ylim(-2, 2)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xlabel(r"$k$ (rad / period)")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.set_title(f"{kind} ribbon, W = {W:g} Å")
    ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ribbons_lowdin_vs_nonortho.png"), dpi=150)
w(f"\nTempo: {time.time() - t0:.0f} s.")
w("")
w("**Convergência no bulk** (modelo com cutoff H 1e-5, S 1e-6): o erro de Löwdin fora da malha")
w("cai com o cutoff, o que confirma a implementação.")
w("")
w("| cutoff Löwdin (eV) | vetores R | erro máx. das bandas do bulk (meV) |")
w("|---|---|---|")
mfine = from_wtb_ham(ham, basis=basis, cutoff=1e-5, s_cutoff=1e-6)
for c in (1e-3, 1e-4, 1e-5):
    Lc = lowdin_orthogonalize(mfine, cutoff=c)
    err = float(Lc.notes[-1].split("): ")[-1].split()[0])
    w(f"| {c:g} | {len(Lc.hoppings)} | {1e3 * err:.2f} |")
w("")
w("**Por que as fitas diferem tanto: renormalização on-site.** As sobreposições do Hückel")
w(f"estendido chegam a |S| = {max(np.abs(M - (np.eye(len(M)) if R == (0, 0, 0) else 0)).max() for R, M in mfine.overlaps.items()):.2f} entre vizinhos. Os termos on-site de Löwdin absorvem")
w("o ambiente do bulk:")
w("")
H0 = np.diag(mfine.hoppings[(0, 0, 0)]).real
HLd = np.diag(Lc.hoppings[(0, 0, 0)]).real
labs = [o for st in mfine.sites for o in st.orbitals]
names = [st.name for st in mfine.sites for _ in st.orbitals]
w("| orbital | on-site original (eV) | on-site Löwdin (eV) |")
w("|---|---|---|")
for i in range(len(H0)):
    w(f"| {names[i]} {labs[i]} | {H0[i]:.2f} | {HLd[i]:.2f} |")
w("")
w("No bulk essa renormalização é compensada pelos hoppings de Löwdin com os vizinhos. Numa")
w("borda os vizinhos faltam e a compensação não acontece: os átomos de borda (e, numa fita")
w("de 12 Å, quase todos os átomos estão perto de uma borda) ficam com níveis errados por eV.")
w("Conclusão: para bases muito não ortogonais (Hückel estendido, bases difusas) Löwdin serve")
w("para o bulk, mas não para nanoestruturas; o padrão do wtb2kwant (manter S e usar")
w("H − E(S − 1)) é exato.")
w("")
w("![lowdin](ribbons_lowdin_vs_nonortho.png)")
w("")
w(f"Tempo total da validação: {time.time() - t_start:.0f} s.")

with open(os.path.join(OUT, "REPORT.md"), "w") as f:
    f.write("\n".join(rep) + "\n")
print("\n".join(rep))
print(f"\nRelatório: {OUT}/REPORT.md")
