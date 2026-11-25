#!/usr/bin/env python
"""Validação de SP e SOC: bandas contra referências e transporte resolvido em spin.

Gera relatório e figuras em tests/results/spin/.

Dados:
  - MoS2 SOC (YAeHMOP): tests/data/mos2_yaehmop_soc (no repo).
  - MoS2 NP (YAeHMOP): tests/data/mos2_yaehmop_np (no repo), para o SP magnético sintético.
  - Grafeno SP (FHI-aims) e h-BN SP (CRYSTAL): fora do repo (arquivos grandes), em
    $WTB2KWANT_INTERFACES_TEST (padrão: ~/work/development/wantibexos-files/interfaces_wtb/test);
    as seções são puladas se os dados não existirem.

Uso:  python tests/validate_spin.py [--skip-iv]       (~10 min com 12 núcleos)
"""
import argparse
import datetime
import glob
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.linalg import eigh  # noqa: E402

from wtb2kwant.codegen import generate_script  # noqa: E402
from wtb2kwant.model import Site, TBModel, from_wtb_ham  # noqa: E402
from wtb2kwant.readers import read_basis, read_structure, read_wtb_ham  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--skip-iv", action="store_true")
cli = ap.parse_args()

OUT = os.path.join(HERE, "results", "spin")
os.makedirs(OUT, exist_ok=True)
DSOC = os.environ.get("WTB2KWANT_MOS2_SOC_DIR", os.path.join(HERE, "data", "mos2_yaehmop_soc"))
DNP = os.environ.get("WTB2KWANT_MOS2_YAEHMOP_DIR", os.path.join(HERE, "data", "mos2_yaehmop_np"))
DINT = os.environ.get("WTB2KWANT_INTERFACES_TEST", os.path.expanduser(
    "~/work/development/wantibexos-files/interfaces_wtb/test"))
DAIMS = os.path.join(DINT, "aims", "graphene-sp")
DCRY = os.path.join(DINT, "crystal", "sp")

rep = []
w = rep.append
t_start = time.time()


def rel(p):
    return os.path.relpath(p, os.path.dirname(HERE)) if p.startswith(os.path.dirname(HERE)) \
        else "$WTB2KWANT_INTERFACES_TEST/" + os.path.relpath(p, DINT)


def load(model, name):
    path = os.path.join(OUT, f"{name}.py")
    with open(path, "w") as f:
        f.write(generate_script(model, ["validate_spin"]))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def spin_block_bands(m, ks, spin):
    """Autovalores do bloco de spin (up/dn) de H(k), S(k): vale se não houver mistura."""
    lab = [o for s in m.sites for o in s.orbitals]
    msk = np.array([x.startswith(spin + ":") for x in lab])
    return np.array([eigh(m.hk(k)[np.ix_(msk, msk)], m.sk(k)[np.ix_(msk, msk)],
                          eigvals_only=True) for k in ks])


def updn_coupling(m):
    lab = [o for s in m.sites for o in s.orbitals]
    up = np.array([x.startswith("up:") for x in lab])
    return max(np.abs(M[np.ix_(up, ~up)]).max() for M in list(m.hoppings.values())
               + list(m.overlaps.values()))


w("# Validação wtb2kwant: sistemas com spin (SP e SOC)")
w("")
w(f"Gerado em {datetime.datetime.now():%Y-%m-%d %H:%M} por `tests/validate_spin.py`.")
w("")
w("Convenção do transporte resolvido em spin: T_σ'σ, com σ = spin injetado no lead 0 e")
w("σ' = spin coletado no lead 1 (`uu`, `ud` = entra ↓ e sai ↑, `du`, `dd`). Padrão (`modes`):")
w("spin dos modos propagantes dos leads, T_σ'σ = Tr[t† W₁^σ' t W₀^σ], com t da matriz S e W^σ o")
w("peso de spin dos modos. Para comparação: NEGF com Γ projetado nos orbitais da interface,")
w("Γ^σ = {P_σ, Γ}/2 (`sym`) ou P_σ Γ P_σ (`block`). Com spin conservado (SP) as três coincidem.")
w("")

# ====================================================================== 1. aims SP
w("## 1. Grafeno SP (FHI-aims): bandas contra o DFT")
w("")
if os.path.isdir(DAIMS):
    t0 = time.time()
    ham = read_wtb_ham(f"{DAIMS}/fhi_tb.ham")
    mg = from_wtb_ham(ham, structure=read_structure(f"{DAIMS}/geometry.in"), cutoff=0, s_cutoff=0)

    def aims_bands(prefix):
        ks, ev = [], []
        for f in sorted(glob.glob(f"{DAIMS}/band{prefix}0*.out")):
            row = open(f).readline().split()
            ks.append([float(x) for x in row[1:4]])
            ev.append([float(x) for x in row[5::2]])
        return np.array(ks), np.array(ev)

    w(f"Dados: `{rel(DAIMS)}` (`fhi_tb.ham` + `geometry.in`); referência: `band1xxx.out` (spin ↑) e "
      "`band2xxx.out` (spin ↓) do FHI-aims, nos 81 pontos k da malha 9×9 usada pela interface.")
    w(f"Modelo: {ham.nbasis} spin-orbitais, {len(mg.sites)} sites com "
      f"{len(mg.sites[0].orbitals)} orbitais (base tier do FHI-aims, 28 por spin e por átomo); "
      f"acoplamento ↑↓ máximo em H e S: {updn_coupling(mg):.1e}.")
    w("")
    w("| spin | bandas DFT | desvio constante (= E_F) | máx. desvio, E−E_F ∈ [−20, 10] eV | máx. desvio, todas |")
    w("|---|---|---|---|---|")
    figdat = {}
    for p, sp in (("1", "up"), ("2", "dn")):
        ks, ref = aims_bands(p)
        E = spin_block_bands(mg, ks, sp)[:, :ref.shape[1]]
        c = np.median(E - ref)
        D = np.abs(E - ref - c)
        win = (ref > -20) & (ref < 10)
        w(f"| {'↑' if sp == 'up' else '↓'} | {ref.shape[1]} | {c:.5f} eV | {D[win].max():.1e} eV | "
          f"{D.max():.1e} eV |")
        figdat[sp] = (ks, E - c, ref)
    w("")
    Eup, Edn = figdat["up"][1], figdat["dn"][1]
    w(f"- ↑ e ↓ idênticos (grafeno não magnético): máx. |E↑ − E↓| = {np.abs(Eup - Edn).max():.1e} eV.")
    w("- Os `band*.out` do FHI-aims dão energias relativas ao E_F (−5,0096 eV), daí o desvio "
      "constante. O FHI-aims escreve 5 casas decimais, o que limita a comparação a ~1e-5 eV.")
    # caminho Γ-M-K-Γ: ponto de Dirac
    path = [np.array(x) for x in ([0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0], [0, 0, 0])]
    kp = np.vstack([path[i] + (path[i + 1] - path[i]) * t
                    for i in range(3) for t in np.linspace(0, 1, 40, endpoint=i == 2)])
    Ek = spin_block_bands(mg, kp, "up") + 5.0096
    iK = 80
    occ = int(np.sum(Ek[0] < 0))
    w(f"- Ponto de Dirac em K = (1/3, 1/3): gap {Ek[iK, occ] - Ek[iK, occ - 1]:.1e} eV, a "
      f"{0.5 * (Ek[iK, occ] + Ek[iK, occ - 1]):.3f} eV de E_F.")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for n in range(Ek.shape[1]):
        a1.plot(Ek[:, n], color="C3", lw=1)
    a1.set_xticks([0, 40, 80, 119]); a1.set_xticklabels(["Γ", "M", "K", "Γ"])
    a1.set_ylim(-20, 10); a1.set_xlim(0, 119)
    a1.axhline(0, color="k", ls=":", lw=0.6)
    a1.set_ylabel(r"$E - E_F$ (eV)"); a1.set_title("graphene SP (FHI-aims), wtb2kwant bands, spin up")
    ks, E, ref = figdat["up"]
    a2.plot(ref.ravel() + 5.0096, (E - ref).ravel() * 1e3, ".", ms=3, color="C3", label="spin up")
    ks, E, ref = figdat["dn"]
    a2.plot(ref.ravel() + 5.0096, (E - ref).ravel() * 1e3, "x", ms=3, color="C0", label="spin down")
    a2.set_xlim(-25, 15); a2.set_ylim(-0.05, 0.05)
    a2.set_xlabel(r"$E_{DFT} - E_F$ (eV)"); a2.set_ylabel("wtb2kwant − FHI-aims (meV)")
    a2.set_title("deviation at the 81 mesh k points"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "graphene_sp_bands_vs_aims.png"), dpi=150)
    plt.close(fig)
    w("")
    w("![grafeno SP](graphene_sp_bands_vs_aims.png)")
    w(f"\nTempo: {time.time() - t0:.0f} s.")
else:
    w(f"Dados não encontrados em `{DAIMS}`: seção pulada.")
w("")

# ====================================================================== 2. CRYSTAL SP
w("## 2. h-BN SP (CRYSTAL)")
w("")
if os.path.isdir(DCRY):
    t0 = time.time()
    ham = read_wtb_ham(f"{DCRY}/hbn_sp_tb.ham")
    mc = from_wtb_ham(ham, structure=read_structure(f"{DCRY}/hbn_sp.out"), norb="B=14,N=14",
                      cutoff=0, s_cutoff=0)
    rng = np.random.default_rng(0)
    ks = np.vstack([[0, 0, 0], [1 / 3, 1 / 3, 0], [0.5, 0, 0],
                    np.c_[rng.random((20, 2)), np.zeros(20)]])
    # referência independente: soma de Fourier direta do arquivo, sem folding nem hermitização
    B = 2 * np.pi * np.linalg.inv(ham.lattice).T
    Rc = ham.rvecs @ ham.lattice
    Ed = []
    for k in ks:
        ph = np.exp(1j * (Rc @ (k @ B)))
        H = np.tensordot(ph, ham.ham, 1)
        S = np.tensordot(ph, ham.ovl, 1)
        Ed.append(eigh(0.5 * (H + H.conj().T), 0.5 * (S + S.conj().T), eigvals_only=True))
    Ed = np.array(Ed)
    Em = mc.bands(ks)
    R = [tuple(r) for r in ham.rvecs]
    idx = {r: i for i, r in enumerate(R)}
    miss = [r for r in R if tuple(-x for x in r) not in idx]
    herm = max(np.abs(ham.ham[i] - ham.ham[idx[tuple(-x for x in r)]].conj().T).max()
               for i, r in enumerate(R) if tuple(-x for x in r) in idx)
    hermS = max(np.abs(ham.ovl[i] - ham.ovl[idx[tuple(-x for x in r)]].conj().T).max()
                for i, r in enumerate(R) if tuple(-x for x in r) in idx)
    ef = ham.efermi
    E1 = Em[:, ::2]
    vbm, cbm = E1[:, 5].max() - ef, E1[:, 6].min() - ef
    w(f"Dados: `{rel(DCRY)}` (`hbn_sp_tb.ham` + `hbn_sp.out`, `--norb B=14,N=14`). O `.out` do CRYSTAL "
      "tem só as matrizes de Fock em espaço real, sem bandas, então não há referência DFT direta.")
    w("")
    w(f"- wtb2kwant × soma de Fourier direta do `.ham` (23 pontos k): {np.abs(Ed - Em).max():.1e} eV. "
      "O leitor, o folding e o agrupamento por site reproduzem o arquivo.")
    w(f"- ↑ e ↓ idênticos (sistema não magnético, acoplamento ↑↓ = {updn_coupling(mc):.0e}): "
      f"{np.abs(Em[:, ::2] - Em[:, 1::2]).max():.1e} eV.")
    w(f"- **O próprio `.ham` está com defeito**: H(R) ≠ H(−R)† (máx. {herm:.1f} eV; em S, "
      f"{hermS:.2f}), {len(miss)} vetor R sem o parceiro −R. As bandas saem metálicas "
      f"(com 6 bandas ocupadas por spin, VBM em {vbm:+.2f} eV e CBM em {cbm:+.2f} eV relativos a E_F), "
      "mas o h-BN é isolante com gap de ~6 eV. O problema está na interface do CRYSTAL "
      "(`crystal2wtb`), não no wtb2kwant: este conjunto não serve como referência até a "
      "interface ser corrigida.")
    w(f"\nTempo: {time.time() - t0:.0f} s.")
else:
    w(f"Dados não encontrados em `{DCRY}`: seção pulada.")
w("")

# ====================================================================== 3. SOC
w("## 3. MoS2 SOC (YAeHMOP): bandas contra o YAeHMOP")
w("")
t0 = time.time()
hs = read_wtb_ham(f"{DSOC}/huckel-SOC.dat")
bs = read_basis(f"{DSOC}/basis_set-SOC")
ms = from_wtb_ham(hs, basis=bs, cutoff=0, s_cutoff=0)
ks, ref = yaehmop_bands(f"{DSOC}/MoS2.bind.band")
E = ms.bands(ks)
NOCC = 18
E -= E[:, :NOCC].max()
ref = ref - ref[:, :NOCC].max()
win = (ref > -6) & (ref < 4)
iK = int(np.argmin(np.linalg.norm(ks - [1 / 3, 1 / 3, 0], axis=1)))
w(f"Dados: `{rel(DSOC)}` (`huckel-SOC.dat` reduzido + `basis_set-SOC`); referência: `MoS2.bind.band` "
  f"(YAeHMOP com SOC, {len(ks)} pontos k em Γ-M-K-Γ). 34 spin-orbitais; acoplamento ↑↓ "
  f"máximo {updn_coupling(ms):.2f} eV (o SOC mistura os spins).")
w("")
w("| grandeza | wtb2kwant | YAeHMOP | diferença |")
w("|---|---|---|---|")
w(f"| bandas em VBM−6 … VBM+4 eV | | | {np.abs(E - ref)[win].max():.1e} eV |")
w(f"| todas as bandas | | | {np.abs(E - ref).max():.1e} eV |")
gm, gr = E[:, NOCC:].min(), ref[:, NOCC:].min()
w(f"| gap | {gm:.6f} eV | {gr:.6f} eV | {abs(gm - gr):.1e} eV |")
sm, sr = E[iK, NOCC - 1] - E[iK, NOCC - 2], ref[iK, NOCC - 1] - ref[iK, NOCC - 2]
w(f"| desdobramento SOC do topo da valência em K | {sm * 1e3:.3f} meV | {sr * 1e3:.3f} meV | "
  f"{abs(sm - sr) * 1e3:.1e} meV |")
w("")
fig, ax = plt.subplots(figsize=(6, 4.6))
for n in range(E.shape[1]):
    ax.plot(ref[:, n], color="0.2", lw=1.3, label="YAeHMOP" if n == 0 else None)
    ax.plot(np.arange(len(ks))[::2], E[::2, n], "o", ms=2.4, mfc="none", color="C3",
            label="wtb2kwant" if n == 0 else None)
ticks = [0, 40, 80, 120]
ax.set_xticks(ticks); ax.set_xticklabels(["Γ", "M", "K", "Γ"])
for t in ticks:
    ax.axvline(t, color="0.85", lw=0.6, zorder=0)
ax.set_ylim(-6, 4); ax.set_xlim(0, len(ks) - 1)
ax.set_ylabel(r"$E - E_{VBM}$ (eV)"); ax.set_title(r"MoS$_2$ with SOC: bulk bands", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "mos2_soc_bands_vs_yaehmop.png"), dpi=150)
plt.close(fig)
w("![MoS2 SOC](mos2_soc_bands_vs_yaehmop.png)")
w(f"\nTempo: {time.time() - t0:.0f} s.")
w("")

# ====================================================================== 4. transporte
w("## 4. Transporte resolvido em spin")
w("")

# ---- 4a. SP magnético sintético
w("### 4a. SP magnético sintético (MoS2 NP do YAeHMOP + troca ±Δ)")
w("")
t0 = time.time()


def magnetic_sp(m, delta):
    """H_up = H − Δ S, H_dn = H + Δ S: bandas ↑ = NP − Δ, ↓ = NP + Δ."""
    sites, iu, idn, i = [], [], [], 0
    for s in m.sites:
        n = len(s.orbitals)
        sites.append(Site(s.name, s.species, s.position,
                          ["up:" + o for o in s.orbitals] + ["dn:" + o for o in s.orbitals]))
        iu += list(range(i, i + n)); idn += list(range(i + n, i + 2 * n)); i += 2 * n
    iu, idn = np.array(iu), np.array(idn)
    Z = np.zeros((m.norb, m.norb), complex)
    H, S = {}, {}
    for R in set(m.hoppings) | set(m.overlaps):
        h, o = m.hoppings.get(R, Z), m.overlaps.get(R, Z)
        Hn, Sn = np.zeros((i, i), complex), np.zeros((i, i), complex)
        Hn[np.ix_(iu, iu)] = h - delta * o; Hn[np.ix_(idn, idn)] = h + delta * o
        Sn[np.ix_(iu, iu)] = o; Sn[np.ix_(idn, idn)] = o
        H[R], S[R] = Hn, Sn
    return TBModel(systype="SP", efermi=m.efermi, scissors=0.0, lattice=m.lattice,
                   periodic_dims=m.periodic_dims, sites=sites, hoppings=H, overlaps=S,
                   notes=["SP sintético: troca ±Δ"])


DELTA = 0.3
mnp = from_wtb_ham(read_wtb_ham(f"{DNP}/huckel-NP.dat"), basis=read_basis(f"{DNP}/basis_set-NP"),
                   cutoff=1e-3, s_cutoff=1e-4, efermi=-0.05)
knp = load(mnp, "kwant_model_np")
ksp = load(magnetic_sp(mnp, DELTA), "kwant_model_sp_magnetic")
Wm = 10.0
Eg = np.round(np.arange(-2.0, 2.0001, 0.05), 6)
ssp = ksp.make_two_terminal("zigzag", Wm, 15.0).finalized()
snp = knp.make_two_terminal("zigzag", Wm, 15.0).finalized()
r = ksp.spin_conductance(ssp, Eg)
nsp = ksp.spin_channel_count(ksp.make_ribbon("zigzag", Wm), Eg)
Tnp_p = knp.conductance(snp, Eg + DELTA)
Tnp_m = knp.conductance(snp, Eg - DELTA)
w(f"Fita zigzag W = {Wm:g} Å (cutoff 1e-3 eV em H, 1e-4 em S), Δ = {DELTA} eV, "
  f"{len(Eg)} energias. Resultado exato esperado: T↑↑(E) = T_NP(E + Δ), T↓↓(E) = T_NP(E − Δ), "
  "sem spin-flip.")
w("")
w(f"- máx. |T↑↑(E) − T_NP(E+Δ)| = {np.abs(r['uu'] - Tnp_p).max():.1e}; "
  f"máx. |T↓↓(E) − T_NP(E−Δ)| = {np.abs(r['dd'] - Tnp_m).max():.1e}")
w(f"- spin-flip: máx. |T↑↓|, |T↓↑| = {max(np.abs(r['ud']).max(), np.abs(r['du']).max()):.1e}; "
  f"máx. |T − ΣT_σ'σ| = {np.abs(r['mix']).max():.1e}")
w(f"- Fita perfeita, canais por spin das bandas (`spin_channel_count`): "
  f"máx. |T↑↑ − N↑| = {np.abs(r['uu'] - nsp['up']).max():.1e}, "
  f"máx. |T↓↓ − N↓| = {np.abs(r['dd'] - nsp['dn']).max():.1e}.")
# conferência independente: lei de conservação S_z do Kwant nos leads
ssc = ksp.make_two_terminal("zigzag", Wm, 15.0, spin_conservation=True).finalized()
rc = ksp.spin_conductance(ssc, Eg, projection="conservation")
w(f"- Rota do próprio Kwant (`conservation`: lei de conservação S_z nos leads, "
  f"`smatrix.transmission((1, s'), (0, s))`): máx. |T↑↑ − T↑↑(modes)| = "
  f"{np.abs(rc['uu'] - r['uu']).max():.1e}, máx. |T↓↓ − T↓↓(modes)| = {np.abs(rc['dd'] - r['dd']).max():.1e}; "
  f"contra o exato: {np.abs(rc['uu'] - Tnp_p).max():.1e} e {np.abs(rc['dd'] - Tnp_m).max():.1e}; "
  f"spin-flip {max(np.abs(rc['ud']).max(), np.abs(rc['du']).max()):.1e}.")
w(f"- Tempo: {time.time() - t0:.0f} s.")
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.plot(Eg, nsp["up"], color="C3", alpha=0.3, lw=3, drawstyle="steps-mid", label="channels ↑ (bands)")
ax.plot(Eg, nsp["dn"], color="C0", alpha=0.3, lw=3, drawstyle="steps-mid", label="channels ↓ (bands)")
ax.plot(Eg, r["total"], color="0.7", lw=2.5, label="total")
ax.plot(Eg, r["uu"], "-", color="C3", lw=1.2, label=r"$T_{\uparrow\uparrow}$")
ax.plot(Eg, r["dd"], "-", color="C0", lw=1.2, label=r"$T_{\downarrow\downarrow}$")
ax.plot(Eg, Tnp_p, "o", ms=3, mfc="none", color="C3", label=r"$T_{NP}(E+\Delta)$")
ax.plot(Eg, Tnp_m, "s", ms=3, mfc="none", color="C0", label=r"$T_{NP}(E-\Delta)$")
ax.set_xlabel(r"$E - E_F$ (eV)"); ax.set_ylabel(r"$T$ ($e^2/h$)")
ax.set_title(f"synthetic ferromagnet: MoS$_2$ zigzag, W = {Wm:g} Å, Δ = {DELTA} eV", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "sp_magnetic_spin_transmission.png"), dpi=150)
plt.close(fig)
w("")
w("![SP magnético](sp_magnetic_spin_transmission.png)")
w("")

# ---- 4b. grafeno SP
w("### 4b. Grafeno SP (FHI-aims): canais ↑ e ↓")
w("")
if os.path.isdir(DAIMS):
    t0 = time.time()
    mg2 = from_wtb_ham(read_wtb_ham(f"{DAIMS}/fhi_tb.ham"),
                       structure=read_structure(f"{DAIMS}/geometry.in"),
                       cutoff=1e-3, s_cutoff=1e-4, efermi=-5.0096)
    kg = load(mg2, "kwant_model_graphene_sp")
    Eg2 = [-1.0, -0.2, 0.3, 1.0]
    sg = kg.make_two_terminal("zigzag", 8.0, 10.0).finalized()
    rg = kg.spin_conductance(sg, Eg2)
    nch = kg.channel_count(kg.make_ribbon("zigzag", 8.0), Eg2)
    w("Fita zigzag W = 8 Å (cutoff 1e-3 eV em H, 1e-4 em S).")
    w("")
    w("| E − E_F (eV) | canais (bandas) | T | T↑↑ | T↓↓ | T↑↓ + T↓↑ |")
    w("|---|---|---|---|---|---|")
    for i, e in enumerate(Eg2):
        w(f"| {e:+.1f} | {nch[i]} | {rg['total'][i]:.6f} | {rg['uu'][i]:.6f} | {rg['dd'][i]:.6f} | "
          f"{rg['ud'][i] + rg['du'][i]:.1e} |")
    w("")
    sgc = kg.make_two_terminal("zigzag", 8.0, 10.0, spin_conservation=True).finalized()
    rgc = kg.spin_conductance(sgc, Eg2, projection="conservation")
    w(f"T↑↑ = T↓↓ = T/2 = canais/2 (máx. desvio {np.abs(rg['uu'] - nch / 2).max():.1e}). "
      f"Rota do próprio Kwant (`conservation`): máx. diferença para `modes` "
      f"{max(np.abs(rgc[k] - rg[k]).max() for k in ('uu', 'ud', 'du', 'dd')):.1e}. "
      f"Tempo: {time.time() - t0:.0f} s.")
else:
    w("Dados do FHI-aims ausentes: seção pulada.")
w("")

# ---- 4c. MoS2 SOC
w("### 4c. MoS2 SOC: componentes de spin e spin-flip")
w("")
t0 = time.time()
ksoc = load(from_wtb_ham(hs, basis=bs, cutoff=1e-3, s_cutoff=1e-4, efermi=-0.05),
            "kwant_model_soc")
WS = 12.0
Es = np.round(np.arange(-2.0, 2.0001, 0.05), 6)
res_soc = {}
for kind in ("zigzag", "armchair"):
    syst = ksoc.make_two_terminal(kind, WS, 20.0).finalized()
    rm = ksoc.spin_conductance(syst, Es)
    rs = ksoc.spin_conductance(syst, Es, projection="sym")
    rb = ksoc.spin_conductance(syst, Es, projection="block")
    rib = ksoc.make_ribbon(kind, WS)
    res_soc[kind] = dict(modes=rm, sym=rs, block=rb, nch=ksoc.channel_count(rib, Es),
                         nsp=ksoc.spin_channel_count(rib, Es))
w(f"Fitas W = {WS:g} Å (cutoff 1e-3 eV em H, 1e-4 em S), {len(Es)} energias em [−2, 2] eV.")
w("")
w("| fita | projeção | máx. \\|T − ΣT_σ'σ\\| | componente mínima | máx. spin-flip / T | "
  "mediana spin-flip / T | máx. \\|N↑ − (T↑↑ + T↑↓)\\| |")
w("|---|---|---|---|---|---|---|")
for kind, d in res_soc.items():
    nz = d["modes"]["total"] > 0.5
    for proj in ("modes", "sym", "block"):
        r = d[proj]
        flip = (np.abs(r["ud"]) + np.abs(r["du"]))[nz] / r["total"][nz]
        cmin = min(r[k][nz].min() for k in ("uu", "ud", "du", "dd"))
        dn = np.abs(d["nsp"]["up"] - (r["uu"] + r["ud"]))[nz].max()
        w(f"| {kind} | `{proj}` | {np.abs(r['mix']).max():.1e} | {cmin:+.3f} | {flip.max():.3f} | "
          f"{np.median(flip):.1e} | {dn:.3f} |")
w("")
w(f"T total = canais das bandas em todas as energias (máx. desvio "
  f"{max(np.abs(d['modes']['total'] - d['nch']).max() for d in res_soc.values()):.1e}).")
w("")
w("Energias com maior spin-flip em `modes`, comparadas com `sym`:")
w("")
w("| fita | E − E_F (eV) | T | `modes`: ↑↑ / ↓↓ / ↑↓ / ↓↑ | `sym`: ↑↑ / ↓↓ / ↑↓ / ↓↑ | canais ↑ / ↓ |")
w("|---|---|---|---|---|---|")
for kind, d in res_soc.items():
    rm, rs = d["modes"], d["sym"]
    fl = np.where(rm["total"] > 0.5, (rm["ud"] + rm["du"]) / np.maximum(rm["total"], 1e-9), 0)
    for i in sorted(np.argsort(-fl)[:4]):
        w(f"| {kind} | {Es[i]:+.2f} | {rm['total'][i]:.3f} | "
          f"{rm['uu'][i]:.3f} / {rm['dd'][i]:.3f} / {rm['ud'][i]:.3f} / {rm['du'][i]:.3f} | "
          f"{rs['uu'][i]:.3f} / {rs['dd'][i]:.3f} / {rs['ud'][i]:+.3f} / {rs['du'][i]:+.3f} | "
          f"{d['nsp']['up'][i]:.2f} / {d['nsp']['dn'][i]:.2f} |")
w("")
w("- `modes` mede o spin nos modos propagantes dos leads, a mesma grandeza dos canais por spin "
  "(N_σ): as componentes são ≥ 0, somam exatamente T, e o T coletado em cada spin acompanha N_σ.")
w("- Perto de E_F e na banda de condução o spin é quase conservado, como esperado para o MoS2, "
  "em que o SOC dominante (tipo Ising) conserva S_z. O spin-flip só cresce em cruzamentos de "
  "bandas da valência em que o SOC mistura orbitais pares e ímpares.")
w("- `sym` e `block` projetam o spin nos orbitais da interface, onde Γ inclui o acoplamento "
  "lead-região, que mistura os spins. Nesses cruzamentos as componentes ficam instáveis: "
  "negativas com `sym` e com soma muito maior que T com `block` (colunas acima). Ficam só "
  "como comparação.")
w(f"- Tempo: {time.time() - t0:.0f} s.")
fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True,
                        gridspec_kw=dict(height_ratios=[2.2, 1]))
for col, (kind, d) in enumerate(res_soc.items()):
    rm, rs = d["modes"], d["sym"]
    ax, af = axs[0][col], axs[1][col]
    ax.plot(Es, d["nch"], color="0.8", lw=3, drawstyle="steps-mid", label="channels")
    ax.plot(Es, d["nsp"]["up"], color="C3", alpha=0.3, lw=3, drawstyle="steps-mid",
            label="channels ↑ (bands)")
    ax.plot(Es, d["nsp"]["dn"], color="C0", alpha=0.3, lw=3, drawstyle="steps-mid",
            label="channels ↓ (bands)")
    ax.plot(Es, rm["total"], "k.", ms=4, label="total")
    ax.plot(Es, rm["uu"], color="C3", lw=1.2, label=r"$T_{\uparrow\uparrow}$")
    ax.plot(Es, rm["dd"], "--", color="C0", lw=1.2, label=r"$T_{\downarrow\downarrow}$")
    ax.set_ylabel(r"$T$ ($e^2/h$)")
    ax.set_title(f"MoS$_2$ SOC, {kind}, W = {WS:g} Å (projection: modes)", fontsize=10)
    ax.legend(fontsize=7, loc="upper center")
    af.plot(Es, rm["ud"], color="C2", lw=1.2, label=r"$T_{\uparrow\downarrow}$ (↓ in, ↑ out), modes")
    af.plot(Es, rm["du"], "--", color="C1", lw=1.2, label=r"$T_{\downarrow\uparrow}$ (↑ in, ↓ out), modes")
    af.plot(Es, rs["ud"], ":", color="C2", lw=0.9, label=r"$T_{\uparrow\downarrow}$, sym (interface)")
    af.plot(Es, rs["du"], ":", color="C1", lw=0.9, label=r"$T_{\downarrow\uparrow}$, sym (interface)")
    af.axhline(0, color="k", lw=0.4)
    af.set_xlabel(r"$E - E_F$ (eV)"); af.set_ylabel(r"spin flip ($e^2/h$)")
    af.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "mos2_soc_spin_transmission.png"), dpi=150)
plt.close(fig)
w("")
w("![MoS2 SOC spin](mos2_soc_spin_transmission.png)")
w("")

# ---- 4d. I-V
if not cli.skip_iv:
    w("### 4d. Curvas I-V resolvidas em spin (MoS2 SOC, `run_iv`)")
    w("")
    t0 = time.time()
    energies, biases, res = ksoc.run_iv((WS, WS), vmax=3.0, nv=31, de=0.02, length=20.0,
                                        output=os.path.join(OUT, "iv_soc.png"))
    w(f"T(E) com passo de 0,02 eV, 300 K; tempo {time.time() - t0:.0f} s.")
    w("")
    sel = [1.0, 2.0, 3.0]
    w("| fita | V (V) | I total (µA) | I↑↑ | I↓↓ | I↑↓ | I↓↑ | I total − ΣI_σ'σ |")
    w("|---|---|---|---|---|---|---|---|")
    for kind in ("zigzag", "armchair"):
        T, I, sres = res[kind]
        for v in sel:
            iv = {k: np.interp(v, biases, sres["I"][k]) for k in ("uu", "dd", "ud", "du", "mix")}
            w(f"| {kind} | {v:g} | {np.interp(v, biases, I):.3f} | {iv['uu']:.3f} | {iv['dd']:.3f} | "
              f"{iv['ud']:.4f} | {iv['du']:.4f} | {iv['mix']:.4f} |")
    w("")
    w("![iv](iv_soc.png)")
    w("")
    w("![iv spin](iv_soc_spin.png)")
    w("")

w(f"Tempo total: {time.time() - t_start:.0f} s.")
for f in glob.glob(os.path.join(OUT, "kwant_model_*.py")):
    os.remove(f)                         # scripts gerados: grandes e reproduzíveis
with open(os.path.join(OUT, "REPORT.md"), "w") as f:
    f.write("\n".join(rep) + "\n")
print("\n".join(rep))
print(f"\nRelatório: {OUT}/REPORT.md")
