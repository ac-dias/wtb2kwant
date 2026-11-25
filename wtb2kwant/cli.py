"""Interface de linha de comando: wtb2kwant."""
from __future__ import annotations

import argparse
import os
import sys

from .codegen import generate_script
from .model import from_wannier, from_wtb_ham
from .readers import read_basis, read_structure, read_wout, read_wtb_ham, read_wtb_wannier


def detect_format(path: str) -> str:
    """'nonortho' se a linha 9 for o comentário '#rcell ...' do .ham, senão 'wannier'."""
    with open(path) as f:
        head = [f.readline() for _ in range(9)]
    return "nonortho" if head[8].lstrip().startswith("#") else "wannier"


EPILOG = """\
exemplos:
  # Wannier90 (ortogonal): centros e átomos do .wout
  wtb2kwant tb_hr.dat --wout wannier90.wout
  wtb2kwant tb_hr.dat --wout wannier90.up.wout wannier90.dn.wout      # SP

  # base não ortogonal (SIESTA, YAeHMOP): posições do .basis
  wtb2kwant tb-NP.ham --basis basis_set-NP.basis
  # base não ortogonal (DFTB+, FHI-aims, CRYSTAL): estrutura + orbitais por espécie
  wtb2kwant fhi_tb.ham --structure geometry.in                  # 1 espécie: --norb opcional
  wtb2kwant hbn_sp_tb.ham --structure hbn_sp.out --norb B=14,N=14
  wtb2kwant dftb.ham --structure geo.gen --norb Mo=9,S=4
"""


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="wtb2kwant",
        description="Converte um hamiltoniano TB do WanTiBEXOS em um script Kwant.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="hamiltoniano do WanTiBEXOS (tb_hr.dat ou .ham)")
    ap.add_argument("--format", choices=["auto", "wannier", "nonortho"], default="auto",
                    help="formato de entrada [auto: detecta pelo cabeçalho]")
    g = ap.add_argument_group("formato Wannier90")
    g.add_argument("--wout", nargs="+",
                   help="wannier90.wout com os centros finais; spin-polarizado: up e dn")
    g.add_argument("--sites", choices=["atoms", "centres"], default="atoms",
                   help="agrupar WFs no átomo mais próximo (padrão) ou um site por WF")
    g.add_argument("--soc-spin-order", choices=["blocks", "interleaved"], default=None,
                   help="SOC: ordem de spin das WFs (blocks: todas up e depois todas dn; "
                        "interleaved: up, dn alternados; confira a ordem das projeções no .win/.wout). "
                        "Necessário para o transporte resolvido em spin")
    g = ap.add_argument_group("base não ortogonal (.ham)")
    g.add_argument("--basis", help="basis_set-*.basis (SIESTA, YAeHMOP)")
    g.add_argument("--structure",
                   help="estrutura: POSCAR/CONTCAR, geometry.in (FHI-aims), .gen (DFTB+), "
                        "saída .out do CRYSTAL, ou outro formato lido pelo ASE")
    g.add_argument("--structure-format", default="auto",
                   help="poscar | aims | gen | crystal | ase[:fmt] [auto]")
    g.add_argument("--norb",
                   help="orbitais (por spin) de cada espécie, ex.: Mo=9,S=4 "
                        "(opcional se houver uma só espécie)")
    g.add_argument("--s-cutoff", type=float, default=1e-6,
                   help="descarta |S| < X na sobreposição [1e-6]")
    ap.add_argument("--cutoff", type=float, default=1e-5,
                    help="descarta |H| < cutoff (eV) [1e-5]")
    ap.add_argument("--efermi", type=float, default=None,
                    help="sobrescreve a energia de Fermi do arquivo (eV)")
    ap.add_argument("-o", "--output", default="kwant_model.py",
                    help="script Kwant gerado [kwant_model.py]")
    args = ap.parse_args(argv)

    fmt = detect_format(args.input) if args.format == "auto" else args.format
    sources = [os.path.abspath(args.input)]
    if fmt == "wannier":
        if not args.wout:
            ap.error("formato Wannier90: informe --wout wannier90.wout")
        wtb = read_wtb_wannier(args.input)
        wouts = [read_wout(f) for f in args.wout]
        model = from_wannier(wtb, wouts, sites_mode=args.sites, cutoff=args.cutoff,
                             efermi=args.efermi, soc_spin_order=args.soc_spin_order)
        sources += [os.path.abspath(f) for f in args.wout]
    else:
        if not (args.basis or args.structure):
            ap.error("base não ortogonal: informe --basis (SIESTA/YAeHMOP) ou --structure")
        ham = read_wtb_ham(args.input)
        basis = read_basis(args.basis) if args.basis else None
        struct = (read_structure(args.structure, args.structure_format)
                  if args.structure and not args.basis else None)
        model = from_wtb_ham(ham, basis=basis, structure=struct, norb=args.norb,
                             cutoff=args.cutoff, s_cutoff=args.s_cutoff, efermi=args.efermi)
        sources += [os.path.abspath(f) for f in (args.basis, args.structure) if f]

    code = generate_script(model, sources)
    with open(args.output, "w") as f:
        f.write(code)
    kind = "não ortogonal" if model.nonorthogonal else "ortogonal"
    print(f"[wtb2kwant] {model.systype} ({kind}): {model.norb} orbitais em "
          f"{len(model.sites)} sites")
    for s in model.sites:
        print(f"    {s.name:6s} {len(s.orbitals):2d} orbitais")
    for n in model.notes:
        print(f"    - {n}")
    print(f"[wtb2kwant] script gerado: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
