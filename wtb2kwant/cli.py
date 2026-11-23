"""Interface de linha de comando: wtb2kwant."""
from __future__ import annotations

import argparse
import os
import sys

from .codegen import generate_script
from .model import from_wannier
from .readers import read_wout, read_wtb_wannier


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="wtb2kwant",
        description="Converte um hamiltoniano TB do WanTiBEXOS em um script Kwant.")
    ap.add_argument("input", help="arquivo de hamiltoniano do WanTiBEXOS (ex.: tb_hr.dat)")
    ap.add_argument("--format", choices=["wannier"], default="wannier",
                    help="formato de entrada (por enquanto só 'wannier')")
    ap.add_argument("--wout", nargs="+", required=True,
                    help="wannier90.wout com os centros finais; spin-polarizado: "
                         "dois arquivos, up e dn, nessa ordem")
    ap.add_argument("--sites", choices=["atoms", "centres"], default="atoms",
                    help="agrupar WFs no átomo mais próximo (padrão) ou um site por WF")
    ap.add_argument("--cutoff", type=float, default=1e-5,
                    help="descarta |H| < cutoff (eV) [1e-5]")
    ap.add_argument("--efermi", type=float, default=None,
                    help="sobrescreve a energia de Fermi do arquivo (eV)")
    ap.add_argument("-o", "--output", default="kwant_model.py",
                    help="script Kwant gerado [kwant_model.py]")
    args = ap.parse_args(argv)

    wtb = read_wtb_wannier(args.input)
    wouts = [read_wout(f) for f in args.wout]
    model = from_wannier(wtb, wouts, sites_mode=args.sites, cutoff=args.cutoff,
                         efermi=args.efermi)
    sources = [os.path.abspath(args.input)] + [os.path.abspath(f) for f in args.wout]
    code = generate_script(model, sources)
    with open(args.output, "w") as f:
        f.write(code)
    print(f"[wtb2kwant] {model.systype}: {model.norb} orbitais em {len(model.sites)} sites")
    for s in model.sites:
        print(f"    {s.name:6s} {len(s.orbitals):2d} orbitais")
    for n in model.notes:
        print(f"    - {n}")
    print(f"[wtb2kwant] script gerado: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
