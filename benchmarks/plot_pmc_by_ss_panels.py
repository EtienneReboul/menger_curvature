"""Three-panel histogram of PMCs per DSSP class (helices, strands, turns and coils).

Reads the ``per_chain/<chain>.npz`` files written by ``tubulin_pf_analysis.py``
and pools all chains and frames. Each panel shows, per bin, the percentage of
that class's PMCs (bars of a class sum to 100 %). A PMC is assigned the DSSP
class of the central residue of its triplet.

usage: python plot_pmc_by_ss_panels.py <analysis_outdir> [--bin-width 0.005] [--out file.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# DSSP codes stored by tubulin_pf_analysis.py: 0 = loop '-', 1 = helix 'H', 2 = strand 'E'
PANELS = [(1, "Helices", "red"), (2, "Strands", "blue"), (0, "Turns and coils", "green")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", help="output directory of tubulin_pf_analysis.py")
    ap.add_argument("--bin-width", type=float, default=0.005, help="Å⁻¹")
    ap.add_argument("--xmax", type=float, default=0.45)
    ap.add_argument("--ymax", type=float, default=50)
    ap.add_argument("--hline", type=float, default=5, help="horizontal reference line (%%)")
    ap.add_argument("--out", default=None, help="default: <outdir>/pmc_by_ss_panels.png")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    files = sorted((outdir / "per_chain").glob("*.npz"))
    curv = np.concatenate([np.load(f)["curvature"].ravel() for f in files])
    ss = np.concatenate([np.load(f)["ss_central"].ravel() for f in files])
    chains = " ".join(f.stem for f in files)

    edges = np.arange(0.0, args.xmax + args.bin_width / 2, args.bin_width)
    above = int((curv > args.xmax).sum())
    if above:
        print(f"warning: {above} PMC values above {args.xmax} Å⁻¹ are outside the bins")

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    csv = ["class,bin_left,bin_right,percent"]
    for ax, (code, label, colour) in zip(axes, PANELS):
        vals = curv[ss == code]
        counts, _ = np.histogram(vals, bins=edges)
        percent = 100 * counts / vals.size
        ax.bar(edges[:-1], percent, width=args.bin_width, align="edge", color=colour, label=label)
        ax.axhline(args.hline, color="black", lw=1.5)
        ax.set_xlim(0, args.xmax)
        ax.set_ylim(0, args.ymax)
        ax.set_xticks(np.arange(0, args.xmax + 1e-9, 0.05))
        ax.set_xlabel(r"PMC ($\AA^{-1}$)", fontsize=18)
        ax.set_ylabel("% of occurrence", fontsize=18)
        ax.legend(loc="upper right", fontsize=16)
        csv += [f"{label},{edges[k]:.4f},{edges[k + 1]:.4f},{p:.6f}" for k, p in enumerate(percent)]
        print(f"{label}: {vals.size} PMCs, max bin {percent.max():.1f} %")
    fig.tight_layout(w_pad=6)
    out = Path(args.out) if args.out else outdir / "pmc_by_ss_panels.png"
    fig.savefig(out, dpi=300)
    out.with_suffix(".csv").write_text("\n".join(csv) + "\n")
    print(f"chains {chains}: wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
