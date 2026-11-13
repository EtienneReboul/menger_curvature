"""Regenerate the figures of notebooks/er03-generate_figures.ipynb on a long MD.

Same analyses, plotting code and styling as the notebook (figure A: LCs;
figure B: LFs with RMSF and Neq on twin axes; both stacked into
LCs_and_LFs.svg), applied to one chain of a multi-chain trajectory instead of
the 1000-frame tubulin chain A test data:

* PMCs / LCs / LFs: ``MengerCurvature(u, select="segid <seg> and name CA", spacing=2)``;
* RMSF: the MDAnalysis User Guide recipe used in the notebook (AverageStructure
  ref_frame=0, AlignTraj in_memory=True, RMSF), with the chain's Cα selection;
* Neq: PBxplore on the chain backbone (the notebook's
  ``pbx.chains_from_trajectory`` would process every chain of the system).

Axis limits are the notebook's; if a profile exceeds them the limit is raised
and reported. The shaded M-loop and C-terminal tail default to the αI-tubulin
boundaries from the DSSP consensus of chain C (M-loop = loop between strand S7,
ending at 273, and helix H9, starting at 288; tail = everything after helix H12,
ending at 435); the notebook's 272-286 / 427-450 are βIII numbering.

Agreement between the figure-B profiles (LF, Neq, RMSF): Spearman ρ and Pearson
r with 95 % moving-block-bootstrap confidence intervals, on all residues, the
M-loop, the tail, the rest of the protein and the DSSP consensus classes (see
profile_agreement.py).

Outputs: LCs.svg/.png, LFs.svg/.png, LCs_and_LFs.svg, profiles.csv (residue,
LC, LF, RMSF, Neq), profile_agreement.md/.csv and run_info.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def span(text: str) -> tuple[int, int]:
    a, b = text.split("-")
    return int(a), int(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--segid", default="PROC")
    ap.add_argument("--spacing", type=int, default=2)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--xlabel", default="Residues of αI-tubulin (chain C)")
    ap.add_argument("--m-loop", default="274-287", help="shaded M-loop (residues, inclusive)")
    ap.add_argument("--ctt", default="436-451", help="shaded C-terminal tail (residues, inclusive)")
    ap.add_argument("--dssp-npz", default=None,
                    help="per-chain npz from tubulin_pf_analysis.py, for the DSSP-class agreement analysis")
    ap.add_argument("--bootstrap-block", type=int, default=10, help="residues per bootstrap block")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    import MDAnalysis as mda
    import pbxplore as pbx
    from MDAnalysis.analysis import align, rms
    from pbxplore.structure.structure import Atom, Chain

    from menger.analysis.mengercurvature import MengerCurvature

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timings = {}
    sel_ca = f"segid {args.segid} and name CA"
    spacing = args.spacing

    # ---------------- LCs and LFs ----------------
    u = mda.Universe(args.topology, args.trajectory)
    t0 = time.perf_counter()
    menger = MengerCurvature(u, select=sel_ca, spacing=spacing)
    menger.run(backend="multiprocessing", n_workers=args.n_workers)
    timings["menger_s"] = time.perf_counter() - t0
    local_curvatures = menger.results.local_curvatures
    local_flexibilities = menger.results.local_flexibilities
    resids = np.asarray(menger.atomgroup.resids)

    # ---------------- Neq (chain backbone) ----------------
    t0 = time.perf_counter()
    backbone = u.select_atoms(f"segid {args.segid} and backbone")
    structure = Chain()
    for atm in backbone:
        structure.add_atom(Atom.read_from_xtc(atm))
    sequences = []
    for _ in u.trajectory:
        structure.set_coordinates(backbone.positions)
        sequences.append(pbx.assign(structure.get_phi_psi_angles()))
    neq_by_position = np.asarray(pbx.analysis.compute_neq(pbx.analysis.count_matrix(sequences)))
    timings["neq_s"] = time.perf_counter() - t0

    # ---------------- RMSF (notebook recipe, in-memory alignment of the whole system) ----------------
    t0 = time.perf_counter()
    average = align.AverageStructure(u, u, select=sel_ca, ref_frame=0).run()
    align.AlignTraj(u, average.results.universe, select=sel_ca, in_memory=True).run()
    rmsf = rms.RMSF(u.select_atoms(sel_ca)).run().results.rmsf
    timings["rmsf_s"] = time.perf_counter() - t0
    n_frames = u.trajectory.n_frames

    number_first_residue, number_last_residue = int(resids[0]), int(resids[-1])
    assert len(neq_by_position) == len(rmsf) == len(resids)
    range_residues = range(number_first_residue, number_last_residue + 1)
    inner = slice(spacing, -spacing)

    np.savetxt(outdir / "profiles.csv",
               np.column_stack([resids[inner], local_curvatures, local_flexibilities, rmsf[inner],
                                neq_by_position[inner]]),
               delimiter=",", header="residue,LC,LF,RMSF,Neq", comments="", fmt=["%d", "%.6f", "%.6f", "%.4f", "%.4f"])

    # ---------------- plotting: notebook parameters ----------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    size_figure_tuple = (20, 7)
    ylim_min = 0
    xlabel = args.xlabel
    size_xlabel = 25
    size_ylabel = 25
    size_xticks = 15
    size_yticks = 15
    linewidth = 4
    size_scatter = 50
    alpha_line = 0.6
    DPI = 300
    m_lo, m_hi = span(args.m_loop)
    c_lo, c_hi = span(args.ctt)
    raised = {}

    def limit(name, default, data, pad=1.05):
        top = float(np.max(data))
        if top > default:
            raised[name] = {"notebook": default, "used": round(top * pad, 3), "data_max": round(top, 4)}
            return top * pad
        return default

    # Figure A: LCs
    color = "black"
    ylim_max = limit("LC", 0.35, local_curvatures)
    ylabel = "Local Curvatures ($Å^{⁻1}$)"
    plt.figure(figsize=size_figure_tuple)
    ax = plt.axes()
    plt.xlim(number_first_residue - 0.5, number_last_residue + 0.5)
    plt.ylim(ylim_min, ylim_max)
    plt.xlabel(xlabel, fontsize=size_xlabel)
    plt.ylabel(ylabel, fontsize=size_ylabel)
    ax.add_patch(patches.Rectangle((m_lo - 0.5, 0), m_hi - m_lo + 1, ylim_max * 1.03, color="grey", alpha=0.3))
    ax.add_patch(patches.Rectangle((c_lo - 0.5, 0), c_hi - c_lo + 1, ylim_max * 1.03, color="grey", alpha=0.3))
    plt.fill_between(x=range(0, number_last_residue + 1), y1=0.01, y2=0.07, color="blue", alpha=0.1)  # beta-sheets
    plt.fill_between(x=range(0, number_last_residue + 1), y1=0.28, y2=0.32, color="red", alpha=0.1)  # alpha-helices
    plt.scatter(range_residues[spacing:-spacing], local_curvatures, color=color, s=size_scatter, label="LCs")
    plt.plot(range_residues[spacing:-spacing], local_curvatures, color=color, linewidth=linewidth, alpha=alpha_line)
    plt.xticks(fontsize=size_xticks)
    plt.yticks(fontsize=size_yticks)
    plt.text(-30, ylim_max, "A", fontsize=50)
    plt.tight_layout()
    plt.savefig(outdir / "LCs.svg", dpi=DPI)
    plt.savefig(outdir / "LCs.png", dpi=150)
    plt.close()

    # Figure B: LFs, RMSF, Neq
    color_LFs, color_Neq, color_RMSF = "black", "blue", "red"
    ylabel = "Local Flexibilities ($Å^{⁻1}$)"
    lf_max = limit("LF", 0.09, local_flexibilities)
    rmsf_max = limit("RMSF", 22, rmsf[inner])
    neq_max = limit("Neq", 8, neq_by_position[inner])
    fig, ax_LFs = plt.subplots(figsize=size_figure_tuple)
    ax_RMSF = ax_LFs.twinx()
    ax_Neq = ax_LFs.twinx()
    plt.xlim(number_first_residue - 0.5, number_last_residue + 0.5)
    ax_Neq.scatter(range_residues[spacing:-spacing], neq_by_position[inner], color=color_Neq, s=size_scatter, zorder=1)
    ax_Neq.plot(range_residues[spacing:-spacing], neq_by_position[inner], color=color_Neq, linewidth=linewidth,
                alpha=alpha_line, zorder=1)
    ax_Neq.set_ylabel("Neq ($a.u.$)", fontsize=size_ylabel, color=color_Neq)
    ax_Neq.set_ylim([0.5, neq_max])
    ax_Neq.tick_params(labelsize=15)
    ax_Neq.tick_params(axis="y", colors=color_Neq)
    ax_RMSF.scatter(range_residues[spacing:-spacing], rmsf[inner], color=color_RMSF, s=size_scatter)
    ax_RMSF.plot(range_residues[spacing:-spacing], rmsf[inner], color=color_RMSF, linewidth=linewidth,
                 alpha=alpha_line, zorder=5)
    ax_RMSF.set_ylabel("RMSF ($Å$)", fontsize=size_ylabel, color=color_RMSF)
    ax_RMSF.set_ylim([0.0, rmsf_max])
    ax_RMSF.spines["right"].set_position(("outward", 60))
    ax_RMSF.tick_params(labelsize=15)
    ax_RMSF.tick_params(axis="y", colors=color_RMSF)
    ax_LFs.add_patch(patches.Rectangle((m_lo - 0.5, 0), m_hi - m_lo + 1, lf_max * 1.35, color="grey", alpha=0.3))
    ax_LFs.add_patch(patches.Rectangle((c_lo - 0.5, 0), c_hi - c_lo + 1, lf_max * 1.35, color="grey", alpha=0.3))
    ax_LFs.set_xlabel(xlabel, fontsize=size_xlabel)
    ax_LFs.set_ylabel(ylabel, fontsize=size_ylabel)
    ax_LFs.scatter(range_residues[spacing:-spacing], local_flexibilities, color=color_LFs, s=size_scatter, zorder=10)
    ax_LFs.plot(range_residues[spacing:-spacing], local_flexibilities, color=color_LFs, linewidth=linewidth,
                alpha=1.0, zorder=10)
    ax_LFs.set_ylim([0.0, lf_max])
    ax_LFs.tick_params(labelsize=15)
    ax_LFs.tick_params(axis="y", colors=color_LFs)
    ax_LFs.axhline(y=0.07, color="purple", linestyle="--", linewidth=4)  # LF of a polyalanine
    ax_LFs.text(-30, lf_max * 1.11, "B", fontsize=50)
    plt.tight_layout()
    plt.savefig(outdir / "LFs.svg", dpi=DPI)
    plt.savefig(outdir / "LFs.png", dpi=150)
    plt.close()

    try:
        import svg_stack as ss

        doc = ss.Document()
        layout = ss.VBoxLayout()
        layout.addSVG(str(outdir / "LCs.svg"), alignment=ss.AlignCenter)
        layout.addSVG(str(outdir / "LFs.svg"), alignment=ss.AlignCenter)
        doc.setLayout(layout)
        doc.save(str(outdir / "LCs_and_LFs.svg"))
    except ImportError:
        print("svg_stack not installed: LCs_and_LFs.svg not written")

    # ---------------- agreement between the figure-B profiles ----------------
    from profile_agreement import (agreement_table, dssp_consensus_from_npz, residue_sets, table_csv,
                                   table_markdown)

    res_inner = resids[inner]
    prof = {"LF": np.asarray(local_flexibilities, float), "Neq": neq_by_position[inner].astype(float),
            "RMSF": np.asarray(rmsf[inner], float)}
    consensus = dssp_consensus_from_npz(args.dssp_npz) if args.dssp_npz else None
    sets = residue_sets(res_inner, {"M-loop": (m_lo, m_hi), "C-terminal tail": (c_lo, c_hi)}, consensus)
    rows = agreement_table(prof, res_inner, sets, block=args.bootstrap_block, n_boot=args.n_boot)
    head = (f"# Agreement between the profiles of figure B ({args.segid}, {n_frames} frames)\n\n"
            f"Residues {res_inner[0]}–{res_inner[-1]} (those with a PMC). Spearman ρ and Pearson r; 95 % "
            f"confidence intervals from a moving-block bootstrap over residues ({args.n_boot} resamples, blocks "
            f"of {args.bootstrap_block} consecutive residues, shorter for short regions; the block column of the "
            f"CSV gives the size used). M-loop {m_lo}–{m_hi}, C-terminal tail {c_lo}–{c_hi}.\n")
    (outdir / "profile_agreement.md").write_text(head + "\n" + "\n".join(table_markdown(rows)) + "\n")
    (outdir / "profile_agreement.csv").write_text("\n".join(table_csv(rows)) + "\n")
    print((outdir / "profile_agreement.md").read_text())

    info = {"args": vars(args), "n_frames": n_frames, "n_residues": len(resids), "timings": timings,
            "axis_limits_raised": raised, "versions": {"MDAnalysis": mda.__version__, "pbxplore": pbx.__version__}}
    (outdir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
