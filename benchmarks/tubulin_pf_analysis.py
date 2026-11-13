"""Convergence check and PMC distributions per secondary structure for a tubulin protofilament.

For each requested chain (one tubulin monomer per segment ``<prefix><chain>``):

1. Menger curvatures (PMCs) of the Cα trace are computed over the whole
   trajectory with :class:`menger.analysis.mengercurvature.MengerCurvature`.
2. The secondary structure of every residue in every frame is assigned with
   MDAnalysis' DSSP (``MDAnalysis.analysis.dssp.DSSP``, MDAnalysis >= 2.11,
   PyDSSP 0.9.1 algorithm), using the distopia backend for the distance
   calculations.

Outputs (``--outdir``):

* ``convergence.md/.csv``: for each chain, the squared Pearson correlation (R²)
  between the local curvatures (LC, time mean of the PMC) and between the local
  flexibilities (LF, time standard deviation of the PMC) computed on the first
  and on the second half of the trajectory, plus the mean absolute differences.
* ``pmc_by_ss.md/.csv``: summary statistics of the PMC distribution for each
  DSSP class (helix ``H``, strand ``E``, coil/loop ``-``), pooled over chains
  and per chain. A PMC computed on the triplet (i - s, i, i + s) is assigned
  the DSSP class of its central residue i in the same frame.
* ``pmc_by_ss_hist.csv``: histogram counts (common bins) per class, pooled and
  per chain, so the figure can be redrawn.
* ``convergence.png`` and ``pmc_by_ss.png``: figures.
* ``per_chain/<chain>.npz``: curvature array, DSSP codes, resids, for reuse.
* ``run_info.json``: versions, timings, settings.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

SS_CODES = {0: "-", 1: "H", 2: "E"}  # order of DSSP.results.dssp_ndarray's last axis
SS_NAMES = {"H": "helix", "E": "strand", "-": "coil"}


def r2(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def analyse_chain(u, segid: str, spacing: int, n_workers: int, dssp_backend: str, stop: int | None):
    from MDAnalysis.analysis.dssp import DSSP

    from menger.analysis.mengercurvature import MengerCurvature

    run_kwargs = {"backend": "multiprocessing", "n_workers": n_workers} if n_workers > 1 else {}
    run_kwargs["stop"] = stop
    timings = {}

    t0 = time.perf_counter()
    mc = MengerCurvature(u, select=f"segid {segid} and name CA", spacing=spacing).run(**run_kwargs)
    timings["menger_s"] = time.perf_counter() - t0
    curvature = np.asarray(mc.results.curvature_array, dtype=np.float32)
    ca_resids = np.asarray(mc.residue_ids)

    t0 = time.perf_counter()
    dssp = DSSP(u.select_atoms(f"segid {segid}"), backend=dssp_backend).run(**run_kwargs)
    timings["dssp_s"] = time.perf_counter() - t0
    ss_codes = np.asarray(dssp.results.dssp_ndarray).argmax(axis=2).astype(np.uint8)
    dssp_resids = np.asarray(dssp.results.resids)

    # DSSP class of the central residue of every Menger triplet
    central = ca_resids[spacing: len(ca_resids) - spacing]
    index = {int(r): i for i, r in enumerate(dssp_resids)}
    missing = [int(r) for r in central if int(r) not in index]
    if missing:
        raise RuntimeError(f"{segid}: {len(missing)} Cα residues without DSSP assignment, e.g. {missing[:5]}")
    ss_central = ss_codes[:, [index[int(r)] for r in central]]
    assert ss_central.shape == curvature.shape, (ss_central.shape, curvature.shape)
    return curvature, ss_central, ca_resids, central, dssp_resids, ss_codes, timings


def convergence_row(chain: str, curvature: np.ndarray) -> dict:
    n = curvature.shape[0]
    half = n // 2
    first, second = curvature[:half], curvature[half: 2 * half]
    lc1, lc2 = first.mean(axis=0), second.mean(axis=0)
    lf1, lf2 = first.std(axis=0), second.std(axis=0)
    return {
        "chain": chain,
        "n_pmc_positions": int(curvature.shape[1]),
        "frames_per_half": int(half),
        "r2_lc": r2(lc1, lc2),
        "r2_lf": r2(lf1, lf2),
        "mean_abs_diff_lc": float(np.abs(lc1 - lc2).mean()),
        "mean_abs_diff_lf": float(np.abs(lf1 - lf2).mean()),
        "_lc": (lc1, lc2),
        "_lf": (lf1, lf2),
    }


def ss_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {"n": int(values.size), "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "p5": float(p5), "q1": float(p25), "median": float(p50), "q3": float(p75), "p95": float(p95)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--chains", nargs="+", default=["C", "E", "G", "I", "K"])
    ap.add_argument("--segid-prefix", default="PRO", help="segment id = prefix + chain letter")
    ap.add_argument("--spacing", type=int, default=2)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--dssp-backend", default="distopia", help="DSSP distance backend ('serial' or 'distopia')")
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--stop", type=int, default=None, help="last frame (exclusive); default: all frames")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    import MDAnalysis as mda

    import menger

    outdir = Path(args.outdir)
    (outdir / "per_chain").mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    u = mda.Universe(args.topology, args.trajectory)
    info = {
        "topology": args.topology, "trajectory": args.trajectory,
        "n_frames": int(u.trajectory.n_frames), "n_atoms": int(u.atoms.n_atoms),
        "dt_ps": float(u.trajectory.dt), "chains": args.chains, "segid_prefix": args.segid_prefix,
        "spacing": args.spacing, "n_workers": args.n_workers, "dssp_backend": args.dssp_backend,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {}, "timings": {},
    }
    for mod in ("MDAnalysis", "numpy", "numba", "distopia", "scipy"):
        try:
            info["versions"][mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001
            info["versions"][mod] = f"unavailable ({exc.__class__.__name__})"
    info["versions"]["menger_curvature"] = getattr(menger, "__version__", "unknown")
    print(json.dumps({k: info[k] for k in ("n_frames", "n_atoms", "dt_ps", "versions")}), flush=True)

    conv_rows, curv_by_chain, ss_by_chain = [], {}, {}
    for chain in args.chains:
        segid = f"{args.segid_prefix}{chain}"
        curvature, ss_central, ca_resids, central, dssp_resids, ss_codes, timings = analyse_chain(
            u, segid, args.spacing, args.n_workers, args.dssp_backend, args.stop)
        info["timings"][chain] = timings
        np.savez_compressed(outdir / "per_chain" / f"{chain}.npz", curvature=curvature,
                            ss_central=ss_central, ca_resids=ca_resids, central_resids=central,
                            dssp_resids=dssp_resids, dssp_codes=ss_codes)
        conv_rows.append(convergence_row(chain, curvature))
        curv_by_chain[chain], ss_by_chain[chain] = curvature, ss_central
        frac = {SS_CODES[k]: float((ss_central == k).mean()) for k in SS_CODES}
        print(f"chain {chain}: {curvature.shape} PMC, menger {timings['menger_s']:.1f} s, "
              f"dssp {timings['dssp_s']:.1f} s, SS fractions {frac}, "
              f"R2 LC {conv_rows[-1]['r2_lc']:.3f} LF {conv_rows[-1]['r2_lf']:.3f}", flush=True)

    # ---------------- convergence table ----------------
    n_frames = curv_by_chain[args.chains[0]].shape[0]
    lines = ["# Convergence: LC and LF, first half vs second half of the trajectory", "",
             f"Trajectory: {n_frames} frames; halves = frames [0, {n_frames // 2}) and "
             f"[{n_frames // 2}, {2 * (n_frames // 2)}). Spacing s = {args.spacing}. "
             "R² = squared Pearson correlation over the PMC positions of the chain.", "",
             "| chain | PMC positions | frames per half | R² LC | R² LF | mean abs. ΔLC (Å⁻¹) | mean abs. ΔLF (Å⁻¹) |",
             "|---|---|---|---|---|---|---|"]
    csv = ["chain,n_pmc_positions,frames_per_half,r2_lc,r2_lf,mean_abs_diff_lc,mean_abs_diff_lf"]
    for r in conv_rows:
        lines.append(f"| {r['chain']} | {r['n_pmc_positions']} | {r['frames_per_half']} | {r['r2_lc']:.3f} | "
                     f"{r['r2_lf']:.3f} | {r['mean_abs_diff_lc']:.4f} | {r['mean_abs_diff_lf']:.4f} |")
        csv.append(f"{r['chain']},{r['n_pmc_positions']},{r['frames_per_half']},{r['r2_lc']:.6f},"
                   f"{r['r2_lf']:.6f},{r['mean_abs_diff_lc']:.6f},{r['mean_abs_diff_lf']:.6f}")
    (outdir / "convergence.md").write_text("\n".join(lines) + "\n")
    (outdir / "convergence.csv").write_text("\n".join(csv) + "\n")

    # ---------------- PMC by secondary structure ----------------
    all_curv = np.concatenate([curv_by_chain[c].ravel() for c in args.chains])
    all_ss = np.concatenate([ss_by_chain[c].ravel() for c in args.chains])
    edges = np.linspace(0.0, float(np.percentile(all_curv, 99.99)), args.bins + 1)
    stat_rows, hist_rows = [], []
    for scope in ["pooled", *args.chains]:
        curv = all_curv if scope == "pooled" else curv_by_chain[scope].ravel()
        ss = all_ss if scope == "pooled" else ss_by_chain[scope].ravel()
        for code, letter in SS_CODES.items():
            vals = curv[ss == code]
            st = ss_stats(vals)
            st.update({"scope": scope, "ss": letter, "fraction": float(vals.size / curv.size)})
            stat_rows.append(st)
            counts, _ = np.histogram(vals, bins=edges)
            hist_rows.append((scope, letter, counts))

    lines = ["# PMC distribution by secondary structure (DSSP class of the central residue)", "",
             f"Chains {', '.join(args.chains)}; {n_frames} frames; DSSP from MDAnalysis "
             f"{info['versions']['MDAnalysis']} (backend `{args.dssp_backend}`). PMC in Å⁻¹.", "",
             "| scope | class | values | fraction | mean | SD | P5 | Q1 | median | Q3 | P95 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    csv = ["scope,ss,n,fraction,mean,sd,p5,q1,median,q3,p95"]
    for st in stat_rows:
        name = f"{SS_NAMES[st['ss']]} ({st['ss']})"
        if st["n"] == 0:
            lines.append(f"| {st['scope']} | {name} | 0 | 0 | – | – | – | – | – | – | – |")
            csv.append(f"{st['scope']},{st['ss']},0,0,,,,,,,")
            continue
        lines.append(f"| {st['scope']} | {name} | {st['n']} | {st['fraction']:.3f} | {st['mean']:.4f} | "
                     f"{st['sd']:.4f} | {st['p5']:.4f} | {st['q1']:.4f} | {st['median']:.4f} | "
                     f"{st['q3']:.4f} | {st['p95']:.4f} |")
        csv.append(",".join(str(st[k]) for k in ("scope", "ss", "n", "fraction", "mean", "sd",
                                                   "p5", "q1", "median", "q3", "p95")))
    (outdir / "pmc_by_ss.md").write_text("\n".join(lines) + "\n")
    (outdir / "pmc_by_ss.csv").write_text("\n".join(csv) + "\n")
    hist_csv = ["scope,ss,bin_left,bin_right,count"]
    for scope, letter, counts in hist_rows:
        for k, c in enumerate(counts):
            hist_csv.append(f"{scope},{letter},{edges[k]:.6f},{edges[k + 1]:.6f},{int(c)}")
    (outdir / "pmc_by_ss_hist.csv").write_text("\n".join(hist_csv) + "\n")

    # ---------------- figures ----------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"H": "#d62728", "E": "#1f77b4", "-": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]
    for scope, letter, counts in hist_rows:
        if scope != "pooled" or counts.sum() == 0:
            continue
        frac = next(s["fraction"] for s in stat_rows if s["scope"] == "pooled" and s["ss"] == letter)
        ax.plot(centres, counts / (counts.sum() * width), color=colours[letter], lw=1.6,
                label=f"{SS_NAMES[letter]} ({frac:.0%} of PMCs)")
    ax.set_xlabel("PMC (Å$^{-1}$)")
    ax.set_ylabel("probability density")
    ax.set_title(f"PMC by DSSP class, chains {' '.join(args.chains)}, {n_frames} frames")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "pmc_by_ss.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(conv_rows), figsize=(3 * len(conv_rows), 6), squeeze=False)
    for j, r in enumerate(conv_rows):
        for i, (key, label) in enumerate((("_lc", "LC"), ("_lf", "LF"))):
            a, b = r[key]
            ax = axes[i, j]
            ax.scatter(a, b, s=4, alpha=0.6)
            lim = [0, max(a.max(), b.max()) * 1.05]
            ax.plot(lim, lim, color="grey", lw=0.8)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_title(f"chain {r['chain']} {label}: R² = {r['r2' + key]:.3f}", fontsize=9)
            ax.set_xlabel("first half (Å$^{-1}$)", fontsize=8)
            ax.set_ylabel("second half (Å$^{-1}$)", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "convergence.png", dpi=200)
    plt.close(fig)

    info["timings"]["total_s"] = time.perf_counter() - t_start
    for r in conv_rows:
        r.pop("_lc")
        r.pop("_lf")
    info["convergence"] = conv_rows
    info["pmc_by_ss"] = stat_rows
    (outdir / "run_info.json").write_text(json.dumps(info, indent=2))
    print((outdir / "convergence.md").read_text())
    print((outdir / "pmc_by_ss.md").read_text())


if __name__ == "__main__":
    main()
