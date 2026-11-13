"""Time-block, moving-window and inter-chain robustness of LC, LF, RMSF and Neq.

Answers the request to "evaluate LF, LC, RMSF and Neq over consecutive time
blocks and overlapping windows, repeat the comparison over independent copies,
and quantify the agreement", using the copies of the same tubulin monomer in a
protofilament (chains C, E, G, I, K by default) in place of independent replicas.

Windows (frames, half-open):
* ``block_1..block_N``: N consecutive, non-overlapping blocks (default 4);
* ``win_XXXX``: overlapping moving windows of ``--window`` frames, every
  ``--stride`` frames (default 2500 / 250 frames = 250 / 25 ns);
* ``full``: the whole trajectory.

Per chain and window, four residue profiles are computed:
* LC and LF: mean and standard deviation of the PMC (Menger curvature, spacing s);
* RMSF of the Cα after the MDAnalysis User Guide recipe (AverageStructure with
  ref_frame = first frame of the window, AlignTraj in memory, RMSF), applied to
  a universe holding only the chain's Cα atoms. The fit uses only the Cα
  selection, so the Cα RMSF is identical to the recipe applied to the whole
  system (checked by ``--check-rmsf``);
* Neq from PBxplore protein blocks assigned on the chain backbone.

All four profiles are compared on the residues that have a PMC (s+1 .. N-s).
Agreement = squared Pearson correlation (R²) between two profiles.

Agreement between the figure-B profiles (LF vs Neq, LF vs RMSF, Neq vs RMSF):
Spearman ρ and Pearson r for every chain (whole trajectory) and every 250 ns
block of every chain, on all residues, each named region, the rest of the
protein and the DSSP consensus classes. Reported as the mean over the chains
(replicas) with a 95 % t confidence interval (n = number of chains), and the
range over all block × chain profiles (``profile_agreement_replicas``).

Outputs (``--outdir``): ``block_agreement``, ``interchain_agreement``,
``window_vs_full``, ``regions``, ``profile_agreement_replicas`` (.md/.csv), figures ``profiles_blocks.png``,
``moving_window_r2.png``, ``regions_windows.png``, ``profiles.npz`` (all
window profiles) and ``run_info.json``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

import numpy as np

METRICS = ("LC", "LF", "RMSF", "Neq")
# Okabe-Ito palette (colour-blind safe)
CHAIN_COLOURS = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2", "#F0E442", "#000000"]

# filled by main() before the worker pool forks
_G: dict = {}


def parse_regions(spec: list[str]) -> dict[str, tuple[int, int]]:
    """'M-loop:274-286' -> {'M-loop': (274, 286)} ; 'Gly144:144' -> (144, 144)."""
    out = {}
    for item in spec:
        name, rng = item.split(":")
        a, _, b = rng.partition("-")
        out[name] = (int(a), int(b or a))
    return out


def windows(n_frames: int, n_blocks: int, width: int, stride: int) -> list[tuple[str, int, int]]:
    w = []
    size = n_frames // n_blocks
    w += [(f"block_{k + 1}", k * size, (k + 1) * size) for k in range(n_blocks)]
    w += [(f"win_{a:05d}", a, a + width) for a in range(0, n_frames - width + 1, stride)]
    w.append(("full", 0, n_frames))
    return w


def r2(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def rmsf_recipe(ca_atoms, coords: np.ndarray) -> np.ndarray:
    """MDAnalysis User Guide RMSF recipe on a Cα-only in-memory universe."""
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms

    u = mda.Merge(ca_atoms).load_new(np.ascontiguousarray(coords, dtype=np.float32), order="fac")
    average = align.AverageStructure(u, u, select="name CA", ref_frame=0).run()
    align.AlignTraj(u, average.results.universe, select="name CA", in_memory=True).run()
    return rms.RMSF(u.atoms).run().results.rmsf


def chain_worker(chain: str) -> tuple[str, dict, dict]:
    """PB assignment + RMSF for every window of one chain (runs in a forked process)."""
    import pbxplore as pbx

    t0 = time.perf_counter()
    bb = _G["bb_coords"][chain]  # (n_frames, n_bb, 3)
    structure = _G["pb_structures"][chain]
    sequences = []
    for frame in bb:
        structure.set_coordinates(frame)
        sequences.append(pbx.assign(structure.get_phi_psi_angles()))
    t_pb = time.perf_counter() - t0

    t0 = time.perf_counter()
    ca = bb[:, _G["ca_index"][chain], :]
    out = {"Neq": {}, "RMSF": {}}
    for name, a, b in _G["windows"]:
        cm = pbx.analysis.count_matrix(sequences[a:b])
        out["Neq"][name] = np.asarray(pbx.analysis.compute_neq(cm), dtype=float)
        out["RMSF"][name] = rmsf_recipe(_G["ca_atoms"][chain], ca[a:b])
    t_win = time.perf_counter() - t0
    return chain, out, {"pb_assign_s": t_pb, "rmsf_neq_windows_s": t_win}


def check_rmsf_equivalence() -> float:
    """Max |ΔRMSF| between the recipe on the full system and on a Cα-only universe."""
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms

    from menger.data import files

    u = mda.Universe(str(files.TUBULIN_CHAIN_A_PDB), str(files.TUBULIN_CHAIN_A_DCD))
    sel = "protein and name CA"
    avg = align.AverageStructure(u, u, select=sel, ref_frame=0).run()
    align.AlignTraj(u, avg.results.universe, select=sel, in_memory=True).run()
    full = rms.RMSF(u.select_atoms(sel)).run().results.rmsf

    u2 = mda.Universe(str(files.TUBULIN_CHAIN_A_PDB), str(files.TUBULIN_CHAIN_A_DCD))
    ca = u2.select_atoms(sel)
    coords = np.array([ca.positions for _ in u2.trajectory])
    return float(np.abs(rmsf_recipe(ca, coords) - full).max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology")
    ap.add_argument("--trajectory")
    ap.add_argument("--chains", nargs="+", default=["C", "E", "G", "I", "K"])
    ap.add_argument("--segid-prefix", default="PRO")
    ap.add_argument("--spacing", type=int, default=2)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--window", type=int, default=2500, help="moving-window width (frames)")
    ap.add_argument("--stride", type=int, default=250, help="moving-window stride (frames)")
    ap.add_argument("--frame-ps", type=float, default=100.0,
                    help="time between saved frames in ps (the DCD header of the Zenodo files is wrong)")
    ap.add_argument("--regions", nargs="+", default=["M-loop:274-287", "C-terminal tail:436-451", "Gly144:144"])
    ap.add_argument("--stop", type=int, default=None, help="only use the first STOP frames (testing)")
    ap.add_argument("--dssp-dir", default=None,
                    help="per_chain/ directory of tubulin_pf_analysis.py (DSSP consensus classes for the "
                         "region-restricted agreement between profiles)")
    ap.add_argument("--outdir")
    ap.add_argument("--check-rmsf", action="store_true", help="only run the RMSF equivalence check and exit")
    args = ap.parse_args()

    if args.check_rmsf:
        print(f"max |RMSF(full system) - RMSF(Cα-only universe)| = {check_rmsf_equivalence():.2e} Å")
        return

    import MDAnalysis as mda
    import pbxplore
    from pbxplore.structure.structure import Atom, Chain

    import menger
    from menger.analysis.mengercurvature import MengerCurvature

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    timings = {}
    regions = parse_regions(args.regions)

    # ---- one pass over the trajectory: backbone coordinates of every chain ----
    u = mda.Universe(args.topology, args.trajectory)
    bb_groups = {c: u.select_atoms(f"segid {args.segid_prefix}{c} and backbone") for c in args.chains}
    n_frames = u.trajectory.n_frames if args.stop is None else min(args.stop, u.trajectory.n_frames)
    t0 = time.perf_counter()
    bb_coords = {c: np.empty((n_frames, g.n_atoms, 3), dtype=np.float32) for c, g in bb_groups.items()}
    for i, _ in enumerate(u.trajectory[:n_frames]):
        for c, g in bb_groups.items():
            bb_coords[c][i] = g.positions
    timings["read_trajectory_s"] = time.perf_counter() - t0
    print(f"read {n_frames} frames in {timings['read_trajectory_s']:.1f} s", flush=True)

    ca_index = {c: np.flatnonzero(g.names == "CA") for c, g in bb_groups.items()}
    ca_atoms = {c: bb_groups[c][ca_index[c]] for c in args.chains}
    resids = {c: ca_atoms[c].resids.copy() for c in args.chains}
    for c in args.chains[1:]:
        if not np.array_equal(resids[c], resids[args.chains[0]]):
            raise SystemExit(f"chain {c} numbering differs from chain {args.chains[0]}")
    resids = resids[args.chains[0]]
    s = args.spacing
    common = resids[s: len(resids) - s]  # residues with a PMC
    keep = np.arange(s, len(resids) - s)  # their index in the per-residue (RMSF, Neq) profiles

    wins = windows(n_frames, args.blocks, args.window, args.stride)

    # ---- PMC -> LC, LF per window (Menger kernel on the Cα-only universe) ----
    t0 = time.perf_counter()
    profiles: dict = {m: {c: {} for c in args.chains} for m in METRICS}
    for c in args.chains:
        ca_xyz = bb_coords[c][:, ca_index[c], :]
        u_ca = mda.Merge(ca_atoms[c]).load_new(np.ascontiguousarray(ca_xyz), order="fac")
        curv = MengerCurvature(u_ca, select="name CA", spacing=s).run().results.curvature_array
        for name, a, b in wins:
            profiles["LC"][c][name] = curv[a:b].mean(axis=0).astype(float)
            profiles["LF"][c][name] = curv[a:b].std(axis=0).astype(float)
    timings["menger_s"] = time.perf_counter() - t0
    print(f"Menger LC/LF done in {timings['menger_s']:.1f} s", flush=True)

    # ---- PB assignment, Neq and RMSF per window, one forked process per chain ----
    pb_structures = {}
    for c, g in bb_groups.items():
        chain = Chain()
        for atm in g:
            chain.add_atom(Atom.read_from_xtc(atm))
        pb_structures[c] = chain
    _G.update(bb_coords=bb_coords, pb_structures=pb_structures, ca_index=ca_index,
              ca_atoms=ca_atoms, windows=wins)
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(len(args.chains)) as pool:
        for c, out, t in pool.imap_unordered(chain_worker, args.chains):
            profiles["Neq"][c] = {k: v[keep] for k, v in out["Neq"].items()}
            profiles["RMSF"][c] = {k: v[keep] for k, v in out["RMSF"].items()}
            timings[f"chain_{c}"] = t
            print(f"chain {c}: PB {t['pb_assign_s']:.0f} s, RMSF+Neq windows {t['rmsf_neq_windows_s']:.0f} s",
                  flush=True)
    timings["rmsf_neq_s"] = time.perf_counter() - t0

    # ---------------- agreement ----------------
    ns = args.frame_ps / 1000.0
    block_names = [w[0] for w in wins if w[0].startswith("block_")]
    mov = [w for w in wins if w[0].startswith("win_")]
    chains = args.chains

    def mean_offdiag(m):
        off = m[~np.eye(len(m), dtype=bool)]
        return float(off.mean()), float(off.min())

    # (a) block vs block, per chain
    block_rows = []
    for metric in METRICS:
        for c in chains:
            mat = np.array([[r2(profiles[metric][c][x], profiles[metric][c][y]) for y in block_names]
                            for x in block_names])
            mean, low = mean_offdiag(mat)
            block_rows.append({"metric": metric, "chain": c, "mean_r2": mean, "min_r2": low,
                               "matrix": mat.round(4).tolist()})
    # (b) inter-chain, per block and full
    inter_rows = []
    for metric in METRICS:
        for name in block_names + ["full"]:
            mat = np.array([[r2(profiles[metric][a][name], profiles[metric][b][name]) for b in chains]
                            for a in chains])
            mean, low = mean_offdiag(mat)
            inter_rows.append({"metric": metric, "window": name, "mean_r2": mean, "min_r2": low,
                               "matrix": mat.round(4).tolist()})
    # (c) moving windows vs full trajectory, per chain
    mov_r2 = {metric: {c: np.array([r2(profiles[metric][c][w[0]], profiles[metric][c]["full"]) for w in mov])
                       for c in chains} for metric in METRICS}
    centres_ns = np.array([(a + b) / 2 * ns for _, a, b in mov])
    # (d) regions: mean value over the region residues and its percentile rank in the chain profile
    region_rows = []
    region_track = {}
    for rname, (lo, hi) in regions.items():
        mask = (common >= lo) & (common <= hi)
        if not mask.any():
            raise SystemExit(f"region {rname} ({lo}-{hi}) has no residue with a PMC")
        for metric in METRICS:
            vals, ranks = [], []
            for c in chains:
                for name in block_names:
                    prof = profiles[metric][c][name]
                    v = float(prof[mask].mean())
                    vals.append(v)
                    ranks.append(float((prof < v).mean() * 100))
            vals, ranks = np.array(vals), np.array(ranks)
            full_vals = [float(profiles[metric][c]["full"][mask].mean()) for c in chains]
            region_rows.append({
                # residues actually used: the region clipped to the residues that have a PMC
                "region": rname, "residues": (f"{common[mask][0]}-{common[mask][-1]}"
                                              if mask.sum() > 1 else str(common[mask][0])),
                "n_residues": int(mask.sum()), "metric": metric,
                "mean_blocks_chains": float(vals.mean()), "sd_blocks_chains": float(vals.std(ddof=1)),
                "cv_percent": float(vals.std(ddof=1) / vals.mean() * 100),
                "percentile_min": float(ranks.min()), "percentile_median": float(np.median(ranks)),
                "percentile_max": float(ranks.max()), "full_by_chain": dict(zip(chains, full_vals)),
            })
            region_track[(rname, metric)] = {
                c: np.array([float(profiles[metric][c][w[0]][mask].mean()) for w in mov]) for c in chains}

    # (e) agreement between the figure-B profiles, over replicas (chains) and blocks
    from scipy import stats

    from profile_agreement import PAIRS, dssp_consensus_from_npz, pearson, residue_sets, spearman

    named = {k: v for k, v in regions.items()}
    agree_rows = []
    per_chain_sets = {}
    for c in chains:
        cons = dssp_consensus_from_npz(Path(args.dssp_dir) / f"{c}.npz") if args.dssp_dir else None
        per_chain_sets[c] = residue_sets(common, named, cons)
    set_names = list(dict.fromkeys(n for c in chains for n in per_chain_sets[c]))
    for set_name in set_names:
        for a, b in PAIRS:
            full_rho, full_r, blk_rho, n_res = [], [], [], []
            for c in chains:
                mask = per_chain_sets[c].get(set_name)
                if mask is None:
                    continue
                n_res.append(int(mask.sum()))
                full_rho.append(spearman(profiles[a][c]["full"][mask], profiles[b][c]["full"][mask]))
                full_r.append(pearson(profiles[a][c]["full"][mask], profiles[b][c]["full"][mask]))
                blk_rho += [spearman(profiles[a][c][w][mask], profiles[b][c][w][mask]) for w in block_names]
            k = len(full_rho)
            mean = float(np.mean(full_rho))
            half = float(stats.t.ppf(0.975, k - 1) * np.std(full_rho, ddof=1) / np.sqrt(k)) if k > 1 else np.nan
            agree_rows.append({
                "residues": set_name, "pair": f"{a} vs {b}", "n_residues": f"{min(n_res)}-{max(n_res)}"
                if min(n_res) != max(n_res) else str(n_res[0]), "n_chains": k,
                "spearman_mean_chains": mean, "spearman_ci_low": mean - half, "spearman_ci_high": mean + half,
                "spearman_by_chain": dict(zip(chains, [round(v, 4) for v in full_rho])),
                "pearson_mean_chains": float(np.mean(full_r)),
                "spearman_blocks_min": float(np.min(blk_rho)), "spearman_blocks_max": float(np.max(blk_rho)),
                "spearman_blocks_mean": float(np.mean(blk_rho)),
            })

    # ---------------- tables ----------------
    total_ns = n_frames * ns
    head = (f"{len(chains)} chains ({', '.join(chains)}), {n_frames} frames = {total_ns:.0f} ns "
            f"({args.frame_ps:.0f} ps per frame). Blocks: {args.blocks} × {n_frames // args.blocks * ns:.0f} ns. "
            f"Moving windows: {args.window * ns:.0f} ns wide every {args.stride * ns:.0f} ns ({len(mov)} windows). "
            f"Profiles compared on residues {common[0]}–{common[-1]} ({len(common)} residues). "
            "R² = squared Pearson correlation between two residue profiles.")
    md = ["# Agreement between consecutive time blocks (per chain)", "", head, "",
          "Mean (min) of the off-diagonal block-vs-block R².", "",
          "| metric | " + " | ".join(chains) + " |", "|---" * (len(chains) + 1) + "|"]
    for metric in METRICS:
        rows = {r["chain"]: r for r in block_rows if r["metric"] == metric}
        md.append(f"| {metric} | " + " | ".join(f"{rows[c]['mean_r2']:.3f} ({rows[c]['min_r2']:.3f})"
                                                for c in chains) + " |")
    (outdir / "block_agreement.md").write_text("\n".join(md) + "\n")
    (outdir / "block_agreement.csv").write_text(
        "metric,chain,mean_r2,min_r2\n" + "\n".join(
            f"{r['metric']},{r['chain']},{r['mean_r2']:.6f},{r['min_r2']:.6f}" for r in block_rows) + "\n")

    md = ["# Agreement between chains (copies of the same monomer), per time block", "", head, "",
          "Mean (min) of the off-diagonal chain-vs-chain R².", "",
          "| metric | " + " | ".join(block_names + ["full"]) + " |", "|---" * (len(block_names) + 2) + "|"]
    for metric in METRICS:
        rows = {r["window"]: r for r in inter_rows if r["metric"] == metric}
        md.append(f"| {metric} | " + " | ".join(f"{rows[n]['mean_r2']:.3f} ({rows[n]['min_r2']:.3f})"
                                                for n in block_names + ["full"]) + " |")
    (outdir / "interchain_agreement.md").write_text("\n".join(md) + "\n")
    (outdir / "interchain_agreement.csv").write_text(
        "metric,window,mean_r2,min_r2\n" + "\n".join(
            f"{r['metric']},{r['window']},{r['mean_r2']:.6f},{r['min_r2']:.6f}" for r in inter_rows) + "\n")

    md = ["# Moving windows vs whole trajectory (per chain)", "", head, "",
          "Mean (min) over the moving windows of R²(window profile, whole-trajectory profile).", "",
          "| metric | " + " | ".join(chains) + " |", "|---" * (len(chains) + 1) + "|"]
    csv = ["metric,chain,window_centre_ns,r2_vs_full"]
    for metric in METRICS:
        md.append(f"| {metric} | " + " | ".join(
            f"{mov_r2[metric][c].mean():.3f} ({mov_r2[metric][c].min():.3f})" for c in chains) + " |")
        csv += [f"{metric},{c},{t:.1f},{v:.6f}" for c in chains for t, v in zip(centres_ns, mov_r2[metric][c])]
    (outdir / "window_vs_full.md").write_text("\n".join(md) + "\n")
    (outdir / "window_vs_full.csv").write_text("\n".join(csv) + "\n")

    md = ["# Agreement between the figure-B profiles across replicas (chains) and time blocks", "", head, "",
          "Spearman ρ between two profiles of the same chain. *Chains*: whole-trajectory profile of each chain; "
          f"mean over the {len(chains)} chains with its 95 % t confidence interval. *Blocks*: the "
          f"{args.blocks} × {len(chains)} block profiles; mean and range. Pearson r (mean over chains) for "
          "reference. DSSP classes are each chain's consensus over the whole trajectory.", "",
          "| residues | n | pair | ρ chains: mean [95 % CI] | ρ per chain | ρ blocks: mean (min–max) | r chains |",
          "|---|---|---|---|---|---|---|"]
    for r in agree_rows:
        md.append(f"| {r['residues']} | {r['n_residues']} | {r['pair']} | {r['spearman_mean_chains']:.2f} "
                  f"[{r['spearman_ci_low']:.2f}, {r['spearman_ci_high']:.2f}] | "
                  + " ".join(f"{v:.2f}" for v in r["spearman_by_chain"].values())
                  + f" | {r['spearman_blocks_mean']:.2f} ({r['spearman_blocks_min']:.2f}–{r['spearman_blocks_max']:.2f})"
                  f" | {r['pearson_mean_chains']:.2f} |")
    (outdir / "profile_agreement_replicas.md").write_text("\n".join(md) + "\n")
    (outdir / "profile_agreement_replicas.csv").write_text(
        "residues,n_residues,pair,n_chains,spearman_mean_chains,spearman_ci_low,spearman_ci_high,"
        + ",".join(f"spearman_{c}" for c in chains)
        + ",spearman_blocks_mean,spearman_blocks_min,spearman_blocks_max,pearson_mean_chains\n"
        + "\n".join(f"{r['residues']},{r['n_residues']},{r['pair']},{r['n_chains']},{r['spearman_mean_chains']:.6f},"
                    f"{r['spearman_ci_low']:.6f},{r['spearman_ci_high']:.6f},"
                    + ",".join(f"{r['spearman_by_chain'].get(c, float('nan')):.6f}" for c in chains)
                    + f",{r['spearman_blocks_mean']:.6f},{r['spearman_blocks_min']:.6f},"
                    f"{r['spearman_blocks_max']:.6f},{r['pearson_mean_chains']:.6f}" for r in agree_rows) + "\n")

    seq = {rn: " ".join(f"{n}{r}" for n, r in zip(ca_atoms[chains[0]].resnames, resids) if lo <= r <= hi)
           for rn, (lo, hi) in regions.items()}
    md = ["# Persistence of the regions across time blocks and chains", "", head, "",
          f"For each region, the metric is averaged over the region's residues in each of the "
          f"{args.blocks} blocks × {len(chains)} chains = {args.blocks * len(chains)} profiles. "
          "Percentile = rank of the region's value within the same profile (100 = highest residue).", "",
          "| region | residues | metric | mean | SD | CV (%) | percentile min | median | max |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in region_rows:
        md.append(f"| {r['region']} | {r['residues']} | {r['metric']} | {r['mean_blocks_chains']:.4f} | "
                  f"{r['sd_blocks_chains']:.4f} | {r['cv_percent']:.1f} | {r['percentile_min']:.0f} | "
                  f"{r['percentile_median']:.0f} | {r['percentile_max']:.0f} |")
    md += ["", "Region residues (chain " + chains[0] + "):", ""] + [f"- {rn}: {seq[rn]}" for rn in regions]
    (outdir / "regions.md").write_text("\n".join(md) + "\n")
    (outdir / "regions.csv").write_text(
        "region,residues,n_residues,metric,mean,sd,cv_percent,percentile_min,percentile_median,percentile_max\n"
        + "\n".join(f"{r['region']},{r['residues']},{r['n_residues']},{r['metric']},{r['mean_blocks_chains']:.6f},"
                    f"{r['sd_blocks_chains']:.6f},{r['cv_percent']:.3f},{r['percentile_min']:.1f},"
                    f"{r['percentile_median']:.1f},{r['percentile_max']:.1f}" for r in region_rows) + "\n")

    # ---------------- figures ----------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    units = {"LC": "Å$^{-1}$", "LF": "Å$^{-1}$", "RMSF": "Å", "Neq": ""}
    colour = dict(zip(chains, CHAIN_COLOURS))
    region_shade = ["#999999", "#BBBBBB", "#777777"]

    # profiles: every block of every chain, regions shaded
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(14, 12), sharex=True)
    for ax, metric in zip(axes, METRICS):
        for k, (rname, (lo, hi)) in enumerate(regions.items()):
            ax.axvspan(lo - 0.5, hi + 0.5, color=region_shade[k % 3], alpha=0.35, lw=0)
        for c in chains:
            for name in block_names:
                ax.plot(common, profiles[metric][c][name], color=colour[c], lw=0.6, alpha=0.7)
        ax.set_ylabel(f"{metric} ({units[metric]})" if units[metric] else metric)
    for k, rname in enumerate(regions):
        lo, hi = regions[rname]
        axes[0].text((lo + hi) / 2, axes[0].get_ylim()[1], rname, ha="center", va="bottom", fontsize=9)
    handles = [plt.Line2D([], [], color=colour[c], label=f"chain {c}") for c in chains]
    axes[0].legend(handles=handles, ncol=len(chains), loc="upper left", fontsize=9, frameon=False)
    axes[-1].set_xlabel("residue")
    fig.suptitle(f"Profiles of the {args.blocks} blocks × {len(chains)} chains (regions shaded)")
    fig.tight_layout()
    fig.savefig(outdir / "profiles_blocks.png", dpi=200)
    plt.close(fig)

    # moving-window R² vs full trajectory
    fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 4.2), sharey=True)
    for ax, metric in zip(axes, METRICS):
        for c in chains:
            ax.plot(centres_ns, mov_r2[metric][c], color=colour[c], marker="o", ms=3, lw=1.2, label=f"chain {c}")
        ax.set_title(metric)
        ax.set_xlabel("window centre (ns)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("R² (window vs whole trajectory)")
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Moving windows ({args.window * ns:.0f} ns wide, every {args.stride * ns:.0f} ns)")
    fig.tight_layout()
    fig.savefig(outdir / "moving_window_r2.png", dpi=200)
    plt.close(fig)

    # regions over the moving windows
    fig, axes = plt.subplots(len(regions), len(METRICS), figsize=(18, 3.4 * len(regions)), squeeze=False)
    for i, rname in enumerate(regions):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for c in chains:
                ax.plot(centres_ns, region_track[(rname, metric)][c], color=colour[c], lw=1.2)
            ax.set_title(f"{rname}: {metric}", fontsize=10)
            ax.set_ylim(bottom=0)
            if i == len(regions) - 1:
                ax.set_xlabel("window centre (ns)")
    axes[0, -1].legend(handles=handles, frameon=False, fontsize=8)
    fig.suptitle("Region averages over the moving windows")
    fig.tight_layout()
    fig.savefig(outdir / "regions_windows.png", dpi=200)
    plt.close(fig)

    np.savez_compressed(outdir / "profiles.npz", residues=common, window_names=np.array([w[0] for w in wins]),
                        window_frames=np.array([[a, b] for _, a, b in wins]), chains=np.array(chains),
                        **{f"{m}_{c}": np.array([profiles[m][c][w[0]] for w in wins])
                           for m in METRICS for c in chains})
    timings["total_s"] = time.perf_counter() - t_start
    info = {"args": vars(args), "n_frames": n_frames, "total_ns": total_ns, "timings": timings,
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": {"MDAnalysis": mda.__version__, "numpy": np.__version__, "pbxplore": pbxplore.__version__,
                         "menger_curvature": getattr(menger, "__version__", "unknown")},
            "block_agreement": block_rows, "interchain_agreement": inter_rows, "regions": region_rows,
            "profile_agreement_replicas": agree_rows}
    (outdir / "run_info.json").write_text(json.dumps(info, indent=2))
    for f in ("block_agreement.md", "interchain_agreement.md", "window_vs_full.md", "regions.md",
              "profile_agreement_replicas.md"):
        print((outdir / f).read_text())
    print(f"total {timings['total_s']:.0f} s")


if __name__ == "__main__":
    main()
