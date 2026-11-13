"""Inter-chain convergence: R² of LC and LF between every pair of chains.

Reads the ``per_chain/<chain>.npz`` files written by ``tubulin_pf_analysis.py``.
For each chain, LC (time mean of the PMC) and LF (time standard deviation) are
computed over the whole trajectory; R² is the squared Pearson correlation
between two chains over the PMC positions (residue by residue). Chains must be
copies of the same monomer (identical central-residue numbering).

Writes ``interchain_r2.png`` (two heatmaps, colour scale 0-1) and
``interchain_r2.md/.csv`` into the analysis directory.

usage: python plot_interchain_r2.py <analysis_outdir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    chains, lc, lf, resids = [], [], [], []
    for f in sorted((outdir / "per_chain").glob("*.npz")):
        d = np.load(f)
        chains.append(f.stem)
        lc.append(d["curvature"].mean(axis=0))
        lf.append(d["curvature"].std(axis=0))
        resids.append(d["central_resids"])
    for c, r in zip(chains[1:], resids[1:]):
        if not np.array_equal(r, resids[0]):
            raise SystemExit(f"chain {c} has a different residue numbering than chain {chains[0]}")

    n_frames = int(np.load(outdir / "per_chain" / f"{chains[0]}.npz")["curvature"].shape[0])
    mats = {"LC": np.corrcoef(np.array(lc)) ** 2, "LF": np.corrcoef(np.array(lf)) ** 2}

    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (name, m) in zip(axes, mats.items()):
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(chains)), chains)
        ax.set_yticks(range(len(chains)), chains)
        ax.set_xlabel("chain")
        ax.set_ylabel("chain")
        ax.set_title(f"R² of {name} between chains")
        for i in range(len(chains)):
            for j in range(len(chains)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        color="black" if m[i, j] > 0.6 else "white", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="R²")
    fig.suptitle(f"Inter-chain agreement, whole trajectory ({n_frames} frames, {len(resids[0])} PMC positions)")
    fig.tight_layout()
    fig.savefig(outdir / "interchain_r2.png", dpi=300)

    md = [f"# Inter-chain R² (whole trajectory, {n_frames} frames, {len(resids[0])} PMC positions)", ""]
    csv = ["metric,chain_a,chain_b,r2"]
    for name, m in mats.items():
        md += [f"## {name}", "", "| | " + " | ".join(chains) + " |", "|---" * (len(chains) + 1) + "|"]
        md += [f"| {a} | " + " | ".join(f"{m[i, j]:.3f}" for j in range(len(chains))) + " |"
               for i, a in enumerate(chains)]
        off = m[~np.eye(len(chains), dtype=bool)]
        md += ["", f"Off-diagonal: min {off.min():.3f}, mean {off.mean():.3f}, max {off.max():.3f}", ""]
        csv += [f"{name},{a},{b},{m[i, j]:.6f}" for i, a in enumerate(chains) for j, b in enumerate(chains)]
        print(f"{name}: off-diagonal R² min {off.min():.3f} mean {off.mean():.3f} max {off.max():.3f}")
    (outdir / "interchain_r2.md").write_text("\n".join(md) + "\n")
    (outdir / "interchain_r2.csv").write_text("\n".join(csv) + "\n")
    print((outdir / "interchain_r2.md").read_text())


if __name__ == "__main__":
    main()
