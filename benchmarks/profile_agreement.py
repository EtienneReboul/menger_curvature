"""Quantitative agreement between residue profiles (LF, Neq, RMSF, ...).

Used by ``regenerate_er03_figures.py`` (one chain, whole trajectory) and
``tubulin_pf_blocks.py`` (replica chains and time blocks).

* Spearman ρ (rank correlation; robust to the very different scales and to the
  skewed distributions of RMSF and Neq) and Pearson r.
* 95 % confidence intervals by moving-block bootstrap over residues: residues
  are resampled in contiguous blocks so that the correlation between
  neighbouring residues along the sequence is preserved (an i.i.d. bootstrap or
  the Fisher z interval would assume independent residues and give intervals
  that are too narrow).
* Region-restricted analysis: the same statistics on predefined residue sets
  (named regions, the rest of the protein, and DSSP consensus classes).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

PAIRS = (("LF", "Neq"), ("LF", "RMSF"), ("Neq", "RMSF"))


def _corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of every row of a with the same row of b."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=-1) / den


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(_corr_rows(rankdata(x)[None], rankdata(y)[None])[0])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(_corr_rows(np.asarray(x, float)[None], np.asarray(y, float)[None])[0])


def block_bootstrap_indices(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) indices made of contiguous residue blocks (moving-block bootstrap)."""
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n]


def agreement(x: np.ndarray, y: np.ndarray, block: int = 10, n_boot: int = 10000, seed: int = 0) -> dict:
    """Spearman ρ and Pearson r with 95 % moving-block-bootstrap confidence intervals."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    out = {"n": n, "spearman": spearman(x, y), "pearson": pearson(x, y)}
    if n < 6:
        out.update(spearman_ci=(np.nan, np.nan), pearson_ci=(np.nan, np.nan), block=None)
        return out
    blk = min(block, max(2, n // 4))  # short regions: shorter blocks so that resampling still varies
    idx = block_bootstrap_indices(n, blk, n_boot, np.random.default_rng(seed))
    xs, ys = x[idx], y[idx]
    rho = _corr_rows(rankdata(xs, axis=1), rankdata(ys, axis=1))
    r = _corr_rows(xs, ys)
    out["spearman_ci"] = tuple(float(v) for v in np.nanpercentile(rho, [2.5, 97.5]))
    out["pearson_ci"] = tuple(float(v) for v in np.nanpercentile(r, [2.5, 97.5]))
    out["block"] = blk
    return out


def residue_sets(residues: np.ndarray, regions: dict[str, tuple[int, int]],
                 dssp_consensus: dict[int, str] | None = None) -> dict[str, np.ndarray]:
    """Boolean masks: whole profile, each named region, the rest, and DSSP consensus classes."""
    sets = {"all residues": np.ones(len(residues), bool)}
    named = np.zeros(len(residues), bool)
    for name, (lo, hi) in regions.items():
        m = (residues >= lo) & (residues <= hi)
        if m.sum() >= 2:  # a single residue (e.g. Gly144) has no correlation
            sets[name] = m
        named |= m
    sets["rest (outside named regions)"] = ~named
    if dssp_consensus is not None:
        ss = np.array([dssp_consensus.get(int(r), "?") for r in residues])
        for letter, label in (("H", "helix (DSSP consensus)"), ("E", "strand (DSSP consensus)"),
                              ("-", "coil (DSSP consensus)")):
            if (ss == letter).sum() >= 2:
                sets[label] = ss == letter
    return sets


def agreement_table(profiles: dict[str, np.ndarray], residues: np.ndarray, sets: dict[str, np.ndarray],
                    pairs=PAIRS, block: int = 10, n_boot: int = 10000) -> list[dict]:
    rows = []
    for set_name, mask in sets.items():
        for a, b in pairs:
            res = agreement(profiles[a][mask], profiles[b][mask], block=block, n_boot=n_boot)
            rows.append({"residues": set_name, "pair": f"{a} vs {b}", **res})
    return rows


def format_ci(value: float, ci) -> str:
    if ci is None or np.isnan(ci[0]):
        return f"{value:.2f}"
    return f"{value:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def table_markdown(rows: list[dict]) -> list[str]:
    md = ["| residues | n | pair | Spearman ρ [95 % CI] | Pearson r [95 % CI] |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['residues']} | {r['n']} | {r['pair']} | {format_ci(r['spearman'], r['spearman_ci'])} | "
                  f"{format_ci(r['pearson'], r['pearson_ci'])} |")
    return md


def table_csv(rows: list[dict]) -> list[str]:
    out = ["residues,n,pair,spearman,spearman_ci_low,spearman_ci_high,pearson,pearson_ci_low,pearson_ci_high,"
           "bootstrap_block"]
    for r in rows:
        out.append(f"{r['residues']},{r['n']},{r['pair']},{r['spearman']:.6f},{r['spearman_ci'][0]:.6f},"
                   f"{r['spearman_ci'][1]:.6f},{r['pearson']:.6f},{r['pearson_ci'][0]:.6f},"
                   f"{r['pearson_ci'][1]:.6f},{r['block']}")
    return out


def dssp_consensus_from_npz(path) -> dict[int, str]:
    """Most frequent DSSP class per residue from a tubulin_pf_analysis.py per-chain npz."""
    d = np.load(path)
    codes, resids = d["dssp_codes"], d["dssp_resids"]
    counts = np.stack([(codes == k).sum(axis=0) for k in range(3)])
    letters = np.array(list("-HE"))
    return {int(r): str(letters[k]) for r, k in zip(resids, counts.argmax(axis=0))}
