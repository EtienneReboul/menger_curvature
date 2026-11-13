# Agreement between the figure-B profiles across replicas (chains) and time blocks

5 chains (C, E, G, I, K), 10000 frames = 1000 ns (100 ps per frame). Blocks: 4 × 250 ns. Moving windows: 250 ns wide every 25 ns (31 windows). Profiles compared on residues 3–449 (447 residues). R² = squared Pearson correlation between two residue profiles.

Spearman ρ between two profiles of the same chain. *Chains*: whole-trajectory profile of each chain; mean over the 5 chains with its 95 % t confidence interval. *Blocks*: the 4 × 5 block profiles; mean and range. Pearson r (mean over chains) for reference. DSSP classes are each chain's consensus over the whole trajectory.

| residues | n | pair | ρ chains: mean [95 % CI] | ρ per chain | ρ blocks: mean (min–max) | r chains |
|---|---|---|---|---|---|---|
| all residues | 447 | LF vs Neq | 0.64 [0.60, 0.68] | 0.66 0.69 0.61 0.62 0.62 | 0.57 (0.52–0.63) | 0.74 |
| all residues | 447 | LF vs RMSF | 0.44 [0.41, 0.48] | 0.49 0.44 0.44 0.43 0.41 | 0.44 (0.36–0.50) | 0.59 |
| all residues | 447 | Neq vs RMSF | 0.43 [0.40, 0.46] | 0.47 0.44 0.42 0.42 0.40 | 0.42 (0.33–0.50) | 0.61 |
| M-loop | 14 | LF vs Neq | 0.47 [0.25, 0.68] | 0.38 0.34 0.78 0.42 0.42 | 0.47 (0.19–0.88) | 0.38 |
| M-loop | 14 | LF vs RMSF | 0.40 [-0.20, 1.00] | 0.61 -0.02 0.83 0.78 -0.22 | 0.28 (-0.18–0.91) | 0.35 |
| M-loop | 14 | Neq vs RMSF | 0.34 [-0.12, 0.80] | 0.53 0.14 0.80 0.41 -0.17 | 0.33 (-0.23–0.83) | 0.39 |
| C-terminal tail | 14 | LF vs Neq | 0.39 [-0.01, 0.78] | -0.04 0.15 0.49 0.71 0.62 | 0.43 (-0.22–0.90) | 0.42 |
| C-terminal tail | 14 | LF vs RMSF | -0.03 [-0.31, 0.24] | -0.33 -0.18 0.10 0.00 0.24 | 0.27 (-0.14–0.82) | 0.03 |
| C-terminal tail | 14 | Neq vs RMSF | 0.23 [-0.08, 0.54] | 0.38 0.31 0.35 -0.21 0.33 | 0.38 (-0.25–0.89) | 0.21 |
| rest (outside named regions) | 418 | LF vs Neq | 0.59 [0.53, 0.64] | 0.61 0.66 0.54 0.56 0.56 | 0.51 (0.46–0.59) | 0.64 |
| rest (outside named regions) | 418 | LF vs RMSF | 0.36 [0.32, 0.39] | 0.39 0.37 0.34 0.34 0.33 | 0.35 (0.25–0.43) | 0.63 |
| rest (outside named regions) | 418 | Neq vs RMSF | 0.35 [0.31, 0.39] | 0.38 0.38 0.33 0.33 0.32 | 0.35 (0.21–0.44) | 0.65 |
| helix (DSSP consensus) | 186-197 | LF vs Neq | 0.38 [0.30, 0.47] | 0.47 0.44 0.33 0.39 0.30 | 0.30 (0.19–0.38) | 0.39 |
| helix (DSSP consensus) | 186-197 | LF vs RMSF | 0.40 [0.34, 0.46] | 0.42 0.33 0.38 0.43 0.44 | 0.41 (0.25–0.52) | 0.40 |
| helix (DSSP consensus) | 186-197 | Neq vs RMSF | 0.23 [0.17, 0.29] | 0.27 0.28 0.16 0.24 0.22 | 0.18 (0.02–0.30) | 0.20 |
| strand (DSSP consensus) | 68-72 | LF vs Neq | 0.40 [0.35, 0.45] | 0.39 0.44 0.39 0.35 0.44 | 0.32 (0.18–0.41) | 0.27 |
| strand (DSSP consensus) | 68-72 | LF vs RMSF | 0.27 [0.16, 0.38] | 0.28 0.37 0.25 0.31 0.14 | 0.37 (0.08–0.59) | 0.35 |
| strand (DSSP consensus) | 68-72 | Neq vs RMSF | 0.15 [0.08, 0.22] | 0.16 0.20 0.06 0.15 0.19 | 0.20 (0.03–0.38) | 0.18 |
| coil (DSSP consensus) | 181-189 | LF vs Neq | 0.51 [0.44, 0.58] | 0.55 0.56 0.42 0.51 0.51 | 0.43 (0.35–0.49) | 0.71 |
| coil (DSSP consensus) | 181-189 | LF vs RMSF | 0.57 [0.51, 0.64] | 0.64 0.59 0.51 0.55 0.58 | 0.55 (0.44–0.66) | 0.58 |
| coil (DSSP consensus) | 181-189 | Neq vs RMSF | 0.56 [0.51, 0.61] | 0.59 0.60 0.50 0.56 0.56 | 0.51 (0.42–0.58) | 0.59 |
