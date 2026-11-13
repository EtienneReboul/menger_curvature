# Moving windows vs whole trajectory (per chain)

5 chains (C, E, G, I, K), 10000 frames = 1000 ns (100 ps per frame). Blocks: 4 × 250 ns. Moving windows: 250 ns wide every 25 ns (31 windows). Profiles compared on residues 3–449 (447 residues). R² = squared Pearson correlation between two residue profiles.

Mean (min) over the moving windows of R²(window profile, whole-trajectory profile).

| metric | C | E | G | I | K |
|---|---|---|---|---|---|
| LC | 0.989 (0.978) | 0.991 (0.984) | 0.990 (0.982) | 0.993 (0.986) | 0.993 (0.989) |
| LF | 0.767 (0.636) | 0.840 (0.738) | 0.839 (0.714) | 0.807 (0.738) | 0.851 (0.733) |
| RMSF | 0.916 (0.503) | 0.981 (0.960) | 0.973 (0.887) | 0.976 (0.939) | 0.915 (0.535) |
| Neq | 0.682 (0.570) | 0.804 (0.682) | 0.802 (0.647) | 0.807 (0.730) | 0.811 (0.596) |
