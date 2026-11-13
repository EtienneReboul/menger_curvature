# Agreement between chains (copies of the same monomer), per time block

5 chains (C, E, G, I, K), 10000 frames = 1000 ns (100 ps per frame). Blocks: 4 × 250 ns. Moving windows: 250 ns wide every 25 ns (31 windows). Profiles compared on residues 3–449 (447 residues). R² = squared Pearson correlation between two residue profiles.

Mean (min) of the off-diagonal chain-vs-chain R².

| metric | block_1 | block_2 | block_3 | block_4 | full |
|---|---|---|---|---|---|
| LC | 0.975 (0.966) | 0.953 (0.933) | 0.949 (0.928) | 0.942 (0.924) | 0.970 (0.959) |
| LF | 0.706 (0.646) | 0.596 (0.475) | 0.628 (0.422) | 0.608 (0.451) | 0.705 (0.569) |
| RMSF | 0.963 (0.941) | 0.832 (0.611) | 0.929 (0.884) | 0.899 (0.783) | 0.969 (0.954) |
| Neq | 0.683 (0.545) | 0.450 (0.250) | 0.527 (0.362) | 0.573 (0.437) | 0.685 (0.600) |
