# Persistence of the regions across time blocks and chains

5 chains (C, E, G, I, K), 10000 frames = 1000 ns (100 ps per frame). Blocks: 4 × 250 ns. Moving windows: 250 ns wide every 25 ns (31 windows). Profiles compared on residues 3–449 (447 residues). R² = squared Pearson correlation between two residue profiles.

For each region, the metric is averaged over the region's residues in each of the 4 blocks × 5 chains = 20 profiles. Percentile = rank of the region's value within the same profile (100 = highest residue).

| region | residues | metric | mean | SD | CV (%) | percentile min | median | max |
|---|---|---|---|---|---|---|---|---|
| M-loop | 274-287 | LC | 0.1725 | 0.0221 | 12.8 | 26 | 36 | 41 |
| M-loop | 274-287 | LF | 0.0225 | 0.0072 | 32.2 | 66 | 86 | 96 |
| M-loop | 274-287 | RMSF | 1.5200 | 0.2472 | 16.3 | 71 | 88 | 94 |
| M-loop | 274-287 | Neq | 1.8480 | 0.4181 | 22.6 | 76 | 82 | 94 |
| C-terminal tail | 436-449 | LC | 0.1357 | 0.0103 | 7.6 | 24 | 28 | 30 |
| C-terminal tail | 436-449 | LF | 0.0497 | 0.0061 | 12.2 | 94 | 97 | 99 |
| C-terminal tail | 436-449 | RMSF | 7.7201 | 1.8921 | 24.5 | 96 | 98 | 99 |
| C-terminal tail | 436-449 | Neq | 4.1668 | 0.8680 | 20.8 | 80 | 97 | 98 |
| Gly144 | 144 | LC | 0.2605 | 0.0174 | 6.7 | 47 | 59 | 63 |
| Gly144 | 144 | LF | 0.0089 | 0.0042 | 47.2 | 9 | 34 | 84 |
| Gly144 | 144 | RMSF | 0.6373 | 0.1308 | 20.5 | 3 | 13 | 31 |
| Gly144 | 144 | Neq | 1.3420 | 0.2087 | 15.5 | 67 | 75 | 83 |

Region residues (chain C):

- M-loop: PRO274 VAL275 ILE276 SER277 ALA278 GLU279 LYS280 ALA281 TYR282 HSD283 GLU284 GLN285 LEU286 SER287
- C-terminal tail: GLY436 VAL437 ASP438 SER439 VAL440 GLU441 GLY442 GLU443 GLY444 GLU445 GLU446 GLU447 GLY448 GLU449 GLU450 TYR451
- Gly144: GLY144
