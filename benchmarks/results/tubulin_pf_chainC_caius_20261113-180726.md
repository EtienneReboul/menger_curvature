# Benchmark: tubulin_pf_chainC_caius

## Environment

| key | value |
|---|---|
| date_utc | 2026-11-13T15:09:24+00:00 |
| python | 3.12.16 |
| python_implementation | CPython |
| os | Linux-4.18.0-425.3.1.el8.x86_64-x86_64-with-glibc2.28 |
| machine | x86_64 |
| logical_cpus | 64 |
| default_start_method_multiprocessing | fork |
| default_start_method_dask | spawn |
| MDAnalysis | 2.11.0 |
| numpy | 2.1.0 |
| numba | 0.65.1 |
| numba_threading_layer_env | default |
| OMP_NUM_THREADS | unset |
| menger_curvature | 1.1.1+2.g4da20c3.dirty |
| pbxplore | 1.4.0 |
| cpu | Intel(R) Xeon(R) Gold 6454S |
| physical_cpus | 64 |
| ram_gib | 219.9 |
| git_commit | 4da20c3 |
| git_dirty | True |

Repetitions per configuration: **5** (each in a fresh process). Times in seconds, mean ± SD (sample SD, n−1).

`total` = load + preload + JIT + setup + avg. struct. + alignment + analysis (for rmsf_split_*, `analysis` is the fused alignment + RMSF pass) (library imports excluded). Peak RSS is the maximum over repetitions; "increment" is the peak above the RSS measured right after imports. Worker RSS is not measurable with forkserver (workers are not children of the benchmark process); with fork it mostly counts pages touched after the fork.

`start method` is the one actually used by each run. MDAnalysis' multiprocessing backend follows Python's default (fork on Linux with Python < 3.14, spawn on macOS); dask's process scheduler uses spawn unless `multiprocessing.context` is set.


Selections used: menger_parallel: `(segid PROC) and (name CA)`; menger_serial: `(segid PROC) and (name CA)`; neq: `(segid PROC) and (backbone)`; rmsf: `(segid PROC) and (protein and name CA)`

| method | backend | start method | I/O | workers | frames | sel. atoms | load | preload | JIT | avg. struct. | alignment | analysis | total | peak RSS main (MiB) | peak increment main (MiB) | peak RSS largest worker (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| menger_serial | serial | – | disk | 1 | 10000 | 451 | 0.331 ± 0.001 | – | 2.288 ± 0.025 | – | – | 25.981 ± 2.185 | 28.606 ± 2.179 | 295 | 96 | – |
| menger_serial | serial | – | memory | 1 | 10000 | 451 | 0.333 ± 0.006 | 24.244 ± 0.154 | 2.279 ± 0.030 | – | – | 2.342 ± 0.013 | 29.204 ± 0.147 | 11667 | 11468 | – |
| menger_parallel | multiprocessing | fork | disk | 1 | 10000 | 451 | 0.332 ± 0.004 | – | 2.289 ± 0.021 | – | – | 26.212 ± 0.424 | 28.839 ± 0.444 | 342 | 143 | 265 |
| menger_parallel | multiprocessing | fork | disk | 2 | 10000 | 451 | 0.328 ± 0.003 | – | 2.283 ± 0.016 | – | – | 13.603 ± 0.062 | 16.219 ± 0.061 | 410 | 211 | 236 |
| menger_parallel | multiprocessing | fork | disk | 4 | 10000 | 451 | 0.329 ± 0.004 | – | 2.287 ± 0.028 | – | – | 7.426 ± 0.034 | 10.048 ± 0.048 | 413 | 214 | 235 |
| menger_parallel | multiprocessing | fork | disk | 8 | 10000 | 451 | 0.329 ± 0.002 | – | 2.281 ± 0.022 | – | – | 5.147 ± 0.031 | 7.763 ± 0.051 | 482 | 283 | 232 |
| menger_parallel | multiprocessing | fork | disk | 16 | 10000 | 451 | 0.331 ± 0.004 | – | 2.280 ± 0.028 | – | – | 4.778 ± 0.032 | 7.395 ± 0.031 | 622 | 423 | 226 |
| menger_parallel | multiprocessing | spawn | disk | 1 | 10000 | 451 | 0.331 ± 0.003 | – | 2.304 ± 0.005 | – | – | 31.239 ± 0.421 | 33.880 ± 0.426 | 283 | 85 | 342 |
| menger_parallel | multiprocessing | spawn | disk | 2 | 10000 | 451 | 0.332 ± 0.003 | – | 2.281 ± 0.020 | – | – | 18.824 ± 0.169 | 21.443 ± 0.165 | 316 | 117 | 310 |
| menger_parallel | multiprocessing | spawn | disk | 4 | 10000 | 451 | 0.330 ± 0.004 | – | 2.297 ± 0.018 | – | – | 13.010 ± 0.104 | 15.642 ± 0.105 | 320 | 121 | 307 |
| menger_parallel | multiprocessing | spawn | disk | 8 | 10000 | 451 | 0.332 ± 0.004 | – | 2.275 ± 0.020 | – | – | 10.950 ± 0.086 | 13.563 ± 0.094 | 389 | 191 | 302 |
| menger_parallel | multiprocessing | spawn | disk | 16 | 10000 | 451 | 0.330 ± 0.002 | – | 2.281 ± 0.007 | – | – | 10.877 ± 0.172 | 13.493 ± 0.171 | 525 | 326 | 300 |
| menger_parallel | multiprocessing | forkserver | disk | 1 | 10000 | 451 | 0.328 ± 0.002 | – | 2.273 ± 0.013 | – | – | 31.214 ± 0.263 | 33.821 ± 0.258 | 283 | 85 | – |
| menger_parallel | multiprocessing | forkserver | disk | 2 | 10000 | 451 | 0.334 ± 0.006 | – | 2.283 ± 0.022 | – | – | 19.054 ± 0.149 | 21.676 ± 0.145 | 316 | 117 | – |
| menger_parallel | multiprocessing | forkserver | disk | 4 | 10000 | 451 | 0.328 ± 0.003 | – | 2.269 ± 0.009 | – | – | 13.089 ± 0.067 | 15.692 ± 0.061 | 318 | 119 | – |
| menger_parallel | multiprocessing | forkserver | disk | 8 | 10000 | 451 | 0.329 ± 0.003 | – | 2.282 ± 0.019 | – | – | 11.090 ± 0.107 | 13.706 ± 0.118 | 389 | 190 | – |
| menger_parallel | multiprocessing | forkserver | disk | 16 | 10000 | 451 | 0.334 ± 0.005 | – | 2.285 ± 0.042 | – | – | 10.860 ± 0.223 | 13.484 ± 0.246 | 527 | 329 | – |
| menger_parallel | dask | fork | disk | 1 | 10000 | 451 | 0.331 ± 0.002 | – | 2.300 ± 0.027 | – | – | 26.512 ± 0.331 | 29.148 ± 0.316 | 359 | 160 | 290 |
| menger_parallel | dask | fork | disk | 2 | 10000 | 451 | 0.331 ± 0.001 | – | 2.255 ± 0.028 | – | – | 14.103 ± 0.085 | 16.694 ± 0.100 | 395 | 196 | 300 |
| menger_parallel | dask | fork | disk | 4 | 10000 | 451 | 0.331 ± 0.002 | – | 2.293 ± 0.031 | – | – | 7.992 ± 0.032 | 10.622 ± 0.050 | 438 | 239 | 313 |
| menger_parallel | dask | fork | disk | 8 | 10000 | 451 | 0.331 ± 0.003 | – | 2.287 ± 0.026 | – | – | 5.727 ± 0.015 | 8.349 ± 0.024 | 504 | 305 | 347 |
| menger_parallel | dask | fork | disk | 16 | 10000 | 451 | 0.329 ± 0.004 | – | 2.286 ± 0.016 | – | – | 5.495 ± 0.022 | 8.114 ± 0.026 | 675 | 477 | 454 |
| menger_parallel | dask | spawn | disk | 1 | 10000 | 451 | 0.331 ± 0.002 | – | 2.304 ± 0.012 | – | – | 32.333 ± 0.247 | 34.974 ± 0.254 | 296 | 97 | 362 |
| menger_parallel | dask | spawn | disk | 2 | 10000 | 451 | 0.330 ± 0.004 | – | 2.286 ± 0.019 | – | – | 19.996 ± 0.219 | 22.618 ± 0.223 | 326 | 128 | 335 |
| menger_parallel | dask | spawn | disk | 4 | 10000 | 451 | 0.337 ± 0.009 | – | 2.299 ± 0.016 | – | – | 14.249 ± 0.052 | 16.891 ± 0.057 | 335 | 136 | 332 |
| menger_parallel | dask | spawn | disk | 8 | 10000 | 451 | 0.332 ± 0.001 | – | 2.295 ± 0.031 | – | – | 12.098 ± 0.133 | 14.730 ± 0.148 | 410 | 212 | 383 |
| menger_parallel | dask | spawn | disk | 16 | 10000 | 451 | 0.332 ± 0.005 | – | 2.324 ± 0.032 | – | – | 12.159 ± 0.171 | 14.820 ± 0.169 | 580 | 381 | 483 |
| menger_parallel | dask | forkserver | disk | 1 | 10000 | 451 | 0.330 ± 0.003 | – | 2.284 ± 0.021 | – | – | 32.210 ± 0.181 | 34.829 ± 0.200 | 294 | 95 | – |
| menger_parallel | dask | forkserver | disk | 2 | 10000 | 451 | 0.332 ± 0.003 | – | 2.293 ± 0.012 | – | – | 20.150 ± 0.125 | 22.780 ± 0.125 | 327 | 128 | – |
| menger_parallel | dask | forkserver | disk | 4 | 10000 | 451 | 0.337 ± 0.003 | – | 2.308 ± 0.019 | – | – | 14.099 ± 0.089 | 16.749 ± 0.093 | 339 | 140 | – |
| menger_parallel | dask | forkserver | disk | 8 | 10000 | 451 | 0.332 ± 0.004 | – | 2.289 ± 0.015 | – | – | 11.972 ± 0.562 | 14.598 ± 0.566 | 411 | 212 | – |
| menger_parallel | dask | forkserver | disk | 16 | 10000 | 451 | 0.333 ± 0.003 | – | 2.304 ± 0.012 | – | – | 13.795 ± 2.824 | 16.438 ± 2.830 | 565 | 367 | – |
| rmsf | serial | – | disk | 1 | 10000 | 451 | 0.331 ± 0.002 | – | – | 104.586 ± 0.968 | 106.754 ± 1.759 | 0.300 ± 0.003 | 211.975 ± 2.607 | 11548 | 11349 | – |
| rmsf | serial | – | memory | 1 | 10000 | 451 | 0.331 ± 0.002 | 24.556 ± 0.388 | – | 83.254 ± 1.048 | 82.465 ± 1.344 | 0.303 ± 0.002 | 190.913 ± 1.988 | 11541 | 11342 | – |
| neq | serial | – | disk | 1 | 10000 | 1803 | 0.335 ± 0.006 | – | – | – | – | 142.011 ± 0.594 | 142.396 ± 0.598 | 199 | 0 | – |
| neq | serial | – | memory | 1 | 10000 | 1803 | 0.334 ± 0.002 | 24.817 ± 0.111 | – | – | – | 120.344 ± 0.698 | 145.536 ± 0.774 | 11556 | 11357 | – |
