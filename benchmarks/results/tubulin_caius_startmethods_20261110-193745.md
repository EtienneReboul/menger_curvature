# Benchmark: tubulin_caius_startmethods

## Environment

| key | value |
|---|---|
| date_utc | 2026-11-10T18:17:47+00:00 |
| python | 3.12.16 |
| python_implementation | CPython |
| os | Linux-4.18.0-425.3.1.el8.x86_64-x86_64-with-glibc2.28 |
| machine | x86_64 |
| logical_cpus | 32 |
| default_start_method_multiprocessing | fork |
| default_start_method_dask | spawn |
| MDAnalysis | 2.10.0 |
| numpy | 2.1.0 |
| numba | 0.65.1 |
| numba_threading_layer_env | default |
| OMP_NUM_THREADS | unset |
| menger_curvature | 0.0.0 |
| pbxplore | 1.4.0 |
| cpu | Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz |
| physical_cpus | 32 |
| ram_gib | 125.3 |
| git_commit | 4da20c3 |
| git_dirty | True |

Repetitions per configuration: **5** (each in a fresh process). Times in seconds, mean ± SD (sample SD, n−1).

`total` = load + preload + JIT + setup + avg. struct. + alignment + analysis (library imports excluded). Peak RSS is the maximum over repetitions; "increment" is the peak above the RSS measured right after imports. Worker RSS is not measurable with forkserver (workers are not children of the benchmark process); with fork it mostly counts pages touched after the fork.

`start method` is the one actually used by each run. MDAnalysis' multiprocessing backend follows Python's default (fork on Linux with Python < 3.14, spawn on macOS); dask's process scheduler uses spawn unless `multiprocessing.context` is set.

| method | backend | start method | I/O | workers | frames | sel. atoms | load | preload | JIT | avg. struct. | alignment | analysis | total | peak RSS main (MiB) | peak increment main (MiB) | peak RSS largest worker (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| menger_serial | serial | – | disk | 1 | 1000 | 450 | 0.046 ± 0.003 | – | 2.375 ± 0.042 | – | – | 0.497 ± 0.006 | 2.920 ± 0.047 | 232 | 49 | – |
| menger_parallel | multiprocessing | fork | disk | 1 | 1000 | 450 | 0.047 ± 0.001 | – | 2.392 ± 0.022 | – | – | 1.278 ± 0.017 | 3.718 ± 0.022 | 234 | 52 | 148 |
| menger_parallel | multiprocessing | fork | disk | 2 | 1000 | 450 | 0.045 ± 0.002 | – | 2.372 ± 0.031 | – | – | 0.671 ± 0.031 | 3.088 ± 0.044 | 239 | 56 | 146 |
| menger_parallel | multiprocessing | fork | disk | 4 | 1000 | 450 | 0.047 ± 0.003 | – | 2.360 ± 0.031 | – | – | 0.428 ± 0.012 | 2.836 ± 0.037 | 241 | 59 | 145 |
| menger_parallel | multiprocessing | fork | disk | 8 | 1000 | 450 | 0.045 ± 0.002 | – | 2.370 ± 0.029 | – | – | 0.336 ± 0.030 | 2.753 ± 0.013 | 246 | 63 | 145 |
| menger_parallel | multiprocessing | spawn | disk | 1 | 1000 | 450 | 0.046 ± 0.002 | – | 2.381 ± 0.041 | – | – | 6.171 ± 0.045 | 8.599 ± 0.049 | 230 | 47 | 236 |
| menger_parallel | multiprocessing | spawn | disk | 2 | 1000 | 450 | 0.044 ± 0.002 | – | 2.375 ± 0.022 | – | – | 5.590 ± 0.082 | 8.010 ± 0.088 | 229 | 46 | 231 |
| menger_parallel | multiprocessing | spawn | disk | 4 | 1000 | 450 | 0.046 ± 0.002 | – | 2.407 ± 0.009 | – | – | 5.627 ± 0.109 | 8.081 ± 0.107 | 230 | 48 | 228 |
| menger_parallel | multiprocessing | spawn | disk | 8 | 1000 | 450 | 0.046 ± 0.003 | – | 2.403 ± 0.011 | – | – | 5.672 ± 0.046 | 8.122 ± 0.047 | 230 | 48 | 228 |
| menger_parallel | multiprocessing | forkserver | disk | 1 | 1000 | 450 | 0.044 ± 0.002 | – | 2.389 ± 0.016 | – | – | 6.288 ± 0.106 | 8.723 ± 0.107 | 229 | 46 | – |
| menger_parallel | multiprocessing | forkserver | disk | 2 | 1000 | 450 | 0.047 ± 0.002 | – | 2.391 ± 0.040 | – | – | 5.711 ± 0.062 | 8.150 ± 0.046 | 229 | 46 | – |
| menger_parallel | multiprocessing | forkserver | disk | 4 | 1000 | 450 | 0.047 ± 0.001 | – | 2.400 ± 0.027 | – | – | 5.589 ± 0.079 | 8.036 ± 0.090 | 230 | 48 | – |
| menger_parallel | multiprocessing | forkserver | disk | 8 | 1000 | 450 | 0.045 ± 0.002 | – | 2.379 ± 0.026 | – | – | 5.740 ± 0.048 | 8.165 ± 0.046 | 230 | 48 | – |
| menger_parallel | dask | fork | disk | 1 | 1000 | 450 | 0.047 ± 0.002 | – | 2.416 ± 0.015 | – | – | 1.772 ± 0.034 | 4.235 ± 0.040 | 252 | 69 | 163 |
| menger_parallel | dask | fork | disk | 2 | 1000 | 450 | 0.047 ± 0.001 | – | 2.409 ± 0.013 | – | – | 1.159 ± 0.020 | 3.615 ± 0.028 | 253 | 71 | 162 |
| menger_parallel | dask | fork | disk | 4 | 1000 | 450 | 0.046 ± 0.002 | – | 2.393 ± 0.018 | – | – | 0.887 ± 0.024 | 3.327 ± 0.036 | 255 | 73 | 163 |
| menger_parallel | dask | fork | disk | 8 | 1000 | 450 | 0.047 ± 0.001 | – | 2.413 ± 0.008 | – | – | 0.801 ± 0.018 | 3.262 ± 0.018 | 262 | 80 | 166 |
| menger_parallel | dask | spawn | disk | 1 | 1000 | 450 | 0.046 ± 0.003 | – | 2.413 ± 0.009 | – | – | 7.209 ± 0.107 | 9.670 ± 0.113 | 242 | 60 | 242 |
| menger_parallel | dask | spawn | disk | 2 | 1000 | 450 | 0.045 ± 0.002 | – | 2.399 ± 0.008 | – | – | 6.730 ± 0.069 | 9.174 ± 0.071 | 244 | 62 | 244 |
| menger_parallel | dask | spawn | disk | 4 | 1000 | 450 | 0.047 ± 0.002 | – | 2.408 ± 0.011 | – | – | 6.654 ± 0.080 | 9.110 ± 0.083 | 246 | 63 | 246 |
| menger_parallel | dask | spawn | disk | 8 | 1000 | 450 | 0.045 ± 0.003 | – | 2.397 ± 0.025 | – | – | 6.916 ± 0.090 | 9.359 ± 0.097 | 249 | 67 | 249 |
| menger_parallel | dask | forkserver | disk | 1 | 1000 | 450 | 0.045 ± 0.001 | – | 2.393 ± 0.012 | – | – | 6.871 ± 0.113 | 9.311 ± 0.116 | 243 | 61 | – |
| menger_parallel | dask | forkserver | disk | 2 | 1000 | 450 | 0.045 ± 0.002 | – | 2.404 ± 0.018 | – | – | 6.334 ± 0.073 | 8.784 ± 0.083 | 246 | 64 | – |
| menger_parallel | dask | forkserver | disk | 4 | 1000 | 450 | 0.046 ± 0.002 | – | 2.406 ± 0.009 | – | – | 6.316 ± 0.078 | 8.769 ± 0.085 | 246 | 64 | – |
| menger_parallel | dask | forkserver | disk | 8 | 1000 | 450 | 0.047 ± 0.001 | – | 2.409 ± 0.018 | – | – | 6.388 ± 0.122 | 8.845 ± 0.128 | 249 | 67 | – |
