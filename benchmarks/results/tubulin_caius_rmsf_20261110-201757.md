# Benchmark: tubulin_caius_rmsf

## Environment

| key | value |
|---|---|
| date_utc | 2026-11-10T19:04:21+00:00 |
| python | 3.12.16 |
| python_implementation | CPython |
| os | Linux-4.18.0-425.3.1.el8.x86_64-x86_64-with-glibc2.28 |
| machine | x86_64 |
| logical_cpus | 24 |
| default_start_method_multiprocessing | fork |
| default_start_method_dask | spawn |
| MDAnalysis | 2.10.0 |
| numpy | 2.1.0 |
| numba | 0.65.1 |
| numba_threading_layer_env | default |
| OMP_NUM_THREADS | unset |
| menger_curvature | 0.0.0 |
| pbxplore | 1.4.0 |
| cpu | Intel(R) Xeon(R) Gold 6126 CPU @ 2.60GHz |
| physical_cpus | 24 |
| ram_gib | 92.8 |
| git_commit | 4da20c3 |
| git_dirty | True |

Repetitions per configuration: **5** (each in a fresh process). Times in seconds, mean ± SD (sample SD, n−1).

`total` = load + preload + JIT + setup + avg. struct. + alignment + analysis (for rmsf_split_*, `analysis` is the fused alignment + RMSF pass) (library imports excluded). Peak RSS is the maximum over repetitions; "increment" is the peak above the RSS measured right after imports. Worker RSS is not measurable with forkserver (workers are not children of the benchmark process); with fork it mostly counts pages touched after the fork.

`start method` is the one actually used by each run. MDAnalysis' multiprocessing backend follows Python's default (fork on Linux with Python < 3.14, spawn on macOS); dask's process scheduler uses spawn unless `multiprocessing.context` is set.

| method | backend | start method | I/O | workers | frames | sel. atoms | load | preload | JIT | avg. struct. | alignment | analysis | total | peak RSS main (MiB) | peak increment main (MiB) | peak RSS largest worker (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rmsf | serial | – | disk | 1 | 1000 | 450 | 0.051 ± 0.001 | – | – | 1.290 ± 0.007 | 1.212 ± 0.008 | 0.043 ± 0.001 | 2.597 ± 0.015 | 183 | 2 | – |
| rmsf_split_serial | serial | – | disk | 1 | 1000 | 450 | 0.051 ± 0.002 | – | – | 0.424 ± 0.007 | – | 0.373 ± 0.009 | 0.848 ± 0.015 | 182 | 0 | – |
| rmsf_split_parallel | multiprocessing | fork | disk | 1 | 1000 | 450 | 0.049 ± 0.002 | – | – | 1.046 ± 0.040 | – | 0.946 ± 0.060 | 2.042 ± 0.076 | 182 | 0 | 88 |
| rmsf_split_parallel | multiprocessing | fork | disk | 2 | 1000 | 450 | 0.050 ± 0.002 | – | – | 0.476 ± 0.023 | – | 0.424 ± 0.006 | 0.951 ± 0.026 | 182 | 0 | 88 |
| rmsf_split_parallel | multiprocessing | fork | disk | 4 | 1000 | 450 | 0.050 ± 0.000 | – | – | 0.294 ± 0.004 | – | 0.263 ± 0.005 | 0.607 ± 0.007 | 182 | 0 | 88 |
| rmsf_split_parallel | multiprocessing | fork | disk | 8 | 1000 | 450 | 0.050 ± 0.001 | – | – | 0.225 ± 0.003 | – | 0.208 ± 0.001 | 0.484 ± 0.002 | 182 | 0 | 88 |
| rmsf_split_parallel | multiprocessing | spawn | disk | 1 | 1000 | 450 | 0.050 ± 0.001 | – | – | 3.076 ± 0.073 | – | 3.002 ± 0.085 | 6.129 ± 0.111 | 182 | 0 | 103 |
| rmsf_split_parallel | multiprocessing | spawn | disk | 2 | 1000 | 450 | 0.049 ± 0.001 | – | – | 2.421 ± 0.047 | – | 2.370 ± 0.035 | 4.841 ± 0.078 | 182 | 0 | 100 |
| rmsf_split_parallel | multiprocessing | spawn | disk | 4 | 1000 | 450 | 0.051 ± 0.000 | – | – | 2.198 ± 0.045 | – | 2.172 ± 0.045 | 4.422 ± 0.076 | 182 | 0 | 102 |
| rmsf_split_parallel | multiprocessing | spawn | disk | 8 | 1000 | 450 | 0.050 ± 0.001 | – | – | 2.496 ± 0.059 | – | 3.422 ± 2.245 | 5.968 ± 2.251 | 182 | 0 | 102 |
| rmsf_split_parallel | multiprocessing | forkserver | disk | 1 | 1000 | 450 | 0.050 ± 0.001 | – | – | 3.119 ± 0.061 | – | 2.863 ± 0.024 | 6.032 ± 0.079 | 182 | 0 | – |
| rmsf_split_parallel | multiprocessing | forkserver | disk | 2 | 1000 | 450 | 0.051 ± 0.002 | – | – | 2.490 ± 0.030 | – | 2.268 ± 0.023 | 4.810 ± 0.046 | 182 | 0 | – |
| rmsf_split_parallel | multiprocessing | forkserver | disk | 4 | 1000 | 450 | 0.050 ± 0.001 | – | – | 2.188 ± 0.040 | – | 2.007 ± 0.045 | 4.246 ± 0.025 | 182 | 0 | – |
| rmsf_split_parallel | multiprocessing | forkserver | disk | 8 | 1000 | 450 | 0.050 ± 0.001 | – | – | 2.452 ± 0.066 | – | 2.204 ± 0.023 | 4.706 ± 0.058 | 182 | 0 | – |
| rmsf_split_parallel | dask | fork | disk | 1 | 1000 | 450 | 0.052 ± 0.001 | – | – | 1.469 ± 0.019 | – | 0.951 ± 0.018 | 2.473 ± 0.027 | 182 | 0 | 95 |
| rmsf_split_parallel | dask | fork | disk | 2 | 1000 | 450 | 0.052 ± 0.002 | – | – | 0.922 ± 0.006 | – | 0.435 ± 0.004 | 1.410 ± 0.009 | 182 | 0 | 97 |
| rmsf_split_parallel | dask | fork | disk | 4 | 1000 | 450 | 0.053 ± 0.003 | – | – | 0.755 ± 0.008 | – | 0.278 ± 0.005 | 1.086 ± 0.008 | 182 | 0 | 101 |
| rmsf_split_parallel | dask | fork | disk | 8 | 1000 | 450 | 0.051 ± 0.001 | – | – | 0.711 ± 0.003 | – | 0.233 ± 0.003 | 0.996 ± 0.006 | 182 | 0 | 107 |
| rmsf_split_parallel | dask | spawn | disk | 1 | 1000 | 450 | 0.053 ± 0.001 | – | – | 3.913 ± 0.042 | – | 3.378 ± 0.034 | 7.345 ± 0.044 | 182 | 0 | 117 |
| rmsf_split_parallel | dask | spawn | disk | 2 | 1000 | 450 | 0.052 ± 0.001 | – | – | 3.190 ± 0.040 | – | 2.719 ± 0.041 | 5.962 ± 0.053 | 182 | 0 | 120 |
| rmsf_split_parallel | dask | spawn | disk | 4 | 1000 | 450 | 0.053 ± 0.002 | – | – | 3.040 ± 0.052 | – | 2.531 ± 0.045 | 5.626 ± 0.043 | 182 | 0 | 120 |
| rmsf_split_parallel | dask | spawn | disk | 8 | 1000 | 450 | 0.051 ± 0.002 | – | – | 3.260 ± 0.051 | – | 2.759 ± 0.069 | 6.072 ± 0.056 | 182 | 0 | 125 |
| rmsf_split_parallel | dask | forkserver | disk | 1 | 1000 | 450 | 0.052 ± 0.001 | – | – | 3.842 ± 0.077 | – | 3.159 ± 0.068 | 7.054 ± 0.108 | 182 | 0 | – |
| rmsf_split_parallel | dask | forkserver | disk | 2 | 1000 | 450 | 0.052 ± 0.001 | – | – | 3.164 ± 0.024 | – | 2.510 ± 0.023 | 5.727 ± 0.035 | 182 | 0 | – |
| rmsf_split_parallel | dask | forkserver | disk | 4 | 1000 | 450 | 0.052 ± 0.002 | – | – | 2.894 ± 0.023 | – | 2.230 ± 0.042 | 5.177 ± 0.053 | 182 | 0 | – |
| rmsf_split_parallel | dask | forkserver | disk | 8 | 1000 | 450 | 0.050 ± 0.001 | – | – | 3.120 ± 0.039 | – | 2.437 ± 0.039 | 5.608 ± 0.077 | 182 | 0 | – |
