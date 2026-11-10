# Benchmark: tubulin_startmethods

## Environment

| key | value |
|---|---|
| date_utc | 2026-11-10T18:10:58+00:00 |
| python | 3.12.13 |
| python_implementation | CPython |
| os | macOS-27.2.1-arm64-arm-64bit |
| machine | arm64 |
| logical_cpus | 14 |
| default_start_method_multiprocessing | spawn |
| default_start_method_dask | spawn |
| MDAnalysis | 2.10.0 |
| numpy | 2.1.0 |
| numba | 0.65.1 |
| numba_threading_layer_env | default |
| OMP_NUM_THREADS | unset |
| menger_curvature | 1.1.0+14.g29f198b.dirty |
| pbxplore | not installed |
| cpu | Apple M4 Max |
| physical_cpus | 14 |
| ram_gib | 36.0 |
| git_commit | 4da20c3 |
| git_dirty | True |

Repetitions per configuration: **5** (each in a fresh process). Times in seconds, mean ± SD (sample SD, n−1).

`total` = load + preload + JIT + setup + avg. struct. + alignment + analysis (library imports excluded). Peak RSS is the maximum over repetitions; "increment" is the peak above the RSS measured right after imports. Worker RSS is not measurable with forkserver (workers are not children of the benchmark process); with fork it mostly counts pages touched after the fork.

`start method` is the one actually used by each run. MDAnalysis' multiprocessing backend follows Python's default (fork on Linux with Python < 3.14, spawn on macOS); dask's process scheduler uses spawn unless `multiprocessing.context` is set.

| method | backend | start method | I/O | workers | frames | sel. atoms | load | preload | JIT | avg. struct. | alignment | analysis | total | peak RSS main (MiB) | peak increment main (MiB) | peak RSS largest worker (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| menger_serial | serial | – | disk | 1 | 1000 | 450 | 0.014 ± 0.000 | – | 0.991 ± 0.006 | – | – | 0.250 ± 0.007 | 1.256 ± 0.006 | 224 | 86 | – |
| menger_parallel | multiprocessing | fork | disk | 1 | 1000 | 450 | 0.014 ± 0.000 | – | 0.991 ± 0.014 | – | – | 0.270 ± 0.010 | 1.276 ± 0.023 | 235 | 96 | 49 |
| menger_parallel | multiprocessing | fork | disk | 2 | 1000 | 450 | 0.014 ± 0.000 | – | 0.977 ± 0.006 | – | – | 0.155 ± 0.006 | 1.146 ± 0.009 | 239 | 101 | 46 |
| menger_parallel | multiprocessing | fork | disk | 4 | 1000 | 450 | 0.014 ± 0.000 | – | 0.978 ± 0.006 | – | – | 0.106 ± 0.004 | 1.098 ± 0.008 | 244 | 106 | 46 |
| menger_parallel | multiprocessing | fork | disk | 8 | 1000 | 450 | 0.014 ± 0.000 | – | 1.011 ± 0.032 | – | – | 0.091 ± 0.012 | 1.116 ± 0.028 | 258 | 121 | 46 |
| menger_parallel | multiprocessing | spawn | disk | 1 | 1000 | 450 | 0.014 ± 0.002 | – | 0.984 ± 0.038 | – | – | 1.708 ± 0.051 | 2.707 ± 0.089 | 235 | 98 | 231 |
| menger_parallel | multiprocessing | spawn | disk | 2 | 1000 | 450 | 0.014 ± 0.000 | – | 0.952 ± 0.001 | – | – | 1.599 ± 0.003 | 2.565 ± 0.003 | 242 | 103 | 230 |
| menger_parallel | multiprocessing | spawn | disk | 4 | 1000 | 450 | 0.014 ± 0.000 | – | 0.957 ± 0.003 | – | – | 1.746 ± 0.004 | 2.717 ± 0.005 | 249 | 111 | 230 |
| menger_parallel | multiprocessing | spawn | disk | 8 | 1000 | 450 | 0.014 ± 0.000 | – | 0.962 ± 0.010 | – | – | 1.855 ± 0.013 | 2.831 ± 0.015 | 256 | 118 | 231 |
| menger_parallel | multiprocessing | forkserver | disk | 1 | 1000 | 450 | 0.014 ± 0.000 | – | 0.959 ± 0.002 | – | – | 1.681 ± 0.010 | 2.654 ± 0.009 | 232 | 94 | – |
| menger_parallel | multiprocessing | forkserver | disk | 2 | 1000 | 450 | 0.014 ± 0.000 | – | 0.959 ± 0.003 | – | – | 1.620 ± 0.004 | 2.593 ± 0.006 | 240 | 102 | – |
| menger_parallel | multiprocessing | forkserver | disk | 4 | 1000 | 450 | 0.014 ± 0.000 | – | 0.959 ± 0.001 | – | – | 1.761 ± 0.007 | 2.734 ± 0.007 | 246 | 109 | – |
| menger_parallel | multiprocessing | forkserver | disk | 8 | 1000 | 450 | 0.014 ± 0.000 | – | 0.959 ± 0.004 | – | – | 1.850 ± 0.012 | 2.823 ± 0.015 | 256 | 118 | – |
| menger_parallel | dask | fork | disk | 1 | 1000 | 450 | 0.018 ± 0.005 | – | 0.961 ± 0.016 | – | – | 0.353 ± 0.033 | 1.333 ± 0.030 | 253 | 108 | 55 |
| menger_parallel | dask | fork | disk | 2 | 1000 | 450 | 0.021 ± 0.004 | – | 0.955 ± 0.005 | – | – | 0.219 ± 0.004 | 1.196 ± 0.004 | 258 | 114 | 51 |
| menger_parallel | dask | fork | disk | 4 | 1000 | 450 | 0.021 ± 0.006 | – | 1.000 ± 0.064 | – | – | 0.172 ± 0.003 | 1.193 ± 0.070 | 264 | 117 | 50 |
| menger_parallel | dask | fork | disk | 8 | 1000 | 450 | 0.018 ± 0.005 | – | 0.970 ± 0.022 | – | – | 0.146 ± 0.004 | 1.134 ± 0.025 | 275 | 131 | 50 |
| menger_parallel | dask | spawn | disk | 1 | 1000 | 450 | 0.021 ± 0.004 | – | 0.990 ± 0.104 | – | – | 1.859 ± 0.012 | 2.870 ± 0.105 | 254 | 110 | 239 |
| menger_parallel | dask | spawn | disk | 2 | 1000 | 450 | 0.022 ± 0.004 | – | 0.985 ± 0.014 | – | – | 2.315 ± 0.573 | 3.323 ± 0.562 | 259 | 115 | 239 |
| menger_parallel | dask | spawn | disk | 4 | 1000 | 450 | 0.020 ± 0.005 | – | 0.974 ± 0.026 | – | – | 2.056 ± 0.032 | 3.050 ± 0.057 | 267 | 122 | 237 |
| menger_parallel | dask | spawn | disk | 8 | 1000 | 450 | 0.024 ± 0.001 | – | 0.970 ± 0.026 | – | – | 2.362 ± 0.385 | 3.357 ± 0.408 | 278 | 131 | 238 |
| menger_parallel | dask | forkserver | disk | 1 | 1000 | 450 | 0.021 ± 0.007 | – | 0.991 ± 0.017 | – | – | 1.850 ± 0.056 | 2.863 ± 0.071 | 253 | 110 | – |
| menger_parallel | dask | forkserver | disk | 2 | 1000 | 450 | 0.021 ± 0.004 | – | 0.971 ± 0.034 | – | – | 1.795 ± 0.168 | 2.787 ± 0.203 | 260 | 115 | – |
| menger_parallel | dask | forkserver | disk | 4 | 1000 | 450 | 0.017 ± 0.005 | – | 1.003 ± 0.049 | – | – | 1.881 ± 0.031 | 2.901 ± 0.081 | 268 | 124 | – |
| menger_parallel | dask | forkserver | disk | 8 | 1000 | 450 | 0.018 ± 0.005 | – | 0.956 ± 0.005 | – | – | 1.937 ± 0.009 | 2.911 ± 0.011 | 283 | 140 | – |
