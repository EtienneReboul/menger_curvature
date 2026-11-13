# Benchmarks

`benchmark_metrics.py` compares the runtime and memory use of four calculations:

* the Menger-curvature analysis (PMCs, LCs and LFs), run serially and in parallel;
* the RMSF from MDAnalysis;
* the Neq from PBxplore.

It replaces the earlier `notebooks/er02-benchmark_metrics.ipynb`. Every run records
its full methodology, so the results can be reported and reproduced.

```bash
pip install pbxplore            # needed only for the Neq benchmark

# Tubulin chain A (shipped with the package: 1000 frames, 450 Cα, 6924 atoms)
python benchmarks/benchmark_metrics.py --reps 5 --workers 1 2 4 8

# Compare parallel backends and multiprocessing start methods
python benchmarks/benchmark_metrics.py --methods menger_serial menger_parallel \
    --io-modes disk --backends multiprocessing dask --start-methods fork spawn forkserver

# Tau protein (Zenodo record 14900596, 20000 frames; the archive is 6.6 GB)
python benchmarks/benchmark_metrics.py --dataset tau --reps 5 --workers 1 2 4 8 \
    --topology .data/.../topology_Tau_minimized_all_atoms_sequence_WT_renumerote.psf \
    --trajectory .data/.../trajectory_Tau_minimized_all_atoms_sequence_WT.dcd
```

Each invocation writes `results/<dataset>_<timestamp>.json`, which holds the environment
and every individual repetition, and a Markdown summary table `results/<dataset>_<timestamp>.md`.

## Methodology

**a. Repetitions.** The number of repetitions is set with `--reps` and defaults to 5.
Each repetition runs in its own fresh Python process. The reports give the mean and
sample SD (n−1), and the JSON file also stores the individual values.

**b. Numba compilation and warm-up.** The JIT compilation of the curvature kernel is
timed on its own (`JIT` column) and is not counted in the `analysis` time. Because
each repetition starts a new process, every repetition pays the compile cost once,
so the results show both the cost of a cold start (`total`) and the steady-state
cost (`analysis`). Compiled code is not cached on disk (no `cache=True`).

Whether workers reuse the parent's compiled code depends on how they are started.
Use `--start-methods fork spawn forkserver` to force a method; each run records the
method it actually used (`start method` column):

| start method | how workers start | Numba code in the workers |
| --- | --- | --- |
| `fork` | copy of the parent process | already compiled, inherited from the parent |
| `spawn` | fresh interpreter | compiled again by every worker on its first frame |
| `forkserver` | forked from a clean server process | compiled again by every worker on its first frame |

The defaults differ by platform and backend. MDAnalysis's `multiprocessing` backend
uses Python's default: `fork` on Linux before Python 3.14, and `spawn` on macOS and
Windows. Dask's process scheduler (the MDAnalysis `dask` backend) uses `spawn` on
every platform unless `multiprocessing.context` is set in the dask config. With
`spawn` or `forkserver`, the recompilation in the workers cannot be separated from
the parallel `analysis` time and is included in it.

**c. Workers and backends.** The number of worker processes is set with `--workers`
and appears in the `workers` column. Parallel runs use the supported MDAnalysis API,
`run(backend=..., n_workers=N)`, with the backends listed in `--backends`
(`multiprocessing` and/or `dask`). With `n_workers=1`, the run measures the parallel
backend's own overhead on a single worker.

**d. Versions.** Every report starts with a table listing:

* the Python, OS, MDAnalysis, NumPy, Numba, PBxplore and menger_curvature versions;
* the CPU model, core counts and RAM;
* the multiprocessing start method;
* the git commit of the code that was run.

**e. I/O.** All methods measure the following phases separately:

* `load`: parsing the topology and opening the trajectory;
* `analysis`: the analysis itself.

There are two I/O modes. In `disk` mode, frames are read from the file during the
analysis, so trajectory reading is part of `analysis` (this is what a user
experiences). In `memory` mode, the frames are first loaded into RAM, timed on their
own in the `preload` column, so `analysis` contains no file reading. Before the timed
repetitions, the script reads the input files once, so every repetition starts with
the same warm OS file cache. Library imports are timed separately and excluded from
`total`. No results are written to disk while timing.

**f. Same frames and selections.** All methods analyse the same trajectory and the same
frames, set with `--start/--stop/--step`; by default this is every frame. Each run
records how many frames and atoms it actually analysed. The selections differ only
where the method requires it:

| method | selection | atoms used per residue | values per frame |
| --- | --- | --- | --- |
| Menger curvature | `name CA` | Cα | N_res − 2s (s = 2) |
| RMSF | alignment and RMSF on `protein and name CA` | Cα | N_res |
| Neq (PBxplore) | `backbone` | N, Cα, C, O (φ/ψ need N, Cα, C) | N_res |

**RMSF and alignment.** Computing an RMSF requires removing global rotation and
translation first, so the RMSF pipeline includes the alignment. The script times its
three steps separately:

1. the average structure (`AverageStructure`);
2. the alignment of every frame to it (`AlignTraj(in_memory=True)`);
3. the RMSF calculation (`RMSF`).

The RMSF calculation alone (`analysis`) is very cheap, a single pass of running
sums. Almost all of the pipeline's cost comes from the two alignment passes. The
Menger curvature and Neq are both invariant to rotation and translation, so neither
needs an alignment.

**Parallel RMSF.** In MDAnalysis 2.10, `AverageStructure`, `AlignTraj` and `RMSF`
support only the serial backend. To compare RMSF with the parallel Menger
curvature on equal terms, [`parallel_rmsf.py`](parallel_rmsf.py) implements the
same pipeline as two passes that work with MDAnalysis's `multiprocessing` and
`dask` backends:

1. `AverageStructureCA` fits every frame onto frame 0 and sums the fitted Cα
   coordinates. The parent then adds up the per-worker sums.
2. `AlignedRMSF` fits every frame onto the average and keeps a running mean and
   variance (Welford). The parent merges the per-worker statistics with Chan's
   formula.

The fit is the same as in MDAnalysis: unweighted centre of geometry plus QCP
rotation. Over the whole tubulin trajectory, the RMSF agrees with the MDAnalysis
pipeline to within 6 × 10⁻⁷ Å, for every backend and start method.

In the second pass, alignment and RMSF are computed together in one loop over the
frames, so their costs can't be separated; that pass is reported as `analysis`.
The split pipeline also differs from the MDAnalysis one in two ways:

* it transforms only the Cα atoms, not the whole system;
* it keeps no aligned trajectory in memory, so its memory is O(N_res) instead of
  O(N_f · N_at).

To separate the algorithmic gain from the parallel speed-up, it is benchmarked
both serially (`rmsf_split_serial`) and in parallel (`rmsf_split_parallel`).
Unlike the Menger curvature, it uses no Numba. With `spawn` or `forkserver`, each
worker only has to start an interpreter and import MDAnalysis, and it pays this
once per pass.

```bash
python benchmarks/benchmark_metrics.py --methods rmsf rmsf_split_serial rmsf_split_parallel \
    --io-modes disk --backends multiprocessing dask --start-methods fork spawn forkserver
```

**g. Memory.** In each repetition, the script measures the peak resident set size
(`ru_maxrss`) of the main process and of the largest worker process. The
`peak increment` column subtracts the RSS measured after the imports. The expected
scaling uses the following symbols:

* N_f: number of frames;
* N_at: total number of atoms;
* N_res: number of residues (Cα);
* s: spacing;
* W: number of workers.

| method | dominant memory term | scaling |
| --- | --- | --- |
| Menger, serial | `curvature_array`, float32, N_f × (N_res − 2s) | O(N_f · N_res); about 35 MB for Tau (20000 × 437 × 4 B) |
| Menger, parallel | the same array, plus a transient 2× copy while the per-worker blocks are stacked, plus W worker processes each holding an interpreter, Numba, MDAnalysis and a trajectory reader | O(N_f · N_res) + W × (per-worker RSS: about 230 MiB with spawn or forkserver, 50–150 MiB with fork, where pages are shared with the parent) |
| RMSF pipeline | the in-memory aligned trajectory of **all atoms**, N_f × N_at × 3 × 4 B | O(N_f · N_at); 79 MiB for tubulin, about 1.5 GiB for 20000 frames of a 6600-atom system |
| RMSF alone | running mean and variance of the selection | O(N_res), independent of N_f |
| split RMSF (`parallel_rmsf.py`) | per-worker Welford accumulators for the selection, plus W worker processes | O(N_res) + W × per-worker RSS |
| Neq | N_f PB strings of length N_res, plus a 16 × N_res count matrix | O(N_f · N_res) |

In `memory` mode, every method also holds the whole trajectory in RAM
(N_f × N_at × 3 × 4 B).
