#!/usr/bin/env python
"""Reproducible benchmark of Menger curvature (PMC/LC/LF) against RMSF and PBxplore Neq.

Design choices (see benchmarks/README.md for the full methodology):

* Every repetition runs in a **fresh Python process**. This makes Numba
  compilation cost identical in every repetition, prevents caches from one
  method leaking into another, and lets peak memory (``ru_maxrss``) be
  attributed to a single method.
* Each run reports **separate phase timings**: universe loading (topology
  parsing + trajectory opening), optional trajectory pre-loading into memory,
  Numba JIT compilation, and the analysis itself. For RMSF, the average
  structure, the trajectory alignment and the RMSF calculation proper are
  timed separately.
* Two I/O modes are available:
    - ``disk``: frames are streamed from the trajectory file during the
      analysis, so trajectory reading is part of the analysis time (this is
      what a user experiences);
    - ``memory``: the trajectory is first loaded into RAM (timed separately as
      ``preload``) so that the analysis time excludes file reading.
* All methods analyse the **same frames** (``--start/--stop/--step``) of the
  same trajectory; the number of frames and selected atoms actually analysed
  are recorded for each run.
* Before the timed repetitions, the trajectory and topology files are read
  once so that every repetition starts with the same (warm) OS file cache.

Usage
-----
    # tubulin test system shipped with the package
    python benchmarks/benchmark_metrics.py --reps 5 --workers 1 2 4 8

    # Tau protein trajectory (Zenodo record 14900596)
    python benchmarks/benchmark_metrics.py --dataset tau \
        --topology path/to/topology.psf --trajectory path/to/trajectory.dcd

PBxplore must be importable for the Neq benchmark (``pip install pbxplore``);
it is skipped with a warning otherwise.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

MENGER_SELECTION = "name CA"
RMSF_SELECTION = "protein and name CA"
ALIGN_SELECTION = "protein and name CA"
PBX_SELECTION = "backbone"  # PBxplore needs N, CA, C to compute phi/psi
SPACING = 2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def peak_rss_mib(who=resource.RUSAGE_SELF) -> float:
    """Peak resident set size in MiB (ru_maxrss is bytes on macOS, KiB on Linux)."""
    maxrss = resource.getrusage(who).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return maxrss * scale / 2**20


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def _dask_default_context() -> str:
    try:
        import dask

        return dask.config.get("multiprocessing.context", "spawn")
    except ImportError:
        return "dask not installed"


def environment_info() -> dict:
    """Collect hardware and software versions needed to reproduce the benchmark."""
    import multiprocessing as mp

    import MDAnalysis
    import numba
    import numpy

    info = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "default_start_method_multiprocessing": mp.get_start_method(),
        "default_start_method_dask": _dask_default_context(),
        "MDAnalysis": MDAnalysis.__version__,
        "numpy": numpy.__version__,
        "numba": numba.__version__,
        "numba_threading_layer_env": os.environ.get("NUMBA_THREADING_LAYER", "default"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
    }
    try:
        import menger

        info["menger_curvature"] = getattr(menger, "__version__", "unknown")
    except ImportError:
        info["menger_curvature"] = "not installed"
    try:
        import pbxplore

        info["pbxplore"] = pbxplore.__version__
    except ImportError:
        info["pbxplore"] = "not installed"

    if sys.platform == "darwin":
        info["cpu"] = _sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["physical_cpus"] = _sh(["sysctl", "-n", "hw.physicalcpu"])
        mem = _sh(["sysctl", "-n", "hw.memsize"])
        info["ram_gib"] = round(int(mem) / 2**30, 1) if mem else None
    else:
        cpu = ""
        try:
            with open("/proc/cpuinfo", encoding="utf8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        info["cpu"] = cpu
        info["physical_cpus"] = _sh(["nproc", "--all"])
        try:
            info["ram_gib"] = round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1
            )
        except (ValueError, OSError):
            info["ram_gib"] = None

    repo = Path(__file__).resolve().parent.parent
    info["git_commit"] = _sh(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"])
    info["git_dirty"] = bool(_sh(["git", "-C", str(repo), "status", "--porcelain"]))
    return info


# --------------------------------------------------------------------------
# worker: one method, one repetition, fresh process
# --------------------------------------------------------------------------
def _load(cfg: dict, timings: dict):
    import MDAnalysis as mda

    t0 = time.perf_counter()
    u = mda.Universe(cfg["topology"], cfg["trajectory"])
    timings["load_universe"] = time.perf_counter() - t0

    frames = slice(cfg["start"], cfg["stop"], cfg["step"])
    run_kwargs = {"start": cfg["start"], "stop": cfg["stop"], "step": cfg["step"]}
    if cfg["io_mode"] == "memory":
        t0 = time.perf_counter()
        u.transfer_to_memory(**run_kwargs)
        timings["preload"] = time.perf_counter() - t0
        frames = slice(None)
        run_kwargs = {}
    return u, frames, run_kwargs


def run_menger(cfg: dict, timings: dict, meta: dict) -> None:
    from menger.analysis.mengercurvature import MengerCurvature, menger_curvature
    from menger.data.dummy_data import DummyData

    u, _, run_kwargs = _load(cfg, timings)

    # Numba JIT compilation in the parent process, timed separately.
    t0 = time.perf_counter()
    menger_curvature(DummyData().frame, SPACING)
    timings["jit_compile"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    analyser = MengerCurvature(u, select=MENGER_SELECTION, spacing=SPACING)
    timings["setup"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if cfg["n_workers"] is None:
        analyser.run(**run_kwargs)
    else:
        analyser.run(backend=cfg["backend"], n_workers=cfg["n_workers"], **run_kwargs)
    timings["analysis"] = time.perf_counter() - t0

    meta["n_frames"] = int(analyser.results.curvature_array.shape[0])
    meta["n_selected_atoms"] = int(analyser.atomgroup.n_atoms)
    meta["n_output_values_per_frame"] = int(analyser.results.curvature_array.shape[1])
    meta["selection"] = MENGER_SELECTION


def run_rmsf(cfg: dict, timings: dict, meta: dict) -> None:
    from MDAnalysis.analysis import align, rms

    u, _, run_kwargs = _load(cfg, timings)

    t0 = time.perf_counter()
    average = align.AverageStructure(u, u, select=ALIGN_SELECTION, ref_frame=0).run(
        **run_kwargs
    )
    timings["average_structure"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    align.AlignTraj(
        u, average.results.universe, select=ALIGN_SELECTION, in_memory=True
    ).run(**run_kwargs)
    timings["alignment"] = time.perf_counter() - t0

    # after in-memory alignment the trajectory only holds the analysed frames
    c_alphas = u.select_atoms(RMSF_SELECTION)
    t0 = time.perf_counter()
    rmsf = rms.RMSF(c_alphas).run()
    timings["analysis"] = time.perf_counter() - t0

    meta["n_frames"] = int(rmsf.n_frames)
    meta["n_selected_atoms"] = int(c_alphas.n_atoms)
    meta["n_output_values_per_frame"] = int(c_alphas.n_atoms)
    meta["selection"] = RMSF_SELECTION
    meta["n_atoms_aligned_in_memory"] = int(u.atoms.n_atoms)


def run_neq(cfg: dict, timings: dict, meta: dict) -> None:
    import pbxplore as pbx
    from pbxplore.structure.structure import Atom, Chain

    u, frames, _ = _load(cfg, timings)

    # Same steps as pbxplore.chains_from_trajectory, minus its progress printing,
    # so that it can share the universe/frames/I-O mode used by the other methods.
    t0 = time.perf_counter()
    selection = u.select_atoms(PBX_SELECTION)
    structure = Chain()
    for atm in selection:
        structure.add_atom(Atom.read_from_xtc(atm))
    timings["setup"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sequences = []
    for _ in u.trajectory[frames]:
        structure.set_coordinates(selection.positions)
        sequences.append(pbx.assign(structure.get_phi_psi_angles()))
    count_matrix = pbx.analysis.count_matrix(sequences)
    neq = pbx.analysis.compute_neq(count_matrix)
    timings["analysis"] = time.perf_counter() - t0

    meta["n_frames"] = len(sequences)
    meta["n_selected_atoms"] = int(selection.n_atoms)
    meta["n_selected_residues"] = int(selection.residues.n_residues)
    meta["n_output_values_per_frame"] = int(len(neq))
    meta["selection"] = PBX_SELECTION


METHODS = {
    "menger_serial": run_menger,
    "menger_parallel": run_menger,
    "rmsf": run_rmsf,
    "neq": run_neq,
}


IMPORTS = {
    "menger_serial": ["MDAnalysis", "menger.analysis.mengercurvature"],
    "menger_parallel": ["MDAnalysis", "menger.analysis.mengercurvature"],
    "rmsf": ["MDAnalysis", "MDAnalysis.analysis.align", "MDAnalysis.analysis.rms"],
    "neq": ["MDAnalysis", "pbxplore", "pbxplore.structure.structure"],
}


def worker(cfg: dict) -> dict:
    import importlib
    import multiprocessing as mp

    effective_start_method = None
    if cfg.get("backend") == "multiprocessing":
        if cfg.get("start_method"):
            mp.set_start_method(cfg["start_method"], force=True)
        effective_start_method = mp.get_start_method()
    elif cfg.get("backend") == "dask":
        import dask

        if cfg.get("start_method"):
            dask.config.set({"multiprocessing.context": cfg["start_method"]})
        effective_start_method = dask.config.get("multiprocessing.context", "spawn")
    timings: dict = {}
    meta: dict = {}
    # Library imports are timed separately and excluded from "total".
    t0 = time.perf_counter()
    for module in IMPORTS[cfg["method"]]:
        importlib.import_module(module)
    timings["imports"] = time.perf_counter() - t0
    baseline = peak_rss_mib()

    t0 = time.perf_counter()
    METHODS[cfg["method"]](cfg, timings, meta)
    timings["total"] = time.perf_counter() - t0
    peak_main = peak_rss_mib()
    return {
        **cfg,
        "start_method_effective": effective_start_method,
        "timings_s": timings,
        "meta": meta,
        "rss_after_imports_mib": baseline,
        "peak_rss_main_mib": peak_main,
        "peak_rss_increment_main_mib": peak_main - baseline,
        # forkserver workers are children of the fork server, not of this
        # process, so RUSAGE_CHILDREN cannot see them.
        "peak_rss_largest_child_mib": (
            peak_rss_mib(resource.RUSAGE_CHILDREN)
            if cfg["n_workers"] and effective_start_method != "forkserver"
            else None
        ),
    }


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------
def warm_file_cache(*paths: str) -> None:
    for path in paths:
        with open(path, "rb") as fh:
            while fh.read(1 << 24):
                pass


def run_in_subprocess(cfg: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, __file__, "--_worker", json.dumps(cfg)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{cfg['method']} failed:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def summarise(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def markdown_report(env: dict, runs: list[dict], args) -> str:
    lines = [
        f"# Benchmark: {args.dataset}",
        "",
        "## Environment",
        "",
        "| key | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in env.items()]
    lines += [
        "",
        f"Repetitions per configuration: **{args.reps}** (each in a fresh process). "
        "Times in seconds, mean ± SD (sample SD, n−1).",
        "",
        "`total` = load + preload + JIT + setup + avg. struct. + alignment + analysis "
        "(library imports excluded). Peak RSS is the maximum over repetitions; "
        "\"increment\" is the peak above the RSS measured right after imports. "
        "Worker RSS is not measurable with forkserver (workers are not children of the "
        "benchmark process); with fork it mostly counts pages touched after the fork.",
        "",
        "`start method` is the one actually used by each run. MDAnalysis' multiprocessing "
        "backend follows Python's default (fork on Linux with Python < 3.14, spawn on macOS); "
        "dask's process scheduler uses spawn unless `multiprocessing.context` is set.",
        "",
        "| method | backend | start method | I/O | workers | frames | sel. atoms | load | preload | JIT | "
        "avg. struct. | alignment | analysis | total | peak RSS main (MiB) | "
        "peak increment main (MiB) | peak RSS largest worker (MiB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups: dict = {}
    for r in runs:
        key = (r["method"], r.get("backend"), r.get("start_method_effective"),
               r["io_mode"], r["n_workers"])
        groups.setdefault(key, []).append(r)

    def cell(rs, phase):
        vals = [r["timings_s"][phase] for r in rs if phase in r["timings_s"]]
        if not vals:
            return "–"
        s = summarise(vals)
        return f"{s['mean']:.3f} ± {s['sd']:.3f}"

    for (method, backend, start_method, io_mode, n_workers), rs in groups.items():
        m = rs[0]["meta"]
        peak_main = max(r["peak_rss_main_mib"] for r in rs)
        peak_inc = max(r["peak_rss_increment_main_mib"] for r in rs)
        children = [r["peak_rss_largest_child_mib"] for r in rs if r["peak_rss_largest_child_mib"]]
        peak_child = f"{max(children):.0f}" if children else "–"
        lines.append(
            f"| {method} | {backend or 'serial'} | {start_method or '–'} | {io_mode} | "
            f"{n_workers or 1} | {m['n_frames']} | "
            f"{m['n_selected_atoms']} | {cell(rs, 'load_universe')} | {cell(rs, 'preload')} | "
            f"{cell(rs, 'jit_compile')} | {cell(rs, 'average_structure')} | "
            f"{cell(rs, 'alignment')} | {cell(rs, 'analysis')} | {cell(rs, 'total')} | "
            f"{peak_main:.0f} | {peak_inc:.0f} | {peak_child} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", default="tubulin", help="label used in the report")
    parser.add_argument("--topology", help="topology file (default: tubulin chain A test data)")
    parser.add_argument("--trajectory", help="trajectory file (default: tubulin chain A test data)")
    parser.add_argument("--reps", type=int, default=5, help="independent timed repetitions")
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8],
                        help="worker counts for the parallel Menger benchmark")
    parser.add_argument("--backends", nargs="+", default=["multiprocessing"],
                        choices=["multiprocessing", "dask"],
                        help="MDAnalysis parallel backends for the parallel Menger benchmark")
    parser.add_argument("--start-methods", nargs="+", default=[None],
                        choices=["fork", "spawn", "forkserver"],
                        help="multiprocessing start methods for the parallel Menger benchmark "
                             "(default: each backend's own default, i.e. the platform default for "
                             "multiprocessing and spawn for dask)")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--io-modes", nargs="+", default=["disk", "memory"],
                        choices=["disk", "memory"])
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--outdir", default="benchmarks/results")
    args = parser.parse_args()

    if args._worker:
        print(json.dumps(worker(json.loads(args._worker))))
        return

    if args.topology is None or args.trajectory is None:
        from menger.data import files

        args.topology = args.topology or str(files.TUBULIN_CHAIN_A_PDB)
        args.trajectory = args.trajectory or str(files.TUBULIN_CHAIN_A_DCD)

    methods = list(args.methods)
    if "neq" in methods:
        try:
            import pbxplore  # noqa: F401
        except ImportError:
            print("pbxplore not importable: skipping the Neq benchmark", file=sys.stderr)
            methods.remove("neq")

    env = environment_info()
    warm_file_cache(args.topology, args.trajectory)

    configs = []
    for method in methods:
        # parallel runs stream from disk: with the in-memory reader every worker
        # would receive a pickled copy of the whole trajectory.
        io_modes = ["disk"] if method == "menger_parallel" else args.io_modes
        worker_counts = args.workers if method == "menger_parallel" else [None]
        start_methods = args.start_methods if method == "menger_parallel" else [None]
        backends = args.backends if method == "menger_parallel" else [None]
        for backend, start_method, io_mode, n_workers in itertools.product(
            backends, start_methods, io_modes, worker_counts
        ):
            configs.append({
                "method": method,
                "io_mode": io_mode,
                "n_workers": n_workers,
                "backend": backend,
                "start_method": start_method,
                "topology": os.path.abspath(args.topology),
                "trajectory": os.path.abspath(args.trajectory),
                "start": args.start,
                "stop": args.stop,
                "step": args.step,
            })

    runs = []
    for cfg in configs:
        for rep in range(args.reps):
            result = run_in_subprocess({**cfg, "rep": rep})
            runs.append(result)
            print(
                f"{cfg['method']:<16} io={cfg['io_mode']:<6} workers={cfg['n_workers'] or 1:<2} "
                f"backend={cfg['backend'] or 'serial':<15} "
                f"start={result['start_method_effective'] or '-':<10} "
                f"rep={rep}  analysis={result['timings_s']['analysis']:.3f}s "
                f"total={result['timings_s']['total']:.3f}s",
                file=sys.stderr,
            )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    (outdir / f"{stem}.json").write_text(
        json.dumps({"environment": env, "args": vars(args), "runs": runs}, indent=2)
    )
    report = markdown_report(env, runs, args)
    (outdir / f"{stem}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
