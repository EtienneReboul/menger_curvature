#!/bin/bash
# Timing/memory benchmark (Menger serial + parallel, MDAnalysis RMSF, PBxplore Neq)
# on chain C (segid PROC) of the alpha1/beta1 tubulin protofilament without Tau
# (Zenodo record 19886983). Every method reads the full 99,393-atom trajectory;
# selections are restricted to chain C. Run from ~/menger_curvature_bench on CAIUS.
#SBATCH --job-name=menger_pf_bench
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail
cd ~/menger_curvature_bench
export TMPDIR=$HOME/tmp
D=data/tubulin_PF_alpha1_beta1_without_tau
echo "node: $(hostname)"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"; echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
./env211/bin/python benchmarks/benchmark_metrics.py --dataset tubulin_pf_chainC_caius \
    --topology $D/PDB_protein_ions_ligands_alpha1_beta1_PF_without_tau.pdb \
    --trajectory $D/DCD_protein_ions_ligands_alpha1_beta1_PF_without_tau.dcd \
    --restrict "segid PROC" \
    --reps 5 --methods menger_serial menger_parallel rmsf neq --io-modes disk memory \
    --workers 1 2 4 8 16 --backends multiprocessing dask --start-methods fork spawn forkserver \
    --outdir benchmarks/results
