#!/bin/bash
# Convergence (LC/LF half vs half) and PMC-by-DSSP-class analysis of the
# alpha1/beta1 tubulin protofilament without Tau (Zenodo record 19886983),
# chains C, E, G, I, K. Run from ~/menger_curvature_bench on CAIUS.
#SBATCH --job-name=menger_pf_analysis
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail
cd ~/menger_curvature_bench
export TMPDIR=$HOME/tmp
D=data/tubulin_PF_alpha1_beta1_without_tau
echo "node: $(hostname)"; lscpu | grep -E "Model name|^CPU\(s\)"; echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
./env211/bin/python benchmarks/tubulin_pf_analysis.py \
    --topology $D/PDB_protein_ions_ligands_alpha1_beta1_PF_without_tau.pdb \
    --trajectory $D/DCD_protein_ions_ligands_alpha1_beta1_PF_without_tau.dcd \
    --chains C E G I K --n-workers 16 --dssp-backend distopia \
    --outdir benchmarks/results/tubulin_pf_analysis_${SLURM_JOB_ID}
