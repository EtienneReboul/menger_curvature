#!/bin/bash
# Regenerate the figures of notebooks/er03-generate_figures.ipynb (LCs; LFs + RMSF + Neq)
# on chain C (alpha-tubulin, segid PROC) of the alpha1/beta1 protofilament without Tau
# (Zenodo record 19886983, 1 us, 10000 frames). Run from ~/menger_curvature_bench on CAIUS.
#SBATCH --job-name=menger_pf_er03
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail
cd ~/menger_curvature_bench
export TMPDIR=$HOME/tmp
D=data/tubulin_PF_alpha1_beta1_without_tau
echo "node: $(hostname)"; lscpu | grep -E "Model name|^CPU\(s\)"
./env211/bin/python benchmarks/regenerate_er03_figures.py \
    --topology $D/PDB_protein_ions_ligands_alpha1_beta1_PF_without_tau.pdb \
    --trajectory $D/DCD_protein_ions_ligands_alpha1_beta1_PF_without_tau.dcd \
    --segid PROC --n-workers 8 --xlabel "Residues of αI-tubulin (chain C)" \
    --m-loop 274-287 --ctt 436-451 \
    --dssp-npz benchmarks/results/tubulin_pf_analysis_18576875/per_chain/C.npz \
    --outdir benchmarks/results/tubulin_pf_er03_figures_${SLURM_JOB_ID}
