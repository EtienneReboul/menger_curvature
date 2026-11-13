#!/bin/bash
# Time-block (4 x 250 ns), moving-window (250 ns every 25 ns) and inter-chain
# agreement of LC, LF, RMSF and Neq for the alpha-tubulin chains C, E, G, I, K
# of the alpha1/beta1 protofilament without Tau (Zenodo record 19886983).
# Run from ~/menger_curvature_bench on CAIUS.
#SBATCH --job-name=menger_pf_blocks
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail
cd ~/menger_curvature_bench
export TMPDIR=$HOME/tmp
D=data/tubulin_PF_alpha1_beta1_without_tau
echo "node: $(hostname)"; lscpu | grep -E "Model name|^CPU\(s\)"
./env211/bin/python benchmarks/tubulin_pf_blocks.py \
    --topology $D/PDB_protein_ions_ligands_alpha1_beta1_PF_without_tau.pdb \
    --trajectory $D/DCD_protein_ions_ligands_alpha1_beta1_PF_without_tau.dcd \
    --chains C E G I K --blocks 4 --window 2500 --stride 250 --frame-ps 100 \
    --regions "M-loop:274-287" "C-terminal tail:436-451" "Gly144:144" \
    --dssp-dir benchmarks/results/tubulin_pf_analysis_18576875/per_chain \
    --outdir benchmarks/results/tubulin_pf_blocks_${SLURM_JOB_ID}
