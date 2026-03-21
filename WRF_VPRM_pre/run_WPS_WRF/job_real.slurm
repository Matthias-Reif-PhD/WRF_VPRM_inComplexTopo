#!/bin/bash
#SBATCH --mail-user=matthias.reif@student.uibk.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=real
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=40
#SBATCH --mem=80GB

set -euo pipefail

# Environment variables are passed via sbatch --export=ALL from parent script
# No need to reload .env here

# ------------------------------------------------------------------
# Load modules and activate Conda environment
# ------------------------------------------------------------------
module purge
module load $CONDA_MODULE

eval "$("$UIBK_CONDA_DIR/bin/conda" shell.bash hook)"
conda activate "$CONDA_ENV"

# ------------------------------------------------------------------
# Load WRF specific environment
# ------------------------------------------------------------------
source ~/wrf_dev.sh

# ------------------------------------------------------------------
# Run REAL with srun for parallel execution (from the directory where sbatch was called)
# ------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
srun ./real.exe
