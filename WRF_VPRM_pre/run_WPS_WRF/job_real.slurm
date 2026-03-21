#!/bin/bash
#SBATCH --mail-user=matthias.reif@student.uibk.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=real
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=40
#SBATCH --mem=80GB

set -euo pipefail

# ------------------------------------------------------------------
# Load environment variables from project root .env
# ------------------------------------------------------------------
ENV_FILE="$(dirname "$(dirname "$(pwd)")")/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: .env file not found at $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

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
# Run REAL with srun for parallel execution
# ------------------------------------------------------------------
srun ./real.exe
