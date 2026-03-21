#!/bin/bash
#SBATCH --mail-user=matthias.reif@student.uibk.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=wps_wrf_seq
#SBATCH --partition=std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

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
# Run sequential WPS and WRF
# ------------------------------------------------------------------
SCRIPT="$GITHUB_PATH"/WRF_VPRM_inComplexTopo/WRF_VPRM_pre/run_WPS_WRF/run_WPS_and_WRF_sequential.sh

bash "$SCRIPT"

echo "WPS and WRF sequential run completed."
