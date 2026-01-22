#!/bin/bash
#SBATCH --mail-user=matthias.reif@student.uibk.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00

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
conda activate "$CONDA_ENV_PYVPRM"
PYTHON="$CONDA_ENV_PYVPRM/bin/python"

# Run the Python script
"$PYTHON" "$SCRATCH_PATH"/DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/vprm_preprocessor_new.py \
  --chunk_x=$CHUNK_X --chunk_y=$CHUNK_Y --n_cpus=96
