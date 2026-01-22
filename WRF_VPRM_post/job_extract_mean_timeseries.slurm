#!/bin/bash
#SBATCH --job-name=extract_wrf_domains_mean_timeseries
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1200


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

sim_typ="_cloudy" # "" or "_cloudy" - run one after the other
start_date="2012-01-01 00:00:00"
end_date="2012-12-31 00:00:00"
srun "$PYTHON" extract_wrf_domains_mean_timeseries.py -s "$start_date" -e "$end_date" -t "$sim_typ"