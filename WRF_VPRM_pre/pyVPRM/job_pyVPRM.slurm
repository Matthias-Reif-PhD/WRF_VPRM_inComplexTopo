#!/bin/bash
#SBATCH --job-name=d01_VPRM_corine
#SBATCH --mail-user=matthias.reif@student.uibk.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1620

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

domain="d01" # add manually in job_name and in preprocessor_config.yaml
year=2012
# !!! Run the Python script with the correct number of CPUs !!!
n_cpus=96 # --n_cpus=2*cpus-per-task 

# To rerun single splits
# vprm_preprocessor_new.py --n_cpus=64  --chunk_x=1 --chunk_y=1

for i in $(seq 1 8);
do
for j in $(seq 1 8);
do
echo "processing domain:"$domain " chunk_x="$i" chunk_y="$j
"$PYTHON" "$SCRATCH_PATH"/DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/vprm_preprocessor_new.py --year=$year --n_cpus=$n_cpus --chunk_x=$i --chunk_y=$j
done
done



