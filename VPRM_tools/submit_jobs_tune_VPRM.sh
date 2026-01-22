#!/bin/bash

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

# Define variables
base_paths="$SCRATCH_PATH"/DATA/Fluxnet2015/Alps/
 
maxiter=42
opt_method="diff_evo_V23_SW_05"  # method and version
VPRM_options=("old") # "migli" "new" "old"

# Loop through each base path
for base_path in "${base_paths[@]}"; do
    # Loop through each VPRM option
    for VPRM_old_or_new in "${VPRM_options[@]}"; do
        # List of folders
        folders=($(find "$base_path" -type d -name "FLX_*"))

        # Loop through each folder
        for folder in "${folders[@]}"; do
            # Extract folder name from path
            folder_name=$(basename "$folder")
            # Create SLURM script for each job
            cat <<EOF >"job_${folder_name}_${VPRM_old_or_new}.sh"
#!/bin/bash
#SBATCH --job-name=tune_VPRM_${folder_name}_${VPRM_old_or_new}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=120


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

srun python main_tune_VPRM.py -p "$base_path" -f "$folder_name" -i "$maxiter" -m "$opt_method" -v "$VPRM_old_or_new"
EOF

            # Submit the job to the cluster
            sbatch "job_${folder_name}_${VPRM_old_or_new}.sh"
        done
    done
done
