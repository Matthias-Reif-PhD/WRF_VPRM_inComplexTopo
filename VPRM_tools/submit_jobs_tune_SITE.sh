#!/bin/bash

# ------------------------------------------------------------------
# SITE tuning (GMD revision): re-tune ONLY the five d03 SITE FLUXNET
# locations for year 2012 with the GPP-based per-site T_opt (T_opt_site
# dict in main_tune_VPRM.py), via the -y/--single_year flag.
# Mirrors submit_jobs_tune_VPRM.sh but restricted to the 5 sites + -y.
# ------------------------------------------------------------------
ENV_FILE="$(dirname "$(pwd)")/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: .env file not found at $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

# Define variables
base_path="$SCRATCH_PATH"/DATA/Fluxnet2015/Alps/

maxiter=42
opt_method="diff_evo_V24_SITE"  # SITE = per-site 2012 tuning, GPP-based Topt (GMD revision)
VPRM_old_or_new="old"
tune_env="$SCRATCH_PATH/conda_envs/pyrealm311"

# The five d03 SITE locations (match FLX_<site>_* folders)
SITES=("AT-Neu" "CH-Dav" "IT-Lav" "IT-MBo" "IT-Ren")

for site in "${SITES[@]}"; do
    folder=$(find "$base_path" -maxdepth 1 -type d -name "FLX_${site}_*" | head -1)
    if [ -z "$folder" ]; then
        echo "WARNING: no FLX_ folder found for $site, skipping" >&2
        continue
    fi
    folder_name=$(basename "$folder")

    cat <<EOF >"job_${folder_name}_SITE.sh"
#!/bin/bash
#SBATCH --job-name=tuneSITE_${folder_name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=120

set -euo pipefail

ENV_FILE="$(dirname "$(pwd)")/.env"

if [ ! -f "\$ENV_FILE" ]; then
  echo "ERROR: .env file not found at \$ENV_FILE" >&2
  exit 1
fi

set -a
source "\$ENV_FILE"
set +a

module purge
module load $CONDA_MODULE

eval "\$(conda shell.bash hook)"
conda activate "$tune_env"

srun python main_tune_VPRM.py -p "$base_path" -f "$folder_name" -i "$maxiter" -m "$opt_method" -v "$VPRM_old_or_new" -y
EOF

    sbatch "job_${folder_name}_SITE.sh"
done
