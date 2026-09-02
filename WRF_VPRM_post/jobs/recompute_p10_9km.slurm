#!/bin/bash
#SBATCH --job-name=recompute_vprm_fluxes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10


set -euo pipefail

# ------------------------------------------------------------------
# Load environment variables from project root .env
# ------------------------------------------------------------------
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_PATH" != /* ]]; then
  SCRIPT_PATH="$SUBMIT_DIR/$SCRIPT_PATH"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd || true)"
if [ -z "$SCRIPT_DIR" ] || [ ! -d "$SCRIPT_DIR" ]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"
fi
KNOWN_REPO_ROOT="${KNOWN_REPO_ROOT:-$HOME/Github/WRF_VPRM_inComplexTopo}"
WORK_DIR=""
for d in "$SCRIPT_DIR/.." "$SCRIPT_DIR" "$SUBMIT_DIR/WRF_VPRM_post" "$SUBMIT_DIR" "$KNOWN_REPO_ROOT/WRF_VPRM_post" "$KNOWN_REPO_ROOT"; do
  if [ -f "$d/recompute_vprm_fluxes.py" ]; then
    WORK_DIR="$d"
    break
  fi
done
if [ -z "$WORK_DIR" ]; then
  WORK_DIR="$KNOWN_REPO_ROOT/WRF_VPRM_post"
fi
ENV_FILE=""
for d in "$WORK_DIR" "$KNOWN_REPO_ROOT/WRF_VPRM_post" "$KNOWN_REPO_ROOT" "$SUBMIT_DIR" "$PWD" "$HOME"; do
  if [ -f "$d/.env" ]; then
    ENV_FILE="$d/.env"
    break
  fi
done

if [ -z "$ENV_FILE" ] || [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: .env file not found near $WORK_DIR" >&2
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
conda activate "$CONDA_ENV_WRF_VPRM"
PYTHON="$CONDA_ENV_WRF_VPRM/bin/python"
RECOMPUTE_SCRIPT="$WORK_DIR/recompute_vprm_fluxes.py"
cd "$WORK_DIR"

# ------------------------------------------------------------------
# Recompute offline VPRM fluxes (GPP/Reco/NEE) with a new parameter set.
# Edit tag/params/dates/res as needed; one resolution per submission.
# ------------------------------------------------------------------
tag="topt_p10"
params="vprm_params_topt_p10.csv"
start_date="2012-01-01 00:00:00"
end_date="2012-12-31 00:00:00"
sim_typ=""                          # "" or "_cloudy" or "_parm_err"

for res in 9km; do
  echo "=== recomputing VPRM fluxes for ${res}${sim_typ} (tag=${tag}) ==="
  params_path="$params"
  if [[ "$params_path" != /* ]]; then
    params_path="$WORK_DIR/$params_path"
  fi
  srun "$PYTHON" "$RECOMPUTE_SCRIPT" \
      -s "$start_date" -e "$end_date" -r "$res" -t "$sim_typ" \
      --tag "$tag" --params "$params_path"
done
