#!/bin/bash
#SBATCH --job-name=recompute_topt_tags
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=30
#SBATCH --array=0-4

set -euo pipefail

# ------------------------------------------------------------------
# Robust WORK_DIR / .env detection (mirrors job_recompute_vprm_fluxes.slurm)
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
for d in "$SCRIPT_DIR" "$SUBMIT_DIR/WRF_VPRM_post" "$SUBMIT_DIR" "$KNOWN_REPO_ROOT/WRF_VPRM_post"; do
  if [ -f "$d/recompute_vprm_fluxes.py" ]; then
    WORK_DIR="$d"
    break
  fi
done
if [ -z "$WORK_DIR" ]; then
  WORK_DIR="$KNOWN_REPO_ROOT/WRF_VPRM_post"
fi
ENV_FILE=""
for d in "$WORK_DIR" "$WORK_DIR/.." "$KNOWN_REPO_ROOT" "$SUBMIT_DIR" "$PWD" "$HOME"; do
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

module purge
module load $CONDA_MODULE
eval "$("$UIBK_CONDA_DIR/bin/conda" shell.bash hook)"
conda activate "$CONDA_ENV_WRF_VPRM"
PYTHON="$CONDA_ENV_WRF_VPRM/bin/python"
RECOMPUTE_SCRIPT="$WORK_DIR/recompute_vprm_fluxes.py"
cd "$WORK_DIR"

# ------------------------------------------------------------------
# Recompute offline VPRM fluxes for the CLOUDY days, one tag per array task.
# The full-day filter selects the 12 complete cloudy days automatically.
# ------------------------------------------------------------------
TAGS=(topt_p10 topt_p25 topt_p50 topt_p75 topt_p90)
tag="${TAGS[$SLURM_ARRAY_TASK_ID]}"
params_path="$WORK_DIR/vprm_params_${tag}.csv"
start_date="2012-01-01 00:00:00"
end_date="2012-12-31 00:00:00"
# Regime is overridable so the same array job serves the clear-sky ("") and
# cloudy ("_cloudy") recomputes; defaults to _cloudy for backwards compatibility.
sim_typ="${SIM_TYPE-_cloudy}"

for res in 1km 9km 54km; do
  echo "=== recompute ${res}${sim_typ} (tag=${tag}) ==="
  srun "$PYTHON" "$RECOMPUTE_SCRIPT" \
      -s "$start_date" -e "$end_date" -r "$res" -t "$sim_typ" \
      --tag "$tag" --params "$params_path" --overwrite
done
