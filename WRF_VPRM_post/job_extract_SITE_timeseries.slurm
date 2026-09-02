#!/bin/bash
#SBATCH --job-name=extract_SITE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --array=0-4

set -euo pipefail

# ------------------------------------------------------------------
# Robust WORK_DIR / .env detection
# ------------------------------------------------------------------
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_PATH" != /* ]]; then SCRIPT_PATH="$SUBMIT_DIR/$SCRIPT_PATH"; fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd || true)"
if [ -z "$SCRIPT_DIR" ] || [ ! -d "$SCRIPT_DIR" ]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"
fi
KNOWN_REPO_ROOT="${KNOWN_REPO_ROOT:-$HOME/Github/WRF_VPRM_inComplexTopo}"
WORK_DIR=""
for d in "$SCRIPT_DIR" "$SUBMIT_DIR/WRF_VPRM_post" "$SUBMIT_DIR" "$KNOWN_REPO_ROOT/WRF_VPRM_post"; do
  if [ -f "$d/extract_SITE_timeseries.py" ]; then WORK_DIR="$d"; break; fi
done
if [ -z "$WORK_DIR" ]; then WORK_DIR="$KNOWN_REPO_ROOT/WRF_VPRM_post"; fi
ENV_FILE=""
for d in "$WORK_DIR" "$WORK_DIR/.." "$KNOWN_REPO_ROOT" "$SUBMIT_DIR" "$PWD" "$HOME"; do
  if [ -f "$d/.env" ]; then ENV_FILE="$d/.env"; break; fi
done
if [ -z "$ENV_FILE" ] || [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: .env file not found near $WORK_DIR" >&2; exit 1
fi

set -a; source "$ENV_FILE"; set +a

module purge
module load $CONDA_MODULE
eval "$("$UIBK_CONDA_DIR/bin/conda" shell.bash hook)"
conda activate "$CONDA_ENV_WRF_VPRM"
PYTHON="$CONDA_ENV_WRF_VPRM/bin/python"
cd "$WORK_DIR"

# ------------------------------------------------------------------
# One tag per array task; both sim types. Full-year range -> the complete-day
# filter selects exactly the 12 recomputed days (clear) / 12 cloudy days.
# ------------------------------------------------------------------
TAGS=(topt_p10 topt_p25 topt_p50 topt_p75 topt_p90)
tag="${TAGS[$SLURM_ARRAY_TASK_ID]}"
start_date="2012-01-01 00:00:00"
end_date="2012-12-31 00:00:00"

for sim_typ in "" "_cloudy"; do
  echo "=== extract SITE (tag=${tag}, sim='${sim_typ}') ==="
  srun "$PYTHON" extract_SITE_timeseries.py \
      -s "$start_date" -e "$end_date" -t "$sim_typ" --tag "$tag"
done
