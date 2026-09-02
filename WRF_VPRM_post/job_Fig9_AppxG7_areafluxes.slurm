#!/bin/bash
#SBATCH --job-name=Fig9_area
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00

set -euo pipefail

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
  if [ -f "$d/Fig9_AppxG7_areafluxes_per_timestep.py" ]; then WORK_DIR="$d"; break; fi
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

echo "=== Fig9_AppxG7_areafluxes_per_timestep.py ==="
srun "$PYTHON" Fig9_AppxG7_areafluxes_per_timestep.py
