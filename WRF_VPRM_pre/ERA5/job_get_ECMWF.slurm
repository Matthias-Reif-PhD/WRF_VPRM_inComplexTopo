#!/bin/bash
#SBATCH --job-name=get_ECMWF2
#SBATCH --mail-user=matthias.reif@student.uibk.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=720

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

PYTHON="$CONDA_ENV/bin/python"

cd "$SCRATCH_PATH"/DATA/ECMWF/pressure
file_path="$HOME/.cdsapirc"
# Replace the line starting with 'url:' using sed
sed -i '/^url:/c\url: https://cds.climate.copernicus.eu/api' "$file_path"
echo "Updated 'url:' line in $file_path."

dates=(
#clearsky 
# "12.01."
# "22.02."
# "27.03."
# "27.04."
# "26.05."
# "28.06."
# "27.07."
# "27.08."
# "21.09."
# "20.10."
# "15.11."
# "24.12."
#cloudy 
# "20.01."
# "19.02."
# "19.03."
# "07.04."
# "16.05."
# "10.06."
# "15.07."
# "30.08."
# "19.09."

# "19.09."
# "15.10."
# "11.11."
"07.12."
)

start_dates=()
end_dates=()
months=()
year=2012

for date in "${dates[@]}"; do
    day=${date:0:2}
    month=${date:3:2}
    # Remove leading zeros for arithmetic
    day_num=$((10#$day))
    # day before
    day_before=$((day_num - 1))
    next_day=$((day_num + 1))
    # Pad with zero if needed
    if [ $next_day -lt 10 ]; then
        next_day="0$next_day"
    fi
    if [ $day_before -lt 10 ]; then
        day_before="0$day_before"
    fi
    start_dates+=("$day_before")
    end_dates+=("$next_day")
    months+=("$month")
done

# Example: print all
for i in "${!dates[@]}"; do
    echo "Start: ${start_dates[$i]}, End: ${end_dates[$i]}, Month: ${months[$i]}, Year: $year"
    "$PYTHON" "$GITHUB_PATH"/WRF_VPRM_inComplexTopo/WRF_VPRM_pre/get_ECMWF_pressure.py --start ${start_dates[$i]} --end "${end_dates[$i]}" --month "${months[$i]}"  --year $year

done


