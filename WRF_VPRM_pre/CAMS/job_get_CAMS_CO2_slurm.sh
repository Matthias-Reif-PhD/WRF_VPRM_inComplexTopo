#!/bin/bash
#SBATCH --job-name=get_CAMS_CO2
#SBATCH --mail-user=matthias.reif@student.uibk.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3000

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

eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
conda activate "$CONDA_ENV"

PYTHON="$CONDA_ENV/bin/python"

# ------------------------------------------------------------------
# Ensure correct CDS API endpoint
# ------------------------------------------------------------------
file_path="$HOME/.cdsapirc"
sed -i '/^url:/c\url: https://ads.atmosphere.copernicus.eu/api' "$file_path"
echo "Updated 'url:' line in $file_path."

# ------------------------------------------------------------------
# Download CAMS CO2 data
# ------------------------------------------------------------------
for month in {01..11}; do
    start_date="2012-$month-01"
    next_month=$(printf "%02d" $((10#$month + 1)))
    end_date="2012-$next_month-01"

    echo "processing $start_date to $end_date"
    "$PYTHON" CAMS_get_CO2_API.py -s "$start_date" -e "$end_date"
done

start_date="2012-12-01"
end_date="2013-01-01"
echo "processing $start_date to $end_date"
"$PYTHON" CAMS_get_CO2_API.py -s "$start_date" -e "$end_date"

# ------------------------------------------------------------------
# Unpack downloaded files
# ------------------------------------------------------------------
for file in ghg-reanalysis_CO2_*.nc; do
    newname="${file%.nc}.zip"
    mv "$file" "$newname"
    unzip "$newname" -d "${newname%.zip}"
done
