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

module load nco 
cd "$SCRATCH_PATH/DATA/CAMS"

for f in ghg-reanalysis_surface_2012*.nc; do
    ncks --mk_rec_dmn valid_time "$f" "tmp_$f"
done
 
ncrcat tmp_ghg-reanalysis_surface_2012*.nc ghg-reanalysis_surface_2012_full.nc

rm tmp_ghg-reanalysis_surface_2012*.nc
