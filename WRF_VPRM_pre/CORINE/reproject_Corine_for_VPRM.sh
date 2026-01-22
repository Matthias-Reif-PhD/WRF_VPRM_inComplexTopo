#!/bin/bash
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

gdalwarp -co COMPRESS=DEFLATE -srcnodata 127 -dstnodata 255 -s_srs EPSG:3035 -t_srs EPSG:4326 \
    "$SCRATCH_PATH"/DATA/CORINE_LC/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif \
    "$SCRATCH_PATH"/DATA/CORINE_LC/reprojected_for_VPRM_U2018_CLC2018_V2020_20u1.tif && \
gdal_translate -of GTiff -b 1 -co COMPRESS=DEFLATE \
    "$SCRATCH_PATH"/DATA/CORINE_LC/reprojected_for_VPRM_U2018_CLC2018_V2020_20u1.tif \
    "$SCRATCH_PATH"/DATA/CORINE_LC/modified_for_VPRM_U2018_CLC2018_V2020_20u1.tif
