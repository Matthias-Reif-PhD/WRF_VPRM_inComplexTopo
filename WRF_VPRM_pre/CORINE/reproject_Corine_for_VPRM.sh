#!/bin/bash
gdalwarp -co COMPRESS=DEFLATE -srcnodata 127 -dstnodata 255 -s_srs EPSG:3035 -t_srs EPSG:4326 \
    /scratch/c7071034/DATA/CORINE_LC/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif \
    /scratch/c7071034/DATA/CORINE_LC/reprojected_for_VPRM_U2018_CLC2018_V2020_20u1.tif && \
gdal_translate -of GTiff -b 1 -co COMPRESS=DEFLATE \
    /scratch/c7071034/DATA/CORINE_LC/reprojected_for_VPRM_U2018_CLC2018_V2020_20u1.tif \
    /scratch/c7071034/DATA/CORINE_LC/modified_for_VPRM_U2018_CLC2018_V2020_20u1.tif
