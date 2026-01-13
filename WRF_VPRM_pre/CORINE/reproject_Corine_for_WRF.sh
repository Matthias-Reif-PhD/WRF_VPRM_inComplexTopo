#!/bin/bash

# Step 1: Change projection from EPSG 3035 to WGS84
        # 3 km resolution:
        # -tr 0.039 0.027
        # 1 km resolution:
        # -tr 0.013 0.009
        # 200 m resolution:
        # -tr 0.0026 0.0018

# old: gdalwarp /scratch/c7071034/DATA/CORINE_LC/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif projection_1km_U2018_CLC2018_V2020_20u1.tif -s_srs EPSG:3035 -t_srs EPSG:4326 -dstnodata 255 -tr 0.0009 0.0009
gdalwarp /scratch/c7071034/DATA/CORINE_LC/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif projection_1km_U2018_CLC2018_V2020_20u1.tif -s_srs EPSG:3035 -t_srs EPSG:4326 -dstnodata 127

# Step 2: Crop the map (change the geographical limits as needed)
gdal_translate -projwin 2. 52. 20. 40. -co COMPRESS=LZW projection_1km_U2018_CLC2018_V2020_20u1.tif cropped_1km_U2018_CLC2018_V2020_20u1.tif

# Step 3: Convert to binary for Geogrid
/home/c707/c7071034/convert_geotiff/convert_geotiff -c 43 -t 1200 -d "CORINE remapped to MODIS" -m 0 cropped_1km_U2018_CLC2018_V2020_20u1.tif

# add missing_value=127 in index file manually
