# TODO: need to install netcdf dependencies to make it work
# could be used to compare results with the python script
# Reproject the Dataset
gdalwarp /scratch/c7071034/DATA/MODIS/MODIS_FPAR/MCD15A3H.061_500m_aid0001.nc /scratch/c7071034/DATA/MODIS/MODIS_FPAR/projection_MCD15A3H.061_500m_aid0001.nc -s_srs EPSG:6842 -t_srs EPSG:4326 -dstnodata 127
# Crop the Reprojected Map:
gdal_translate -projwin 2. 52. 20. 40. -co COMPRESS=LZW /scratch/c7071034/DATA/MODIS/MODIS_FPAR/projection_MCD15A3H.061_500m_aid0001.nc /scratch/c7071034/DATA/MODIS/MODIS_FPAR/cropped_MCD15A3H.061_500m_aid0001.nc
