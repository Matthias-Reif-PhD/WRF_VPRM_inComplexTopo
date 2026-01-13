module load nco 
cd /scratch/c7071034/DATA/CAMS

for f in ghg-reanalysis_surface_2012*.nc; do
    ncks --mk_rec_dmn valid_time "$f" "tmp_$f"
done
 
ncrcat tmp_ghg-reanalysis_surface_2012*.nc ghg-reanalysis_surface_2012_full.nc

rm tmp_ghg-reanalysis_surface_2012*.nc
