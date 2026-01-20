#!/bin/bash
module load nco

folder="/scratch/c7071034/DATA/CAMS"

infile="$folder/ghg-reanalysis_surface_2012_full.nc"
outfile="$folder/ghg-reanalysis_surface_2012_WRFVPRM_ready.nc"
tmp_ssr="$folder/tmp_ssr.nc"
tmp_par="$folder/tmp_par.nc"


echo "ssr is accumulated, converting to flux..."
ncks -O -v ssr $infile $tmp_ssr
ncap2 -O -s 'ssr_diff[time_counter,lat,lon]=ssr' $tmp_ssr $tmp_ssr
ncap2 -O -s 'ssr_diff(0,:,:)=ssr(0,:,:); ssr_diff(1:,:,:)=ssr(1:,:,:)-ssr(0:-2,:,:)' $tmp_ssr $tmp_ssr
ncap2 -O -s 'SWDOWN=ssr_diff/10800.0' $tmp_ssr $tmp_ssr
ncks -O -v SWDOWN $tmp_ssr $tmp_ssr
# 2. Process PAR
echo "par is accumulated, converting to FAPAR (W/m2)..."
ncks -O -v par $infile $tmp_par
ncap2 -O -s 'par_diff[time_counter,lat,lon]=par' $tmp_par $tmp_par
ncap2 -O -s 'par_diff(0,:,:)=par(0,:,:); par_diff(1:,:,:)=par(1:,:,:)-par(0:-2,:,:)' $tmp_par $tmp_par
ncap2 -O -s 'FAPAR=par_diff/10800.0' $tmp_par $tmp_par
ncks -O -v FAPAR $tmp_par $tmp_par

# if not accumulated
#     echo "ssr is not accumulated, dividing by 10800 to get flux..."
#     ncks -O -v ssr $infile $tmp_ssr
#     ncap2 -O -s 'SWDOWN=ssr/10800.0' $tmp_ssr $tmp_ssr
#     ncks -O -v SWDOWN $tmp_ssr $tmp_ssr
# echo "par is not accumulated, dividing by 10800 to get flux..."
# ncks -O -v par $infile $tmp_par
# ncap2 -O -s 'FAPAR=par/10800.0' $tmp_par $tmp_par
# ncks -O -v FAPAR $tmp_par $tmp_par


# 3. Merge new variables into output file
cp $infile $outfile
ncks -A -v SWDOWN $tmp_ssr $outfile
ncks -A -v FAPAR $tmp_par $outfile

# 4. Clean up
rm $tmp_ssr $tmp_par

echo "Done. Output: $outfile"