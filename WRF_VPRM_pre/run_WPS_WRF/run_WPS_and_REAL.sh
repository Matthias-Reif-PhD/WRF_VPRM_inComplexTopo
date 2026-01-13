#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error
# set -x  # Print commands and their arguments as they are executed

##### Settings ########
START_DATE=$1
END_DATE=$2
shift 2
resx=("$@")
#######################

for res in "${resx[@]}"
do
  # echo "Resolution is: $res" >&2
  # echo "Running WPS." >&2
   cp /scratch/c7071034/WPS/namelist.wps_$res /scratch/c7071034/WPS/namelist.wps
   namelist_wps="/scratch/c7071034/WPS/namelist.wps"
   cd /scratch/c7071034/WPS || exit 1
   # Update namelist.wps
   sed -i "s/^ *start_date *=.*/ start_date = '${START_DATE}', '${START_DATE}',/" "$namelist_wps"
   sed -i "s/^ *end_date *=.*/ end_date   = '${END_DATE}', '${END_DATE}',/" "$namelist_wps"
   sed -i "s/^\s*prefix\s*=.*$/ prefix = 'FILE',/" "$namelist_wps"
   # Link and run WPS
   ./link_grib.csh /scratch/c7071034/DATA/ECMWF/pressure/era5_data_2012
   ln -sf ./ungrib/Variable_Tables/Vtable.ECMWF ./Vtable
   ./ungrib.exe
   rm -f GRIBFILE.*
   sed -i "s/^\s*prefix\s*=.*$/ prefix = 'SFILE',/" "$namelist_wps"
   ./link_grib.csh /scratch/c7071034/DATA/ECMWF/surface/era5_surface_2012
   ln -sf ./ungrib/Variable_Tables/Vtable.ECMWF ./Vtable
   ./ungrib.exe
   ./geogrid.exe
   ./metgrid.exe
  # echo "Finished WPS." >&2
   # Create symbolic links for met_em files
   cd /scratch/c7071034/WRF_$res/test/em_real/ || exit 1
   mv /scratch/c7071034/WPS/met_em.d0* .

   # Extract components
   START=$(echo $START_DATE | cut -d'_' -f1) # e.g. 2012-07-02
   START_HOUR=$(echo $START_DATE | cut -d'_' -f2 | cut -d':' -f1) # e.g. 18
   END=$(echo $END_DATE | cut -d'_' -f1)
   # Convert to date format for loop
   current_date=$(date -d "$START" +%Y-%m-%d)
   end_date=$(date -d "$END" +%Y-%m-%d)
   # Loop over each day from start to end
   while [[ "$current_date" < "$end_date" || "$current_date" == "$end_date" ]]; do
      START_YEAR=$(date -d "$current_date" +%Y)
      START_MONTH=$(date -d "$current_date" +%m)
      START_DAY=$(date -d "$current_date" +%d)

     # echo "Copying VPRM files for $current_date..." >&2

      cp "/scratch/c7071034/DATA/VPRM_input/vprm_corine_${res}/vprm_input_d01_${START_YEAR}-${START_MONTH}-${START_DAY}_00:00:00.nc" \
         "/scratch/c7071034/WRF_$res/test/em_real/vprm_input_d01_${START_YEAR}-${START_MONTH}-${START_DAY}_${START_HOUR}:00:00.nc"

      # if res == 3km do:
         if [[ "$res" == "3km" ]]; then
            cp "/scratch/c7071034/DATA/VPRM_input/vprm_corine_1km/vprm_input_d02_${START_YEAR}-${START_MONTH}-${START_DAY}_00:00:00.nc" \
            "/scratch/c7071034/WRF_$res/test/em_real/vprm_input_d02_${START_YEAR}-${START_MONTH}-${START_DAY}_${START_HOUR}:00:00.nc"
         fi
      # Increment by one day
      current_date=$(date -I -d "$current_date + 1 day")
   done

  # echo "Running real.exe and updating boundary conditions." >&2
   source ~/wrf_dev.sh 
   namelist_input="/scratch/c7071034/WRF_$res/test/em_real/namelist.input"

   # Find and replace chem_in_opt in namelist.input with chem_in_opt = 0,
   sed -i 's/^chem_in_opt\s*=.*/chem_in_opt = 0,0,/' "$namelist_input"

   # Extract components
   START_YEAR=$(echo "$START_DATE" | cut -d'-' -f1)
   START_MONTH=$(echo "$START_DATE" | cut -d'-' -f2)
   START_DAY=$(echo "$START_DATE" | cut -d'-' -f3 | cut -d'_' -f1)
   START_HOUR=$(echo "$START_DATE" | cut -d'_' -f2 | cut -d':' -f1)

   END_YEAR=$(echo "$END_DATE" | cut -d'-' -f1)
   END_MONTH=$(echo "$END_DATE" | cut -d'-' -f2)
   END_DAY=$(echo "$END_DATE" | cut -d'-' -f3 | cut -d'_' -f1)
   END_HOUR=$(echo "$END_DATE" | cut -d'_' -f2 | cut -d':' -f1)

   ### --- Update namelist.input ---
   # Replace start_* and end_* lines
   sed -i "s/^\s*start_year\s*=.*/ start_year                          = ${START_YEAR}, ${START_YEAR},/" "$namelist_input"
   sed -i "s/^\s*start_month\s*=.*/ start_month                         = ${START_MONTH}, ${START_MONTH},/" "$namelist_input"
   sed -i "s/^\s*start_day\s*=.*/ start_day                           = ${START_DAY}, ${START_DAY},/" "$namelist_input"
   sed -i "s/^\s*start_hour\s*=.*/ start_hour                          = ${START_HOUR}, ${START_HOUR},/" "$namelist_input"

   sed -i "s/^\s*end_year\s*=.*/ end_year                            = ${END_YEAR}, ${END_YEAR},/" "$namelist_input"
   sed -i "s/^\s*end_month\s*=.*/ end_month                           = ${END_MONTH}, ${END_MONTH},/" "$namelist_input"
   sed -i "s/^\s*end_day\s*=.*/ end_day                             = ${END_DAY}, ${END_DAY},/" "$namelist_input"
   sed -i "s/^\s*end_hour\s*=.*/ end_hour                            = ${END_HOUR}, ${END_HOUR},/" "$namelist_input"

  # echo "Starting real.exe for $res" >&2
   cd /scratch/c7071034/WRF_$res/test/em_real || exit 1

   # Running Real
# if res == 9km or 3km then:
   if [[ "$res" == "9km" || "$res" == "3km" ]]; then
      # Additional commands for 9km and 3km resolutions
        # echo "Starting Real" >&2
      # Submit WRF job and capture JobID
      sbatch_out=$(sbatch job_real.slurm)
      jobid=$(echo "$sbatch_out" | awk '{print $4}')
      # echo "Submitted job $jobid for resolution $res" >&2
      jobids+=($jobid)
   else
     ./real.exe
   fi

done

# Print all jobids (space-separated)
echo "${jobids[@]}"

