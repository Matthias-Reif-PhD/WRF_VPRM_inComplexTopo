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
   cd /scratch/c7071034/WRF_$res/test/em_real || exit 1
   # Extract components
   START_YEAR=$(echo "$START_DATE" | cut -d'-' -f1)
   START_MONTH=$(echo "$START_DATE" | cut -d'-' -f2)
   START_DAY=$(echo "$START_DATE" | cut -d'-' -f3 | cut -d'_' -f1)
   START_HOUR=$(echo "$START_DATE" | cut -d'_' -f2 | cut -d':' -f1)

   END_YEAR=$(echo "$END_DATE" | cut -d'-' -f1)
   END_MONTH=$(echo "$END_DATE" | cut -d'-' -f2)
   END_DAY=$(echo "$END_DATE" | cut -d'-' -f3 | cut -d'_' -f1)
   END_HOUR=$(echo "$END_DATE" | cut -d'_' -f2 | cut -d':' -f1)

   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   # !!! you have to adopt the path_CAMS_file  (and WRF_V2?)
   conda run -n py_basic python /home/c707/c7071034/Github/WRF_VPRM_pre/prep_boundary_cond_CO2_BCK.py $res $START_YEAR-$START_MONTH-$START_DAY $START_HOUR:00:00 $END_YEAR-$END_MONTH-$END_DAY $END_HOUR:00:00 
   conda run -n py_basic python /home/c707/c7071034/Github/WRF_VPRM_pre/prep_initial_cond_CO2_BCK.py $res $START_YEAR-$START_MONTH-$START_DAY $START_HOUR:00:00 $END_YEAR-$END_MONTH-$END_DAY $END_HOUR:00:00

   mv /scratch/c7071034/WRF_$res/test/em_real/wrfbdy_d01_updated /scratch/c7071034/WRF_$res/test/em_real/wrfbdy_d01
   mv /scratch/c7071034/WRF_$res/test/em_real/wrfinput_d01_updated /scratch/c7071034/WRF_$res/test/em_real/wrfinput_d01
   if [[ "$res" == "3km" ]]; then
      mv /scratch/c7071034/WRF_$res/test/em_real/wrfinput_d02_updated /scratch/c7071034/WRF_$res/test/em_real/wrfinput_d02
   fi
   # Find and replace chem_in_opt in namelist.input with chem_in_opt = 1,
   namelist_input="/scratch/c7071034/WRF_$res/test/em_real/namelist.input"
   sed -i 's/^chem_in_opt\s*=.*/chem_in_opt       = 1,1,/' "$namelist_input"

  # echo "Running real.exe and updating boundary conditions complete." >&2

   # Submit WRF job and capture JobID
   sbatch_out=$(sbatch job_WRF.slurm_$res)
   jobid=$(echo "$sbatch_out" | awk '{print $4}')
  # echo "Submitted WRF job $jobid for resolution $res" >&2
   jobids+=($jobid)
done

# Print all jobids (space-separated)
echo "${jobids[@]}"
