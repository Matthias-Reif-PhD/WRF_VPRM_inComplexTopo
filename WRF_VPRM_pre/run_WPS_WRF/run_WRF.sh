#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error
# set -x  # Print commands and their arguments as they are executed

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

##### Settings ########
START_DATE=$1
END_DATE=$2
shift 2
resx=("$@")
jobids=()
#######################

for res in "${resx[@]}"
do
   cd "$SCRATCH_PATH"/WRF_$res/test/em_real || exit 1
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
   echo "Running prep_boundary_cond_CO2_BCK.py for $res..."
   "$CONDA_ENV/bin/python" "$GITHUB_PATH"/WRF_VPRM_inComplexTopo/WRF_VPRM_pre/CAMS/prep_boundary_cond_CO2_BCK.py $res $START_YEAR-$START_MONTH-$START_DAY $START_HOUR:00:00 $END_YEAR-$END_MONTH-$END_DAY $END_HOUR:00:00 || { echo "ERROR: prep_boundary_cond failed for $res"; exit 1; }
   echo "Running prep_initial_cond_CO2_BCK.py for $res..."
   "$CONDA_ENV/bin/python" "$GITHUB_PATH"/WRF_VPRM_inComplexTopo/WRF_VPRM_pre/CAMS/prep_initial_cond_CO2_BCK.py $res $START_YEAR-$START_MONTH-$START_DAY $START_HOUR:00:00 $END_YEAR-$END_MONTH-$END_DAY $END_HOUR:00:00 || { echo "ERROR: prep_initial_cond failed for $res"; exit 1; }

   # Verify updated files exist before mv
   if [ ! -f "$SCRATCH_PATH"/WRF_$res/test/em_real/wrfbdy_d01_updated ]; then
      echo "ERROR: wrfbdy_d01_updated not found for $res" >&2
      exit 1
   fi
   if [ ! -f "$SCRATCH_PATH"/WRF_$res/test/em_real/wrfinput_d01_updated ]; then
      echo "ERROR: wrfinput_d01_updated not found for $res" >&2
      exit 1
   fi

   mv "$SCRATCH_PATH"/WRF_$res/test/em_real/wrfbdy_d01_updated "$SCRATCH_PATH"/WRF_$res/test/em_real/wrfbdy_d01
   mv "$SCRATCH_PATH"/WRF_$res/test/em_real/wrfinput_d01_updated "$SCRATCH_PATH"/WRF_$res/test/em_real/wrfinput_d01
   if [[ "$res" == "3km" ]]; then
      mv "$SCRATCH_PATH"/WRF_$res/test/em_real/wrfinput_d02_updated "$SCRATCH_PATH"/WRF_$res/test/em_real/wrfinput_d02
   fi
   # Find and replace chem_in_opt in namelist.input with chem_in_opt = 1,
   namelist_input="$SCRATCH_PATH"/WRF_$res/test/em_real/namelist.input
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
