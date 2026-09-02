#!/bin/bash
#SBATCH --mail-user=matthias.reif@student.uibk.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=wrf_chain
#SBATCH --partition=std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00


set -euo pipefail
# set -x

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

# ------------------------------------------------------------------
# Load modules and activate Conda environment
# ------------------------------------------------------------------
module purge
module load $CONDA_MODULE
eval "$("$UIBK_CONDA_DIR/bin/conda" shell.bash hook)"
conda activate "$CONDA_ENV"

# ------------------------------------------------------------------
# CONFIGURATION: Choose simulation type
# ------------------------------------------------------------------

USER_ID="c7071034" # adjust your User ID
REAL_SCRIPT="$GITHUB_PATH"/WRF_VPRM_inComplexTopo/WRF_VPRM_pre/run_WPS_WRF/run_WPS_and_REAL.sh
WRF_SCRIPT="$GITHUB_PATH"/WRF_VPRM_inComplexTopo/WRF_VPRM_pre/run_WPS_WRF/run_WRF.sh
COPY_SCRIPT="$GITHUB_PATH"/WRF_VPRM_inComplexTopo/WRF_VPRM_pre/run_WPS_WRF/copy_wrf_output.sh
resx=("3km")
# Options: "" (clear sky) or "_cloudy" (cloudy and rainy)
# Anything other than "_cloudy" selects clear_sky_dates; the string is also the
# suffix of the output dir ($SCRATCH/DATA/WRFOUT/WRFOUT_ALPS_${res}${type}).
# "_toptfix" keeps the corrected runs separate from the archived buggy-Topt runs
# in WRFOUT_ALPS_54km/, which are the before-reference for the regression check.
type="_toptfix"

# START/END dates (6h spinup, end = 0h next day)
# Define date sets for each scenario
clear_sky_dates=(
  # Trimmed to the single Zenodo date (2012-07-27) for the Topt-fix verification.
  # Restore the full list below to run the whole clear-sky campaign again.
  "2012-07-26_18:00:00 2012-07-28_00:00:00"
# "2012-01-11_18:00:00 2012-01-13_00:00:00"
# "2012-02-21_18:00:00 2012-02-23_00:00:00"
# "2012-03-26_18:00:00 2012-03-28_00:00:00"
# "2012-04-26_18:00:00 2012-04-28_00:00:00"
# "2012-05-25_18:00:00 2012-05-27_00:00:00"
# "2012-06-27_18:00:00 2012-06-29_00:00:00"
# "2012-08-26_18:00:00 2012-08-28_00:00:00"
# "2012-09-20_18:00:00 2012-09-22_00:00:00"
# "2012-10-19_18:00:00 2012-10-21_00:00:00"
# "2012-11-14_18:00:00 2012-11-16_00:00:00"
# "2012-12-23_18:00:00 2012-12-25_00:00:00"
)

cloudy_dates=(
  "2012-01-19_18:00:00 2012-01-21_00:00:00"
  "2012-02-18_18:00:00 2012-02-20_00:00:00"
  "2012-03-18_18:00:00 2012-03-20_00:00:00"
  "2012-04-06_18:00:00 2012-04-08_00:00:00"
  "2012-05-15_18:00:00 2012-05-17_00:00:00"
  "2012-06-09_18:00:00 2012-06-11_00:00:00"
  "2012-07-14_18:00:00 2012-07-16_00:00:00"
  "2012-08-29_18:00:00 2012-08-31_00:00:00"
  "2012-09-18_18:00:00 2012-09-20_00:00:00"
  "2012-10-14_18:00:00 2012-10-16_00:00:00"
  "2012-11-10_18:00:00 2012-11-12_00:00:00"
  "2012-12-06_18:00:00 2012-12-08_00:00:00"
)

# Select dates based on type
if [ "$type" == "_cloudy" ]; then
  dates=("${cloudy_dates[@]}")
else
  dates=("${clear_sky_dates[@]}")
fi

for d in "${dates[@]}"; do
  START_DATE=$(echo $d | awk '{print $1}')
  END_DATE=$(echo $d | awk '{print $2}')
  echo "Submitting jobs for $START_DATE → $END_DATE"

  jobids_r=($($REAL_SCRIPT "$START_DATE" "$END_DATE" "${resx[@]}" | grep -Eo '\b[0-9]{7}\b'))
  if [ ${#jobids_r[@]} -eq 0 ]; then
    echo "ERROR: No REAL job IDs extracted!" >&2
    exit 1
  fi
  echo "Submitted JobIDs: ${jobids_r[*]}"
  sleep 10

  # Wait until all jobids_r are done
  for jid in "${jobids_r[@]}"; do
    while squeue -j "$jid" -u "$USER_ID" | grep -q "$jid"; do
      echo "Real Job $jid still running..."
      sleep 60
    done
    echo "Job $jid finished."
    # Check HOW it finished, not just that it stopped. real.exe writes wrfbdy_d01
    # progressively, so a job killed part-way (e.g. OUT_OF_MEMORY, job 7590899)
    # leaves a truncated but existing wrfbdy that the -f test below happily
    # accepts -- which would launch the multi-hour WRF job on corrupt boundary
    # data. Fail here instead.
    state=$(sacct -j "$jid" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}')
    if [ "$state" != "COMPLETED" ]; then
      echo "ERROR: real job $jid ended as '$state', not COMPLETED." >&2
      echo "       Any wrfbdy_d01 it wrote is likely truncated - not proceeding." >&2
      exit 1
    fi
  done
  # Verify wrfbdy files exist
  for res in "${resx[@]}"; do
    wrfbdy_file="$SCRATCH_PATH/WRF_$res/test/em_real/wrfbdy_d01"
    if [ -f "$wrfbdy_file" ]; then
      echo "✓ wrfbdy found for $res: $(ls -lh $wrfbdy_file | awk '{print $9, $5}')"
    else
      echo "ERROR: wrfbdy NOT found for $res at $wrfbdy_file" >&2
      exit 1
    fi
  done
  jobids=($($WRF_SCRIPT "$START_DATE" "$END_DATE" "${resx[@]}" | grep -Eo '\b[0-9]{7}\b' ))
  if [ ${#jobids[@]} -eq 0 ]; then
    echo "ERROR: No WRF job IDs extracted!" >&2
    exit 1
  fi
  echo "Submitted JobIDs: ${jobids[*]}"
  sleep 10

  # Wait until all jobids are done
  for jid in "${jobids[@]}"; do
    while squeue -j "$jid" -u "$USER_ID" | grep -q "$jid"; do
      echo "WRF Job $jid still running..."
      sleep 60
    done
    echo "Job $jid finished."
  done

  echo "All jobs finished for $START_DATE → $END_DATE, copying results..."
  bash $COPY_SCRIPT $START_DATE "$type" "${resx[@]}"

done

echo "All simulations completed."
