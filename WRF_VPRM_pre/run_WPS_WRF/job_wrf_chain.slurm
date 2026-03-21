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

# adjust your User ID
USER_ID="c7071034" 

REAL_SCRIPT="$GITHUB_PATH"/WRF_VPRM_inComplexTopo/WRF_VPRM_pre/run_WPS_WRF/run_WPS_and_REAL.sh
WRF_SCRIPT="$GITHUB_PATH"/WRF_VPRM_inComplexTopo/WRF_VPRM_pre/run_WPS_WRF/run_WRF.sh
COPY_SCRIPT="$GITHUB_PATH"/WRF_VPRM_inComplexTopo/WRF_VPRM_pre/run_WPS_WRF/copy_wrf_output.sh
resx=("54km")

# START/END dates (6h spinup, end = 0h next day)
# Clear sky 
  type=""
  dates=(
  # "2012-01-11_18:00:00 2012-01-13_00:00:00"
  # "2012-02-21_18:00:00 2012-02-23_00:00:00"
  "2012-03-26_18:00:00 2012-03-28_00:00:00"
  "2012-04-26_18:00:00 2012-04-28_00:00:00"
  "2012-05-25_18:00:00 2012-05-27_00:00:00"
  "2012-06-27_18:00:00 2012-06-29_00:00:00"
  "2012-07-26_18:00:00 2012-07-28_00:00:00"
  "2012-08-26_18:00:00 2012-08-28_00:00:00"
  "2012-09-20_18:00:00 2012-09-22_00:00:00"
  "2012-10-19_18:00:00 2012-10-21_00:00:00"
  "2012-11-14_18:00:00 2012-11-16_00:00:00"
  "2012-12-23_18:00:00 2012-12-25_00:00:00"
  )

  # Cloudy and rainy
  # type="_cloudy"
  # dates=(
  # "2012-01-19_18:00:00 2012-01-21_00:00:00"
  # "2012-02-18_18:00:00 2012-02-20_00:00:00"
  # "2012-03-18_18:00:00 2012-03-20_00:00:00"
  # "2012-04-06_18:00:00 2012-04-08_00:00:00"
  # "2012-05-15_18:00:00 2012-05-17_00:00:00"
  # "2012-06-09_18:00:00 2012-06-11_00:00:00"
  # "2012-07-14_18:00:00 2012-07-16_00:00:00"
  # "2012-08-29_18:00:00 2012-08-31_00:00:00"
  # "2012-09-18_18:00:00 2012-09-20_00:00:00"
  # "2012-10-14_18:00:00 2012-10-16_00:00:00"
  # "2012-11-10_18:00:00 2012-11-12_00:00:00"
  # "2012-12-06_18:00:00 2012-12-08_00:00:00"
  # )

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