#!/bin/bash
#TODO: adjust paths

# Load environment variables from project root .env
ENV_FILE="$(dirname "$(dirname "$(pwd)")")/.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: .env file not found at $ENV_FILE" >&2
  exit 1
fi
set -a
source "$ENV_FILE"
set +a

START_DATE=$1
type=$2
# Parse resx from arguments, or use defaults
resx=("${@:3}")
if [ ${#resx[@]} -eq 0 ]; then
  resx=("3km" "9km" "27km" "54km")
fi

for res in "${resx[@]}"
do
  echo "Moving outfiles for resolution: $res"
  # Define source and destination directories
  # change folder here and below dest_dir2 
  folder_name="WRFOUT_ALPS_${res}${type}" 
  source_dir="$SCRATCH_PATH/WRF_$res/test/em_real"
  destination_dir="$SCRATCH_PATH/DATA/WRFOUT/$folder_name"
  # Create the destination directory
  mkdir -p "$destination_dir"

  # Move the specified files from source to destination
  cp "$source_dir/job_WRF.slurm_$res" "$destination_dir"
  cp "$source_dir/namelist.input" "$destination_dir"
  # cp "$source_dir/rsl.error.0000" "$destination_dir"
  # cp "$source_dir/rsl.out.0000" "$destination_dir"
  cp $SCRATCH_PATH/WPS/namelist.wps "$destination_dir" 
  cp $SCRATCH_PATH/WPS/geo_em.d0* "$destination_dir" 
  rm $source_dir/rsl.*
  rm $source_dir/met_em*

  # Use a loop to move files matching the wildcards, checking if they exist first
  for file in "$source_dir"/wrfbdy* "$source_dir"/wrfinput* "$source_dir"/wrfout_d01* "$source_dir"/wrfrst*; do
    if [ -e "$file" ]; then
      mv "$file" "$destination_dir"
    fi
  done

  if [[ "$res" == "3km" ]]; then
    folder_name2="WRFOUT_ALPS_1km${type}"
    destination_dir2="$SCRATCH_PATH/DATA/WRFOUT/$folder_name2"
    mkdir -p "$destination_dir2"
    for file in "$source_dir"/wrfout_d02* ; do
      if [ -e "$file" ]; then
        mv "$file" "$destination_dir2"
      fi
    done
  fi

  # Print success message
  echo "Files moved to $destination_dir successfully." # WRFOUT_20241120_202043_ALPS_corene_d01
done
