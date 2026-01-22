#!/bin/bash

set -euo pipefail

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

# Define the start and end dates
start_date="2020-08-01"
end_date="2020-08-08"

# Convert start and end dates to seconds since the epoch
current_date=$(date -d "$start_date" +%Y-%m-%d)
end_date=$(date -d "$end_date" +%Y-%m-%d)

# Loop through the date range
while [[ "$current_date" < "$end_date" || "$current_date" == "$end_date" ]]; do
    # Loop through domain numbers (d01 to d03)
    for domain in d01 d02 d03; do
        # Define the source file name
        file_name="vprm_input_${domain}_${current_date}_00:00:00.nc"
        
        # Define the target file name without the .nc extension
        target_file=$SCRATCH_PATH/WRF/test/em_real/vprm_input_${domain}_${current_date}_00:00:00.nc

        # Copy the file to the target directory without the .nc ending
        cp ${file_name} ${target_file}
        
        # Check if the copy was successful
        if [ $? -eq 0 ]; then
            echo "Successfully copied: ${file_name} to ${target_file}"
        else
            echo "Failed to copy: ${file_name}"
        fi
    done
    
    # Increment the date by one day
    current_date=$(date -I -d "$current_date + 1 day")
done
