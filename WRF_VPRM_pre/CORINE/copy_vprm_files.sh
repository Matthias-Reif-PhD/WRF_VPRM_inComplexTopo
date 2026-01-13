#!/bin/bash

# Define the start and end dates
start_date="2012-07-01"
end_date="2012-07-31"

# Convert start and end dates to seconds since the epoch
current_date=$(date -d "$start_date" +%Y-%m-%d)
end_date=$(date -d "$end_date" +%Y-%m-%d)

# Loop through the date range
while [[ "$current_date" < "$end_date" || "$current_date" == "$end_date" ]]; do
    # Loop through domain numbers (d01 to d03)
    for domain in d02; do
        # Define the source file name
        file_name="vprm_input_${domain}_${current_date}_00:00:00.nc"
        
        # Define the target file name without the .nc extension
        target_file="/scratch/c7071034/WRF/test/em_real/vprm_input_${domain}_${current_date}_00:00:00.nc"

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
