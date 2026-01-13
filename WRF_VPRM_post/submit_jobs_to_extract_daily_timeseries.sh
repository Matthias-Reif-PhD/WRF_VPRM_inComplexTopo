#!/bin/bash

# Define variables
start_date="2012-07-01 00:00:00"
end_date="2012-07-31 00:00:00"
wrf_path="/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241230_093202_ALPS_3km"

# Convert dates to seconds since epoch for looping
start_sec=$(date -d "$start_date" +%s)
end_sec=$(date -d "$end_date" +%s)
one_day=$((24 * 60 * 60))

# Loop through each day
current_sec=$start_sec
while [ "$current_sec" -lt "$end_sec" ]; do
    # Format the current date
    current_date=$(date -d "@$current_sec" +"%Y-%m-%d")
    next_date=$(date -d "@$((current_sec + one_day))" +"%Y-%m-%d")

    # Create SLURM script for each day
    cat <<EOF >"job_${current_date}.sh"
#!/bin/bash
#SBATCH --job-name=extract_timeseries_${current_date}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=100

module purge
module load Anaconda3/2023.03/miniconda-base-2023.03
eval "\$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"
conda activate /scratch/c7071034/conda_envs/vprm4
srun python extract_timeseries.py -w "$wrf_path" -s "${current_date} 00:00:00" -e "${next_date} 00:00:00"
EOF

    # Submit the job to the cluster
    sbatch "job_${current_date}.sh"

    # Increment the date by one day
    current_sec=$((current_sec + one_day))
done
