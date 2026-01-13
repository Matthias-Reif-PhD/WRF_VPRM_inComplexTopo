#!/bin/bash
#SBATCH --job-name=extract_wrf_dPdT_REF
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1200

module purge
module load Anaconda3/2023.03/miniconda-base-2023.03
# eval "$(/bin/conda shell.bash hook)"
eval "$(/usr/site/hpc/x86_64/generic/Anaconda3/2023.03/bin/conda shell.bash hook)"
conda activate /scratch/c7071034/conda_envs/pyrealm311

sim_typ="_cloudy" # "" for clear sky or "_cloudy" - run one after the other
start_date="2012-01-01 00:00:00"
end_date="2012-12-31 00:00:00"
srun python extract_dPdT_timeseries.py -s "$start_date" -e "$end_date" -t "$sim_typ"

