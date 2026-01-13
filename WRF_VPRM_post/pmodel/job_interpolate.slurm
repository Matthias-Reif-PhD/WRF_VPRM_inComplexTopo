#!/bin/bash
#SBATCH --job-name=interp_Modis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1800

module purge
module load Anaconda3/2023.03/miniconda-base-2023.03
eval "$(/bin/conda shell.bash hook)"
conda activate /scratch/c7071034/conda_envs/pyrealm311
srun python interpolate_fpar_gapfilled_on_wrf.py