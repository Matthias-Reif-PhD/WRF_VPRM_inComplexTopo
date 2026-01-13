#!/bin/bash
#SBATCH --mail-user=matthias.reif@student.uibk.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00

# Load Anaconda module and activate the environment
module load Anaconda3/2023.03/miniconda-base-2023.03
eval "$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"
conda activate /scratch/c7071034/conda_envs/pyvprm4

# Run the Python script
/scratch/c7071034/conda_envs/pyvprm4/bin/python /scratch/c7071034/DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/vprm_preprocessor_new.py \
  --chunk_x=$CHUNK_X --chunk_y=$CHUNK_Y --n_cpus=96
