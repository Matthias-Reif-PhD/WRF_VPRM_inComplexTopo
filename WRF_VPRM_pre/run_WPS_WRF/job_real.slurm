#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=real
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=40
# 20GB is enough for 9km but NOT for 3km: real.exe there peaked at ~4.8GB per
# task x 4 tasks and was OOM-killed (job 7590899). 80GB (20GB/task) is the value
# this script originally carried and is what 3km needs.
#SBATCH --mem=80GB

set -euo pipefail

# Environment variables are passed via sbatch --export=ALL from parent script
# No need to reload .env here

# ------------------------------------------------------------------
# Load modules and activate Conda environment
# ------------------------------------------------------------------
module purge
module load $CONDA_MODULE

eval "$("$UIBK_CONDA_DIR/bin/conda" shell.bash hook)"
conda activate "$CONDA_ENV"

# ------------------------------------------------------------------
# Load WRF specific environment
# ------------------------------------------------------------------
source ~/wrf_dev.sh

# ------------------------------------------------------------------
# Run REAL with srun for parallel execution (from the directory where sbatch was called)
# ------------------------------------------------------------------
# This job is submitted from inside job_wrf_chain.slurm with --export=ALL, so it
# inherits the parent's SLURM_MEM_PER_CPU (the cluster sets DefMemPerCPU=1000).
# Combined with this job's own "--mem" (= SLURM_MEM_PER_NODE) that makes srun
# abort with "SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are
# mutually exclusive". Drop the inherited per-CPU value and keep --mem.
# NB: this is needed for ANY --mem value -- lowering --mem does not avoid it.
# job_real_small.slurm needs no such line only because it runs real.exe without srun.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

cd "$SLURM_SUBMIT_DIR" || exit 1
srun ./real.exe
