N_CHUNKS=8 # set this also in preprocessor_config.yaml

for i in $(seq 1 $N_CHUNKS); do
   for j in $(seq 1 $N_CHUNKS); do
        sbatch --export=CHUNK_X=$i,CHUNK_Y=$j --job-name=job_VPRM_par_${i}_${j} job_preprocessor_parallel.slurm
    done
done
