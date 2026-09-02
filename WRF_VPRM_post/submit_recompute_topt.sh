#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/jobs"
MEMBERS=(p10 p25 p50 p75 p90)
RESOLUTIONS=(1km 9km 54km)
START="2012-01-01 00:00:00"
END="2012-12-31 00:00:00"
SUBMIT=1
for m in "${MEMBERS[@]}"; do
  for r in "${RESOLUTIONS[@]}"; do
    jobfile="$HERE/jobs/recompute_${m}_${r}.slurm"
    cp "$HERE/job_recompute_vprm_fluxes.slurm" "$jobfile"
    sed -i "s/^tag=.*/tag=\"topt_${m}\"/" "$jobfile"
    sed -i "s|^params=.*|params=\"vprm_params_topt_${m}.csv\"|" "$jobfile"
    sed -i "s|^start_date=.*|start_date=\"$START\"|" "$jobfile"
    sed -i "s|^end_date=.*|end_date=\"$END\"|" "$jobfile"
    sed -i "s/^for res in .*$/for res in ${r}; do/" "$jobfile"
    echo "submitting $jobfile"
    sbatch "$jobfile"
  done
done
