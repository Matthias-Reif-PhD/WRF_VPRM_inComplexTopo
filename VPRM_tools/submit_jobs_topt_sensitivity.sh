#!/bin/bash
# ------------------------------------------------------------------
# Topt-sensitivity tuning sweep (GMD revision, RC1/RC2 request).
# Re-tune the GPP params (PAR0, lambda) for the ENF/DBF/GRA sites at each Topt
# percentile member. p50 == V24 central (already tuned as diff_evo_V24_SW_05) ->
# only p10/p25/p75/p90 here. The per-member Topt comes from topt_percentiles.csv
# via the _pXX tag in opt_method (handled in main_tune_VPRM.py). alpha/beta are
# Topt-independent, so only GPP params differ between members.
# ------------------------------------------------------------------
ENV_FILE="$(dirname "$(pwd)")/.env"
[ -f "$ENV_FILE" ] || { echo "ERROR: .env not found at $ENV_FILE" >&2; exit 1; }
set -a; source "$ENV_FILE"; set +a

base_path="$SCRATCH_PATH"/DATA/Fluxnet2015/Alps/
maxiter=42
VPRM_old_or_new="old"
tune_env="$SCRATCH_PATH/conda_envs/pyrealm311"
percentiles=(p10 p25 p75 p90)   # p50 == V24, already done

# Only ENF/DBF/GRA sites get a Topt range (others fixed at V24 -> no re-tune needed)
sites_to_tune=(CH-Dav IT-La2 IT-Ren DE-Lkb IT-Lav \
               IT-Isp IT-PT1 FR-Fon DE-Hai \
               CH-Fru CH-Oe1 IT-MBo IT-Tor CH-Cha AT-Neu)

folders=($(find "$base_path" -type d -name "FLX_*"))

for pct in "${percentiles[@]}"; do
    opt_method="diff_evo_V24_SW_05_${pct}"
    for folder in "${folders[@]}"; do
        folder_name=$(basename "$folder")
        site_code=$(echo "$folder_name" | sed -E 's/^FLX_([^_]+)_.*/\1/')
        keep=0
        for s in "${sites_to_tune[@]}"; do [ "$s" = "$site_code" ] && keep=1; done
        [ "$keep" -eq 1 ] || continue

        job="job_${folder_name}_${VPRM_old_or_new}_${pct}.sh"
        cat <<EOF >"$job"
#!/bin/bash
#SBATCH --job-name=tune_${site_code}_${pct}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=60

set -euo pipefail
ENV_FILE="\$(dirname "\$(pwd)")/.env"
set -a; source "\$ENV_FILE"; set +a
module purge
module load \$CONDA_MODULE
eval "\$(conda shell.bash hook)"
conda activate "$tune_env"
srun python main_tune_VPRM.py -p "$base_path" -f "$folder_name" -i "$maxiter" -m "$opt_method" -v "$VPRM_old_or_new"
EOF
        sbatch "$job"
    done
done
