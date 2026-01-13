#!/bin/bash
#SBATCH --job-name=get_ECMWF2
#SBATCH --mail-user=matthias.reif@student.uibk.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=720

# Load Anaconda module and activate the environment
module purge
module load Anaconda3/2023.03/miniconda-base-2023.03
eval "$(/usr/site/hpc/x86_64/generic/Anaconda3/2023.03/bin/conda shell.bash hook)"
conda activate /scratch/c7071034/conda_envs/py_basic

cd /scratch/c7071034/DATA/ECMWF/pressure
file_path="$HOME/.cdsapirc"
# Replace the line starting with 'url:' using sed
sed -i '/^url:/c\url: https://cds.climate.copernicus.eu/api' "$file_path"
echo "Updated 'url:' line in $file_path."

# "12.01."
# "22.02."
# "15.03."
# "27.03."
# "27.04."
# "08.05."
# "26.05."
# "15.06."
# "28.06."
# "27.07."
# "18.08."
# "27.08."
# "08.09."
# "21.09."
# "20.10."
# "15.11."
# "24.12."

#cloudy 
# "07.04."
# "16.05."
# "10.06."
# "15.07."
# "30.08."
# "19.09."

# "19.09."
# "15.10."
# "11.11."
# "07.12."
dates=(
"20.01."
"19.02."
"19.03."
)


start_dates=()
end_dates=()
months=()
year=2012

for date in "${dates[@]}"; do
    day=${date:0:2}
    month=${date:3:2}
    # Remove leading zeros for arithmetic
    day_num=$((10#$day))
    # day before
    day_before=$((day_num - 1))
    next_day=$((day_num + 1))
    # Pad with zero if needed
    if [ $next_day -lt 10 ]; then
        next_day="0$next_day"
    fi
    if [ $day_before -lt 10 ]; then
        day_before="0$day_before"
    fi
    start_dates+=("$day_before")
    end_dates+=("$next_day")
    months+=("$month")
done

# Example: print all
for i in "${!dates[@]}"; do
    echo "Start: ${start_dates[$i]}, End: ${end_dates[$i]}, Month: ${months[$i]}, Year: $year"
    /scratch/c7071034/conda_envs/py_basic/bin/python /home/c707/c7071034/Github/WRF_VPRM_pre/get_ECMWF_pressure.py --start ${start_dates[$i]} --end "${end_dates[$i]}" --month "${months[$i]}"  --year $year
done


# /scratch/c7071034/conda_envs/py_basic/bin/python /home/c707/c7071034/Github/WRF_VPRM_pre/get_ECMWF_pressure.py --start 16 --end 16 --month 3 --year 2012


# for CAMS use
# sed -i '/^url:/c\url: https://ads.atmosphere.copernicus.eu/api' "$file_path"