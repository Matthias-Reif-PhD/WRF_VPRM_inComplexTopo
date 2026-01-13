#!/bin/bash
#SBATCH --job-name=get_CAMS_surface
#SBATCH --mail-user=matthias.reif@student.uibk.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3000

# Load Anaconda module and activate the environment
module purge
module load Anaconda3/2023.03/miniconda-base-2023.03
eval "$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"
conda activate /scratch/c7071034/conda_envs/py_basic

# use the correct url in ~/.cdsapirc
# url: https://ads.atmosphere.copernicus.eu/api
# Replace the line starting with 'url:' using sed
file_path="$HOME/.cdsapirc"
sed -i '/^url:/c\url: https://ads.atmosphere.copernicus.eu/api' "$file_path"
echo "Updated 'url:' line in $file_path."

for month in {01..11}; do
    start_date="2012-$month-01"
    next_month=$(printf "%02d" $((10#$month + 1)))
    end_date="2012-$next_month-01"
    echo "processing $start_date to $end_date"
    /scratch/c7071034/conda_envs/py_basic/bin/python CAMS_get_surface_API.py -s $start_date -e $end_date
done

start_date="2012-12-01"
end_date="2013-01-01"
echo "processing $start_date to $end_date"
/scratch/c7071034/conda_envs/py_basic/bin/python CAMS_get_surface_API.py -s $start_date -e $end_date


# /scratch/c7071034/conda_envs/py_basic/bin/python CAMS_get_CO2_API.py

# for ECMWF use
# file_path="$HOME/.cdsapirc"
# sed -i '/^url:/c\url: https://cds.climate.copernicus.eu/api' "$file_path"