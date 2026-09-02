# WRF/WPS Execution Scripts

## Overview
Scripts for running WPS (WRF Preprocessing System) and WRF (Weather Research & Forecasting) model with VPRM integration.

Clone the modified WRF and WPS code to your scratch directory, or search through the commits:

``` 
clone https://github.com/Matthias-Reif-PhD/WRF/tree/WRF-P
clone https://github.com/Matthias-Reif-PhD/WPS/tree/WRF-VPRM-CLC
```

## Scripts

### `job_wrf_chain.slurm`
Main SLURM job script that orchestrates the full WRF chain for multiple simulation dates. Submits parallel REAL and WRF jobs, monitors completion, and copies results.
Edit the USER_ID, this will only work on similar architectures

**Usage:**
- Edit the `dates` array with desired date ranges (format: `"YYYY-MM-DD_HH:MM:SS YYYY-MM-DD_HH:MM:SS"`)
- Edit `resx` array for desired resolutions (54km, 27km, 9km, 3km)
- Submit: `sbatch job_wrf_chain.slurm`

### `run_WPS_and_REAL.sh`
Processes WPS (geogrid, metgrid, ungrib) and runs REAL.exe with CO₂ boundary/initial condition updates.

**Usage:** `./run_WPS_and_REAL.sh "START_DATE" "END_DATE" res1 res2 ...`

### `run_WRF.sh`
Prepares and submits WRF jobs with updated chemistry options.

**Usage:** `./run_WRF.sh "START_DATE" "END_DATE" res1 res2 ...`

### `run_WPS_and_WRF_sequential.sh`
Sequential version of entire WPS+REAL+WRF chain for single runs. Includes date hardcoding.

### `job_real.slurm` / `job_real_small.slurm`
SLURM wrappers around REAL.exe, submitted by `job_wrf_chain.slurm`/`run_WPS_and_REAL.sh`
for the full-size and reduced-domain runs respectively.

### `job_WRF.slurm_{27,9,3,54}km`
Per-resolution SLURM wrappers around wrf.exe, one for each of the four
independent grid spacings used in the paper (54/9/3 km are independent
ERA5-driven runs; only 3 km nests down to 1 km).

### `copy_wrf_output.sh`
Copies finished `wrfout_*` files from the run directory to
`$SCRATCH_PATH/DATA/WRFOUT/`, the location the `WRF_VPRM_post/extract_*.py`
scripts read from.

### `namelists/`
The `namelist.input_{27,9,3,54}km` templates `job_wrf_chain.slurm` copies
into place per resolution before submitting.

## Key Features
- Automated VPRM input file copying
- CO₂ boundary condition updates via Python scripts
- Dynamic namelist.input/namelist.wps configuration
- Multi-resolution support with nested domains (3km has nested d02)
- Job dependency management via SLURM

## Requirements
- Conda environment: `pyvprm4` or `py_basic`
- Input data: ERA5 GRIB files, VPRM NetCDF files
- WRF/WPS compiled binaries


