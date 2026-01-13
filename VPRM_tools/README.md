# VPRM_tools

Workflows for optimizing VPRM parameters using FLUXNET observations and MODIS satellite data.

## Workflow

### Step 1: Prepare Input Data

**Download FLUXNET2015 data:**
```bash
# From https://fluxnet.org/data/fluxnet2015-dataset/
```

**Generate MODIS time series:**
```bash
Rscript Modis_timeseries_FluxNet.r
```

### Step 2: Upload to HPC Cluster

```bash
zip FLX_AT-Mie_files.zip FLX_AT-Mie_M*
scp FLX_AT-Mie_files.zip c7071034@leo5.uibk.ac.at:/scratch/c7071034/DATA/Fluxnet2015/
```

### Step 3: Run Parameter Optimization

```bash
./submit_jobs_tune_VPRM.sh
```
Optimizes VPRM parameters for both old and new formulations using differential evolution (experimental pmodel option is available).

### runs `tune_VPRM.py`
Main optimization script that:
- Reads FLUXNET half-hourly data and MODIS satellite data for each site
- Preprocesses data (QC filtering, missing value handling, variable calculation)
- Calculates vegetation indices (LSWI, EVI) from MODIS bands
- Optimizes VPRM parameters using differential evolution algorithm and NNSE
- Supports multiple VPRM versions (old/new formulation) and target variables (NEE, RECO, GPP)
- Evaluates results using regression metrics (RMSE, R², NNSE)
- Generates diagnostic plots and saves optimized parameters to Excel files

**Dependencies (codes used by `tune_VPRM.py`):**
- `VPRM.py`: VPRM model implementations (old/new versions, RECO/GPP calculations)
- `pModel.py`: P-model for sub-daily GPP predictions (alternative to VPRM)
- `plots_for_VPRM.py`: Visualization functions for optimization results
- `Modis_timeseries_FluxNet.r`: Pre-processing script to generate MODIS input files




### Step 4: Visualize Results

**Generate parameter plots:**
```bash
# plots_for_VPRM_from_excel.ipynb - saves results for all sites
```

### Other Key Scripts

- `Modis_timeseries_FluxNet.r`: Extract MODIS time series for FLUXNET sites
- `submit_jobs_tune_VPRM.sh`: Submit parameter optimization jobs to cluster (runs `tune_VPRM.py`)
- `plots_for_VPRM_from_excel.ipynb`: Generate parameter distribution plots from optimization results





