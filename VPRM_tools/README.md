# VPRM_tools

Workflows for optimizing VPRM parameters using FLUXNET observations and MODIS satellite data.
Conda env: `pyrealm311` (differential-evolution tuning dependencies; no exported `.yml` in
this repo). Despite their names, `../VPRM_tools_pre.yml` and `../VPRM_tools_post.yml` are
**not** this directory's env files -- they're named for the pipeline's pre-/post-processing
stages and set up `WRF_VPRM_pre/pyVPRM/`'s `pyvprm4` and `WRF_VPRM_post/`'s `wrf_vprm`
envs respectively (see the root `.env`'s `CONDA_ENV_PYVPRM`/`CONDA_ENV_WRF_VPRM`).

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
mv FLX_AT-Mie_files.zip "$SCRATCH_PATH"/DATA/Fluxnet2015/
```

### Step 3: Run Parameter Optimization

```bash
./submit_jobs_tune_VPRM.sh
```
Optimizes VPRM parameters for both old and new formulations using differential evolution (experimental pmodel option is available).

### runs `main_tune_VPRM.py`
Main optimization script that:
- Reads FLUXNET half-hourly data and MODIS satellite data for each site
- Preprocesses data (QC filtering, missing value handling, variable calculation)
- Calculates vegetation indices (LSWI, EVI) from MODIS bands
- Optimizes VPRM parameters using differential evolution algorithm and NNSE
- Supports multiple VPRM versions (old/new formulation) and target variables (NEE, RECO, GPP)
- Evaluates results using regression metrics (RMSE, R², NNSE)
- Generates diagnostic plots and saves optimized parameters to Excel files

**Dependencies (codes used by `main_tune_VPRM.py`):**
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
- `submit_jobs_tune_VPRM.sh`: Submit parameter optimization jobs to cluster (runs `main_tune_VPRM.py`)
- `plots_for_VPRM_from_excel.ipynb`: Generate parameter distribution plots from optimization results
- `submit_jobs_tune_SITE.sh`: Site-specific-parameter tuning jobs (the "SITE" parameter set used
  in Table 1's FLUXNET evaluation, distinct from the domain-wide "ALPS" set)

### Topt-percentile ensemble (5-member sensitivity, feeds Figs. 3b/5/7/10/11's shaded bands)

The GPP-based $T_\text{opt}$ per-site-year fits (`WRF_VPRM_post/Fig3a_Topt_perSiteYear_fit.py`
and `Fig3b_AppxG10_G11_Topt_tuneParam.py`, run after `main_tune_VPRM.py`) produce raw
per-site-year dumps; these two scripts turn them into the 5-member (p10/p25/p50/p75/p90)
percentile ensemble used throughout the paper's uncertainty bands:

- `derive_topt_members.py`: builds `topt_percentiles.csv` (per-PFT $T_\text{opt}$ percentiles,
  from `topt_raw_persiteyear.csv`)
- `derive_param_members.py`: builds `param_percentiles.csv` (per-PFT PAR0/lambda percentiles,
  from `params_raw_persiteyear.csv`), re-tuned against each $T_\text{opt}$ member
- `submit_jobs_topt_sensitivity.sh`: submits the per-percentile re-tuning jobs these two
  scripts' inputs are built from

Both output CSVs are then consumed by `WRF_VPRM_post/build_member_param_csvs.py` to build
the 5 full VPRM parameter sets, and by `WRF_VPRM_post/recompute_vprm_fluxes.py` to recompute
fluxes for each member offline (no WRF re-run).





