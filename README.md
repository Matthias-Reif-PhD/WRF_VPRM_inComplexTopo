# WRF_VPRM_inComplexTopo

Workflow for WRF-VPRM simulations and analysis in complex topography.

## Folder Structure

### **VPRM_tools/**
Workflows for VPRM parameter optimization at FLUXNET sites:
- setup conda environment using "VPRM_tools_post.yml"
- Download FLUXNET2015 data and generate MODIS time series
- Run parameter tuning using differential evolution
- Process results (convert EPS to PNG, organize by site)
- Generate plots comparing optimized parameters with literature values

### **WRF_VPRM_pre/**
Setup and preprocessing steps for WRF/VPRM simulations:
- setup environment from pyVPRM and tools are HPC specific 
- Configure directory structure on home and scratch drives
- Download and process CAMS CO₂ data via CDS API
- Fetch ERA5 boundary conditions
- Reproject CORINE land cover data
- Prepare VPRM inputs (via pyVPRM and vprm_shapeshifter)
- Generate files needed for WRF domain setup and boundary conditions

### **WRF_VPRM_post/**
Post-processing and analysis of WRF-VPRM simulations:
- setup conda environment using "VPRM_tools_post.yml"
- Extract time series data from WRF output (site-level and domain-averaged)
- Generate publication figures and perform correlation/evaluation analyses
- P-model implementation for computing GPP and RECO (not included in publication; requires long-term simulations)
- Calculate performance metrics (RMSE, R²) at various resolutions

## Workflow

1. **Pre-processing** (WRF_VPRM_pre/): Set up data and prepare inputs
2. **VPRM Optimization (optional)** (VPRM_tools/): Tune VPRM parameters using FLUXNET observations
3. **WRF Simulations**: Run coupled WRF-VPRM model (scripts in WRF_VPRM_pre/)
4. **Post-processing** (WRF_VPRM_post/): Extract results and generate analysis/figures

## Changes in v0.4

v0.4 accompanies the second revision of the GMD manuscript. Only the
post-processing changed; the modified WRF and WPS code is identical to v0.3
and is re-released as v0.4 for consistent versioning.

- `extract_dPdT_timeseries.py`: the T2m < 5 °C threshold is applied to GPP only.
  dGPP/dT2m grows unrealistically large near the minimum temperature of
  photosynthesis, whereas dR_eco/dT2m is constant, so R_eco is now evaluated over
  the whole area of analysis. The tracked `csv/dPdT_timeseries_*.csv` were
  regenerated accordingly (GPP columns unchanged).
- `Fig11_linPertComp_hourly_mean.py`: optional dump (`DUMP_LINPERT=1`) of the
  per-timestep linear-perturbation contributions, also restricted to
  T2m >= 5 °C, with an exact temperature-only check. The results and their
  24-day shares are in `WRF_VPRM_post/csv/linpert/`.
- Round-2 figure and table updates: terrain panel in Fig. 8, temperature
  contribution in front in Fig. 9, distinct markers in Fig. 3b, VPRM class area
  fractions (`vprm_class_area_fractions.py`), and Table D2 referenced to the
  central ALPS calibration.
