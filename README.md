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
