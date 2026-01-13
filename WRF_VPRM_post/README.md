# WRF_VPRM_post

Post-processing scripts for WRF-VPRM model output analysis and visualization.

## Contents

- **Figure scripts** (`Fig*.py`): Generate publication-ready figures and analysis plots
- **Extraction scripts** (`extract_*.py`): Extract time series data from WRF and FLUXNET outputs
- **Analysis scripts**: Correlation analysis, parameter tuning, model evaluation
- **pmodel/**: P-model related analysis and MODIS data processing
  - Includes post-processing capability to compute GPP from WRF output using the P-model
  - RECO estimation using the Migliavacca method is also implemented
  - Note: WRF post-processing with pmodel is not included in the publication and would require long-term simulations for validation
- **plots/**: Output directory for generated figures

## Key Scripts

- `extract_SITE_timeseries.py`: Extract site-level time series from WRF output
- `extract_wrf_domains_mean_timeseries.py`: Calculate domain-averaged time series
- `merge_timeseries.py`: Combine multiple time series files
- `rmse_r2_out_1km.py`: Calculate RMSE and R² metrics at 1km resolution

## SLURM Jobs

Batch scripts for HPC cluster execution of extraction and post-processing tasks.

