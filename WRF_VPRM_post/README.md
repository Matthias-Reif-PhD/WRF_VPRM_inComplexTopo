# WRF_VPRM_post

Post-processing: turns `wrfout` NetCDF (and offline-recomputed VPRM fluxes) into the
timeseries/CSV intermediates, then into the paper's figures and tables. Conda env:
`wrf_vprm` (`$CONDA_ENV_WRF_VPRM`, from `../VPRM_tools_post.yml`).

Script names encode which figure/table each one produces, using the manuscript's
**current** numbering (`Draft_CO2inRealTopo_REVISION_pre.tex`): main figures 1-11, main
tables 1-2, Appendix F = additional tables, Appendix G = additional figures, Appendix H =
seasonal decomposition.

## Pipeline order

1. **Extract** — `extract_SITE_timeseries.py`, `extract_dPdT_timeseries.py`,
   `extract_wrf_domains_mean_timeseries.py` (+ `_ensemble.py` variant, one call per Topt
   percentile member) read `wrfout` and write the CSV intermediates in `csv/`. Driven by
   the matching `job_extract_*.slurm`.
2. **Recompute** (offline VPRM flux recalculation, no WRF re-run) — `recompute_vprm_fluxes.py`,
   driven by `jobs/recompute_p{10,25,50,75,90}_{1,9,54}km.slurm` (one job per Topt-percentile
   member per resolution) and `job_recompute_vprm_fluxes.slurm`; `build_member_param_csvs.py`
   and `build_site_param_csv.py` build the parameter-set CSVs it reads (`vprm_params_*.csv`).
3. **Figures & tables** — the `Fig*.py` scripts below, each writing into `plots/`.
4. **Ship** — `ship_tables.py` copies the generated `.tex` tables into `plots/plots_final/`
   (what the manuscript actually `\input`s) and re-verifies them there.

## Figure/table scripts (`Fig*.py`)

| Script | Produces |
|---|---|
| `Fig1_domains_topo_sites.py` | Fig. 1 (domain/topo/site map) |
| `Fig2_PFTs_d03_d01.py` | Fig. 2 (vegetation classes) |
| `Fig3a_Topt_perSiteYear_fit.py` | Fig. 3a (per-site-year Topt fit) |
| `Fig3b_AppxG10_G11_Topt_tuneParam.py` | Fig. 3b, Fig. G10, Fig. G11 (Topt regression + PFT/NNSE boxplots; run after `VPRM_tools/main_tune_VPRM.py`) |
| `Fig4_AppxG1_G4_Table1_TableF5_F6_FLUXNET_eval.py` | Fig. 4 + Figs. G1-G4 (5-site FLUXNET validation), Table 1, Table F5, Table F6 |
| `Fig5_6_AppxG5_G6_Table2_TableH_WRFout_hourly_means_and_timeseries.py` | Fig. 5, Fig. 6, Figs. G5-G6, Table 2, Tables H1-H5/H7/H9 (domain-mean diurnal composites + seasonal decomposition tables) |
| `Fig7_AppxH6_H8_effects_seasonal.py` | Fig. 7, Fig. H6, Fig. H8 (seasonal effect-size bars; reads the tables the previous script ships) |
| `Fig8_VPRM_params_dFldT.py` | Fig. 8 (VPRM GPP/Reco vs. T and their derivatives, per PFT) |
| `Fig9_AppxG7_areafluxes_per_timestep.py` | Fig. 9 + Fig. G7 (spatial ∂GPP/∂T and radiation-scaling maps) |
| `Fig10_dFldT_hourly_mean.py` | Fig. 10 (hourly ΔGPP/ΔReco vs. temperature contribution) |
| `Fig11_linPertComp_hourly_mean.py` | Fig. 11 (linear-perturbation driver decomposition) |

`topt_box.py` is a shared helper (not a figure script itself) for the fitted-Topt boxplot
panel used by both Fig. 3a and Fig. 3b.

## Other scripts

- `check_Y_sign_constancy.py`: verifies a manuscript claim (driver-contribution sign
  constancy through the year) from `Fig11`'s dumped per-timestep records.
- `diagnose_itlav.py`: read-only plausibility diagnosis of the IT-Lav FLUXNET site;
  supplementary/diagnostic, but its numeric findings are quoted in the manuscript's
  IT-Lav discussion (Sect. `sec:FLUX`).
- `swint_delta_res_single_hour.py`: single-hour swint_opt=1 counterfactual supporting the
  Appendix G radiation-figure discussion.
- `report_coarse_eval.py`: 9/54 km FLUXNET evaluation follow-up (`COARSE_EVAL_15e.md`).
- `mk_bias_percentile_sensitivity.py`: **exploratory**, not part of the shipped figure/table
  set (its own docstring says so) — checks whether the Topt-percentile choice moves tower
  bias enough to matter.
- `noFig_wrf_T2_correlations.py`: **not in the publication** (its own docstring says so) —
  T2-vs-topography correlation exploration.

## Not part of the published paper

- `pmodel/`: P-model GPP/Reco implementation, excluded from the publication (would need
  long-term simulations to validate; own README).

## Data layout

- `csv/`: extracted timeseries intermediates (tracked in git where a `Fig*.py`/table
  script actually reads them back as input — see each script's `pd.read_csv` calls;
  large multi-GB WRF-derived NetCDF fields live on the paper's Zenodo data release, not
  in this repo).
- `plots/`: figure/table output; `plots/plots_final/` is what the manuscript `\input`s.
- `jobs/`: per-member SLURM scripts for the offline recompute step.

## SLURM jobs

Batch scripts (`job_*.slurm`, `jobs/*.slurm`) for HPC cluster execution of extraction,
recompute and figure-generation tasks — real, adaptable job scripts (edit account/
partition/module lines for your own cluster).
