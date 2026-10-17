## Summary

Server-side work for the GMD revision (RC1 + RC2) of *"Sensitivity of CO₂ exchange
in WRF-VPRM to model resolution and parameter settings over Alpine topography"*.
Brings `REVISION` up to the state the revised manuscript is written against:
GPP-based Tₒₚₜ, an offline percentile ensemble that replaces re-running WRF, the
renumbered figure/table producers, and the extraction chain that feeds them.

24 commits, 166 files, +29 987 / −3 507.

**No new WRF simulations are introduced by this PR** — everything is calibration
and post-processing of the existing ~3 TB of output on `/scratch`.

## What changed

**Tₒₚₜ now derived from GPP, not NEE** (RC2)
- `VPRM_tools/main_tune_VPRM.py`: per-PFT `T_opt` from `GPP_NT_VUT_USTAR50`
  (ENF 16.4, DBF 23.9, MF 19.6, WET 20.2, GRA 18.6; SHB 20, CRO 22), plus a fix for
  the `T_max` RangeIndex bug that crashed every "old"-VPRM job in `data_to_append`.
- `Fig3b_AppxG10_G11_Topt_tuneParam.py`: upward Gaussian → maximum with a cubic-max
  fallback, negative GPP clipped before fitting, informed `p0`, and a guard so sites
  missing from `site_info` (AT-Mie 2022) are skipped instead of crashing.

**Offline percentile ensemble** — the core new capability
- `recompute_vprm_fluxes.py` + `jobs/recompute_p{10,25,50,75,90}_{1,9,54}km.slurm`
  recompute VPRM fluxes from WRF meteorology (T2M, PAR, EVI, LSWI) for five Tₒₚₜ
  percentile calibrations, so parameter sensitivity is evaluated without re-running WRF.
- Member parameter sets built by `build_member_param_csvs.py` / `build_site_param_csv.py`
  from the `VPRM_tools` percentile tables (`derive_topt_members.py`, `derive_param_members.py`).

**Extraction chain**
- `extract_wrf_domains_mean_timeseries{,_ensemble}.py`, `extract_SITE_timeseries.py`,
  `extract_dPdT_timeseries.py` now carry NEE (= RECO − GPP) as an explicit column, with
  their `job_extract_*.slurm` drivers. Without those columns Fig. 5/6 cannot build
  Table 2 or the seasonal tables from a fresh clone.
- The resulting csv are tracked (26 new files, ≤ 232 KB each) so the figure scripts
  run from a clone without re-extraction.

**Figures and tables**
- `Fig*.py` renamed to the manuscript's current numbering: two appendix sections were
  inserted during revision, so old Appendix D/E become F/G/H, and the Tₒₚₜ panels and
  the FLUXNET bias table moved from appendix to main text.
- New `Fig7_AppxH6_H8_effects_seasonal.py`: the DJF/MAM/JJA/SON + yearly robustness
  breakdown requested in RC2.
- New producer for `Table_par_members_2012_all.tex` — the manuscript `\input`s this
  table but nothing generated it.
- Seasonal-effects tables emit a canonical `*_full.tex` alongside the manuscript's
  column view, cross-checked by `_assert_paper_view_subset`; `ship_tables.py` copies
  them into `plots_final/` and re-asserts against the shipped copies.

**Run-chain fixes** (pre-existing job-script bugs, found while producing corrected runs)
- `job_real.slurm`: unset the inherited `SLURM_MEM_PER_CPU` before `srun`, which
  otherwise collides with this script's `--mem` and prevents `real.exe` from starting.
- `job_wrf_chain.slurm`: the wrfbdy gate tested existence only and would accept the
  output of an OOM-killed real job; it now requires the job's final sacct state to be
  `COMPLETED`.
- `job_WRF.slurm_9km`: `ntasks` 99 → 64, matching every archived 9 km run; at 99 tasks
  the 113×95 grid decomposes to ~10×10-cell patches, marginal against WRF's halo width.
- `namelist.input_9km`: `radt` was empty, set to 9 min.

**Diagnostics and utilities**
`verify_toptfix_channels.py` (regression check for the Tₒₚₜ fix — asserts control
channels unchanged, corrected channels moved, five percentile channels present,
non-zero, distinct and NaN-free), `diagnose_itlav.py`, `report_coarse_eval.py`,
`check_Y_sign_constancy.py`, `swint_delta_res_single_hour.py`,
`mk_bias_percentile_sensitivity.py`, `stage_zenodo_wrfout.py`, `zenodo_new_version.py`.

**Repo hygiene**
`.gitignore` reworked so the csv intermediates and SLURM scripts needed to reproduce
the figures stay tracked, instead of being swallowed by a blanket `**/*.csv`; logs,
caches and working notes stay ignored. The Glauch-parameter and Tₒₚₜ-acclimation
side-experiments are excluded (neither is referenced in the final manuscript).
READMEs and `.env.example` brought in line with the actual scripts.

## Why the offline route

The per-channel `Topt` in `chem/module_ghg_fluxes.F` was set with `DATA Topt /.../`
statements placed inside a runtime `SELECT CASE`. Fortran `DATA` is a compile-time
initializer and ignores runtime control flow, so ifort 18 silently took the first
in source order (`VPRM_table_US`, `/20,20,20,20,20,22,18/`) — every tracer channel
ran at Tₒₚₜ = 20/18 rather than the intended per-channel values. Confirmed both
empirically (offline GPP reproduces the online result at Tₒₚₜ = 20 to ~1 %) and with
a minimal ifort-18 reproducer; no compiler diagnostic was emitted.

The source fix (executable `Topt = (/ ... /)` assignments in both subroutines) needs a
WRF recompile and is **out of scope here**; this PR applies the intended Tₒₚₜ offline
via `recompute_vprm_fluxes.py` and documents the bug so future runs don't silently
inherit Tₒₚₜ = 20.

## Verification

- `verify_toptfix_channels.py` passes on the corrected output (tolerance- and
  NaN-aware; EBIO_GEE carries a few pre-existing NaN cells at 9/3 km where VPRM's
  Wscale divides by an LSWI_MAX fill value).
- Seasonal and full-year tables cross-check via `_assert_paper_view_subset`, and
  `ship_tables.py` re-asserts against the shipped copies.
- The GPP-based per-PFT Tₒₚₜ values were reproduced by a standalone recompute.
- The Tₒₚₜ NEE→GPP switch changes model skill only marginally (ΔNNSE ≤ ~0.3 %) while
  shifting PAR0/λ — equifinality, not a skill change; the driver-attribution
  conclusions hold.

## Open items (deliberately not in this PR)

- **CAMS is still present** in the extraction scripts and in Fig. 4 / Fig. 5–6. The
  revision plan calls for removing it; that removal has not been made in the code yet.
- The WRF source fix for the `DATA Topt` bug belongs on a dedicated branch with a
  recompile.
- 3 km and 1 km Tₒₚₜ-fix reruns are still outstanding, which is why the WRF_pre run
  configuration is only partially updated.
- ENF Tₒₚₜ is 16.42 in the `main_tune_VPRM.py` dict vs 16.36 from the appendix figure
  script (different filtering); one value should be chosen for consistency.

## Review notes

Large diff, but it is dominated by tracked csv and generated `.tex`/parameter tables.
The code worth reading is `recompute_vprm_fluxes.py`, the `build_*_param_csv*.py` pair,
`Fig7_AppxH6_H8_effects_seasonal.py`, and the three run-chain job-script fixes.
