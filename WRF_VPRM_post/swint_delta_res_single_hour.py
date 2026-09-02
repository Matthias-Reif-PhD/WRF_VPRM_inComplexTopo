"""
swint_delta_res_single_hour.py
==============================================
Hour-matched swint_opt=1 counterfactual for the Appendix G radiation figure.

Appendix G's radiation area maps (areafluxes_SWDOWN_*_2012-07-27_05h.pdf, built
by Fig9_AppxG7_areafluxes_per_timestep.py with wrf_tag_d01="_radt54_swint1") show
27 Jul 2012 05 UTC, clear sky. The only domain-averaged number available so far
for "what would omitting swint_opt=1 cost" was a cloudy 24-h mean, mismatched to
the clear-sky morning-hour figure it sits under. This script computes the same
Delta_res S-down number (54km - 1km, STD_TOPO>200 masked) for that exact hour, in
both the swint1 and no-swint1 54km configurations, reusing the reprojection/
masking pattern from Fig9_AppxG7_areafluxes_per_timestep.py.

If either 54km wrfout for the requested hour is missing on disk, this script
prints a plain statement and stops -- no substitution, no interpolation.
==============================================
"""

import os
import sys
import numpy as np
import netCDF4 as nc
from dotenv import load_dotenv
from pathlib import Path
from scipy.interpolate import griddata
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
SCRATCH_PATH = os.getenv("SCRATCH_PATH")

DATE_HOUR = "2012-07-27_05"
STD_TOPO = 200
INTERP_METHOD = "nearest"

WRFOUT_1KM = os.path.join(SCRATCH_PATH, f"DATA/WRFOUT/WRFOUT_ALPS_1km/wrfout_d02_{DATE_HOUR}:00:00")
WRFOUT_54KM_SWINT1 = os.path.join(
    SCRATCH_PATH, f"DATA/WRFOUT/WRFOUT_ALPS_54km_radt54_swint1/wrfout_d01_{DATE_HOUR}:00:00"
)
WRFOUT_54KM_NOSWINT = os.path.join(
    SCRATCH_PATH, f"DATA/WRFOUT/WRFOUT_ALPS_54km_radt54/wrfout_d01_{DATE_HOUR}:00:00"
)


def generate_coastal_mask(veg_type, buffer_km=30.0, grid_spacing_km=50.0):
    """Same as Fig9_AppxG7_areafluxes_per_timestep.py's helper -- 30 km buffer,
    50 km grid spacing (matched to its 54 km-grid call, since IVGTYP here also
    comes from the 54 km grid)."""
    landmask = np.ones_like(veg_type, dtype=np.uint8)
    landmask[veg_type == 44] = 0
    land_binary = landmask.astype(bool)
    eroded_land = binary_erosion(land_binary)
    coastline = land_binary & (~eroded_land)
    distance = distance_transform_edt(~coastline) * grid_spacing_km
    new_landmask = landmask.copy()
    new_landmask[distance <= buffer_km] = 0
    return new_landmask


def proj_on_finer_WRF_grid(lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_1km, method_in):
    proj_var = griddata(
        (lats_coarse.flatten(), lons_coarse.flatten()),
        var_coarse.flatten(),
        (lats_fine, lons_fine),
        method=method_in,
    ).reshape(WRF_var_1km.shape)
    return proj_var


def _masked_mean(field, mask):
    m = field.copy()
    m[mask == 0] = np.nan
    return float(np.nanmean(m))


def main():
    missing = [p for p in (WRFOUT_1KM, WRFOUT_54KM_SWINT1, WRFOUT_54KM_NOSWINT) if not os.path.exists(p)]
    if missing:
        print(f"MISSING WRFOUT for {DATE_HOUR} -- cannot compute the swint counterfactual for this hour.")
        for p in missing:
            print(f"  not found: {p}")
        print("Stopping without substitution or interpolation, as instructed.")
        sys.exit(1)

    nc_1km = nc.Dataset(WRFOUT_1KM, "r")
    nc_swint1 = nc.Dataset(WRFOUT_54KM_SWINT1, "r")
    nc_noswint = nc.Dataset(WRFOUT_54KM_NOSWINT, "r")

    SWDOWN_1km = nc_1km.variables["SWDOWN"][0, 10:-10, 10:-10]
    SWDOWN_54km_swint1 = nc_swint1.variables["SWDOWN"][0]
    SWDOWN_54km_noswint = nc_noswint.variables["SWDOWN"][0]

    lats_fine = nc_1km.variables["XLAT"][0, 10:-10, 10:-10]
    lons_fine = nc_1km.variables["XLONG"][0, 10:-10, 10:-10]
    stdh_topo_1km = nc_1km.variables["VAR"][0, 10:-10, 10:-10]
    stdh_mask = stdh_topo_1km >= STD_TOPO

    # veg_type/IVGTYP is static land cover -- identical between the swint1 and
    # no-swint1 runs, so either 54km file works; use swint1's, matching Fig6_8.
    veg_type = nc_swint1.variables["IVGTYP"][0, :, :]
    lats_54km = nc_swint1.variables["XLAT"][0, :, :]
    lons_54km = nc_swint1.variables["XLONG"][0, :, :]

    new_landmask = generate_coastal_mask(veg_type, buffer_km=30.0, grid_spacing_km=50.0)
    proj_landmask_54km = proj_on_finer_WRF_grid(
        lats_54km, lons_54km, new_landmask, lats_fine, lons_fine, SWDOWN_1km, INTERP_METHOD
    )
    combined_mask = proj_landmask_54km * stdh_mask
    n_unmasked = int(np.sum(combined_mask != 0))
    assert n_unmasked > 0, "STD_TOPO+coastal mask left zero unmasked pixels -- check inputs"

    proj_SWDOWN_54km_swint1 = proj_on_finer_WRF_grid(
        lats_54km, lons_54km, SWDOWN_54km_swint1, lats_fine, lons_fine, SWDOWN_1km, INTERP_METHOD
    )
    proj_SWDOWN_54km_noswint = proj_on_finer_WRF_grid(
        lats_54km, lons_54km, SWDOWN_54km_noswint, lats_fine, lons_fine, SWDOWN_1km, INTERP_METHOD
    )

    dSWDOWN_swint1 = np.asarray(proj_SWDOWN_54km_swint1) - np.asarray(SWDOWN_1km)
    dSWDOWN_noswint = np.asarray(proj_SWDOWN_54km_noswint) - np.asarray(SWDOWN_1km)

    print(f"unmasked pixel count: {n_unmasked}")
    print(f"raw (unmasked) domain means: 1km={np.nanmean(SWDOWN_1km):.2f}, "
          f"54km swint1={np.nanmean(proj_SWDOWN_54km_swint1):.2f}, "
          f"54km no-swint={np.nanmean(proj_SWDOWN_54km_noswint):.2f} W/m2")

    mean_1km = _masked_mean(np.asarray(SWDOWN_1km), combined_mask)
    mean_dres_swint1 = _masked_mean(dSWDOWN_swint1, combined_mask)
    mean_dres_noswint = _masked_mean(dSWDOWN_noswint, combined_mask)
    pct_swint1 = abs(mean_dres_swint1 / mean_1km * 100)
    pct_noswint = abs(mean_dres_noswint / mean_1km * 100)

    print(f"\nSTD_TOPO>{STD_TOPO} masked domain mean, {DATE_HOUR} UTC (clear sky):")
    print(f"  1km S-down                 = {mean_1km:.2f} W/m2")
    print(f"  Delta_res S-down (swint1)  = {mean_dres_swint1:.2f} W/m2 ({pct_swint1:.0f}% of 1km)")
    print(f"  Delta_res S-down (no swint)= {mean_dres_noswint:.2f} W/m2 ({pct_noswint:.0f}% of 1km)")
    print(f"\nSentence form: {mean_dres_swint1:.2f} vs {mean_dres_noswint:.2f} W/m2, "
          f"{pct_swint1:.0f}% vs {pct_noswint:.0f}%")


if __name__ == "__main__":
    main()
