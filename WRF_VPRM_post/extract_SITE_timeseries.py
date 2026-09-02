"""
extract_SITE_timeseries.py
==============================================
Extract SITE timeseries of WRF-VPRM output variables over different WRF domains
for specified date ranges and simulation types and save to CSV files for further analysis.
==============================================
"""

import os
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from scipy.interpolate import griddata
import argparse
import sys
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from collections import defaultdict
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Reuse the offline VPRM-old flux engine for the per-site (point) recompute.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute_vprm_fluxes import compute_fluxes, load_params, _load_vprm_input

# ==================== Configuration ====================
SCRATCH_PATH = os.getenv("SCRATCH_PATH")
GITHUB_PATH = os.getenv("GITHUB_PATH")
OUTFOLDER = os.getenv("OUTFOLDER")
CSVFOLDER = os.getenv("CSVFOLDER")

# GPP-based per-site full parameter set (GMD revision).
# Loaded from vprm_params_site.csv (built by build_site_param_csv.py after
# main_tune_VPRM.py with diff_evo_V24_SITE).  Keyed by "{site}_SITE" to match
# location["name"].  Each value is a dict: t_opt, par0, lambda, alpha, beta.
def _load_site_params():
    csv = Path(__file__).resolve().parent / "vprm_params_site.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"vprm_params_site.csv not found at {csv}. "
            "Run build_site_param_csv.py after the SITE tuning jobs complete."
        )
    df = pd.read_csv(csv)
    result = {}
    for _, row in df.iterrows():
        result[f"{row['site']}_SITE"] = {
            "t_opt":   float(row["t_opt"]),
            "par0":    float(row["par0"]),
            "lambda":  float(row["lambda"]),
            "alpha":   float(row["alpha"]),
            "beta":    float(row["beta"]),
        }
    return result

SITE_PARAMS = _load_site_params()


def site_point_flux(base_params, pft, site_params, swdown_cell, t2c_cell, vin_cell):
    """Point recompute of VPRM-old GPP/RECO/NEE at a single grid cell using the
    base (tag) parameter table with this site's full per-site params substituted
    for its PFT row (t_opt, par0, lambda, alpha, beta).
    site_params: dict with those five scalar keys (from SITE_PARAMS).
    Returns scalars in umol m-2 s-1."""
    m = pft - 1  # vprm_veg_id 1..7 -> index 0..6
    p = dict(base_params)
    for key in ("t_opt", "par0", "lambda", "alpha", "beta"):
        arr = base_params[key].copy()
        arr[m] = site_params[key]
        p[key] = arr
    gpp, reco, nee = compute_fluxes(swdown_cell, t2c_cell, vin_cell, p)
    return float(gpp.ravel()[0]), float(reco.ravel()[0]), float(nee.ravel()[0])


def compute_slope_aspect(hgt, lats, lons):
    lat_rad = np.radians(lats)
    dy = 111000  # meters per degree latitude
    dx = 111000 * np.cos(lat_rad)  # meters per degree longitude

    dz_dy = np.gradient(hgt, axis=0) / dy
    dz_dx = np.gradient(hgt, axis=1) / dx

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope = np.degrees(slope_rad)

    aspect = (np.degrees(np.arctan2(dz_dy, -dz_dx)) + 360) % 360
    return slope, aspect


def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def find_best_fluxnet_match(
    lat_target,
    lon_target,
    lats,
    lons,
    location_pft,
    veg_frac_map,
    hgt,
    hgt_site,
    radius,
    slope=None,
    aspect=None,
    nearest_only=False,
):
    """Pick the grid cell representing a FLUXNET tower.

    Normally this minimises a terrain/vegetation cost function over all cells
    within `radius`. With `nearest_only`, it returns the minimum-distance cell
    instead and the cost function is not applied.

    `nearest_only` exists for the 54 km domain, where the cost function has no
    room to act: at dx = 54 km a 30 km radius contains 0-2 cell centres (AT-Neu
    has none -- its nearest centre is 32.9 km away, which would raise below).
    Widening the radius until the cost function nominally "runs" would select
    cells up to 80 km from the tower and dress an arbitrary choice as a terrain
    match, so the honest option is to take the nearest cell and say so.
    Returns `n_cand`, the number of cells the rule actually chose between, so
    the caller can report how much of a choice it was.
    """
    if slope is None or aspect is None:
        slope, aspect = compute_slope_aspect(hgt, lats, lons)

    # Site reference
    flat_idx = np.abs(lats - lat_target) + np.abs(lons - lon_target)
    lat_idx, lon_idx = np.unravel_index(np.argmin(flat_idx), lats.shape)
    target_slope = slope[lat_idx, lon_idx]
    target_aspect = aspect[lat_idx, lon_idx]

    # Distance constraint
    dist_km = haversine_dist(lat_target, lon_target, lats, lons)
    dist_mask = dist_km <= abs(radius)

    # Terrain differences
    height_diff = np.abs(hgt - hgt_site)
    slope_diff = np.abs(slope - target_slope)
    aspect_diff = np.abs((aspect - target_aspect + 180) % 360 - 180)

    # Vegetation fraction (axis 1 = PFT)
    veg_frac = veg_frac_map[0, location_pft - 1, :, :]  # shape (ny, nx)

    # Define weighted cost function
    cost = (
        0.5 * (height_diff / np.nanmax(height_diff))
        + 0.1 * (slope_diff / 90.0)
        + 0.1 * (aspect_diff / 180.0)
        + 0.6 * (1.0 - veg_frac) ** 2
    )

    n_cand = int(dist_mask.sum())

    if nearest_only:
        # No radius mask and no cost minimisation: the nearest cell, full stop.
        min_idx = np.unravel_index(np.argmin(dist_km), dist_km.shape)
        n_cand = 1
    else:
        # Apply radius mask
        cost = np.where(dist_mask, cost, np.inf)

        if not np.any(np.isfinite(cost)):
            raise ValueError("No valid grid cell within radius.")

        min_idx = np.unravel_index(np.argmin(cost), cost.shape)

    min_dist = dist_km[min_idx]

    return (
        min_dist,
        min_idx,
        target_slope,
        slope_diff[min_idx],
        target_aspect,
        aspect_diff[min_idx],
        veg_frac[min_idx],
        height_diff[min_idx],
        cost[min_idx],
        n_cand,
    )


def find_nearest_grid_hgt_sa(
    lat_target, lon_target, lats, lons, location_pft, IVGTYP_vprm, hgt, hgt_site, radius
):
    slope, aspect = compute_slope_aspect(hgt, lats, lons)

    flat_idx = np.abs(lats - lat_target) + np.abs(lons - lon_target)
    min_flat_idx = np.argmin(flat_idx)
    lat_idx, lon_idx = np.unravel_index(min_flat_idx, lats.shape)
    target_slope = slope[lat_idx, lon_idx]
    target_aspect = aspect[lat_idx, lon_idx]

    pft_mask = IVGTYP_vprm == location_pft
    dist_km = haversine_dist(lat_target, lon_target, lats, lons)
    dist_mask = dist_km <= abs(radius)

    height_diff = np.abs(hgt - hgt_site)
    slope_diff = np.abs(slope - target_slope)
    aspect_diff = np.abs((aspect - target_aspect + 180) % 360 - 180)

    aspect_mask = aspect_diff <= 20
    slope_mask = slope_diff <= 10

    combined_mask = pft_mask & dist_mask & aspect_mask & slope_mask

    if not np.any(combined_mask):
        relaxed_mask = pft_mask & dist_mask
        if np.any(relaxed_mask):
            masked_height_diff = np.where(relaxed_mask, height_diff, np.inf)
            min_idx = np.unravel_index(np.argmin(masked_height_diff), hgt.shape)
            min_dist = dist_km[min_idx]
            return min_dist, min_idx, target_slope, target_aspect
        else:
            fallback_flat_idx = np.argmin(dist_km)
            fallback_idx = np.unravel_index(fallback_flat_idx, lats.shape)
            fallback_dist = dist_km[fallback_idx]
            return fallback_dist, fallback_idx, target_slope, target_aspect

    masked_height_diff = np.where(combined_mask, height_diff, np.inf)
    min_idx = np.unravel_index(np.argmin(masked_height_diff), hgt.shape)
    min_dist = dist_km[min_idx]

    return min_dist, min_idx, target_slope, target_aspect


def extract_datetime_from_filename(filename):
    """
    Extract datetime from WRF filename assuming format 'wrfout_d0x_YYYY-MM-DD_HH:MM:SS'.
    """
    base_filename = os.path.basename(filename)
    date_str = base_filename.split("_")[-2] + "_" + base_filename.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")


def recalc_file_path(tag, res, sim_type, time):
    """
    Deterministic path of an offline-recomputed VPRM flux NetCDF for a given
    parameter tag, resolution, sim_type and timestamp (see recompute_vprm_fluxes.py).
    Returns None if no tag is requested.
    """
    if tag is None:
        return None
    base = os.path.join(
        SCRATCH_PATH, "DATA/VPRM_recalc", f"vprm_recalc_{tag}_{res}{sim_type}"
    )
    ts = time.strftime("%Y-%m-%d_%H:%M:%S")
    return os.path.join(base, f"vprm_recalc_{tag}_{res}{sim_type}_{ts}.nc")


def extract_timeseries(
    wrf_path, start_date, end_date, res, sim_type, radius, tag=None
):

    # Grid-point rule. At 1 km and 9 km the terrain/vegetation cost function
    # has a real pool to choose from within `radius` (~2800 and ~35 cells
    # respectively). At 54 km it does not -- 0-2 cell centres lie within 30 km,
    # and AT-Neu has none -- so the nearest cell is taken and the cost function
    # is not applied. See find_best_fluxnet_match's docstring.
    nearest_only = res == "54km"

    run_Pmodel = False
    subday = ""
    if run_Pmodel:
        gpp_pmodel_path = os.path.join(
            SCRATCH_PATH, "DATA/MODIS/MODIS_FPAR/gpp_pmodel/"
        )
        migli_path = os.path.join(SCRATCH_PATH, "DATA/RECO_Migli")
        subday = "subdailyC3_"

    # When a recompute tag is given, add offline-recomputed GPP/RECO/NEE columns
    # for the ALPS channel (CO2_ID == "") alongside the WRF EBIO_* columns.
    use_recomputed = tag is not None
    # the recompute 1km grid is pre-trimmed [10:-10]; grid_idx is on the full grid
    recalc_offset = 10 if res == "1km" else 0

    output_dir = CSVFOLDER
    d0X = "wrfout_d01"
    if res == "1km":
        d0X = "wrfout_d02"
        vprm_input_path_1km = os.path.join(
            SCRATCH_PATH,
            "DATA/VPRM_input/vprm_corine_1km/vprm_input_d02_2012-06-23_00:00:00.nc",
        )
        ds = xr.open_dataset(vprm_input_path_1km)
        # ['Times', 'XLONG', 'XLAT', 'EVI_MIN', 'EVI_MAX', 'EVI', 'LSWI_MIN', 'LSWI_MAX', 'LSWI', 'VEGFRA_VPRM']
        veg_frac_map = ds["VEGFRA_VPRM"].values
        veg_frac_map = np.nan_to_num(veg_frac_map, nan=0.0)
    else:
        vprm_input_path = os.path.join(
            SCRATCH_PATH,
            f"DATA/VPRM_input/vprm_corine_{res}/vprm_input_d01_2012-06-23_00:00:00.nc",
        )
        ds = xr.open_dataset(vprm_input_path)
        # ['Times', 'XLONG', 'XLAT', 'EVI_MIN', 'EVI_MAX', 'EVI', 'LSWI_MIN', 'LSWI_MAX', 'LSWI', 'VEGFRA_VPRM']
        veg_frac_map = ds["VEGFRA_VPRM"].values
        veg_frac_map = np.nan_to_num(veg_frac_map, nan=0.0)

    # Convert to datetime (but ignore time part for full-day selection)
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").date()
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").date()

    # Collect all files
    all_files = sorted(glob.glob(os.path.join(wrf_path, f"{d0X}_*")))
    file_by_day = defaultdict(list)

    for f in all_files:
        dt = extract_datetime_from_filename(f)
        day = dt.date()
        if start_date_obj <= day <= end_date_obj:
            file_by_day[day].append((dt, f))

    # Filter for full days (24 hourly files starting from 00:00 to 23:00)
    file_list = []
    for day in sorted(file_by_day.keys()):
        files = sorted(file_by_day[day])
        if len(files) == 24 and all(dt.hour == i for i, (dt, _) in enumerate(files)):
            file_list.extend(f for _, f in files)

    # Define target locations (latitude, longitude)

    locations = [
        {
            "name": "CH-Dav_REF",
            "CO2_ID": "_REF",
            "lat": 46.8153,
            "lon": 9.8559,
            "pft": 1,  # "ENF",
            "hgt_site": 1639,
        },
        {
            "name": "IT-Lav_REF",
            "CO2_ID": "_REF",
            "lat": 45.9562,
            "lon": 11.2813,
            "pft": 1,  # "ENF",
            "hgt_site": 1353,
        },
        {
            "name": "IT-Ren_REF",
            "CO2_ID": "_REF",
            "lat": 46.5869,
            "lon": 11.4337,
            "pft": 1,  # "ENF",
            "hgt_site": 1730,
        },
        {
            "name": "AT-Neu_REF",
            "CO2_ID": "_REF",
            "lat": 47.1167,
            "lon": 11.3175,
            "pft": 7,  # "GRA",
            "hgt_site": 970,
        },
        {
            "name": "IT-MBo_REF",
            "CO2_ID": "_REF",
            "lat": 46.0147,
            "lon": 11.0458,
            "pft": 7,  # "GRA",
            "hgt_site": 1550,
        },
        {
            "name": "CH-Dav_ALPS",
            "CO2_ID": "",
            "lat": 46.8153,
            "lon": 9.8559,
            "pft": 1,  # "ENF",
            "hgt_site": 1639,
        },
        {
            "name": "IT-Lav_ALPS",
            "CO2_ID": "",
            "lat": 45.9562,
            "lon": 11.2813,
            "pft": 1,  # "ENF",
            "hgt_site": 1353,
        },
        {
            "name": "IT-Ren_ALPS",
            "CO2_ID": "",
            "lat": 46.5869,
            "lon": 11.4337,
            "pft": 1,  # "ENF",
            "hgt_site": 1730,
        },
        {
            "name": "AT-Neu_ALPS",
            "CO2_ID": "",
            "lat": 47.1167,
            "lon": 11.3175,
            "pft": 7,  # "GRA",
            "hgt_site": 970,
        },
        {
            "name": "IT-MBo_ALPS",
            "CO2_ID": "",
            "lat": 46.0147,
            "lon": 11.0458,
            "pft": 7,  # "GRA",
            "hgt_site": 1550,
        },
        {
            "name": "CH-Dav_SITE",
            "CO2_ID": "_2",
            "lat": 46.8153,
            "lon": 9.8559,
            "pft": 1,  # "ENF",
            "hgt_site": 1639,
        },
        {
            "name": "IT-Lav_SITE",
            "CO2_ID": "_4",
            "lat": 45.9562,
            "lon": 11.2813,
            "pft": 1,  # "ENF",
            "hgt_site": 1353,
        },
        {
            "name": "IT-Ren_SITE",
            "CO2_ID": "_3",
            "lat": 46.5869,
            "lon": 11.4337,
            "pft": 1,  # "ENF",
            "hgt_site": 1730,
        },
        {
            "name": "AT-Neu_SITE",
            "CO2_ID": "_4",
            "lat": 47.1167,
            "lon": 11.3175,
            "pft": 7,  # "GRA",
            "hgt_site": 970,
        },
        {
            "name": "IT-MBo_SITE",
            "CO2_ID": "_3",
            "lat": 46.0147,
            "lon": 11.0458,
            "pft": 7,  # "GRA",
            "hgt_site": 1550,
        },
    ]

    locations_d01 = [
        {
            "name": "CH-Cha_REF",
            "CO2_ID": "_REF",
            "lat": 47.2102,
            "lon": 8.4104,
            "pft": 7,  # GRA
            "hgt_site": 393,
        },
        {
            "name": "CH-Cha_ALPS",
            "CO2_ID": "",
            "lat": 47.2102,
            "lon": 8.4104,
            "pft": 7,
            "hgt_site": 393,
        },
        {
            "name": "CH-Fru_REF",
            "CO2_ID": "_REF",
            "lat": 47.1158,
            "lon": 8.5378,
            "pft": 7,  # GRA
            "hgt_site": 982,
        },
        {
            "name": "CH-Fru_ALPS",
            "CO2_ID": "",
            "lat": 47.1158,
            "lon": 8.5378,
            "pft": 7,
            "hgt_site": 982,
        },
        {
            "name": "CH-Oe1_REF",
            "CO2_ID": "_REF",
            "lat": 47.2858,
            "lon": 7.7319,
            "pft": 7,  # GRA
            "hgt_site": 450,
        },
        {
            "name": "CH-Oe1_ALPS",
            "CO2_ID": "",
            "lat": 47.2858,
            "lon": 7.7319,
            "pft": 7,
            "hgt_site": 450,
        },
        {
            "name": "DE-Lkb_REF",
            "CO2_ID": "_REF",
            "lat": 49.0996,
            "lon": 13.3047,
            "pft": 1,  # ENF
            "hgt_site": 1308,
        },
        {
            "name": "DE-Lkb_ALPS",
            "CO2_ID": "",
            "lat": 49.0996,
            "lon": 13.3047,
            "pft": 1,
            "hgt_site": 1308,
        },
        {
            "name": "IT-Isp_REF",
            "CO2_ID": "_REF",
            "lat": 45.8126,
            "lon": 8.6336,
            "pft": 4,  # DBF
            "hgt_site": 210,
        },
        {
            "name": "IT-Isp_ALPS",
            "CO2_ID": "",
            "lat": 45.8126,
            "lon": 8.6336,
            "pft": 4,
            "hgt_site": 210,
        },
        {
            "name": "IT-La2_REF",
            "CO2_ID": "_REF",
            "lat": 45.9542,
            "lon": 11.2853,
            "pft": 1,  # ENF
            "hgt_site": 1350,
        },
        {
            "name": "IT-La2_ALPS",
            "CO2_ID": "",
            "lat": 45.9542,
            "lon": 11.2853,
            "pft": 1,
            "hgt_site": 1350,
        },
        {
            "name": "IT-PT1_REF",
            "CO2_ID": "_REF",
            "lat": 45.2009,
            "lon": 9.061,
            "pft": 4,  # DBF
            "hgt_site": 60,
        },
        {
            "name": "IT-PT1_ALPS",
            "CO2_ID": "",
            "lat": 45.2009,
            "lon": 9.061,
            "pft": 4,
            "hgt_site": 60,
        },
        {
            "name": "CH-Oe2_REF",
            "CO2_ID": "_REF",
            "lat": 47.2863,
            "lon": 7.7343,
            "pft": 6,  # "CRO",
            "hgt_site": 452,
        },
        {
            "name": "CH-Oe2_ALPS",
            "CO2_ID": "",
            "lat": 47.2863,
            "lon": 7.7343,
            "pft": 6,  # "CRO",
            "hgt_site": 452,
        },
        {
            "name": "CH-Oe2_SITE",
            "CO2_ID": "_2",
            "lat": 47.2863,
            "lon": 7.7343,
            "pft": 6,  # "CRO",
            "hgt_site": 452,
        },
        {
            "name": "IT-Tor_REF",
            "CO2_ID": "_REF",
            "lat": 45.8444,
            "lon": 7.5781,
            "pft": 7,  # "GRA",
            "hgt_site": 2160,
        },
        {
            "name": "IT-Tor_ALPS",
            "CO2_ID": "",
            "lat": 45.8444,
            "lon": 7.5781,
            "pft": 7,  # "GRA",
            "hgt_site": 2160,
        },
        {
            "name": "IT-Tor_SITE",
            "CO2_ID": "_2",
            "lat": 45.8444,
            "lon": 7.5781,
            "pft": 7,  # "GRA",
            "hgt_site": 2160,
        },
        {
            "name": "CH-Lae_REF",
            "CO2_ID": "_REF",
            "lat": 47.4781,
            "lon": 8.3644,
            "pft": 3,  # "MF",
            "hgt_site": 689,
        },
        {
            "name": "CH-Lae_ALPS",
            "CO2_ID": "",
            "lat": 47.4781,
            "lon": 8.3644,
            "pft": 3,  # "MF",
            "hgt_site": 689,
        },
        {
            "name": "CH-Lae_SITE",
            "CO2_ID": "_2",
            "lat": 47.4781,
            "lon": 8.3644,
            "pft": 3,  # "MF",
            "hgt_site": 689,
        },
    ]
    if res != "1km":
        locations = locations_d01 + locations

    # Define the remapping dictionary for CORINE vegetation types
    corine_to_vprm = {
        24: 1,  # Coniferous Forest (Evergreen)
        23: 2,  # Broad-leaved Forest (Deciduous)
        25: 3,
        29: 3,  # Mixed Forest and Transitional Woodland-Shrub
        27: 4,
        28: 4,  # Moors and Heathland, Sclerophyllous Vegetation (Shrubland)
        35: 5,
        36: 5,
        37: 5,  # Wetlands: Inland Marshes, Peat Bogs, Salt Marshes
        12: 6,
        13: 6,
        14: 6,
        15: 6,
        16: 6,
        17: 6,
        19: 6,
        20: 6,
        21: 6,
        22: 6,  # Cropland
        18: 7,
        26: 7,  # Grassland: Pastures, Natural Grasslands
        # Others mapped to 8 (gray)
        1: 8,
        2: 8,
        3: 8,
        4: 8,
        5: 8,
        6: 8,
        7: 8,
        8: 8,
        9: 8,
        10: 8,
        11: 8,
        30: 8,
        31: 8,
        32: 8,
        33: 8,
        34: 8,
        38: 8,
        39: 8,
        40: 8,
        41: 8,
        42: 8,
        43: 8,
        44: 8,
    }

    # Initialize an empty DataFrame with time as the index and locations as columns
    columns = (
        [f"{location['name']}_GPP_WRF" for location in locations]
        + [f"{location['name']}_RECO_WRF" for location in locations]
        + [f"{location['name']}_T2_WRF" for location in locations]
    )
    if run_Pmodel:
        columns = (
            [f"{location['name']}_GPP_WRF" for location in locations]
            + [f"{location['name']}_RECO_WRF" for location in locations]
            + [f"{location['name']}_T2_WRF" for location in locations]
            + [f"{location['name']}_GPP_Pmodel" for location in locations]
            + [f"{location['name']}_RECO_Migli" for location in locations]
        )

    # Offline-recompute columns: ALPS channel (domain-uniform tag, read from the
    # vprm_recalc NetCDF) and SITE channel (full per-site params, point recompute here).
    recalc_locations = [
        loc
        for loc in locations
        if loc["CO2_ID"] == "" or loc["name"] in SITE_PARAMS
    ]
    if use_recomputed:
        columns = (
            columns
            + [f"{loc['name']}_GPP_recalc" for loc in recalc_locations]
            + [f"{loc['name']}_RECO_recalc" for loc in recalc_locations]
            + [f"{loc['name']}_NEE_recalc" for loc in recalc_locations]
        )

    # Base parameter table for the SITE point recompute = the same CSV as --tag,
    # with each site's own Topt substituted for its PFT (see site_point_flux).
    base_params = None
    vin_by_day = {}  # cache of per-day full-grid VPRM input
    if use_recomputed:
        tag_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"vprm_params_{tag}.csv"
        )
        base_params = load_params(tag_csv)

    df_out = pd.DataFrame(columns=columns)
    # Process each WRF file (representing one timestep)
    for nc_f1 in file_list:
        nc_fid1 = nc.Dataset(nc_f1, "r")
        wrf_file = nc_f1.split("/")[-1]
        date_time = wrf_file.split("_")[2] + "_" + wrf_file.split("_")[3]
        file_end = res + "_" + date_time
        print(f"Processing {file_end}")
        xlat = nc_fid1.variables["XLAT"][0]  # Assuming the first time slice
        xlon = nc_fid1.variables["XLONG"][0]
        WRF_T2 = nc_fid1.variables["T2"][0]
        WRF_SWDOWN = nc_fid1.variables["SWDOWN"][0]  # for SITE point recompute
        hgt = nc_fid1.variables["HGT"][0]
        IVGTYP = nc_fid1.variables["IVGTYP"][0]
        IVGTYP_vprm = np.vectorize(corine_to_vprm.get)(
            IVGTYP[:, :]
        )  # Create a new array for the simplified vegetation categories
        # dx = (xlat[1, 0] - xlat[0, 0]) * 111

        if run_Pmodel:
            # find file in migli_path which ends with date_time
            reco_migli_files = [
                f
                for f in sorted(glob.glob(os.path.join(migli_path, "reco_migli*")))
                if file_end in f
            ]
            if not reco_migli_files:
                raise FileNotFoundError(f"No reco_migli file found for {file_end}")
            reco_migli_file = reco_migli_files[0]
            reco_migli = xr.open_dataset(reco_migli_file)
            reco_migli = reco_migli["RECO_Migli"].values
            # find file in gpp which ends with date_time
            gpp_pmodel_file = [
                f
                for f in sorted(
                    glob.glob(os.path.join(gpp_pmodel_path, f"gpp_pmodel_{subday}*"))
                )
                if file_end in f
            ][0]

            gpp_pmodel = xr.open_dataset(gpp_pmodel_file)
            gpp_pmodel = gpp_pmodel["GPP_Pmodel"].values

        # Offline-recomputed fluxes for this timestep (already in umol m-2 s-1).
        gpp_rc = reco_rc = nee_rc = None
        vin_day = None
        if use_recomputed:
            # ALPS channel: domain-uniform recompute NetCDF (pre-trimmed at 1km).
            rc_path = recalc_file_path(
                tag, res, sim_type, extract_datetime_from_filename(nc_f1)
            )
            if os.path.exists(rc_path):
                with nc.Dataset(rc_path) as f_rc:
                    gpp_rc = f_rc.variables["GPP"][:, :]
                    reco_rc = f_rc.variables["RECO"][:, :]
                    nee_rc = f_rc.variables["NEE"][:, :]
            # SITE channel: per-day full-grid VPRM input for the point recompute.
            day = extract_datetime_from_filename(nc_f1).strftime("%Y-%m-%d")
            if day not in vin_by_day:
                if res == "1km":
                    vpath = os.path.join(
                        SCRATCH_PATH,
                        f"DATA/VPRM_input/vprm_corine_1km/vprm_input_d02_{day}_00:00:00.nc",
                    )
                else:
                    vpath = os.path.join(
                        SCRATCH_PATH,
                        f"DATA/VPRM_input/vprm_corine_{res}/vprm_input_d01_{day}_00:00:00.nc",
                    )
                vin_by_day[day] = (
                    _load_vprm_input(vpath, trim=False)
                    if os.path.exists(vpath)
                    else None
                )
            vin_day = vin_by_day[day]

        # Initialize lists to store data for the current timestep
        data_row = {col: None for col in df_out.columns}  # Map columns to values

        # Extract data for each location
        for location in locations:
            lat_target, lon_target = location["lat"], location["lon"]
            WRF_gee = nc_fid1.variables[f"EBIO_GEE{location['CO2_ID']}"][0, 0, :, :]
            WRF_res = nc_fid1.variables[f"EBIO_RES{location['CO2_ID']}"][0, 0, :, :]

            # TODO: just calculate once...
            (
                dist_km,
                grid_idx,
                target_slope,
                target_slope_diff,
                target_aspect,
                target_aspect_diff,
                veg_frac_idx,
                height_diff_idx,
                cost,
                n_cand,
            ) = find_best_fluxnet_match(
                lat_target,
                lon_target,
                xlat,
                xlon,
                location["pft"],
                veg_frac_map,
                hgt,
                location["hgt_site"],
                radius,
                nearest_only=nearest_only,
            )
            print(
                f"Cost: \n  [{location['name']}] "
                f"Rule={'nearest' if nearest_only else 'cost'} | "
                f"Candidates={n_cand} | "
                f"Dist={dist_km:.2f} km | "
                f"Height Diff={height_diff_idx:.2f} m"
                f"Idx={grid_idx} | "
                f"Slope={target_slope:.1f}° | "
                f"Slope diff={target_slope_diff:.1f}° | "
                f"Aspect={target_aspect:.1f}° | "
                f"Aspect diff={target_aspect_diff:.1f}° | "
                f"VegFrac={veg_frac_idx:.2f} | "
                f"Cost={cost:.3f}"
            )

            # add dist to the large locations dict which contains all the locations
            for loc in locations:
                if loc["name"] == location["name"]:

                    loc["dist"] = dist_km
                    loc["veg_frac_idx"] = veg_frac_idx
                    loc["hgt_wrf"] = hgt[grid_idx[0], grid_idx[1]]
                    loc["lat_wrf"] = xlat[grid_idx[0], grid_idx[1]]
                    loc["lon_wrf"] = xlon[grid_idx[0], grid_idx[1]]
                    # Which cell was picked, by which rule, out of how many --
                    # the coarse-domain evaluation is only interpretable with
                    # this alongside it.
                    loc["grid_i"] = int(grid_idx[0])
                    loc["grid_j"] = int(grid_idx[1])
                    loc["n_candidates"] = n_cand
                    loc["rule"] = "nearest" if nearest_only else "cost"
                    break

            # Assign values to their respective columns
            data_row[f"{location['name']}_GPP_WRF"] = (
                WRF_gee[grid_idx[0], grid_idx[1]] / 3600
            )
            data_row[f"{location['name']}_RECO_WRF"] = (
                WRF_res[grid_idx[0], grid_idx[1]] / 3600
            )
            data_row[f"{location['name']}_T2_WRF"] = WRF_T2[grid_idx[0], grid_idx[1]]
            if run_Pmodel:
                data_row[f"{location['name']}_GPP_Pmodel"] = gpp_pmodel[
                    grid_idx[0], grid_idx[1]
                ]
                data_row[f"{location['name']}_RECO_Migli"] = reco_migli[
                    grid_idx[0], grid_idx[1]
                ]
            # ALPS channel: take offline-recomputed flux at the matched cell.
            # grid_idx is on the full grid; recompute 1km is trimmed [10:-10].
            if use_recomputed and location["CO2_ID"] == "" and gpp_rc is not None:
                ri = grid_idx[0] - recalc_offset
                rj = grid_idx[1] - recalc_offset
                if 0 <= ri < gpp_rc.shape[0] and 0 <= rj < gpp_rc.shape[1]:
                    data_row[f"{location['name']}_GPP_recalc"] = gpp_rc[ri, rj]
                    data_row[f"{location['name']}_RECO_recalc"] = reco_rc[ri, rj]
                    data_row[f"{location['name']}_NEE_recalc"] = nee_rc[ri, rj]
            # SITE channel: per-site Topt point recompute at the matched cell.
            # Built from full-grid inputs at grid_idx (no trim offset).
            elif (
                use_recomputed
                and location["name"] in SITE_PARAMS
                and vin_day is not None
            ):
                i0, j0 = grid_idx[0], grid_idx[1]
                vin_cell = {
                    k: v[:, :, i0 : i0 + 1, j0 : j0 + 1] for k, v in vin_day.items()
                }
                swd_cell = np.asarray(
                    WRF_SWDOWN[i0 : i0 + 1, j0 : j0 + 1], dtype=np.float64
                )
                t2c_cell = (
                    np.asarray(WRF_T2[i0 : i0 + 1, j0 : j0 + 1], dtype=np.float64)
                    - 273.15
                )
                g, r, n = site_point_flux(
                    base_params,
                    location["pft"],
                    SITE_PARAMS[location["name"]],
                    swd_cell,
                    t2c_cell,
                    vin_cell,
                )
                data_row[f"{location['name']}_GPP_recalc"] = g
                data_row[f"{location['name']}_RECO_recalc"] = r
                data_row[f"{location['name']}_NEE_recalc"] = n

        # Append the current timestep data as a new row in the DataFrame
        temp_df_out = pd.DataFrame([data_row])
        # Filter out empty or all-NA DataFrames before concatenation to avoid FutureWarning
        frames_to_concat = [
            df
            for df in [df_out, temp_df_out]
            if not df.empty and not df.isna().all(axis=None)
        ]
        if frames_to_concat:
            df_out = pd.concat(frames_to_concat, ignore_index=True)
        nc_fid1.close()

    # Set the time as the index of the DataFrame
    df_out.index = [extract_datetime_from_filename(f) for f in file_list]
    # Optionally, save the DataFrame to CSV
    recalc_suffix = f"_recalc_{tag}" if use_recomputed else ""
    output_filename = f"wrf_FLUXNET_sites_{res}{sim_type}{recalc_suffix}_{start_date.split('_')[0]}_{end_date.split('_')[0]}_r{radius}.csv"

    df_out.to_csv(
        os.path.join(
            output_dir,
            output_filename,
        )
    )
    # write another csv file with the distances but use only the _REF sites
    dist_rows = []
    for loc in locations:
        if "ALPS" in loc["name"]:
            dist_rows.append(
                {
                    "name": loc["name"],
                    "rule": loc["rule"],
                    "n_candidates": loc["n_candidates"],
                    "dist": loc["dist"],
                    "grid_i": loc["grid_i"],
                    "grid_j": loc["grid_j"],
                    "hgt_site": loc["hgt_site"],
                    "hgt_wrf": loc["hgt_wrf"],
                    "lat_wrf": loc["lat_wrf"],
                    "lon_wrf": loc["lon_wrf"],
                    "pft": loc["pft"],
                    "veg_frac_idx": loc["veg_frac_idx"],
                }
            )
    df_out_dist = pd.DataFrame(
        dist_rows,
        columns=[
            "name",
            "rule",
            "n_candidates",
            "dist",
            "grid_i",
            "grid_j",
            "hgt_site",
            "hgt_wrf",
            "lat_wrf",
            "lon_wrf",
            "pft",
            "veg_frac_idx",
        ],
    )
    df_out_dist.to_csv(
        os.path.join(
            output_dir,
            f"distances_{res}{sim_type}_{start_date.split('_')[0]}_{end_date.split('_')[0]}_r{radius}.csv",
        )
    )

    return


def main():

    if len(sys.argv) > 1:  # to run on cluster
        parser = argparse.ArgumentParser(description="Description of your script")
        parser.add_argument(
            "-s", "--start", type=str, help="Format: 2012-01-01 01:00:00"
        )
        parser.add_argument("-e", "--end", type=str, help="Format: 2012-12-31 00:00:00")
        parser.add_argument(
            "-t",
            "--type",
            type=str,
            help="Format: '' or '_cloudy'",
            default="",
        )
        parser.add_argument(
            "--tag",
            type=str,
            default=None,
            help="Recompute parameter tag (e.g. topt_p50). If set, adds offline "
            "VPRM_recalc GPP/RECO/NEE columns for the ALPS channel.",
        )
        parser.add_argument(
            "--res",
            type=str,
            default="1km",
            help="Comma-separated resolutions to extract, e.g. '1km' or "
            "'9km,54km'. Coarse domains read wrfout_d01 and evaluate the "
            "coarse runs against the towers (Follow-up 15e).",
        )
        args = parser.parse_args()
        start_date = args.start
        end_date = args.end
        sim_type = args.type
        tag = args.tag
        resolutions = [r.strip() for r in args.res.split(",") if r.strip()]
    else:  # to run locally
        start_date = "2012-01-01 00:00:00"
        end_date = "2012-12-31 00:00:00"
        sim_type = "_cloudy"  # "" or "_cloudy" - run one after the other
        tag = None  # e.g. "topt_p50" to add offline-recomputed flux columns
        resolutions = ["1km"]

    wrf_paths = [
        f"{SCRATCH_PATH}/DATA/WRFOUT/WRFOUT_ALPS_{r}" for r in resolutions
    ]
    radius = 30  # radius in which the best fitting locaiton is searched

    for wrf_path in wrf_paths:
        res = wrf_path.split("_")[-1]
        wrf_path = wrf_path + sim_type
        print((wrf_path, start_date, end_date, res, sim_type))
        # single pass with the actual sim_type so the wrfout data, the recompute
        # NetCDF lookup, and the output filename all stay consistent.
        extract_timeseries(
            wrf_path, start_date, end_date, res, sim_type, radius, tag=tag
        )


if __name__ == "__main__":
    main()
