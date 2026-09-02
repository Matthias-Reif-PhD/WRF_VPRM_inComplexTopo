"""
Fig11_linPertComp_hourly_mean.py
==============================================
Fig. 11 (fig:componentslinper).
Plots linear perturbation analysis results:
- Contribution maps of each driver to GPP differences
- Residual map of unexplained differences
- Hourly mean diurnal cycles of contributions and residuals
- Comparison across WRF resolutions
==============================================
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import netCDF4 as nc
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import binary_erosion, distance_transform_edt
import xarray as xr
import pandas as pd
import os
import glob
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

############# INPUT ############
save_plot_maps = False
save_plot2 = True
print_output = True
# use all dates within this range
# start_date = "2012-01-01 00:00:00"
# end_date = "2012-12-31 00:00:00"

# for zenodo data there is only available 2012-07-27
start_date = os.getenv("FIG10_START", "2012-01-01 00:00:00")
end_date = os.getenv("FIG10_END", "2012-12-31 00:00:00")
sim_type = os.getenv("FIG10_SIM", "_cloudy")  # "", "_cloudy"
# for Fig. 10 you need the full wrf output, there is no extraction into csv files
SCRATCH_PATH = os.getenv("SCRATCH_PATH")
wrf_basepath = f"{SCRATCH_PATH}/DATA/WRFOUT/WRFOUT_ALPS"  # without resolution suffix
dx_all = ["_54km", "_9km"]  # "_54km",
wrf_tag = os.getenv("FIG10_WRFTAG", "")  # "" for standard/cloudy; "_radt54_swint1" for swint1
_tag_to_label = {"_radt54_swint1": "_swint1"}
file_label_extra = _tag_to_label.get(wrf_tag, "")
OUTFOLDER = os.getenv("OUTFOLDER", "./plots/")
plots_folder = f"{OUTFOLDER}components{sim_type}_L2_"
interp_method = "nearest"  # 'linear', 'nearest', 'cubic'
STD_TOPO = 200
##############################


def plot_lin_pert_results(contribs_grid, residual, driver_names=None):
    """
    Plot contributions and residual from linear perturbation analysis.

    contribs_grid: array (6, ny, nx)
    residual: array (ny, nx)
    driver_names: list of 6 strings
    """
    if driver_names is None:
        driver_names = ["Lambda", "T", "W", "P", "R", "E"]

    n_drivers = contribs_grid.shape[0]
    ny, nx = contribs_grid.shape[1], contribs_grid.shape[2]

    fig, axes = plt.subplots(2, 4, figsize=(6.7, 2.4))  # 2x the 0.48\linewidth width
    axes = axes.flatten()

    # Plot each driver contribution
    vmin = np.nanmin(contribs_grid)
    vmax = np.nanmax(contribs_grid)
    for i in range(n_drivers):
        im = axes[i].imshow(
            contribs_grid[i], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        axes[i].set_title(driver_names[i], fontsize=14)
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Plot residual
    im = axes[n_drivers].imshow(
        residual, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    axes[n_drivers].set_title("Residual", fontsize=14)
    axes[n_drivers].axis("off")
    fig.colorbar(im, ax=axes[n_drivers], fraction=0.046, pad=0.04)

    # Hide any remaining axes
    for j in range(n_drivers + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{plots_folder}_lin_pertubation_panels_{date_time}h_{dx}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def linear_perturbation_analysis(
    GPP_1km,
    GPP_d01,
    Lambda_1km,
    Lambda_d01,
    T_1km,
    T_d01,
    W_1km,
    W_d01,
    P_1km,
    P_d01,
    R_1km,
    R_d01,
    E_1km,
    E_d01,
    regularize=True,
    alpha=1e-6,
):
    # Flatten and convert to float
    GPP_1km = np.asarray(GPP_1km, dtype=np.float64)
    GPP_d01 = np.asarray(GPP_d01, dtype=np.float64)
    dG = (GPP_d01 - GPP_1km).ravel()

    drivers = [
        Lambda_d01 - Lambda_1km,
        T_d01 - T_1km,
        W_d01 - W_1km,
        P_d01 - P_1km,
        R_d01 - R_1km,
        E_d01 - E_1km,
    ]

    D = np.stack([np.asarray(d, dtype=np.float64).ravel() for d in drivers], axis=1)

    # Remove NaNs
    mask = np.all(np.isfinite(D), axis=1) & np.isfinite(dG)
    D_valid = D[mask]
    dG_valid = dG[mask]

    # Solve
    if regularize and alpha > 0.0:
        A = np.linalg.solve(
            np.dot(D_valid.T, D_valid) + alpha * np.eye(6), np.dot(D_valid.T, dG_valid)
        )
    else:
        A, *_ = np.linalg.lstsq(D_valid, dG_valid, rcond=None)

    # Contributions
    contribs_flat = D_valid * A  # (N_valid,6)

    # Fill full grid
    ny, nx = GPP_1km.shape
    contribs_grid = np.full((6, ny, nx), np.nan)
    residual = np.full((ny, nx), np.nan)
    valid_idx = np.where(mask)[0]
    for i in range(6):
        contribs_grid[i].flat[valid_idx] = contribs_flat[:, i]
    residual.flat[valid_idx] = dG_valid - contribs_flat.sum(axis=1)

    return A, contribs_grid, residual


def styled_imshow_plot_d01(data, vmin, vmax, cmap, label, filename):
    fig, ax = plt.subplots(
        figsize=(6.7, 8.4), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    im = ax.imshow(
        data,
        extent=[lons_d01.min(), lons_d01.max(), lats_d01.min(), lats_d01.max()],
        cmap=cmap,
        origin="lower",
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(
        im, ax=ax, orientation="vertical", shrink=0.3, fraction=0.046, pad=0.06
    )
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label, fontsize=16)

    gl = ax.gridlines(
        draw_labels=True, linewidth=1.5, color="black", alpha=0.2, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 20}
    gl.ylabel_style = {"size": 20}

    ax.set_xlabel("Longitude", fontsize=16)
    ax.set_ylabel("Latitude", fontsize=16)
    plt.tick_params(labelsize=14)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)

    plt.tight_layout()

    plt.savefig(f"{plots_folder}{filename}_{date_time}h.pdf", bbox_inches="tight")
    plt.close()


def styled_imshow_plot(data, vmin, vmax, cmap, label, filename):
    fig, ax = plt.subplots(
        figsize=(6.7, 8.4), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    im = ax.imshow(
        data,
        extent=[lons_fine.min(), lons_fine.max(), lats_fine.min(), lats_fine.max()],
        cmap=cmap,
        origin="lower",
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(
        im, ax=ax, orientation="vertical", shrink=0.3, fraction=0.046, pad=0.06
    )
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label, fontsize=16)

    gl = ax.gridlines(
        draw_labels=True, linewidth=1.5, color="black", alpha=0.2, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 20}
    gl.ylabel_style = {"size": 20}

    ax.set_xlabel("Longitude", fontsize=16)
    ax.set_ylabel("Latitude", fontsize=16)
    plt.tick_params(labelsize=14)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"{plots_folder}{filename}_{date_time}h.pdf", bbox_inches="tight")
    plt.close()


def generate_coastal_mask(
    veg_type: np.ndarray, buffer_km: float = 50.0, grid_spacing_km: float = 3.0
) -> np.ndarray:
    """
    Returns a new landmask where any point within `buffer_km` of the coastline is set to 0 (masked).

    Args:
        veg_type: 2D array with 44 for water.
        buffer_km: Distance from coastline to mask, in km.
        grid_spacing_km: Grid spacing in km (e.g., 3 for WRF 1km grid).

    Returns:
        new_landmask: same shape as input, with coastal zone masked out.
    """
    # Create landmask: 1 for land, 0 for water
    landmask = np.ones_like(veg_type, dtype=np.uint8)
    landmask[veg_type == 44] = 0
    land_binary = landmask.astype(bool)
    eroded_land = binary_erosion(land_binary)
    coastline = land_binary & (~eroded_land)

    # Compute distance (in grid cells) from coastline
    distance = distance_transform_edt(~coastline) * grid_spacing_km

    # Mask everything within `buffer_km` from coastline
    new_landmask = landmask.copy()
    new_landmask[distance <= buffer_km] = 0

    return new_landmask


def proj_on_finer_WRF_grid(
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_1km, method_in
):
    proj_var = griddata(
        (lats_coarse.flatten(), lons_coarse.flatten()),
        var_coarse.flatten(),
        (lats_fine, lons_fine),
        method=method_in,
    ).reshape(WRF_var_1km.shape)
    return proj_var


def extract_datetime_from_filename(filename):
    """
    Extract datetime from WRF filename assuming format 'wrfout_d0x_YYYY-MM-DD_HH:MM:SS'.
    """
    base_filename = os.path.basename(filename)
    date_str = base_filename.split("_")[-2] + "_" + base_filename.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")


date_str_min = start_date.split(" ")[0]
date_str_max = end_date.split(" ")[0]

# Load GPP-based ALPS parameters once (replaces hardcoded NEE-based dicts)
_vprm_csv = pd.read_csv(Path(__file__).parent / "vprm_params_newGPP_V24.csv", comment="#")
RAD0_of_PFT   = dict(zip(_vprm_csv["pft"], _vprm_csv["par0"]))
lambda_of_PFT = dict(zip(_vprm_csv["pft"], _vprm_csv["lambda"]))
Tvar_of_PFT   = {r["pft"]: (r["t_opt"], r["t_min"], r["t_max"])
                 for _, r in _vprm_csv.iterrows()}
RAD0_of_PFT["OTH"]   = 0.0
lambda_of_PFT["OTH"] = 0.0
Tvar_of_PFT["OTH"]   = (0.0, 0, 40)
if sim_type == "_pram_err":
    lambda_of_PFT = {
        "ENF": 0.304, "DBF": 0.216, "MF": 0.114,
        "SHB": 0.0874, "SAV": 0.114, "CRO": 0.140,
        "GRA": 0.448, "OTH": 0.00,
    }

# =====================================================================
# DEFAULT config (sign-constancy check, 9d): the parameter table actually
# baked into the WRF binary that produced EBIO_GEE for the archived
# WRFOUT_ALPS_* runs (vprm_params_alps_asrun.csv's own header: "the ACTUAL
# build behind the archived wrfout"). Distinct from the offline ALPS recalc
# above (vprm_params_newGPP_V24.csv). Same dict shape as a MEMBERS entry, so
# it can be passed straight into compute_member_fields.
# =====================================================================
_vprm_csv_default = pd.read_csv(Path(__file__).parent / "vprm_params_alps_asrun.csv", comment="#")
_lambda_default = dict(zip(_vprm_csv_default["pft"], _vprm_csv_default["lambda"]))
_rad0_default = dict(zip(_vprm_csv_default["pft"], _vprm_csv_default["par0"]))
_tvar_default = {r["pft"]: (r["t_opt"], r["t_min"], r["t_max"])
                 for _, r in _vprm_csv_default.iterrows()}
_lambda_default["OTH"] = 0.0
_rad0_default["OTH"] = 0.0
_tvar_default["OTH"] = (0.0, 0, 40)
DEFAULT_MEMBER = {"lambda": _lambda_default, "RAD0": _rad0_default, "Tvar": _tvar_default}

# accumulates one row per (config, resolution, timestep) for the sign-count
# check on Y_Wscale/Y_Pscale/Y_EVI; written to disk once at the end of each
# dx iteration (see bottom of the dx loop).
_Y_SIGNCHECK_RECORDS = []


def _check_modis_8day_cadence():
    """Confirm the manuscript's '8-day update interval' claim from the MODIS
    MOD09A1 file-naming pattern (day-of-year in the filename), not just
    trusted from the text."""
    modis_dir = "/scratch/c7071034/DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/data/modis/hdf/2012/"
    if not os.path.isdir(modis_dir):
        print(f"  [WARN] MODIS hdf directory not found, skipping 8-day cadence check: {modis_dir}")
        return
    doys = sorted({
        int(f.split(".")[1][5:8]) for f in os.listdir(modis_dir)
        if f.startswith("MOD09A1") and f.endswith(".hdf")
    })
    gaps = np.diff(doys)
    # allow a short final tail (day 361 -> day 366/1, less than a full 8-day step)
    bad = [g for g in gaps[:-1] if g != 8] if len(gaps) > 1 else []
    print(f"  MODIS MOD09A1 day-of-year steps found: {doys[:3]} ... {doys[-3:]} "
          f"(gaps: {sorted(set(gaps))})")
    assert not bad, f"MODIS cadence is not 8 days everywhere (excl. final tail): irregular gaps {bad}"
    print("  [ok] MODIS 8-day update cadence confirmed from the MOD09A1 file-naming pattern")


_check_modis_8day_cadence()

# =====================================================================
# Parameter-spread ensemble (p10-p90 of Topt, lambda, PAR0) for the band.
# Central member (p50) reproduces the V24 result exactly; only ENF/DBF/GRA
# are perturbed (the PFTs with a Topt ensemble).
# =====================================================================
EPS_VF = 1e-7
MEMBER_TAGS = ["p10", "p25", "p50", "p75", "p90"]
CENTRAL_IDX = 2
SPREAD_PFTS = ["ENF", "DBF", "GRA"]

_topt_perc = pd.read_csv(ROOT / "VPRM_tools" / "topt_percentiles.csv").set_index("PFT")
_par_perc = pd.read_csv(ROOT / "VPRM_tools" / "param_percentiles.csv")
_perc_RAD0 = _par_perc[_par_perc.param == "RAD0"].set_index("PFT")
_perc_lam = _par_perc[_par_perc.param == "lambd"].set_index("PFT")


def _build_member(tag):
    lam = dict(lambda_of_PFT)
    rad = dict(RAD0_of_PFT)
    tv = dict(Tvar_of_PFT)
    for pft in SPREAD_PFTS:
        rad[pft] = float(_perc_RAD0.loc[pft, tag])
        lam[pft] = float(_perc_lam.loc[pft, tag])
        tv[pft] = (float(_topt_perc.loc[pft, tag]), tv[pft][1], tv[pft][2])
    return {"lambda": lam, "RAD0": rad, "Tvar": tv}


MEMBERS = [_build_member(t) for t in MEMBER_TAGS]

# driver names (defined once; also used inside the loop / at plotting)
driver_names = ["dlambda", "dTscale", "dWscale", "dPscale", "dRAD", "dEVI"]
driver_names_plot = [
    r"$\overline{Y_{\lambda}}$",
    r"$\overline{Y_{\text{T}_\text{scale}}}$",
    r"$\overline{Y_{\text{W}_\text{scale}}}$",
    r"$\overline{Y_{\text{P}_\text{scale}}}$",
    r"$\overline{Y_{\text{RAD}}}$",
    r"$\overline{Y_{\text{EVI}}}$",
]
BAND_DRIVERS = {"dlambda", "dTscale", "dRAD"}  # only these get a spread band

# nearest-neighbour projection index (coarse d01 -> fine 1km), cached per resolution
_PROJ_IDX = {}


def get_proj_idx(key, lats_c, lons_c, lats_f, lons_f):
    if key not in _PROJ_IDX:
        tree = cKDTree(np.column_stack([np.asarray(lats_c).ravel(),
                                        np.asarray(lons_c).ravel()]))
        _, idx = tree.query(np.column_stack([np.asarray(lats_f).ravel(),
                                             np.asarray(lons_f).ravel()]))
        _PROJ_IDX[key] = idx
    return _PROJ_IDX[key]


def project_nn(field_coarse, idx, fine_shape):
    return np.asarray(field_coarse).ravel()[idx].reshape(fine_shape)


def compute_member_fields(mp, f1, fd):
    """Accumulate the VPRM driver fields for one parameter member on the 1km grid
    (f1) and the coarse d01 grid (fd). Faithful port of the in-loop m-loop, so the
    central (p50) member reproduces the unperturbed fields. f1/fd are dicts with
    keys SW,T2,veg,evi,evi_min,evi_max,lswi,lswi_min,lswi_max."""
    lam, rad, tv = mp["lambda"], mp["RAD0"], mp["Tvar"]
    keys = list(lam.keys())
    eps = EPS_VF
    SW1, T21 = f1["SW"], f1["T2"]
    SWd, T2d = fd["SW"], fd["T2"]

    o1 = {k: np.zeros_like(SW1) for k in
          ("Lambda", "Tscale", "RAD", "Wscale", "Pscale", "EVI", "GPP")}
    od = {k: np.zeros_like(SWd) for k in
          ("Lambda", "Tscale", "RAD", "Wscale", "Pscale", "EVI", "GPP")}

    for m in range(7):
        key = keys[m]
        # --- 1 km ---
        vf = f1["veg"][0, m, :, :]
        if np.all(vf < eps):
            continue
        vf = np.where(np.isnan(vf), 0.0, vf)
        vf = np.where(vf < eps, 0.0, vf)
        o1["Lambda"] += lam[key] * vf
        evi_t = f1["evi"][0, m, :, :]
        o1["EVI"] += evi_t * vf
        rad_t = np.zeros_like(SW1)
        if rad[key] > 0:
            rad_t = (1.0 / (1.0 + SW1 / rad[key])) * SW1
            rad_t = np.nan_to_num(rad_t, nan=0.0, posinf=0.0, neginf=0.0)
        o1["RAD"] += rad_t * vf
        a1 = T21 - tv[key][1]
        a2 = T21 - tv[key][2]
        a3 = T21 - tv[key][0]
        tsc = np.where((a1 < 0) | (a2 > 0), 0, a1 * a2 / (a1 * a2 - a3 ** 2))
        tsc = np.nan_to_num(tsc, nan=0.0, posinf=0.0, neginf=0.0)
        tsc = np.where(tsc < 0, 0, tsc)
        o1["Tscale"] += tsc * vf
        if m == 3 or m == 6:
            num = f1["lswi"][0, m] - f1["lswi_min"][0, m]
            den = f1["lswi_max"][0, m] - f1["lswi_min"][0, m]
            wsc = np.divide(num, den, out=np.zeros_like(num), where=den >= eps)
        else:
            wsc = (1 + f1["lswi"][0, m]) / (1 + f1["lswi_max"][0, m])
        wsc[np.isnan(wsc)] = 0
        o1["Wscale"] += wsc * vf
        if m == 0:
            psc = np.ones_like(SW1)
        elif m == 4 or m == 6:
            psc = (1 + f1["lswi"][0, m]) / 2.0
        else:
            thr = f1["evi_min"][0, m] + 0.55 * (f1["evi_max"][0, m] - f1["evi_min"][0, m])
            psc = np.where(f1["evi"][0, m] >= thr, 1.0, (1 + f1["lswi"][0, m]) / 2.0)
        psc = np.nan_to_num(psc, nan=0.0)
        o1["Pscale"] += psc * vf
        g = lam[key] * tsc * wsc * psc * rad_t * evi_t * vf
        g[g < 0] = 0
        o1["GPP"] += g

        # --- d01 ---
        vfd = fd["veg"][0, m, :, :]
        if np.all(vfd < eps):
            continue
        vfd = np.where(np.isnan(vfd), 0.0, vfd)
        vfd = np.where(vfd < eps, 0.0, vfd)
        od["Lambda"] += lam[key] * vfd
        evi_td = fd["evi"][0, m, :, :]
        od["EVI"] += evi_td * vfd
        rad_td = np.zeros_like(SWd)
        if rad[key] > 0:
            rad_td = (1.0 / (1.0 + SWd / rad[key])) * SWd
            rad_td = np.nan_to_num(rad_td, nan=0.0, posinf=0.0, neginf=0.0)
        od["RAD"] += rad_td * vfd
        a1 = T2d - tv[key][1]
        a2 = T2d - tv[key][2]
        a3 = T2d - tv[key][0]
        tscd = np.where((a1 < 0) | (a2 > 0), 0, a1 * a2 / (a1 * a2 - a3 ** 2))
        tscd = np.nan_to_num(tscd, nan=0.0, posinf=0.0, neginf=0.0)
        tscd = np.where(tscd < 0, 0, tscd)
        od["Tscale"] += tscd * vfd
        if m == 3 or m == 6:
            num = fd["lswi"][0, m] - fd["lswi_min"][0, m]
            den = fd["lswi_max"][0, m] - fd["lswi_min"][0, m]
            wscd = np.divide(num, den, out=np.zeros_like(num), where=den >= eps)
        else:
            wscd = (1 + fd["lswi"][0, m]) / (1 + fd["lswi_max"][0, m])
        wscd[np.isnan(wscd)] = 0
        od["Wscale"] += wscd * vfd
        if m == 0:
            pscd = np.ones_like(SWd)
        elif m == 4 or m == 6:
            pscd = (1 + fd["lswi"][0, m]) / 2.0
        else:
            thr = fd["evi_min"][0, m] + 0.55 * (fd["evi_max"][0, m] - fd["evi_min"][0, m])
            pscd = np.where(fd["evi"][0, m] >= thr, 1.0, (1 + fd["lswi"][0, m]) / 2.0)
        pscd = np.nan_to_num(pscd, nan=0.0)
        od["Pscale"] += pscd * vfd
        gd = lam[key] * tscd * wscd * pscd * rad_td * evi_td * vfd
        gd[gd < 0] = 0
        od["GPP"] += gd

    return o1, od


records = []
for dx in dx_all:
    wrf_dx_suffix = wrf_tag if (dx == "_54km" and wrf_tag) else ""
    # Convert to datetime (but ignore time part for full-day selection)
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").date()
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").date()
    # initialize final dfs
    lin_pert_mean_diffs_df = pd.DataFrame(
        columns=["dlambda", "dTscale", "dWscale", "dPscale", "dRAD", "dEVI", "Residual"]
    )
    mean_diffs_df = pd.DataFrame(
        columns=[
            "dlambda",
            "dTscale",
            "dWscale",
            "dPscale",
            "dRAD",
            "dEVI",
            # "dGPP",
            # "dGPP_validate",
            # "dGPP_lin_per",
        ]
    )
    # per-member contribution stores for the non-central spread members (band)
    spread_cols = ["dlambda", "dTscale", "dWscale", "dPscale", "dRAD", "dEVI", "Residual"]
    spread_dfs = {mi: pd.DataFrame(columns=spread_cols)
                  for mi in range(len(MEMBERS)) if mi != CENTRAL_IDX}

    # Collect all files
    files_d01 = sorted(
        glob.glob(os.path.join(wrf_basepath + dx + wrf_dx_suffix + sim_type, f"wrfout_d01*"))
    )
    files_d01 = [os.path.basename(f) for f in files_d01]

    file_by_day = defaultdict(list)
    for f in files_d01:
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

    timestamps = [extract_datetime_from_filename(f) for f in file_list]
    time_index = pd.to_datetime(timestamps)

    for wrf_file in file_list:
        time = extract_datetime_from_filename(wrf_file)
        print(f"processing: {time} at {dx}")
        date_time = wrf_file[11:24]
        date = date_time.split("_")[0]
        wrfinput_path_1km = f"{wrf_basepath}_1km{sim_type}/wrfout_d02_{date_time}:00:00"
        wrfinput_path_d01 = f"{wrf_basepath}{dx}{wrf_dx_suffix}{sim_type}/wrfout_d01_{date_time}:00:00"
        vprm_input_path_1km = f"{SCRATCH_PATH}/DATA/VPRM_input/vprm_corine_1km/vprm_input_d02_{date}_00:00:00.nc"
        vprm_input_path_d01 = f"{SCRATCH_PATH}/DATA/VPRM_input/vprm_corine{dx}/vprm_input_d01_{date}_00:00:00.nc"

        # Load the NetCDF file
        conv_factor = 1 / 3600
        nc_fid1km = nc.Dataset(wrfinput_path_1km, "r")
        nc_fid54km = nc.Dataset(wrfinput_path_d01, "r")
        GPP_WRF_1km = (
            -nc_fid1km.variables["EBIO_GEE"][0, 0, 10:-10, 10:-10] * conv_factor
        )
        GPP_WRF_d01 = -nc_fid54km.variables["EBIO_GEE"][0, 0, :, :] * conv_factor
        SWDOWN_1km = nc_fid1km.variables["SWDOWN"][0, 10:-10, 10:-10]
        SWDOWN_d01 = nc_fid54km.variables["SWDOWN"][0]
        T2_1km = nc_fid1km.variables["T2"][0, 10:-10, 10:-10] - 273.15
        T2_d01 = nc_fid54km.variables["T2"][0] - 273.15
        lats_fine = nc_fid1km.variables["XLAT"][0, 10:-10, 10:-10]
        lons_fine = nc_fid1km.variables["XLONG"][0, 10:-10, 10:-10]
        stdh_topo_1km = nc_fid1km.variables["VAR"][0, 10:-10, 10:-10]
        stdh_mask = stdh_topo_1km >= STD_TOPO
        lats_d01 = nc_fid54km.variables["XLAT"][0, :, :]
        lons_d01 = nc_fid54km.variables["XLONG"][0, :, :]
        veg_type = nc_fid54km.variables["IVGTYP"][0, :, :]
        new_landmask = generate_coastal_mask(
            veg_type, buffer_km=30.0, grid_spacing_km=50.0
        )

        # --- Load vvprm_input ---
        eps = 1e-7  # numerical safeguard
        ds = xr.open_dataset(vprm_input_path_1km)
        ds_d01 = xr.open_dataset(vprm_input_path_d01)
        # ['Times', 'XLONG', 'XLAT', 'EVI_MIN', 'EVI_MAX', 'EVI', 'LSWI_MIN', 'LSWI_MAX', 'LSWI', 'VEGFRA_VPRM']
        veg_frac_map = (
            ds["VEGFRA_VPRM"]
            .isel(south_north=slice(10, -10), west_east=slice(10, -10))
            .values
        )
        veg_frac_map_d01 = ds_d01["VEGFRA_VPRM"].values
        veg_frac_map = np.nan_to_num(veg_frac_map, nan=0.0)
        veg_frac_map_d01 = np.nan_to_num(veg_frac_map_d01, nan=0.0)

        evi_map = (
            ds["EVI"].isel(south_north=slice(10, -10), west_east=slice(10, -10)).values
        )
        evi_map_d01 = ds_d01["EVI"].values
        evi_min_map = (
            ds["EVI_MIN"]
            .isel(south_north=slice(10, -10), west_east=slice(10, -10))
            .values
        )
        evi_min_map_d01 = ds_d01["EVI_MIN"].values
        evi_max_map = (
            ds["EVI_MAX"]
            .isel(south_north=slice(10, -10), west_east=slice(10, -10))
            .values
        )
        evi_max_map_d01 = ds_d01["EVI_MAX"].values

        evi_map = np.nan_to_num(evi_map, nan=0)
        evi_min_map = np.nan_to_num(evi_min_map, nan=0)
        evi_max_map = np.nan_to_num(evi_max_map, nan=0)
        evi_map_d01 = np.nan_to_num(evi_map_d01, nan=0)
        evi_min_map_d01 = np.nan_to_num(evi_min_map_d01, nan=0)
        evi_max_map_d01 = np.nan_to_num(evi_max_map_d01, nan=0)

        lswi_map = (
            ds["LSWI"].isel(south_north=slice(10, -10), west_east=slice(10, -10)).values
        )
        lswi_map_d01 = ds_d01["LSWI"].values
        lswi_min_map = (
            ds["LSWI_MIN"]
            .isel(south_north=slice(10, -10), west_east=slice(10, -10))
            .values
        )
        lswi_min_map_d01 = ds_d01["LSWI_MIN"].values
        lswi_max_map = (
            ds["LSWI_MAX"]
            .isel(south_north=slice(10, -10), west_east=slice(10, -10))
            .values
        )
        lswi_max_map_d01 = ds_d01["LSWI_MAX"].values

        lswi_map = np.nan_to_num(lswi_map, nan=-1.0)
        lswi_min_map = np.nan_to_num(lswi_min_map, nan=-1.0)
        lswi_max_map = np.nan_to_num(lswi_max_map, nan=-1.0)
        lswi_map_d01 = np.nan_to_num(lswi_map_d01, nan=-1.0)
        lswi_min_map_d01 = np.nan_to_num(lswi_min_map_d01, nan=-1.0)
        lswi_max_map_d01 = np.nan_to_num(lswi_max_map_d01, nan=-1.0)

        # parameters loaded above from vprm_params_newGPP_V24.csv

        # --- parameter-spread ensemble fields (central p50 == unperturbed) ---
        proj_idx = get_proj_idx(dx, lats_d01, lons_d01, lats_fine, lons_fine)
        fine_shape = SWDOWN_1km.shape
        _f1_in = dict(SW=SWDOWN_1km, T2=T2_1km, veg=veg_frac_map, evi=evi_map,
                      evi_min=evi_min_map, evi_max=evi_max_map, lswi=lswi_map,
                      lswi_min=lswi_min_map, lswi_max=lswi_max_map)
        _fd_in = dict(SW=SWDOWN_d01, T2=T2_d01, veg=veg_frac_map_d01, evi=evi_map_d01,
                      evi_min=evi_min_map_d01, evi_max=evi_max_map_d01, lswi=lswi_map_d01,
                      lswi_min=lswi_min_map_d01, lswi_max=lswi_max_map_d01)
        raw_members = [compute_member_fields(mp, _f1_in, _fd_in) for mp in MEMBERS]

        # DEFAULT-config driver fields (9d): same satellite/land-cover inputs,
        # DEFAULT_MEMBER's as-run parameter table instead of the ALPS one.
        o1_default, od_default = compute_member_fields(DEFAULT_MEMBER, _f1_in, _fd_in)

        # Self-consistency check: Wscale/Pscale/EVI have no VPRM parameter
        # dependence (see compute_member_fields), so they must be identical
        # between the DEFAULT and ALPS (central/p50) passes -- this is the
        # thing that lets the DEFAULT regression below reuse the ALPS pass's
        # W/P/E arrays instead of reprojecting them a second time.
        _central_o1, _central_od = raw_members[CENTRAL_IDX]
        for _key in ("Wscale", "Pscale", "EVI"):
            assert np.allclose(o1_default[_key], _central_o1[_key], equal_nan=True), (
                f"{_key} (1km) differs between DEFAULT and ALPS parameter sets "
                f"at {time} {dx} -- expected parameter-independence"
            )
            assert np.allclose(od_default[_key], _central_od[_key], equal_nan=True), (
                f"{_key} (d01) differs between DEFAULT and ALPS parameter sets "
                f"at {time} {dx} -- expected parameter-independence"
            )

        # init arrays
        RAD_1km = np.zeros_like(SWDOWN_1km)
        Tscale_1km = np.zeros_like(SWDOWN_1km)
        Wscale_1km = np.zeros_like(SWDOWN_1km)
        Pscale_1km = np.zeros_like(SWDOWN_1km)
        EVI_1km = np.zeros_like(SWDOWN_1km)
        Lambda_1km = np.zeros_like(SWDOWN_1km)
        GPP_validate_1km = np.zeros_like(SWDOWN_1km)

        RAD_d01 = np.zeros_like(SWDOWN_d01)
        Tscale_d01 = np.zeros_like(SWDOWN_d01)
        Wscale_d01 = np.zeros_like(SWDOWN_d01)
        Pscale_d01 = np.zeros_like(SWDOWN_d01)
        EVI_d01 = np.zeros_like(SWDOWN_d01)
        Lambda_d01 = np.zeros_like(SWDOWN_d01)
        GPP_validate_d01 = np.zeros_like(SWDOWN_d01)

        for m in range(7):

            # --- vegetation fraction ---
            vegfrac_1km = veg_frac_map[0, m, :, :]
            if np.all(vegfrac_1km < eps):
                continue
            vegfrac_1km = np.where(np.isnan(vegfrac_1km), 0.0, vegfrac_1km)
            vegfrac_1km = np.where(vegfrac_1km < eps, 0.0, vegfrac_1km)

            # --- Lambda ---
            Lambda_temp_1km = lambda_of_PFT[list(lambda_of_PFT.keys())[m]]
            Lambda_1km += Lambda_temp_1km * vegfrac_1km

            # --- EVI ---
            EVI_temp_1km = evi_map[0, m, :, :]
            EVI_1km += EVI_temp_1km * vegfrac_1km

            # --- PAR ---
            RAD_temp_1km = np.zeros_like(SWDOWN_1km)
            RAD0 = RAD0_of_PFT[list(Tvar_of_PFT.keys())[m]]
            if RAD0 > 0:
                RAD_temp_1km = ((1 / (1 + SWDOWN_1km / RAD0))) * SWDOWN_1km
                if np.any(np.isnan(RAD_temp_1km)):
                    print(
                        "Count of NaNs in RAD_temp_1km:", np.sum(np.isnan(RAD_temp_1km))
                    )
                    RAD_temp_1km = np.nan_to_num(
                        RAD_temp_1km, nan=0.0, posinf=0.0, neginf=0.0
                    )

            RAD_1km += RAD_temp_1km * vegfrac_1km

            # --- Tscale ---
            a1 = T2_1km - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][1]
            a2 = T2_1km - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][2]
            a3 = T2_1km - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][0]
            Tscale_temp_1km = np.where(
                (a1 < 0) | (a2 > 0), 0, a1 * a2 / (a1 * a2 - a3**2)
            )
            Tscale_temp_1km = np.nan_to_num(
                Tscale_temp_1km, nan=0.0, posinf=0.0, neginf=0.0
            )
            Tscale_1km += (
                np.where(Tscale_temp_1km < 0, 0, Tscale_temp_1km) * vegfrac_1km
            )

            # --- Wscale ---
            if m == 3 or m == 6:  # grassland / shrubland (xeric systems)
                num = lswi_map[0, m, :, :] - lswi_min_map[0, m, :, :]
                den = lswi_max_map[0, m, :, :] - lswi_min_map[0, m, :, :]
                # Fortran: if den < 1e-7 → Wscale = 0
                Wscale_temp_1km = np.divide(
                    num, den, out=np.zeros_like(num), where=den >= eps
                )
            else:
                Wscale_temp_1km = (1 + lswi_map[0, m, :, :]) / (
                    1 + lswi_max_map[0, m, :, :]
                )

            Wscale_temp_1km[np.isnan(Wscale_temp_1km)] = 0
            Wscale_1km += Wscale_temp_1km * vegfrac_1km

            # --- Pscale ---
            if m == 0:  # evergreen
                Pscale_temp_1km = np.ones_like(SWDOWN_1km)
            elif m == 4 or m == 6:  # savanna / grassland
                Pscale_temp_1km = (1 + lswi_map[0, m, :, :]) / 2.0
            else:
                evithresh = evi_min_map[0, m, :, :] + 0.55 * (
                    evi_max_map[0, m, :, :] - evi_min_map[0, m, :, :]
                )
                Pscale_temp_1km = np.where(
                    evi_map[0, m, :, :] >= evithresh,
                    1.0,
                    (1 + lswi_map[0, m, :, :]) / 2.0,
                )
            if np.any(np.isnan(Pscale_temp_1km)):
                Pscale_temp_1km = np.nan_to_num(Pscale_temp_1km, nan=0.0)

            Pscale_1km += Pscale_temp_1km * vegfrac_1km

            # --- GPP for comparison ---
            GPP_temp_1km = (
                Lambda_temp_1km
                * Tscale_temp_1km
                * Wscale_temp_1km
                * Pscale_temp_1km
                * RAD_temp_1km
                * EVI_temp_1km
                * vegfrac_1km
            )
            GPP_temp_1km[GPP_temp_1km < 0] = 0
            GPP_validate_1km += GPP_temp_1km

            ### --- Domain d01 ---
            # --- vegetation fraction ---
            vegfrac_d01 = veg_frac_map_d01[0, m, :, :]
            if np.all(vegfrac_d01 < eps):
                continue
            vegfrac_d01 = np.where(np.isnan(vegfrac_d01), 0.0, vegfrac_d01)
            vegfrac_d01 = np.where(vegfrac_d01 < eps, 0.0, vegfrac_d01)

            # --- Lambda ---
            Lambda_temp_d01 = lambda_of_PFT[list(lambda_of_PFT.keys())[m]]
            Lambda_d01 += Lambda_temp_d01 * vegfrac_d01

            # --- EVI ---
            EVI_temp_d01 = evi_map_d01[0, m, :, :]
            EVI_d01 += EVI_temp_d01 * vegfrac_d01

            # --- PAR ---
            RAD_temp_d01 = np.zeros_like(SWDOWN_d01)
            RAD0 = RAD0_of_PFT[list(Tvar_of_PFT.keys())[m]]
            if RAD0 > 0:
                RAD_temp_d01 = ((1 / (1 + SWDOWN_d01 / RAD0))) * SWDOWN_d01
                if np.any(np.isnan(RAD_temp_d01)):
                    print(
                        "Count of NaNs in RAD_temp_d01:", np.sum(np.isnan(RAD_temp_d01))
                    )
                    RAD_temp_d01 = np.nan_to_num(
                        RAD_temp_d01, nan=0.0, posinf=0.0, neginf=0.0
                    )

            RAD_d01 += RAD_temp_d01 * vegfrac_d01

            # --- Tscale ---
            a1 = T2_d01 - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][1]
            a2 = T2_d01 - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][2]
            a3 = T2_d01 - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][0]
            Tscale_temp_d01 = np.where(
                (a1 < 0) | (a2 > 0), 0, a1 * a2 / (a1 * a2 - a3**2)
            )
            Tscale_temp_d01 = np.nan_to_num(
                Tscale_temp_d01, nan=0.0, posinf=0.0, neginf=0.0
            )
            Tscale_d01 += (
                np.where(Tscale_temp_d01 < 0, 0, Tscale_temp_d01) * vegfrac_d01
            )

            # --- Wscale ---
            if m == 3 or m == 6:  # grassland / shrubland (xeric systems)
                num = lswi_map_d01[0, m, :, :] - lswi_min_map_d01[0, m, :, :]
                den = lswi_max_map_d01[0, m, :, :] - lswi_min_map_d01[0, m, :, :]
                # Fortran: if den < 1e-7 → Wscale = 0
                Wscale_temp_d01 = np.divide(
                    num, den, out=np.zeros_like(num), where=den >= eps
                )
            else:
                Wscale_temp_d01 = (1 + lswi_map_d01[0, m, :, :]) / (
                    1 + lswi_max_map_d01[0, m, :, :]
                )

            Wscale_temp_d01[np.isnan(Wscale_temp_d01)] = 0
            Wscale_d01 += Wscale_temp_d01 * vegfrac_d01

            # --- Pscale ---
            if m == 0:  # evergreen
                Pscale_temp_d01 = np.ones_like(SWDOWN_d01)
            elif m == 4 or m == 6:  # savanna / grassland
                Pscale_temp_d01 = (1 + lswi_map_d01[0, m, :, :]) / 2.0
            else:
                evithresh = evi_min_map_d01[0, m, :, :] + 0.55 * (
                    evi_max_map_d01[0, m, :, :] - evi_min_map_d01[0, m, :, :]
                )
                Pscale_temp_d01 = np.where(
                    evi_map_d01[0, m, :, :] >= evithresh,
                    1.0,
                    (1 + lswi_map_d01[0, m, :, :]) / 2.0,
                )
            if np.any(np.isnan(Pscale_temp_d01)):
                Pscale_temp_d01 = np.nan_to_num(Pscale_temp_d01, nan=0.0)

            Pscale_d01 += Pscale_temp_d01 * vegfrac_d01

            # --- GPP for comparison ---
            GPP_temp_d01 = (
                Lambda_temp_d01
                * Tscale_temp_d01
                * Wscale_temp_d01
                * Pscale_temp_d01
                * RAD_temp_d01
                * EVI_temp_d01
                * vegfrac_d01
            )
            GPP_temp_d01[GPP_temp_d01 < 0] = 0
            GPP_validate_d01 += GPP_temp_d01

        proj_landmask_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            new_landmask,
            lats_fine,
            lons_fine,
            SWDOWN_1km,
            interp_method,
        )

        proj_SWDOWN_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            SWDOWN_d01,
            lats_fine,
            lons_fine,
            SWDOWN_1km,
            interp_method,
        )

        proj_RAD_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            RAD_d01,
            lats_fine,
            lons_fine,
            RAD_1km,
            interp_method,
        )
        proj_Tscale_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            Tscale_d01,
            lats_fine,
            lons_fine,
            Tscale_1km,
            interp_method,
        )

        proj_Wscale_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            Wscale_d01,
            lats_fine,
            lons_fine,
            Wscale_1km,
            interp_method,
        )

        proj_Pscale_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            Pscale_d01,
            lats_fine,
            lons_fine,
            Pscale_1km,
            interp_method,
        )
        proj_EVI_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            EVI_d01,
            lats_fine,
            lons_fine,
            EVI_1km,
            interp_method,
        )
        proj_Lambda_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            Lambda_d01,
            lats_fine,
            lons_fine,
            Lambda_1km,
            interp_method,
        )
        proj_GPP_validate_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            GPP_validate_d01,
            lats_fine,
            lons_fine,
            GPP_validate_1km,
            interp_method,
        )
        proj_GPP_WRF_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            GPP_WRF_d01,
            lats_fine,
            lons_fine,
            GPP_WRF_1km,
            interp_method,
        )

        # DEFAULT-config reprojection (9d): only the parameter-dependent
        # drivers need their own reprojection -- Wscale/Pscale/EVI are reused
        # from the ALPS pass (proj_Wscale_d01/proj_Pscale_d01/proj_EVI_d01,
        # already confirmed identical above).
        proj_RAD_d01_default = proj_on_finer_WRF_grid(
            lats_d01, lons_d01, od_default["RAD"],
            lats_fine, lons_fine, o1_default["RAD"], interp_method,
        )
        proj_Tscale_d01_default = proj_on_finer_WRF_grid(
            lats_d01, lons_d01, od_default["Tscale"],
            lats_fine, lons_fine, o1_default["Tscale"], interp_method,
        )
        proj_Lambda_d01_default = proj_on_finer_WRF_grid(
            lats_d01, lons_d01, od_default["Lambda"],
            lats_fine, lons_fine, o1_default["Lambda"], interp_method,
        )

        # --- Apply masks ---
        gpp_mask = ~np.isnan(GPP_WRF_1km) & ~np.isnan(GPP_validate_1km)
        common_mask = ~(proj_landmask_d01.astype(bool) & stdh_mask & gpp_mask)
        all_fields = [
            RAD_1km,
            Tscale_1km,
            Wscale_1km,
            Pscale_1km,
            EVI_1km,
            GPP_WRF_1km,
            GPP_validate_1km,
            proj_RAD_d01,
            proj_Tscale_d01,
            proj_Wscale_d01,
            proj_Pscale_d01,
            proj_EVI_d01,
            proj_GPP_validate_d01,
            proj_GPP_WRF_d01,
        ]  # Residual_1km

        for arr in all_fields:
            arr[common_mask] = np.nan

        all_fields_default = [
            o1_default["RAD"], o1_default["Tscale"], o1_default["Lambda"],
            proj_RAD_d01_default, proj_Tscale_d01_default, proj_Lambda_d01_default,
        ]
        for arr in all_fields_default:
            arr[common_mask] = np.nan

        # GPP_validate uses the ALPS GPP-based parameters (vprm_params_newGPP_V24.csv),
        # while GPP_WRF comes from EBIO_GEE (online WRF run with as-run DEFAULT parameters).
        # A non-zero GPP_diff is therefore expected: it reflects the parameter change
        # (higher ALPS lambda and Topt closer to Alpine site conditions → more GPP).
        # The perturbation decomposition itself only uses GPP_validate, so GPP_diff
        # does not affect the figure — it is printed as a diagnostic only.
        GPP_diff_1km = GPP_validate_1km - GPP_WRF_1km
        GPP_diff_d01 = proj_GPP_validate_d01 - proj_GPP_WRF_d01

        if print_output:
            print("GPP_validate_1km: ", np.nanmean(GPP_validate_1km))
            print("GPP_WRF_1km: ", np.nanmean(GPP_WRF_1km))
            print("GPP_diff: ", np.nanmean(GPP_diff_1km))
            print("GPP_validate_1km NaN count: ", np.isnan(GPP_validate_1km).sum())
            print("GPP_WRF_1km NaN count: ", np.isnan(GPP_WRF_1km).sum())

            print("GPP_validate_d01: ", np.nanmean(proj_GPP_validate_d01))
            print("GPP_WRF_d01: ", np.nanmean(proj_GPP_WRF_d01))
            print("GPP_diff_d01: ", np.nanmean(GPP_diff_d01))
            print("GPP_validate_d01 NaN count: ", np.isnan(proj_GPP_validate_d01).sum())
            print("GPP_WRF_d01 NaN count: ", np.isnan(proj_GPP_WRF_d01).sum())

        dGPP = proj_GPP_WRF_d01 - GPP_WRF_1km
        dGPP_validate = proj_GPP_validate_d01 - GPP_validate_1km
        dRAD = proj_RAD_d01 - RAD_1km
        dTscale = proj_Tscale_d01 - Tscale_1km
        dWscale = proj_Wscale_d01 - Wscale_1km
        dPscale = proj_Pscale_d01 - Pscale_1km
        dEVI = proj_EVI_d01 - EVI_1km
        dLambda = proj_Lambda_d01 - Lambda_1km
        dRAD_mean = np.nanmean(dRAD)
        dTscale_mean = np.nanmean(dTscale)
        dWscale_mean = np.nanmean(dWscale)
        dPscale_mean = np.nanmean(dPscale)
        dEVI_mean = np.nanmean(dEVI)
        dLambda_mean = np.nanmean(dLambda)
        dGPP_mean = np.nanmean(dGPP)
        dGPP_validate_mean = np.nanmean(dGPP_validate)

        # --- linear_perturbation_analysis ---
        alphas, contribs, residual = linear_perturbation_analysis(
            GPP_validate_1km,
            proj_GPP_validate_d01,
            Lambda_1km,
            proj_Lambda_d01,
            Tscale_1km,
            proj_Tscale_d01,
            Wscale_1km,
            proj_Wscale_d01,
            Pscale_1km,
            proj_Pscale_d01,
            RAD_1km,
            proj_RAD_d01,
            EVI_1km,
            proj_EVI_d01,
            regularize=True,
            alpha=1e-6,
        )
        # Then alphas gives your sensitivities; contribs[0] is ΔGPP_Lambda, contribs[1] is ΔGPP_T, etc.; residual is the unexplained fraction.
        driver_names = ["dlambda", "dTscale", "dWscale", "dPscale", "dRAD", "dEVI"]
        driver_names_plot = [
            r"$\overline{Y_{\lambda}}$",
            r"$\overline{Y_{\text{T}_\text{scale}}}$",
            r"$\overline{Y_{\text{W}_\text{scale}}}$",
            r"$\overline{Y_{\text{P}_\text{scale}}}$",
            r"$\overline{Y_{\text{RAD}}}$",
            r"$\overline{Y_{\text{EVI}}}$",
        ]

        mean_contribs = np.nanmean(contribs, axis=(1, 2))  # shape (6,)
        mean_residual = np.nanmean(residual)
        lin_pert_mean_diffs_df.loc[time] = [
            mean_contribs[0],
            mean_contribs[1],
            mean_contribs[2],
            mean_contribs[3],
            mean_contribs[4],
            mean_contribs[5],
            mean_residual,
        ]

        # --- DEFAULT-config regression (9d): dG from EBIO_GEE (GPP_WRF, the
        # as-run online flux) instead of the offline ALPS recalc, drivers for
        # Lambda/Tscale/RAD rebuilt with DEFAULT_MEMBER (vprm_params_alps_asrun.csv);
        # Wscale/Pscale/EVI reused from the ALPS pass since they carry no
        # parameter dependence (confirmed above). driver order is fixed
        # ([Lambda, T, W, P, R, E]) so mean_contribs_default[2]=Y_Wscale,
        # [3]=Y_Pscale, [5]=Y_EVI, matching the ALPS-pass indexing.
        _alphas_default, _contribs_default, _residual_default = linear_perturbation_analysis(
            GPP_WRF_1km, proj_GPP_WRF_d01,
            o1_default["Lambda"], proj_Lambda_d01_default,
            o1_default["Tscale"], proj_Tscale_d01_default,
            Wscale_1km, proj_Wscale_d01,
            Pscale_1km, proj_Pscale_d01,
            o1_default["RAD"], proj_RAD_d01_default,
            EVI_1km, proj_EVI_d01,
            regularize=True,
            alpha=1e-6,
        )
        mean_contribs_default = np.nanmean(_contribs_default, axis=(1, 2))
        _Y_SIGNCHECK_RECORDS.append({
            "config": "ALPS", "res": dx, "time": time,
            "Y_Wscale": mean_contribs[2], "Y_Pscale": mean_contribs[3], "Y_EVI": mean_contribs[5],
        })
        _Y_SIGNCHECK_RECORDS.append({
            "config": "DEFAULT", "res": dx, "time": time,
            "Y_Wscale": mean_contribs_default[2], "Y_Pscale": mean_contribs_default[3],
            "Y_EVI": mean_contribs_default[5],
        })

        # --- spread ensemble: contributions for the non-central members (band) ---
        # reuses the central common_mask so the domain is identical across members
        for mi, (f1o, fdo) in enumerate(raw_members):
            if mi == CENTRAL_IDX:
                continue
            pj = {k: project_nn(fdo[k], proj_idx, fine_shape) for k in fdo}
            for _arr in (f1o["RAD"], f1o["Tscale"], f1o["Wscale"], f1o["Pscale"],
                         f1o["EVI"], f1o["GPP"], f1o["Lambda"],
                         pj["RAD"], pj["Tscale"], pj["Wscale"], pj["Pscale"],
                         pj["EVI"], pj["GPP"], pj["Lambda"]):
                _arr[common_mask] = np.nan
            _a2, _c2, _r2 = linear_perturbation_analysis(
                f1o["GPP"], pj["GPP"], f1o["Lambda"], pj["Lambda"],
                f1o["Tscale"], pj["Tscale"], f1o["Wscale"], pj["Wscale"],
                f1o["Pscale"], pj["Pscale"], f1o["RAD"], pj["RAD"],
                f1o["EVI"], pj["EVI"], regularize=True, alpha=1e-6,
            )
            _mc2 = np.nanmean(_c2, axis=(1, 2))
            spread_dfs[mi].loc[time] = [
                _mc2[0], _mc2[1], _mc2[2], _mc2[3], _mc2[4], _mc2[5],
                np.nanmean(_r2),
            ]

        lin_pert_mean = mean_contribs.sum() + mean_residual
        mean_diffs_df.loc[time] = [
            dLambda_mean,
            dTscale_mean,
            dWscale_mean,
            dPscale_mean,
            dRAD_mean,
            dEVI_mean,
            # dGPP_mean,
            # dGPP_validate_mean,
            # lin_pert_mean,
        ]

        if print_output:
            for name, val in zip(driver_names, mean_contribs):
                print(f"{name}: {val:.3f} [μmol/m²/s]")
            print(f"Residual: {mean_residual:.3f} [μmol/m²/s]")

        if save_plot_maps:
            plot_lin_pert_results(contribs, residual, driver_names)

            styled_imshow_plot(
                GPP_validate_1km,
                np.nanmin(GPP_validate_1km),
                np.nanmax(GPP_validate_1km),
                "YlOrRd",
                r"[-]",
                "GPP_validate_1km",
            )
            styled_imshow_plot(
                GPP_WRF_1km,
                np.nanmin(GPP_WRF_1km),
                np.nanmax(GPP_WRF_1km),
                "YlOrRd",
                r"[-]",
                "GPP_WRF_1km",
            )
            styled_imshow_plot(
                Tscale_1km,
                np.nanmin(Tscale_1km),
                np.nanmax(Tscale_1km),
                "YlOrRd",
                r"T$_{scale}$ [-]",
                "Tscale",
            )
            styled_imshow_plot(
                Wscale_1km,
                np.nanmin(Wscale_1km),
                np.nanmax(Wscale_1km),
                "YlOrRd",
                r"W$_{scale}$ [-]",
                "Wscale",
            )
            styled_imshow_plot(
                Pscale_1km,
                np.nanmin(Pscale_1km),
                np.nanmax(Pscale_1km),
                "YlOrRd",
                r"P$_{scale}$ [-]",
                "Pscale",
            )
            styled_imshow_plot(
                EVI_1km,
                np.nanmin(EVI_1km),
                np.nanmax(EVI_1km),
                "YlOrRd",
                "[-]",
                r"EVI",
            )

            styled_imshow_plot(
                dPscale,
                np.nanmin(dPscale),
                np.nanmax(dPscale),
                "YlOrRd",
                r"P$_{scale}$ [-]",
                "dPscale",
            )
            styled_imshow_plot(
                dEVI,
                np.nanmin(dEVI),
                np.nanmax(dEVI),
                "YlOrRd",
                "[-]",
                "dEVI",
            )
            styled_imshow_plot(
                dWscale,
                np.nanmin(dWscale),
                np.nanmax(dWscale),
                "YlOrRd",
                r"W$_{scale}$ [-]",
                "dWscale",
            )
            styled_imshow_plot(
                dTscale,
                np.nanmin(dTscale),
                np.nanmax(dTscale),
                "YlOrRd",
                r"T$_{scale}$ [-]",
                "dTscale",
            )
            # plot proj_Wscale_d01
            styled_imshow_plot_d01(
                Wscale_d01,
                np.nanmin(Wscale_d01),
                np.nanmax(Wscale_d01),
                "YlOrRd",
                r"W$_{scale}$ [-]",
                "Wscale_d01",
            )

    lin_pert_mean_diffs_df["hour"] = lin_pert_mean_diffs_df.index.hour
    numeric_columns = lin_pert_mean_diffs_df.select_dtypes(include=["number"]).columns
    lin_pert_mean_diffs_df_hour = (
        lin_pert_mean_diffs_df[numeric_columns].groupby("hour").mean()
    )

    # hourly means of every ensemble member (central + spread) -> min/max band.
    # Keyed by MEMBER_TAGS so the p25/p75 pair below cannot be transposed by a
    # change in dict ordering.
    member_hours_by_tag = {MEMBER_TAGS[CENTRAL_IDX]: lin_pert_mean_diffs_df_hour}
    for mi in sorted(spread_dfs):
        _dm = spread_dfs[mi].copy()
        _dm["hour"] = _dm.index.hour
        _nc = _dm.select_dtypes(include=["number"]).columns
        member_hours_by_tag[MEMBER_TAGS[mi]] = _dm[_nc].groupby("hour").mean()
    member_hours = list(member_hours_by_tag.values())          # all 5 -> p10-p90
    _iqr_tags = ["p25", MEMBER_TAGS[CENTRAL_IDX], "p75"]        # inner -> p25-p75

    # Diagnostic dump of the per-member hourly curves behind the bands, so the
    # p10/p25 (and p50/p75) coincidences can be checked numerically rather than
    # read off the rendered PDF. Off unless DUMP_MEMBERS is set.
    if os.getenv("DUMP_MEMBERS"):
        _dump = pd.concat(member_hours_by_tag, names=["tag"])
        _dp = (f"{plots_folder}members_{dx[1:]}-1km{file_label_extra}"
               f"_{date_str_min}_{date_str_max}.csv")
        _dump.to_csv(_dp)
        print(f"  [dump] member curves -> {_dp}")
    member_hours_iqr = [
        member_hours_by_tag[t] for t in _iqr_tags if t in member_hours_by_tag
    ]

    if save_plot2:
        fig, ax = plt.subplots(figsize=(6.7, 3.8))
        _hours = lin_pert_mean_diffs_df_hour.index
        _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for _di, _name in enumerate(driver_names):
            _col = _colors[_di % len(_colors)]
            ax.plot(_hours, lin_pert_mean_diffs_df_hour[_name], marker="o", color=_col)
            if _name in BAND_DRIVERS:
                _stack = np.stack(
                    [h[_name].reindex(_hours).values for h in member_hours], axis=0
                )
                ax.fill_between(_hours, np.nanmin(_stack, axis=0),
                                np.nanmax(_stack, axis=0),
                                color=_col, alpha=0.2, linewidth=0)
                _stack_iqr = np.stack(
                    [h[_name].reindex(_hours).values for h in member_hours_iqr],
                    axis=0,
                )
                ax.fill_between(_hours, np.nanmin(_stack_iqr, axis=0),
                                np.nanmax(_stack_iqr, axis=0),
                                color=_col, alpha=0.4, linewidth=0)
        ax.plot(_hours, lin_pert_mean_diffs_df_hour["Residual"], marker="x",
                linestyle="--", color="k")
        ax.set_ylabel(
            r"contributions $\overline{Y_{x_i}}$ [μmol m$^{-2}$ s$^{-1}$]", fontsize=13
        )
        ax.set_xlabel("UTC [h]", fontsize=16)
        # set xlabels to 1-23h
        # every 3rd hour: 24 labels overlapped into an unreadable strip
        _xt = np.arange(0, len(_hours), 3)
        ax.set_xticks(_xt)
        ax.set_xticklabels([f"{i}" for i in _xt])
        plt.tick_params(labelsize=14)
        if dx == "_54km" and sim_type == "":
            _handles = [Line2D([0], [0], color=_colors[_di % len(_colors)], marker="o")
                        for _di in range(len(driver_names))]
            _handles.append(Line2D([0], [0], color="k", marker="x", linestyle="--"))
            _handles.append(Patch(facecolor="0.5", alpha=0.2))
            _handles.append(Patch(facecolor="0.5", alpha=0.4))
            # Standalone strip (legend_components.pdf) for placement under the
            # 2x2 panel grid: in-axes it covered the curves and overflowed the
            # canvas, and it would have made this one panel taller than the rest.
            _labels = driver_names_plot + [
                "Residual",
                r"p10-p90 (ens.)",
                r"p25-p75 (ens.)",
            ]
            _lf = plt.figure(figsize=(13.6, 1.1))
            _lf.legend(_handles, _labels, loc="center", ncol=5, fontsize=16,
                       frameon=False, handlelength=2.2, columnspacing=1.6)
            plt.axis("off")
            _lf.savefig(f"{OUTFOLDER}legend_components.pdf", bbox_inches="tight")
            plt.close(_lf)
            print("wrote", f"{OUTFOLDER}legend_components.pdf")
            plt.figure(fig.number)  # back to the panel figure
        ax.grid(True)
        # ax.set_ylim(-1.2, 0.6)
        plt.tight_layout()
        # plt.show()

        plt.savefig(
            f"{plots_folder}lin_pert_mean_diffs_{dx[1:]}-1km{file_label_extra}_{date_str_min}_{date_str_max}.pdf",
            dpi=300,
        )
        plt.close()

    print(f"finished {dx}")

    # Sign-check records (9d): dumped after every dx iteration so the file
    # reflects everything accumulated so far; the final dx iteration in
    # dx_all leaves the complete set for this sim_type on disk. Combine with
    # the other sim_type's file (this script only covers one sim_type per
    # invocation) before tallying signs.
    _signcheck_df = pd.DataFrame(_Y_SIGNCHECK_RECORDS)
    _signcheck_path = f"{OUTFOLDER}signcheck_Y_components{sim_type}.csv"
    _signcheck_df.to_csv(_signcheck_path, index=False)
    print(f"  [dump] Y_Wscale/Y_Pscale/Y_EVI sign-check records -> {_signcheck_path} "
          f"({len(_signcheck_df)} rows so far)")
