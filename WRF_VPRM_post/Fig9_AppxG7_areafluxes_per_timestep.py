"""
Fig9_AppxG7_areafluxes_per_timestep.py
==============================================
Fig. 9 (fig:dGPPdT2, 12:00 UTC case) and Fig. G7 (fig:dGPPdT_swdown0508, 05:00
UTC case).
Plots area-averaged flux differences per timestep between WRF-VPRM 1km and 54km simulations.
- Interpolates 54km data onto 1km grid
- Calculates differences in GPP, RECO, SWDOWN, and RAD scaling
- Applies coastal and topographic masks
- Generates styled imshow plots for each variable
==============================================
"""

import sys
import numpy as np
import matplotlib
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ==================== Configuration ====================
SCRATCH_PATH = os.getenv("SCRATCH_PATH")
GITHUB_PATH = os.getenv("GITHUB_PATH")
OUTFOLDER = os.getenv("OUTFOLDER")

sys.path.insert(0, str(Path(__file__).parent))
from recompute_vprm_fluxes import load_params, compute_fluxes, _load_vprm_input  # noqa: E402

params = load_params(str(Path(__file__).parent / "vprm_params_newGPP_V24.csv"))

# use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import binary_erosion, distance_transform_edt
import xarray as xr

############# INPUT ############
plots_folder = os.path.join(OUTFOLDER, "areafluxes_")
save_plot = True
dx = "_54km"
wrf_tag_d01 = "_radt54_swint1"  # extra tag for the coarse wrfout path ("" for standard/cloudy)
interp_method = "nearest"  # 'linear', 'nearest', 'cubic'
temp_gradient = -6.5  # K/km
STD_TOPO = 200
# Set time
dateime = os.getenv("FIG68_DATE", "2012-07-27_12")  # e.g. "2012-07-27_05" for the radiation snapshot
subfolder = ""  # "" or "_cloudy"
wrfinput_path_1km = os.path.join(
    SCRATCH_PATH, f"DATA/WRFOUT/WRFOUT_ALPS_1km{subfolder}/wrfout_d02_{dateime}:00:00"
)
wrfinput_path_54km = os.path.join(
    SCRATCH_PATH, f"DATA/WRFOUT/WRFOUT_ALPS{dx}{wrf_tag_d01}{subfolder}/wrfout_d01_{dateime}:00:00"
)
t_file_fra = os.path.join(
    SCRATCH_PATH,
    "DATA/VPRM_input/vprm_corine_1km/vprm_input_d02_2012-07-27_00:00:00.nc",
)
t_file_fra_d01 = os.path.join(
    SCRATCH_PATH,
    "DATA/VPRM_input/vprm_corine_54km/vprm_input_d01_2012-07-27_00:00:00.nc",
)
################################


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


def proj_on_finer_WRF_grid_3D(
    lats_coarse: np.ndarray,
    lons_coarse: np.ndarray,
    var_coarse: np.ndarray,
    lats_fine: np.ndarray,
    lons_fine: np.ndarray,
    WRF_var_1km: np.ndarray,
    method_in: str,
) -> np.ndarray:
    """
    Interpolates 3D WRF data (z, y, x) from coarse grid to finer 1km WRF grid using cubic interpolation.
    """
    z_levels = var_coarse.shape[0]
    interp_shape = WRF_var_1km.shape
    proj_var = np.empty(interp_shape, dtype=np.float32)

    for z in range(z_levels):
        proj_var[z] = griddata(
            (lats_coarse.flatten(), lons_coarse.flatten()),
            var_coarse[z].flatten(),
            (lats_fine, lons_fine),
            method=method_in,
        ).reshape(
            interp_shape[1:]
        )  # (y_fine, x_fine)

    return proj_var


def _compute_dgpp_dt(t2c, swdown, vin, p):
    """d(GPP)/dT summed over PFTs using the VPRM Tscale quotient-rule derivative."""
    out = np.zeros_like(t2c, dtype=float)
    for m in range(len(p["pft"])):
        if p["par0"][m] == 0:
            continue
        t_opt, t_min, t_max = p["t_opt"][m], p["t_min"][m], p["t_max"][m]
        a1 = t2c - t_min
        a2 = t2c - t_max
        a3 = t2c - t_opt
        denom = a1 * a2 - a3 ** 2
        inside = (a1 >= 0) & (a2 <= 0) & (denom != 0)
        num = (2 * t2c - t_min - t_max) * denom - a1 * a2 * (2 * t_opt - t_min - t_max)
        dTscale_dT = np.where(inside, num / denom ** 2, 0.0)
        lswi     = vin["lswi"][0, m]
        lswi_max = vin["lswi_max"][0, m]
        lswi_min = vin["lswi_min"][0, m]
        evi      = vin["evi"][0, m]
        evi_min  = vin["evi_min"][0, m]
        evi_max  = vin["evi_max"][0, m]
        vegfra   = vin["vegfra"][0, m]
        if m in (3, 6):
            num_w = lswi - lswi_min
            den_w = lswi_max - lswi_min
            wscale = np.where(den_w >= 1e-7, num_w / den_w, 0.0)
        else:
            wscale = (1.0 + lswi) / (1.0 + lswi_max)
        # zero-fraction pixels have lswi=lswi_max=-1 (fill) -> 0/0 = NaN; set to 0
        # so NaN*vegfra(0) does not poison the accumulated sum (as compute_fluxes does)
        wscale = np.nan_to_num(np.clip(wscale, 0.0, 1.0), nan=0.0)
        if m == 0:
            pscale = np.ones_like(t2c)
        elif m in (4, 6):
            pscale = (1.0 + lswi) / 2.0
        else:
            evithresh = evi_min + 0.55 * (evi_max - evi_min)
            pscale = np.where(evi >= evithresh, 1.0, (1.0 + lswi) / 2.0)
        pscale = np.clip(np.nan_to_num(pscale, nan=0.0), 0.0, 1.0)
        rad_scale = swdown / (1.0 + swdown / p["par0"][m])
        out += p["lambda"][m] * dTscale_dT * wscale * pscale * rad_scale * evi * vegfra
    return out


# Load the NetCDF file
nc_fid1km = nc.Dataset(wrfinput_path_1km, "r")
nc_fid54km = nc.Dataset(wrfinput_path_54km, "r")
SWDOWN_1km = nc_fid1km.variables["SWDOWN"][0, 10:-10, 10:-10]
SWDOWN_54km = nc_fid54km.variables["SWDOWN"][0]
T2_1km = nc_fid1km.variables["T2"][0, 10:-10, 10:-10] - 273.15
T2_54km = nc_fid54km.variables["T2"][0] - 273.15
dRECOdT = nc_fid54km.variables["EBIO_RES_DPDT"][:]
dRECOdT_1km = nc_fid1km.variables["EBIO_RES_DPDT"][0, 0, 10:-10, 10:-10]
HGT_1km = nc_fid1km.variables["HGT"][0, 10:-10, 10:-10]
HGT_54km = nc_fid54km.variables["HGT"][0]
# compute GPP and dGPP/dT analytically using GPP-based ALPS parameters
_vin_1km = _load_vprm_input(t_file_fra, trim=True)
_vin_d01 = _load_vprm_input(t_file_fra_d01, trim=False)
GPP_1km  = compute_fluxes(SWDOWN_1km, T2_1km, _vin_1km, params)[0]
GPP_54km = compute_fluxes(SWDOWN_54km, T2_54km, _vin_d01, params)[0]
dGPPdT_1km = _compute_dgpp_dt(T2_1km, SWDOWN_1km, _vin_1km, params)
dGPPdT     = _compute_dgpp_dt(T2_54km, SWDOWN_54km, _vin_d01, params)
print("DEBUG AFTER compute: dGPPdT_1km shape={}, nonzero={}, min={:.4f}, max={:.4f}".format(
    dGPPdT_1km.shape, int(np.sum(dGPPdT_1km != 0)), float(np.nanmin(dGPPdT_1km)), float(np.nanmax(dGPPdT_1km))))
print("DEBUG T2_1km shape={}, SWDOWN_1km shape={}, _vin_1km vegfra shape={}".format(
    T2_1km.shape, SWDOWN_1km.shape, _vin_1km["vegfra"].shape))
lats_fine = nc_fid1km.variables["XLAT"][0, 10:-10, 10:-10]
lons_fine = nc_fid1km.variables["XLONG"][0, 10:-10, 10:-10]
# landmask = nc_fid1km.variables["LANDMASK"][0, :, :]
veg_type = nc_fid54km.variables["IVGTYP"][0, :, :]
stdh_topo_1km = nc_fid1km.variables["VAR"][0, 10:-10, 10:-10]
stdh_mask = stdh_topo_1km >= STD_TOPO

lats_54km = nc_fid54km.variables["XLAT"][0, :, :]
lons_54km = nc_fid54km.variables["XLONG"][0, :, :]
CLDFRC_1km = nc_fid1km.variables["CLDFRA"][0, :, 10:-10, 10:-10]
CLDFRC_54km = nc_fid54km.variables["CLDFRA"][0, :, :, :]

# --- Load vegetation fraction map ---
ds = xr.open_dataset(t_file_fra)
ds_d01 = xr.open_dataset(t_file_fra_d01)
veg_frac_map = ds["VEGFRA_VPRM"].isel(
    Time=0, south_north=slice(10, -10), west_east=slice(10, -10)
)
veg_frac_map_d01 = ds_d01["VEGFRA_VPRM"].isel(Time=0)


PAR0_of_PFT = {pft: float(params["par0"][m]) for m, pft in enumerate(params["pft"])}
PAR0_of_PFT["OTH"] = 0.0
SWDOWN_TO_PAR = 1

# GPP_54km
proj_GPP_54km = proj_on_finer_WRF_grid(
    lats_54km,
    lons_54km,
    GPP_54km,
    lats_fine,
    lons_fine,
    GPP_1km,
    interp_method,
)
new_landmask = generate_coastal_mask(veg_type, buffer_km=30.0, grid_spacing_km=50.0)


proj_landmask_54km = proj_on_finer_WRF_grid(
    lats_54km,
    lons_54km,
    new_landmask,
    lats_fine,
    lons_fine,
    HGT_1km,
    interp_method,
)
proj_HGT_54km = proj_on_finer_WRF_grid(
    lats_54km,
    lons_54km,
    HGT_54km,
    lats_fine,
    lons_fine,
    HGT_1km,
    interp_method,
)
proj_T2_54km = proj_on_finer_WRF_grid(
    lats_54km,
    lons_54km,
    T2_54km,
    lats_fine,
    lons_fine,
    T2_1km,
    interp_method,
)
proj_dGPPdT_54km = proj_on_finer_WRF_grid(
    lats_54km,
    lons_54km,
    dGPPdT,
    lats_fine,
    lons_fine,
    HGT_1km,
    interp_method,
)
proj_CLDFRC_54km = proj_on_finer_WRF_grid_3D(
    lats_54km,
    lons_54km,
    CLDFRC_54km,
    lats_fine,
    lons_fine,
    CLDFRC_1km,
    interp_method,
)
# dRECOdT
proj_dRECOdT_54km = proj_on_finer_WRF_grid(
    lats_54km,
    lons_54km,
    dRECOdT,
    lats_fine,
    lons_fine,
    dRECOdT_1km,
    interp_method,
)

# add "SWDOWN"
proj_SWDOWN_54km = proj_on_finer_WRF_grid(
    lats_54km,
    lons_54km,
    SWDOWN_54km,
    lats_fine,
    lons_fine,
    SWDOWN_1km,
    interp_method,
)
# RAD_scale_54km
RAD_scale_54km = np.zeros_like(SWDOWN_54km)
for idx, pft in enumerate(PAR0_of_PFT.keys()):
    PAR0 = PAR0_of_PFT[pft]
    if PAR0 > 0:
        vegfrac = veg_frac_map_d01[idx, :, :].values
        RAD_scale_54km += (
            (1 / (1 + (SWDOWN_54km * SWDOWN_TO_PAR) / PAR0))
            * SWDOWN_54km
            * SWDOWN_TO_PAR
        ) * vegfrac

proj_RAD_scale_54km = proj_on_finer_WRF_grid(
    lats_54km,
    lons_54km,
    RAD_scale_54km,
    lats_fine,
    lons_fine,
    SWDOWN_1km,
    interp_method,
)

diff_HGT = proj_HGT_54km - HGT_1km
diff_HGT[proj_landmask_54km * stdh_mask == 0] = np.nan
conv_factor = 1 / 3600

# limit value for max dGPPdT between 0-5°, below 0 its set to nan
val_at5C = 1
dGPPdT_1km[T2_1km < 0] = np.nan
mask_0to5 = (T2_1km >= 0) & (T2_1km <= 5)
dGPPdT_1km[mask_0to5] = val_at5C

dT_calc = diff_HGT / 1000 * temp_gradient
dT_model = proj_T2_54km - T2_1km  # TODO why converting sign?
# No conv_factor for dGPPdT_1km: _compute_dgpp_dt returns [umol m-2 s-1 K-1],
# same units as compute_fluxes output, already per-second.
# conv_factor (1/3600) is only needed for WRF-stored EBIO_* variables (hourly units).
dGPP_calc = dGPPdT_1km * dT_calc
dGPP_model = dGPPdT_1km * dT_model
dGPP_real = proj_GPP_54km - GPP_1km  # both from compute_fluxes → [umol m-2 s-1]
dRECO_model = dRECOdT_1km * conv_factor * dT_model  # EBIO_RES_DPDT is in WRF hourly units
# Avoid division by very small values by masking or thresholding dT_model
dT_threshold = 1.3  # K, or set to a value appropriate for your data
safe_dT_model = np.where(np.abs(dT_model) > dT_threshold, dT_model, np.nan)
dGPPdT_real = dGPP_real / safe_dT_model
dSWDOWN = proj_SWDOWN_54km - SWDOWN_1km


PAR0 = 400
RAD_scale_1km_test = (
    (1 / (1 + (SWDOWN_1km * SWDOWN_TO_PAR) / PAR0)) * SWDOWN_1km * SWDOWN_TO_PAR
)

RAD_scale_1km = np.zeros_like(SWDOWN_1km)
for idx, pft in enumerate(PAR0_of_PFT.keys()):
    PAR0 = PAR0_of_PFT[pft]
    if PAR0 > 0:
        vegfrac = veg_frac_map[idx, :, :].values
        RAD_scale_1km += (
            (1 / (1 + (SWDOWN_1km * SWDOWN_TO_PAR) / PAR0)) * SWDOWN_1km * SWDOWN_TO_PAR
        ) * vegfrac
mask_idx8_100 = veg_frac_map[7, :, :].values < 1.0
dRAD_scale = proj_RAD_scale_54km - RAD_scale_1km
RAD_scale_1km[proj_landmask_54km * stdh_mask * mask_idx8_100 == 0] = np.nan
# # --- Apply masks to all vars ---
all_fields = [
    SWDOWN_1km,
    proj_SWDOWN_54km,
    dSWDOWN,
    proj_RAD_scale_54km,
    dRAD_scale,
    dGPPdT_1km,
    GPP_1km,
    proj_GPP_54km,
    dGPP_real,
    dRECO_model,
]
_mask1 = proj_landmask_54km * stdh_mask
print("DEBUG BEFORE first mask: stdh_mask True={}, proj_landmask nonzero={}, product nonzero={}".format(
    int(np.sum(stdh_mask)), int(np.sum(proj_landmask_54km != 0)), int(np.sum(_mask1 != 0))))
print("DEBUG dGPPdT_1km BEFORE first mask: nonnan={}".format(int(np.sum(~np.isnan(dGPPdT_1km)))))
for arr in all_fields:
    arr[proj_landmask_54km * stdh_mask == 0] = np.nan
print("DEBUG dGPPdT_1km AFTER first mask: nonnan={}".format(int(np.sum(~np.isnan(dGPPdT_1km)))))
# mask out fluxes where T is below 5°C
stdh_mask[T2_1km < 5] = False
all_fields = [dT_calc, dT_model, dGPP_calc, dGPP_model, dGPPdT_real, T2_1km]
for arr in all_fields:
    arr[proj_landmask_54km * stdh_mask == 0] = np.nan


CLDFRC_1km_max = np.nanmax(CLDFRC_1km, axis=0)
CLDFRC_54km_max = np.nanmax(proj_CLDFRC_54km, axis=0)


def styled_imshow_plot(data, vmin, vmax, cmap, label, filename):
    fig, ax = plt.subplots(
        figsize=(4.2, 5.25), subplot_kw={"projection": ccrs.PlateCarree()}
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
    # The colorbar annotation is deliberately the lightest text in the panel: the
    # label is long and vertical, so at the same size as the gridline labels (14)
    # it dominated the map. Drawn at 2x printed size, so 12/11 -> ~6.1/5.6 pt on
    # the page (gridlines ~7.1 pt).
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(label, fontsize=12)

    gl = ax.gridlines(
        draw_labels=True, linewidth=1.5, color="black", alpha=0.2, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}

    ax.set_xlabel("Longitude", fontsize=16)
    ax.set_ylabel("Latitude", fontsize=16)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)

    plt.tight_layout()
    if save_plot:
        plt.savefig(f"{plots_folder}{filename}_{dateime}h.pdf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


styled_imshow_plot(
    proj_SWDOWN_54km,
    np.nanmin(SWDOWN_1km),
    np.nanmax(SWDOWN_1km),
    "YlOrRd",
    r"S$_\downarrow$ [W m$^{-2}$]",
    "SWDOWN_54km" + wrf_tag_d01 + subfolder,
)

styled_imshow_plot(
    SWDOWN_1km,
    np.nanmin(SWDOWN_1km),
    np.nanmax(SWDOWN_1km),
    "YlOrRd",
    r"S$_\downarrow$ [W m$^{-2}$]",
    "SWDOWN_1km" + subfolder,
)

styled_imshow_plot(
    dSWDOWN,
    np.nanmin(dSWDOWN),
    np.nanmax(dSWDOWN),
    "RdBu",
    r"$\Delta_\text{res}$ S$_\downarrow$ [W m$^{-2}$]",
    "SWDOWN_54-1km" + wrf_tag_d01 + subfolder,
)

styled_imshow_plot(
    dT_model,
    -15,
    15,
    "coolwarm_r",
    r"$\Delta_\text{res}$T$_\text{2m}$ [°C]",
    "dT_model" + subfolder,
)

styled_imshow_plot(
    dRAD_scale,
    np.nanmin(dRAD_scale),
    np.nanmax(dRAD_scale),
    "RdBu",
    r"$\Delta_\text{res}$RAD [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "RAD_scale_54-1km" + wrf_tag_d01 + subfolder,
)

styled_imshow_plot(
    proj_RAD_scale_54km,
    np.nanmin(0),
    np.nanmax(RAD_scale_1km),
    "YlOrRd",
    r"RAD [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "RAD_scale_54km" + wrf_tag_d01 + subfolder,
)

styled_imshow_plot(
    RAD_scale_1km,
    np.nanmin(0),
    np.nanmax(RAD_scale_1km),
    "YlOrRd",
    r"RAD [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "RAD_scale_1km" + subfolder,
)


import warnings as _w; _w.filterwarnings('ignore')
print("DEBUG masks: proj_landmask nonzero={}, stdh_mask True={}, product nonzero={}".format(
    int(np.sum(proj_landmask_54km != 0)), int(np.sum(stdh_mask)),
    int(np.sum(proj_landmask_54km * stdh_mask != 0))))
print("DEBUG dGPPdT_1km BEFORE plot: nonnan={}  (should match combined mask)".format(
    int(np.sum(~np.isnan(dGPPdT_1km)))))
print("DEBUG dGPP_calc: min={:.4f} max={:.4f} mean={:.4f} nonnan={}".format(
    np.nanmin(dGPP_calc), np.nanmax(dGPP_calc), np.nanmean(dGPP_calc), int(np.sum(~np.isnan(dGPP_calc)))))
print("DEBUG dGPPdT_1km: min={:.4f} max={:.4f} mean={:.4f} nonnan={}".format(
    np.nanmin(dGPPdT_1km), np.nanmax(dGPPdT_1km), np.nanmean(dGPPdT_1km), int(np.sum(~np.isnan(dGPPdT_1km)))))
print("DEBUG dT_calc: min={:.4f} max={:.4f} mean={:.4f} nonnan={}".format(
    np.nanmin(dT_calc), np.nanmax(dT_calc), np.nanmean(dT_calc), int(np.sum(~np.isnan(dT_calc)))))

styled_imshow_plot(
    dGPP_calc,
    -15,
    15,
    "PiYG",
    r"$\Delta_{\partial \text{T}}$GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "dGPP_model_02" + wrf_tag_d01,
)
# # GPP calc again (duplicated in earlier batch, but now renamed to not overwrite)


# dGPP/dT sensitivity
styled_imshow_plot(
    dGPPdT_1km,
    -2,
    2,
    "PiYG",
    r"$\frac{\partial \text{GPP}}{\partial \text{T}}$ ([$\mu$mol m$^{-2}$ s$^{-1}$ °C$^{-1}$]",
    "dGPPdT_1km" + wrf_tag_d01,
)


# # CLDFRC_max
# styled_imshow_plot(
#     CLDFRC_1km_max,
#     np.nanmin(CLDFRC_1km_max),
#     np.nanmax(CLDFRC_1km_max),
#     "Blues",
#     "cloud fraction [%]",
#     "CLDFRC_1km"+subfolder,
# )
# styled_imshow_plot(
#     CLDFRC_54km_max,
#     np.nanmin(CLDFRC_54km_max),
#     np.nanmax(CLDFRC_54km_max),
#     "Blues",
#     "cloud fraction [%]",
#     "CLDFRC_54km"+subfolder,
# )

# # Temperature
# styled_imshow_plot(T2_1km, 0, 35, "coolwarm_r", "[°C]", "T2_1km"+subfolder)
# styled_imshow_plot(proj_T2_54km, 0, 35, "coolwarm_r", "[°C]", "T2_54km"+subfolder)


# # GPP 1km
# styled_imshow_plot(
#     GPP_1km * conv_factor,
#     0,
#     30,
#     "PiYG",
#     r"GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
#     "GPP_1km"+subfolder,
# )

# # GPP 54km (reprojected)
# styled_imshow_plot(
#     proj_GPP_54km * conv_factor,
#     0,
#     30,
#     "PiYG",
#     r"GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
#     "GPP_54"+subfolder,
# )

# # GPP model diff (54km - 1km)
# styled_imshow_plot(
#     dGPP_real,
#     -15,
#     15,
#     "PiYG",
#     r"$\Delta_\text{res}$GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
#     "GPP_54-1km"+subfolder,
# )


# styled_imshow_plot(
#     proj_dGPPdT_54km * conv_factor,
#     -2,
#     2,
#     "PiYG",
#     r"dGPP/dT ([$\mu$mol m$^{-2}$ s$^{-1}$ °C$^{-1}$]",
#     "dGPPdT_54km"+subfolder,
# )

# # Temperature differences
# styled_imshow_plot(
#     dT_model - dT_calc,
#     -15,
#     15,
#     "coolwarm_r",
#     "$\Delta_\text{res}$T [C]",
#     "dT_model-calc"+subfolder,
# )

# # GPP differences
# styled_imshow_plot(
#     dGPP_calc,
#     -15,
#     15,
#     "PiYG",
#     r"$\Delta_\text{res}$GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
#     "dGPP_calc"+subfolder,
# )
# styled_imshow_plot(
#     dGPP_model,
#     -15,
#     15,
#     "PiYG",
#     r"$\Delta_\text{res}$GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
#     "dGPP_model"+subfolder,
# )
# styled_imshow_plot(
#     dGPP_model - dGPP_calc,
#     -15,
#     15,
#     "PiYG",
#     r"$\Delta_\text{res}$GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
#     "dGPP_model-calc"+subfolder,
# )

# # RECO difference
# styled_imshow_plot(
#     dRECO_model,
#     -15,
#     15,
#     "PiYG",
#     r"$\Delta_\text{res}$RECO [$\mu$mol m$^{-2}$ s$^{-1}$]",
#     "dRECO_model"+subfolder,
# )
# print("Plots done.")
