import numpy as np
import matplotlib

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
plots_folder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/areafluxes_"
save_plot = True
dx = "_54km"
interp_method = "nearest"  # 'linear', 'nearest', 'cubic'
temp_gradient = -6.5  # K/km
STD_TOPO = 200
# Set time
dateime = "2012-07-27_12"  # second plot time "2012-07-27_05"
subfolder = ""  # "" or "_cloudy"
wrfinput_path_1km = f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_1km{subfolder}/wrfout_d02_{dateime}:00:00"
wrfinput_path_54km = f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS{dx}{subfolder}/wrfout_d01_{dateime}:00:00"
t_file_fra = "/scratch/c7071034/DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/out_d02_2012_1km/VPRM_input_VEG_FRA_d02_2012.nc"
t_file_fra_d01 = "/scratch/c7071034/DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/out_d01_2012_54km/VPRM_input_VEG_FRA_d01_2012.nc"
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


# Load the NetCDF file
nc_fid1km = nc.Dataset(wrfinput_path_1km, "r")
nc_fid54km = nc.Dataset(wrfinput_path_54km, "r")
dGPPdT = -nc_fid54km.variables["EBIO_GEE_DPDT"][:]
dGPPdT_1km = -nc_fid1km.variables["EBIO_GEE_DPDT"][0, 0, 10:-10, 10:-10]
GPP_1km = -nc_fid1km.variables["EBIO_GEE"][0, 0, 10:-10, 10:-10]
GPP_54km = -nc_fid54km.variables["EBIO_GEE"][0, 0, :, :]
# add "SWDOWN"
SWDOWN_1km = nc_fid1km.variables["SWDOWN"][0, 10:-10, 10:-10]
SWDOWN_54km = nc_fid54km.variables["SWDOWN"][0]

dRECOdT = nc_fid54km.variables["EBIO_RES_DPDT"][:]
dRECOdT_1km = nc_fid1km.variables["EBIO_RES_DPDT"][0, 0, 10:-10, 10:-10]
HGT_1km = nc_fid1km.variables["HGT"][0, 10:-10, 10:-10]
HGT_54km = nc_fid54km.variables["HGT"][0]
T2_1km = nc_fid1km.variables["T2"][0, 10:-10, 10:-10] - 273.15
T2_54km = nc_fid54km.variables["T2"][0] - 273.15
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
veg_frac_map = ds["vegetation_fraction_map"].isel(
    south_north=slice(10, -10), west_east=slice(10, -10)
)

veg_frac_map_d01 = ds_d01["vegetation_fraction_map"]

# lat = ds["lat"].isel(south_north=slice(1, -1), west_east=slice(1, -1)).values
# lon = ds["lon"].isel(south_north=slice(1, -1), west_east=slice(1, -1)).values
# lat_d01 = ds_d01["lat"].isel(south_north=slice(1, -1), west_east=slice(1, -1)).values
# lon_d01 = ds_d01["lon"].isel(south_north=slice(1, -1), west_east=slice(1, -1)).values


PAR0_of_PFT = {
    "ENF": 316.96,
    "DBF": 310.78,
    "MF": 428.83,
    "SHB": 363.00,
    "WET": 682.00,
    "CRO": 595.86,
    "GRA": 406.28,
    "OTH": 0.00,
}
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
dGPP_calc = dGPPdT_1km * conv_factor * dT_calc
dGPP_model = dGPPdT_1km * conv_factor * dT_model
dGPP_real = (proj_GPP_54km - GPP_1km) * conv_factor
dRECO_model = dRECOdT_1km * conv_factor * dT_model
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
for arr in all_fields:
    arr[proj_landmask_54km * stdh_mask == 0] = np.nan
# mask out fluxes where T is below 5°C
stdh_mask[T2_1km < 5] = False
all_fields = [dT_calc, dT_model, dGPP_calc, dGPP_model, dGPPdT_real, T2_1km]
for arr in all_fields:
    arr[proj_landmask_54km * stdh_mask == 0] = np.nan


CLDFRC_1km_max = np.nanmax(CLDFRC_1km, axis=0)
CLDFRC_54km_max = np.nanmax(proj_CLDFRC_54km, axis=0)


def styled_imshow_plot(data, vmin, vmax, cmap, label, filename):
    fig, ax = plt.subplots(
        figsize=(12, 15), subplot_kw={"projection": ccrs.PlateCarree()}
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
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label(label, fontsize=30)

    gl = ax.gridlines(
        draw_labels=True, linewidth=1.5, color="black", alpha=0.2, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 30}
    gl.ylabel_style = {"size": 30}

    ax.set_xlabel("Longitude", fontsize=30)
    ax.set_ylabel("Latitude", fontsize=30)

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
    dT_model,
    -15,
    15,
    "coolwarm_r",
    r"$\Delta_\text{res}$T$_\text{2m}$ [°C]",
    "dT_model",
)

styled_imshow_plot(
    dRAD_scale,
    np.nanmin(dRAD_scale),
    np.nanmax(dRAD_scale),
    "RdBu",
    r"$\Delta_\text{res}$RAD [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "RAD_scale_54-1km",
)

styled_imshow_plot(
    proj_RAD_scale_54km,
    np.nanmin(0),
    np.nanmax(RAD_scale_1km),
    "YlOrRd",
    r"RAD [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "RAD_scale_54km",
)

styled_imshow_plot(
    RAD_scale_1km,
    np.nanmin(0),
    np.nanmax(RAD_scale_1km),
    "YlOrRd",
    r"RAD [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "RAD_scale_1km",
)

styled_imshow_plot(
    proj_SWDOWN_54km,
    np.nanmin(SWDOWN_1km),
    np.nanmax(SWDOWN_1km),
    "YlOrRd",
    r"S$_\downarrow$ [W m$^{-2}$]",
    "SWDOWN_54km",
)

styled_imshow_plot(
    SWDOWN_1km,
    np.nanmin(SWDOWN_1km),
    np.nanmax(SWDOWN_1km),
    "YlOrRd",
    r"S$_\downarrow$ [W m$^{-2}$]",
    "SWDOWN_1km",
)

styled_imshow_plot(
    dSWDOWN,
    np.nanmin(dSWDOWN),
    np.nanmax(dSWDOWN),
    "RdBu",
    r"$\Delta_\text{res}$ S$_\downarrow$ [W m$^{-2}$]",
    "SWDOWN_54-1km",
)

styled_imshow_plot(
    dGPP_calc,
    -15,
    15,
    "PiYG",
    r"$\Delta_{\partial \text{T}}$GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "dGPP_model_02",
)
# GPP calc again (duplicated in earlier batch, but now renamed to not overwrite)


# dGPP/dT sensitivity
styled_imshow_plot(
    dGPPdT_1km * conv_factor,
    -2,
    2,
    "PiYG",
    r"$\frac{\partial \text{GPP}}{\partial \text{T}}$ ([$\mu$mol m$^{-2}$ s$^{-1}$ °C$^{-1}$]",
    "dGPPdT_1km",
)


# CLDFRC_max
styled_imshow_plot(
    CLDFRC_1km_max,
    np.nanmin(CLDFRC_1km_max),
    np.nanmax(CLDFRC_1km_max),
    "Blues",
    "cloud fraction [%]",
    "CLDFRC_1km",
)
styled_imshow_plot(
    CLDFRC_54km_max,
    np.nanmin(CLDFRC_54km_max),
    np.nanmax(CLDFRC_54km_max),
    "Blues",
    "cloud fraction [%]",
    "CLDFRC_54km",
)

# Temperature
styled_imshow_plot(T2_1km, 0, 35, "coolwarm_r", "[°C]", "T2_1km")
styled_imshow_plot(proj_T2_54km, 0, 35, "coolwarm_r", "[°C]", "T2_54km")


# GPP 1km
styled_imshow_plot(
    GPP_1km * conv_factor,
    0,
    30,
    "PiYG",
    r"GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "GPP_1km",
)

# GPP 54km (reprojected)
styled_imshow_plot(
    proj_GPP_54km * conv_factor,
    0,
    30,
    "PiYG",
    r"GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "GPP_54",
)

# GPP model diff (54km - 1km)
styled_imshow_plot(
    dGPP_real,
    -15,
    15,
    "PiYG",
    r"$\Delta_\text{res}$GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "GPP_54-1km",
)


styled_imshow_plot(
    proj_dGPPdT_54km * conv_factor,
    -2,
    2,
    "PiYG",
    r"dGPP/dT ([$\mu$mol m$^{-2}$ s$^{-1}$ °C$^{-1}$]",
    "dGPPdT_54km",
)

# Temperature differences
styled_imshow_plot(
    dT_model - dT_calc,
    -15,
    15,
    "coolwarm_r",
    "$\Delta_\text{res}$T [C]",
    "dT_model-calc",
)

# GPP differences
styled_imshow_plot(
    dGPP_calc,
    -15,
    15,
    "PiYG",
    r"$\Delta_\text{res}$GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "dGPP_calc",
)
styled_imshow_plot(
    dGPP_model,
    -15,
    15,
    "PiYG",
    r"$\Delta_\text{res}$GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "dGPP_model",
)
styled_imshow_plot(
    dGPP_model - dGPP_calc,
    -15,
    15,
    "PiYG",
    r"$\Delta_\text{res}$GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "dGPP_model-calc",
)

# RECO difference
styled_imshow_plot(
    dRECO_model,
    -15,
    15,
    "PiYG",
    r"$\Delta_\text{res}$RECO [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "dRECO_model",
)
print("Plots done.")
