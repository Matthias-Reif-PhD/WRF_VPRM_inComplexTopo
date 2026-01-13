import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import binary_erosion, distance_transform_edt
import xarray as xr
import pandas as pd
import os
import glob
from datetime import datetime, timedelta
from collections import defaultdict

############# INPUT ############
save_plot_maps = False
save_plot2 = True
print_output = False
# date = "2012-07-27"  # "2012-06-28"
start_date = "2012-01-01 00:00:00"
end_date = "2012-12-31 00:00:00"
wrf_basepath = "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS"  # without resolution suffix
dx_all = ["_54km", "_9km"]  # "_54km",
sim_type = "_cloudy"  # "", "_cloudy"
plots_folder = (
    f"/home/c707/c7071034/Github/WRF_VPRM_post/plots/components{sim_type}_L2_"
)
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

    fig, axes = plt.subplots(2, 4, figsize=(16, 5))
    axes = axes.flatten()

    # Plot each driver contribution
    vmin = np.nanmin(contribs_grid)
    vmax = np.nanmax(contribs_grid)
    for i in range(n_drivers):
        im = axes[i].imshow(
            contribs_grid[i], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        axes[i].set_title(driver_names[i])
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Plot residual
    im = axes[n_drivers].imshow(
        residual, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    axes[n_drivers].set_title("Residual")
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
        figsize=(12, 15), subplot_kw={"projection": ccrs.PlateCarree()}
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
    cbar.set_label(label, fontsize=20)

    gl = ax.gridlines(
        draw_labels=True, linewidth=1.5, color="black", alpha=0.2, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 20}
    gl.ylabel_style = {"size": 20}

    ax.set_xlabel("Longitude", fontsize=20)
    ax.set_ylabel("Latitude", fontsize=20)
    plt.tick_params(labelsize=18)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)

    plt.tight_layout()

    plt.savefig(f"{plots_folder}{filename}_{date_time}h.pdf", bbox_inches="tight")
    plt.close()


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
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label, fontsize=20)

    gl = ax.gridlines(
        draw_labels=True, linewidth=1.5, color="black", alpha=0.2, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 20}
    gl.ylabel_style = {"size": 20}

    ax.set_xlabel("Longitude", fontsize=20)
    ax.set_ylabel("Latitude", fontsize=20)
    plt.tick_params(labelsize=18)
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
records = []
for dx in dx_all:
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

    # Collect all files
    files_d01 = sorted(
        glob.glob(os.path.join(wrf_basepath + dx + sim_type, f"wrfout_d01*"))
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
        wrfinput_path_d01 = f"{wrf_basepath}{dx}{sim_type}/wrfout_d01_{date_time}:00:00"
        vprm_input_path_1km = f"/scratch/c7071034/DATA/VPRM_input/vprm_corine_1km/vprm_input_d02_{date}_00:00:00.nc"
        vprm_input_path_d01 = f"/scratch/c7071034/DATA/VPRM_input/vprm_corine{dx}/vprm_input_d01_{date}_00:00:00.nc"

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

        # --- define parameters ---

        RAD0_of_PFT = {
            "ENF": 207.685,
            "DBF": 183.799,
            "MF": 240.386,
            "SHB": 363.00,
            "SAV": 682.00,
            "CRO": 364.145,
            "GRA": 284.875,
            "OTH": 0.00,
        }
        lambda_of_PFT = {
            "ENF": 0.467,
            "DBF": 0.361,
            "MF": 0.248,
            "SHB": 0.087,
            "SAV": 0.114,
            "CRO": 0.230,
            "GRA": 0.771,
            "OTH": 0.00,
        }
        if sim_type == "_pram_err":
            PAR0_of_PFT = {
                "ENF": 316.96,
                "DBF": 310.78,
                "MF": 428.83,
                "SHB": 363.00,
                "SAV": 682.00,
                "CRO": 595.86,
                "GRA": 406.28,
                "OTH": 0.00,
            }
            lambda_of_PFT = {
                "ENF": 0.304,
                "DBF": 0.216,
                "MF": 0.114,
                "SHB": 0.0874,
                "SAV": 0.114,
                "CRO": 0.140,
                "GRA": 0.448,
                "OTH": 0.00,
            }
        Tvar_of_PFT = {
            "ENF": (14.250, 0, 40),
            "DBF": (23.580, 0, 40),
            "MF": (17.440, 0, 40),
            "SHB": (20.000, 0, 40),
            "SAV": (20.000, 0, 40),
            "CRO": (22.000, 0, 40),
            "GRA": (15.880, 0, 40),
            "OTH": (0.000, 0, 40),
        }

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

    if save_plot2:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            lin_pert_mean_diffs_df_hour.index,
            lin_pert_mean_diffs_df_hour[driver_names],
            marker="o",
        )
        ax.plot(
            lin_pert_mean_diffs_df_hour.index,
            lin_pert_mean_diffs_df_hour["Residual"],
            marker="x",
            linestyle="--",
            color="k",
            label="Residual",
        )
        ax.set_ylabel(
            r"contributions $\overline{Y_{x_i}}$ [μmol m$^{-2}$ s$^{-1}$]", fontsize=20
        )
        ax.set_xlabel("UTC [h]", fontsize=20)
        # set xlabels to 1-23h
        ax.set_xticks(np.arange(len(lin_pert_mean_diffs_df_hour.index)))
        ax.set_xticklabels(
            [f"{i}" for i in range(len(lin_pert_mean_diffs_df_hour.index))]
        )
        plt.tick_params(labelsize=18)
        if dx == "_54km" and sim_type == "":
            ax.legend(
                driver_names_plot + ["Residual"],
                loc="upper left",
                fontsize=20,
                frameon=True,
                framealpha=0.4,
            )
        ax.grid(True)
        # ax.set_ylim(-1.2, 0.6)
        plt.tight_layout()
        # plt.show()

        plt.savefig(
            f"{plots_folder}lin_pert_mean_diffs_{dx[1:]}-1km_{date_str_min}_{date_str_max}.pdf",
            dpi=300,
        )
        plt.close()

    print(f"finished {dx}")
