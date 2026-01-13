#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 11:36:35 2021

@author: madse
"""

import netCDF4 as nc
import glob
import os
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad, lon2_rad = np.radians([lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    # Haversine formula
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def proj_on_finer_WRF_grid(
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_1km
):
    proj_var = griddata(
        (lats_coarse.flatten(), lons_coarse.flatten()),
        var_coarse.flatten(),
        (lats_fine, lons_fine),
        method="linear",
    ).reshape(WRF_var_1km.shape)
    return proj_var


def proj_CAMS_on_WRF_grid(
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_1km
):
    # Corrected meshgrid order
    lats_coarse_2d, lons_coarse_2d = np.meshgrid(lats_coarse, lons_coarse)
    # Reverse the order of latitude coordinates
    lat_CAMS_2d_reversed = lats_coarse_2d[::-1]
    # Reverse the order of the variable values
    var_coarse_reversed = np.flipud(var_coarse)
    # Flatten the coordinates
    points_coarse = np.column_stack(
        (lat_CAMS_2d_reversed.flatten(), lons_coarse_2d.flatten())
    )
    points_fine = np.column_stack((lats_fine.flatten(), lons_fine.flatten()))
    # Perform interpolation
    proj_var = griddata(
        points_coarse, var_coarse_reversed.flatten(), points_fine, method="nearest"
    ).reshape(WRF_var_1km.shape)

    return proj_var


def proj_on_WRF_grid(
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_1km
):
    # Corrected meshgrid order
    lats_coarse_2d, lons_coarse_2d = np.meshgrid(lats_coarse, lons_coarse)
    # Flatten the coordinates
    points_coarse = np.column_stack(
        (lats_coarse_2d.flatten(), lons_coarse_2d.flatten())
    )
    points_fine = np.column_stack((lats_fine.flatten(), lons_fine.flatten()))
    # Perform interpolation
    proj_var = griddata(
        points_coarse, var_coarse.flatten(), points_fine, method="nearest"
    ).reshape(WRF_var_1km.shape)

    return proj_var


def extract_datetime_from_filename(filename):
    """
    Extract datetime from WRF filename assuming format 'wrfout_d0x_YYYY-MM-DD_HH:MM:SS'.
    """
    base_filename = os.path.basename(filename)
    date_str = base_filename.split("_")[-2] + "_" + base_filename.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")


################################# INPUT ##############################################
plot_coeff = True
plotting_scatter = False
plotting_scatter_all = False  # TODO: fix this need to be false currently.
start_date = "2012-01-01 00:00:00"
end_date = "2012-12-31 00:00:00"
# start_date = "2012-06-01 00:00:00"
# end_date = "2012-09-01 00:00:00"
hour_start = 6
hour_end = 17
T_bin_size = 1
T_ref_min = 0
T_ref_max = 31
STD_TOPOs = [200]
STD_TOPO_flags = ["gt"]  # "lt" lower than or "gt" greater than STD_TOPO
ref_sim = "_REF"  # "_REF" to use also REF simulation or "" for only tuned values
subdaily = ""  # "_subdailyC3" or ""
coarse_domains = ["54km", "9km"]  # , "27km", "9km", "3km"
subfolder = ""  # "" or "_cloudy"
cloudy_and_clear = False  # if False they are run reperately
filename_str = subfolder if not cloudy_and_clear else "_clearsky_and_cloudy"
subfolders = [subfolder] if not cloudy_and_clear else ["", "_cloudy"]
wrf_files = "wrfout_d01*"
wrf_files_1km = "wrfout_d02*"
outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"
pmodel_path = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/gpp_pmodel/"
migli_path = "/scratch/c7071034/DATA/RECO_Migli/"

# WRF_vars = ["GPP_pmodel", "RECO_Migli", "NEE_PM", "EBIO_GEE"+ref_sim, "EBIO_RES"+ref_sim, "NEE","T2"]
WRF_vars = [
    "EBIO_GEE",
    "EBIO_RES",
    "NEE",
    "EBIO_GEE" + ref_sim,
    "EBIO_RES" + ref_sim,
    "NEE" + ref_sim,
    "T2",
]
units = [
    " [mmol m² s⁻¹]",
    " [mmol m² s⁻¹]",
    " [mmol m² s⁻¹]",
    " [mmol m² s⁻¹]",
    " [mmol m² s⁻¹]",
    " [mmol m² s⁻¹]",
    " [K]",
]
name_vars = {
    # "GPP_pmodel": "GPP P-Model",
    # "RECO_Migli": "RECO Migliavacca",
    # "NEE_PM": "NEE Migli-P_Model",
    "EBIO_GEE": "WRF GPP",
    "EBIO_RES": "WRF RECO",
    "NEE": "WRF NEE",
    "EBIO_GEE" + ref_sim: "WRF GPP" + ref_sim,
    "EBIO_RES" + ref_sim: "WRF RECO" + ref_sim,
    "NEE" + ref_sim: "WRF NEE" + ref_sim,
    "T2": "WRF T2M",
}
# WRF_factors = [1, 1, 1, -1 / 3600, 1 / 3600, 1 / 3600, 273.15]
WRF_factors = [-1 / 3600, 1 / 3600, 1 / 3600, -1 / 3600, 1 / 3600, 1 / 3600, 273.15]

# Initialize an empty DataFrame with time as the index and locations as columns
start_date_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
end_date_obj = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

############################### start the looop #############################################


for coarse_domain in coarse_domains:
    file_list_1km, file_list_hour_1km = [], []
    file_list, file_list_hour = [], []

    for subfolder in subfolders:
        if coarse_domain == "3km":
            wrf_path_i = f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km{subfolder}"
        elif coarse_domain == "9km":
            wrf_path_i = f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_9km{subfolder}"
        elif coarse_domain == "27km":
            wrf_path_i = f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_27km{subfolder}"
        elif coarse_domain == "54km":
            wrf_path_i = f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km{subfolder}"

        wrf_paths = [
            f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_1km{subfolder}",
            wrf_path_i,
        ]

        files_1km = [
            f
            for f in sorted(glob.glob(os.path.join(wrf_paths[0], wrf_files_1km)))
            if start_date_obj
            <= extract_datetime_from_filename(os.path.basename(f))
            <= end_date_obj
        ]
        file_list_1km.extend(files_1km)

        file_list_hour_1km.extend(
            f
            for f in files_1km
            if hour_start
            <= extract_datetime_from_filename(os.path.basename(f)).hour
            <= hour_end
        )

        files = [
            f
            for f in sorted(glob.glob(os.path.join(wrf_paths[1], wrf_files)))
            if start_date_obj
            <= extract_datetime_from_filename(os.path.basename(f))
            <= end_date_obj
        ]
        file_list.extend(files)

        file_list_hour.extend(
            f
            for f in files
            if hour_start
            <= extract_datetime_from_filename(os.path.basename(f)).hour
            <= hour_end
        )

    # optional: ensure chronological order after merging clear+cloudy
    file_list_hour_1km.sort()
    file_list_hour.sort()

    timestamps = [extract_datetime_from_filename(f) for f in file_list_hour]
    time_index = pd.to_datetime(timestamps)
    diff_to_1km_4D = []
    T2_coarse_domain_4D = []
    # set standard deviation of topography
    for STD_TOPO in STD_TOPOs:

        for STD_TOPO_flag in STD_TOPO_flags:

            for wrf_file, wrf_file_1km in zip(file_list_hour, file_list_hour_1km):
                ini_switch = True
                time = extract_datetime_from_filename(wrf_file)
                print("processing ", time)
                for (
                    WRF_var,
                    unit,
                    WRF_factor,
                ) in zip(
                    WRF_vars,
                    units,
                    WRF_factors,
                ):
                    # WRF
                    i = 0
                    nc_fid1km = nc.Dataset(wrf_file_1km, "r")
                    nc_fid_coarse_domain = nc.Dataset(wrf_file, "r")

                    times_variable = nc_fid1km.variables["Times"]
                    start_date_bytes = times_variable[0, :].tobytes()
                    start_date_str = start_date_bytes.decode("utf-8")
                    lats_fine = nc_fid1km.variables["XLAT"][0, 10:-10, 10:-10]
                    lons_fine = nc_fid1km.variables["XLONG"][0, 10:-10, 10:-10]
                    landmask = nc_fid1km.variables["LANDMASK"][0, 10:-10, 10:-10]
                    hgt_1km = nc_fid1km.variables["HGT"][0, 10:-10, 10:-10]

                    land_mask = landmask == 1

                    if WRF_var == "T2":
                        WRF_var_1km = (
                            nc_fid1km.variables[WRF_var][0, 10:-10, 10:-10] - WRF_factor
                        )
                        WRF_var_coarse_domain = (
                            nc_fid_coarse_domain.variables[WRF_var][0, :, :]
                            - WRF_factor
                        )
                    elif WRF_var == "NEE":
                        WRF_var_1km = (
                            nc_fid1km.variables["EBIO_GEE"][0, 0, 10:-10, 10:-10]
                            + nc_fid1km.variables["EBIO_RES"][0, 0, 10:-10, 10:-10]
                        ) * WRF_factor
                        WRF_var_coarse_domain = (
                            nc_fid_coarse_domain.variables["EBIO_GEE"][0, 0, :, :]
                            + nc_fid_coarse_domain.variables["EBIO_RES"][0, 0, :, :]
                        ) * WRF_factor
                    elif WRF_var == "NEE_REF":
                        WRF_var_1km = (
                            nc_fid1km.variables["EBIO_GEE" + ref_sim][
                                0, 0, 10:-10, 10:-10
                            ]
                            + nc_fid1km.variables["EBIO_RES" + ref_sim][
                                0, 0, 10:-10, 10:-10
                            ]
                        ) * WRF_factor
                        WRF_var_coarse_domain = (
                            nc_fid_coarse_domain.variables["EBIO_GEE" + ref_sim][
                                0, 0, :, :
                            ]
                            + nc_fid_coarse_domain.variables["EBIO_RES" + ref_sim][
                                0, 0, :, :
                            ]
                        ) * WRF_factor
                    elif WRF_var == "GPP_pmodel":
                        # get pmodel gpp
                        time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
                        gpp_pmodel_1km = xr.open_dataset(
                            f"{pmodel_path}gpp_pmodel{subdaily}_3km_{time_str}.nc"
                        )
                        gpp_pmodel_coarse_domain = xr.open_dataset(
                            f"{pmodel_path}gpp_pmodel{subdaily}_{coarse_domain}_{time_str}.nc"
                        )
                        WRF_var_1km = gpp_pmodel_1km["GPP_Pmodel"].to_numpy()
                        WRF_var_coarse_domain = gpp_pmodel_coarse_domain[
                            "GPP_Pmodel"
                        ].to_numpy()
                    elif WRF_var == "RECO_Migli":
                        # get migli reco
                        time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
                        reco_migli_1km = xr.open_dataset(
                            f"{migli_path}reco_migliavacca_subdailyC3_3km_{time_str}.nc"
                        )
                        reco_migli_coarse_domain = xr.open_dataset(
                            f"{migli_path}reco_migliavacca_subdailyC3_{coarse_domain}_{time_str}.nc"
                        )
                        WRF_var_1km = reco_migli_1km["RECO_Migli"].to_numpy()
                        WRF_var_coarse_domain = reco_migli_coarse_domain[
                            "RECO_Migli"
                        ].to_numpy()
                    elif WRF_var == "NEE_PM":
                        time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
                        gpp_pmodel_1km = xr.open_dataset(
                            f"{pmodel_path}gpp_pmodel{subdaily}_3km_{time_str}.nc"
                        )
                        gpp_pmodel_coarse_domain = xr.open_dataset(
                            f"{pmodel_path}gpp_pmodel{subdaily}_{coarse_domain}_{time_str}.nc"
                        )
                        reco_migli_1km = xr.open_dataset(
                            f"{migli_path}reco_migliavacca_subdailyC3_3km_{time_str}.nc"
                        )
                        reco_migli_coarse_domain = xr.open_dataset(
                            f"{migli_path}reco_migliavacca_subdailyC3_{coarse_domain}_{time_str}.nc"
                        )
                        WRF_var_1km = (
                            reco_migli_1km["RECO_Migli"].to_numpy()
                            - gpp_pmodel_1km["GPP_Pmodel"].to_numpy()
                        )
                        WRF_var_coarse_domain = (
                            reco_migli_coarse_domain["RECO_Migli"].to_numpy()
                            - gpp_pmodel_coarse_domain["GPP_Pmodel"].to_numpy()
                        )
                    else:
                        WRF_var_1km = (
                            nc_fid1km.variables[WRF_var][0, 0, 10:-10, 10:-10]
                            * WRF_factor
                        )
                        WRF_var_coarse_domain = (
                            nc_fid_coarse_domain.variables[WRF_var][0, 0, :, :]
                            * WRF_factor
                        )

                    lats_coarse_domain = nc_fid_coarse_domain.variables["XLAT"][0, :, :]
                    lons_coarse_domain = nc_fid_coarse_domain.variables["XLONG"][
                        0, :, :
                    ]
                    landmask_4 = nc_fid_coarse_domain.variables["LANDMASK"][0, :, :]
                    hgt_coarse_domain = nc_fid_coarse_domain.variables["HGT"][0, :, :]
                    stdh_topo_coarse_domain = nc_fid_coarse_domain.variables["VAR"][
                        0, :, :
                    ]
                    WRF_var_coarse_domain[landmask_4 == 0] = np.nan
                    proj_WRF_var_coarse_domain = proj_on_finer_WRF_grid(
                        lats_coarse_domain,
                        lons_coarse_domain,
                        WRF_var_coarse_domain,
                        lats_fine,
                        lons_fine,
                        WRF_var_1km,
                    )
                    proj_hgt_coarse_domain = proj_on_finer_WRF_grid(
                        lats_coarse_domain,
                        lons_coarse_domain,
                        hgt_coarse_domain,
                        lats_fine,
                        lons_fine,
                        WRF_var_1km,
                    )
                    proj_stdh_topo_coarse_domain = proj_on_finer_WRF_grid(
                        lats_coarse_domain,
                        lons_coarse_domain,
                        stdh_topo_coarse_domain,
                        lats_fine,
                        lons_fine,
                        WRF_var_1km,
                    )

                    if STD_TOPO_flag == "gt":
                        stdh_mask = proj_stdh_topo_coarse_domain >= STD_TOPO
                    elif STD_TOPO_flag == "lt":
                        stdh_mask = proj_stdh_topo_coarse_domain < STD_TOPO
                    mask = land_mask * stdh_mask

                    WRF_diff = proj_WRF_var_coarse_domain[mask] - WRF_var_1km[mask]
                    hgt_diff = (proj_hgt_coarse_domain[mask] - hgt_1km[mask]) / 1000

                    # Plotting
                    if plotting_scatter:
                        if WRF_var == "T2":
                            fig, ax = plt.subplots(figsize=(8, 6))

                            idx = np.isfinite(WRF_diff) & np.isfinite(hgt_diff)
                            # coeff = np.polyfit(hgt_diff[idx], WRF_diff[idx], deg=1)
                            coeff, _, _, _ = np.linalg.lstsq(
                                hgt_diff[idx][:, np.newaxis], WRF_diff[idx], rcond=None
                            )
                            x_poly = np.linspace(
                                hgt_diff[idx].min(), hgt_diff[idx].max()
                            )
                            y_poly = coeff[0] * x_poly

                            # calculate the R2 and the RMSE value
                            r2 = r2_score(WRF_diff[idx], hgt_diff[idx])
                            rmse = np.sqrt(
                                mean_squared_error(WRF_diff[idx], hgt_diff[idx])
                            )

                            ax.text(
                                0.05,
                                0.95,
                                f"R² = {r2:.2f}\nRMSE = {rmse:.2f}",
                                transform=ax.transAxes,
                                fontsize=12,
                                verticalalignment="top",
                                bbox=dict(
                                    boxstyle="round,pad=0.3",
                                    edgecolor="black",
                                    facecolor="white",
                                ),
                            )

                            ax.scatter(
                                hgt_diff[idx], WRF_diff[idx], s=5, c="k", alpha=0.5
                            )
                            ax.plot(
                                x_poly,
                                y_poly,
                                color="b",
                                lw=1.5,
                                linestyle="--",
                                label=f"y = {coeff[0]:.2f}x",
                            )
                            ax.legend()
                            ax.set_title(
                                f"WRF {WRF_var} diff. {coarse_domain} - 3km vs. height diff with stdev {STD_TOPO_flag} {STD_TOPO}"
                            )
                            ax.set_xlabel("Height Difference [km]")
                            ax.set_ylabel(f"{WRF_var} Difference")
                            ax.set_xlim([-1.5, 1.5])
                            ax.set_ylim([-10, 10])
                            plt.tight_layout()
                            plt.tight_layout()
                            plt.savefig(
                                f"{outfolder}correlations_of_{WRF_var}{ref_sim}_{coarse_domain}_vs_topo_diff_{STD_TOPO_flag}_{STD_TOPO}_{time}.png"
                            )
                            plt.close

                    if WRF_var == "T2":
                        WRF_T2_2d = np.where(mask, proj_WRF_var_coarse_domain, np.nan)
                        T2_coarse_domain_4D.append({"time": time.hour, "T2": WRF_T2_2d})
                        del WRF_T2_2d

                    WRF_var_diff_to_1km_2D = np.where(
                        mask, proj_WRF_var_coarse_domain - WRF_var_1km, np.nan
                    )
                    diff_to_1km_4D.append(
                        {"time": time.hour, WRF_var: WRF_var_diff_to_1km_2D}
                    )
                    del WRF_var_diff_to_1km_2D

            # Convert lists to 3D arrays
            var_data = {}
            hours = [entry["time"] for entry in diff_to_1km_4D]

            for entry in diff_to_1km_4D:
                for var_name, data in entry.items():
                    if var_name != "time":  # Skip the 'time' key
                        if var_name not in var_data:
                            var_data[var_name] = []
                        var_data[var_name].append(data)
            diff_to_1km_3D = {
                var: np.stack(data_list, axis=0) for var, data_list in var_data.items()
            }
            var_data = {}
            for entry in T2_coarse_domain_4D:
                for var_name, data in entry.items():
                    if var_name != "time":  # Skip the 'time' key
                        if var_name not in var_data:
                            var_data[var_name] = []
                        var_data[var_name].append(data)
            T2_coarse_domain_3D = {
                var: np.stack(data_list, axis=0) for var, data_list in var_data.items()
            }

            T_ref_values = np.arange(T_ref_min, T_ref_max, T_bin_size)
            if ini_switch == True:
                df_coeff = pd.DataFrame(index=T_ref_values)
                ini_switch = False

            # correlate T2 and WRF_var for each step in T_ref_values
            for WRF_var in WRF_vars[:-1]:
                coeff_all_T = []
                for T_ref in T_ref_values:
                    try:
                        temp_mask = (T2_coarse_domain_3D["T2"] >= T_ref) & (
                            T2_coarse_domain_3D["T2"] <= T_ref + T_bin_size
                        )
                        masked_diff_T2 = diff_to_1km_3D["T2"][temp_mask]
                        masked_diff_var = diff_to_1km_3D[WRF_var][temp_mask]
                        idx = np.isfinite(masked_diff_var) & np.isfinite(masked_diff_T2)
                        diff_T2_t = masked_diff_T2[idx]
                        diff_var_t = masked_diff_var[idx]
                        diff_T2_t = np.array(diff_T2_t)
                        diff_var_t = np.array(diff_var_t)
                        coeff, _, _, _ = np.linalg.lstsq(
                            masked_diff_T2[idx][:, np.newaxis],
                            masked_diff_var[idx],
                            rcond=None,
                        )

                        if plotting_scatter_all:
                            fig, ax = plt.subplots()
                            ax.scatter(
                                masked_diff_T2[idx],
                                masked_diff_var[idx],
                                s=0.1,
                                c="red",
                            )
                            if masked_diff_T2[idx].size > 0:
                                x_poly = np.linspace(
                                    masked_diff_T2[idx].min(),
                                    masked_diff_T2[idx].max(),
                                )
                                y_poly = coeff[0] * x_poly
                                ax.plot(
                                    x_poly,
                                    y_poly,
                                    color="b",
                                    lw=1.5,
                                    linestyle="--",
                                    label=f"y = {coeff[0]:.2f} * x",
                                )
                            ax.legend()
                            ax.xaxis.grid(True, which="major")
                            ax.yaxis.grid(True, which="major")
                            ax.set_xlabel("T2 diff [°C]")
                            ax.set_ylabel(f"{name_vars[WRF_var]} diff")
                            formatted_T_ref = "{:.2f}".format(T_ref).replace(".", "_")[
                                :4
                            ]
                            r2 = r2_score(masked_diff_T2[idx], masked_diff_var[idx])
                            rmse = np.sqrt(
                                mean_squared_error(
                                    masked_diff_T2[idx], masked_diff_var[idx]
                                )
                            )

                            ax.text(
                                0.05,
                                0.95,
                                f"R² = {r2:.2f}\nRMSE = {rmse:.2f}",
                                transform=ax.transAxes,
                                fontsize=12,
                                verticalalignment="top",
                                bbox=dict(
                                    boxstyle="round,pad=0.3",
                                    edgecolor="black",
                                    facecolor="white",
                                ),
                            )

                            plt.title(
                                f"WRF {coarse_domain}- 3km T2 and {name_vars[WRF_var]} correlation at {formatted_T_ref} °C T_ref"
                            )
                            figname = (
                                outfolder
                                + f"WRF_T2_{WRF_var}_{coarse_domain}_1km_corr{subdaily}_{STD_TOPO_flag}_STD_{STD_TOPO}_T_ref_{formatted_T_ref}_{time}.png"
                            )
                            plt.savefig(figname)
                            plt.close()
                        coeff_all_T.append(coeff[0])
                    except:
                        print("Not enough Data for T_ref=%s" % T_ref)
                        coeff_all_T.append(np.nan)
                df_coeff[name_vars[WRF_var]] = coeff_all_T
                del coeff_all_T

            diff_hgt = np.where(mask, proj_hgt_coarse_domain, np.nan) - np.where(
                mask, hgt_1km, np.nan
            )
            diff_hgt_mean = np.nanmean(diff_hgt)
            diff_hgt_mean_nonproj = np.nanmean(hgt_coarse_domain) - np.nanmean(hgt_1km)
            print(
                f"Mean difference in height between {coarse_domain} and 1km is {diff_hgt_mean:.2f}m"
            )
            print(
                f"Unmasked means of 1km: {np.nanmean(hgt_1km):.2f}m and of {coarse_domain}: {np.nanmean(hgt_coarse_domain):.2f}m with difference  {diff_hgt_mean_nonproj:.2f}m  "
            )

            if plot_coeff:
                ax = df_coeff.plot(linestyle="-", figsize=(10, 6), grid=True)
                ax.set_ylim(-2, 2)
                ax.set_xlim(T_ref_min, T_ref_max)
                ax.set_xlabel(r"$T_{\mathrm{ref}}$ °C", fontsize=20)
                ax.set_ylabel("[μmol m² s⁻¹ °C⁻¹]", fontsize=20)

                ax.tick_params(labelsize=20)
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(fontsize=16, frameon=True, framealpha=0.4)

                # ax.set_title(f"Coefficient Values for NEE, GPP, and RECO")
                figname = (
                    outfolder
                    + f"WRF_Tref_coefficients{ref_sim}{filename_str}_{coarse_domain}_{STD_TOPO_flag}_STD_{STD_TOPO}_{hour_start}-{hour_end}h_till_{end_date}.pdf"
                )
                plt.savefig(figname, bbox_inches="tight")
                plt.close()
                df_coeff.to_csv(
                    figname.replace(".pdf", ".csv"), index_label="T_ref [°C]"
                )
            del df_coeff
