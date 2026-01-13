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
from datetime import datetime, timedelta


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
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_d3
):
    proj_var = griddata(
        (lats_coarse.flatten(), lons_coarse.flatten()),
        var_coarse.flatten(),
        (lats_fine, lons_fine),
        method="nearest",
    ).reshape(WRF_var_d3.shape)
    return proj_var


def proj_CAMS_on_WRF_grid(
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_d3
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
    ).reshape(WRF_var_d3.shape)

    return proj_var


def proj_on_WRF_grid(
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_d3
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
    ).reshape(WRF_var_d3.shape)

    return proj_var


################################# INPUT ##############################################

# set standard deviation of topography
STD_VAL = 50
VEGFRA_percentage = 10
T_bin_size = 1

# set True if you want plots
plotting_refs = True
plotting_scatter = True
plotting_scatter_all = True
T_bin_flag = True

# start_date = "2012-07-01 01:00:00"
# end_date = "2012-07-31 00:00:00"
wrf_paths = [
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_1km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_9km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_27km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km",
]
wrf_file = "wrfout_d01_2012-07-27_15:00:00"
wrf_file_d2 = "wrfout_d02_2012-07-27_15:00:00"
outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"

timedelt = 0
month = "07"


#######################################################################################
# load CAMS data
CAMS_path = (
    "/scratch/c7071034/DATA/CAMS/ghg-reanalysis_surface_2012-07-01_2012-07-31.nc"
)

CAMS_data = nc.Dataset(CAMS_path)
times_CAMS = CAMS_data.variables["valid_time"]

CAMS_vars = ["fco2gpp", "fco2rec", "t2m"]
factor_kgC = -1000 / 44 * 1000000  # conversion from kgCO2/m2/s to  mumol/m2/s
CAMS_factors = [factor_kgC, factor_kgC, 273.15]


# Use glob to list all files in the directory
# file_list_d1 = sorted(glob.glob(os.path.join(WRF_path, "wrfout_d01*")))
WRF_vars = ["EBIO_GEE", "EBIO_RES", "T2"]
units = [" [mmol m² s⁻¹]", " [mmol m² s⁻¹]", " [K]"]
name_vars = {"EBIO_GEE": "WRF GPP", "EBIO_RES": "WRF RECO", "T2": "WRF T2M"}
WRF_factors = [1 / 3600, 1 / 3600, 273.15]

VPRM_flags = [False, False, False, False, False]
VPRM_vars = ["NEE", "GEE", "RESP", "NaN"]
VPRM_factors = [1, -1, 1, 1]
flag_ini = True

for (
    WRF_var,
    CAMS_var,
    factor,
    unit,
    VPRM_flag,
    VPRM_var,
    VPRM_factor,
    WRF_factor,
) in zip(
    WRF_vars,
    CAMS_vars,
    CAMS_factors,
    units,
    VPRM_flags,
    VPRM_vars,
    VPRM_factors,
    WRF_factors,
):
    # WRF
    CAMS_mean = []
    time_steps = []
    i = 0
    # Loop through the files for the period
    # for nc_f1 in file_list_d1:
    nc_fid00 = nc.Dataset(os.path.join(wrf_paths[4], wrf_file), "r")
    nc_fid0 = nc.Dataset(os.path.join(wrf_paths[3], wrf_file), "r")
    nc_fid1 = nc.Dataset(os.path.join(wrf_paths[2], wrf_file), "r")
    nc_fid2 = nc.Dataset(os.path.join(wrf_paths[1], wrf_file), "r")
    nc_fid3 = nc.Dataset(os.path.join(wrf_paths[0], wrf_file_d2), "r")

    times_variable = nc_fid3.variables["Times"]
    start_date_bytes = times_variable[0, :].tobytes()
    start_date_str = start_date_bytes.decode("utf-8")
    print("Processing WRF Date:", start_date_str)
    print("... ")
    lats_fine = nc_fid3.variables["XLAT"][0, :, :]
    lons_fine = nc_fid3.variables["XLONG"][0, :, :]
    landmask = nc_fid3.variables["LANDMASK"][0, :, :]
    land_mask = landmask == 1

    stdh_mask = nc_fid3.variables["VAR"][0, :, :]
    stdh_mask = stdh_mask > STD_VAL
    mask = land_mask * stdh_mask

    if WRF_var == "T2":
        WRF_var_d3 = nc_fid3.variables[WRF_var][0, :, :] - WRF_factor
    else:
        WRF_var_d3 = nc_fid3.variables[WRF_var][0, 0, :, :] * WRF_factor
    WRF_var_d3[landmask == 0] = np.nan
    WRF_var_d3_topo = np.ma.masked_where(~mask, WRF_var_d3)

    WRF_var_d3_topo_m = np.nanmean(WRF_var_d3_topo)

    if WRF_var == "T2":
        WRF_var_d2 = nc_fid2.variables[WRF_var][0, :, :] - WRF_factor
    else:
        WRF_var_d2 = nc_fid2.variables[WRF_var][0, 0, :, :] * WRF_factor
    lats_d2 = nc_fid2.variables["XLAT"][0, :, :]
    lons_d2 = nc_fid2.variables["XLONG"][0, :, :]
    landmask_2 = nc_fid2.variables["LANDMASK"][0, :, :]
    WRF_var_d2[landmask_2 == 0] = np.nan

    proj_WRF_var_d2 = proj_on_finer_WRF_grid(
        lats_d2, lons_d2, WRF_var_d2, lats_fine, lons_fine, WRF_var_d3
    )
    WRF_var_d2_topo = np.nanmean(np.ma.masked_where(~mask, proj_WRF_var_d2))

    if WRF_var == "T2":
        WRF_var_d1 = nc_fid1.variables[WRF_var][0, :, :] - WRF_factor
    else:
        WRF_var_d1 = nc_fid1.variables[WRF_var][0, 0, :, :] * WRF_factor
    lats_d1 = nc_fid1.variables["XLAT"][0, :, :]
    lons_d1 = nc_fid1.variables["XLONG"][0, :, :]
    landmask_1 = nc_fid1.variables["LANDMASK"][0, :, :]
    WRF_var_d1[landmask_1 == 0] = np.nan

    model_TSK_d1 = nc_fid1.variables["TSK"][:]

    proj_WRF_var_d1 = proj_on_finer_WRF_grid(
        lats_d1, lons_d1, WRF_var_d1, lats_fine, lons_fine, WRF_var_d3
    )
    WRF_var_d1_topo = np.nanmean(np.ma.masked_where(~mask, proj_WRF_var_d1))

    if WRF_var == "T2":
        WRF_var_d0 = nc_fid0.variables[WRF_var][0, :, :] - WRF_factor
    else:
        WRF_var_d0 = nc_fid0.variables[WRF_var][0, 0, :, :] * WRF_factor
    lats_d0 = nc_fid0.variables["XLAT"][0, :, :]
    lons_d0 = nc_fid0.variables["XLONG"][0, :, :]
    landmask_0 = nc_fid0.variables["LANDMASK"][0, :, :]
    WRF_var_d0[landmask_0 == 0] = np.nan
    model_TSK_d0 = nc_fid0.variables["TSK"][:]
    proj_WRF_var_d0 = proj_on_finer_WRF_grid(
        lats_d0, lons_d0, WRF_var_d0, lats_fine, lons_fine, WRF_var_d3
    )
    WRF_var_d0_topo = np.nanmean(np.ma.masked_where(~mask, proj_WRF_var_d0))

    if WRF_var == "T2":
        WRF_var_ref = nc_fid00.variables[WRF_var][0, :, :] - WRF_factor
    else:
        WRF_var_ref = nc_fid00.variables[WRF_var][0, 0, :, :] * WRF_factor
    lats_ref = nc_fid00.variables["XLAT"][0, :, :]
    lons_ref = nc_fid00.variables["XLONG"][0, :, :]
    landmask_00 = nc_fid00.variables["LANDMASK"][0, :, :]
    WRF_var_ref[landmask_00 == 0] = np.nan
    model_TSK_ref = nc_fid00.variables["TSK"][:]
    proj_WRF_var_ref = proj_on_finer_WRF_grid(
        lats_ref, lons_ref, WRF_var_ref, lats_fine, lons_fine, WRF_var_d3
    )
    WRF_var_ref_topo = np.nanmean(np.ma.masked_where(~mask, proj_WRF_var_ref))

    # VPRM:
    if VPRM_flag:
        file_path_VPRM = (
            "/media/madse/scratch/VPRM_data/VPRM_ECMWF_" + VPRM_var + "_2020_CP.nc"
        )
        VPRM_data = nc.Dataset(file_path_VPRM)
        times_VPRM = VPRM_data.variables["time"]

    # process CAMS data if times fit
    start_date_nc_f1 = datetime.strptime(start_date_str, "%Y-%m-%d_%H:%M:%S")
    new_time = start_date_nc_f1 - timedelta(minutes=timedelt)
    j = 0
    for time_CAMS in times_CAMS:
        date_CAMS = datetime(1970, 1, 1) + timedelta(seconds=int(time_CAMS))
        j = j + 1
        if new_time == date_CAMS:
            lat_CAMS = CAMS_data.variables["latitude"][:]
            lon_CAMS = CAMS_data.variables["longitude"][:]
            if WRF_var == "T2":
                var_CAMS = (
                    CAMS_data.variables[CAMS_var][j - 1, :, :].data - factor
                )  # convert unit to mmol m-2 s-1
            else:
                var_CAMS = (
                    CAMS_data.variables[CAMS_var][j - 1, :, :].data * factor
                )  # convert unit to mmol m-2 s-1
            CAMS_proj = proj_CAMS_on_WRF_grid(
                lat_CAMS,
                lon_CAMS,
                var_CAMS,
                lats_fine,
                lons_fine,
                WRF_var_d3,
            )

            CAMS_var_x_topo = np.ma.masked_where(~mask, CAMS_proj)
            CAMS_average_over_topo = np.mean(CAMS_var_x_topo)

    if VPRM_flag:
        j = 0
        for time_VPRM in times_VPRM:
            date_VPRM = datetime(2020, 1, 1) + timedelta(hours=int(time_VPRM))
            j = j + 1
            if new_time == date_VPRM:
                lat_VPRM = VPRM_data.variables["lat"][:]
                lon_VPRM = VPRM_data.variables["lon"][:]
                var_VPRM = (
                    VPRM_data.variables[VPRM_var][j - 1, :, :].data * VPRM_factor
                )  # unit is mmol m-2 s-1
                VPRM_proj = proj_on_WRF_grid(
                    lat_VPRM,
                    lon_VPRM,
                    var_VPRM,
                    lats_fine,
                    lons_fine,
                    WRF_var_d3,
                )
                VPRM_var_x_topo = np.ma.masked_where(~mask, VPRM_proj)
                VPRM_average_over_topo = np.mean(VPRM_var_x_topo)
                print("projection of VPRM data")
                print("... at ", date_VPRM)
    else:
        VPRM_average_over_topo = np.nan

    time_step = nc_fid3.variables["Times"][:].tobytes().decode("utf-8")
    time_steps.append(time_step)
    i += 1
    # close the NetCDF files when done

    # Create a bar chart
    labels = [
        "WRF 1km",
        "WRF 3km",
        "WRF 9km",
        "WRF 27km",
        "WRF 54km",
        "CAMS",
        "VPRM",
    ]
    values = [
        WRF_var_d3_topo_m,
        WRF_var_d2_topo,
        WRF_var_d1_topo,
        WRF_var_d0_topo,
        WRF_var_ref_topo,
        CAMS_average_over_topo,
        VPRM_average_over_topo,
    ]

    if plotting_refs:
        figname = (
            outfolder
            + "means_of_WRF_CAMS_VPRM_"
            + name_vars[WRF_var]
            + "_stdv_"
            + str(STD_VAL)
            + ".png"
        )
        fig, ax = plt.subplots()
        bars = ax.bar(
            labels, values, color=["blue", "green", "red", "magenta", "orange", "brown", "purple", "cyan"]
        )
        # Add labels to each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                round(yval, 2),
                ha="center",
                va="bottom",
            )
        plt.ylabel("Mean Values")
        plt.title("Interpol. on WRF 3km for " + name_vars[WRF_var])
        # set angle of labels to 45 °
        # plt.xticks(rotation=45)
        # plt.show()
        plt.savefig(figname)
        plt.close()

    model_TSK_d3 = nc_fid3.variables["TSK"][0, :, :]

    model_TSK_d3_topo = np.ma.masked_where(~mask, model_TSK_d3)
    proj_model_TSK_d1 = proj_on_finer_WRF_grid(
        lats_d1, lons_d1, model_TSK_d1, lats_fine, lons_fine, model_TSK_d3
    )
    model_TSK_d1_topo = np.ma.masked_where(~mask, proj_model_TSK_d1)

    diff_TSK = model_TSK_d1_topo - model_TSK_d3_topo

    # scale by vegetation fraction to normalize for temp difference effect TODO: check if this is correct
    if WRF_var != "TSK":
        WRF_VEGFRA_ref = nc_fid00.variables["VEGFRA"][0, :, :]
        WRF_var_ref[WRF_VEGFRA_ref < VEGFRA_percentage] = np.nan
        WRF_var_ref = WRF_var_ref * WRF_VEGFRA_ref / 100

    proj_WRF_var_ref = proj_on_finer_WRF_grid(
        lats_ref, lons_ref, WRF_var_ref, lats_fine, lons_fine, WRF_var_d3
    )

    if WRF_var != "TSK":
        WRF_VEGFRA_d3 = nc_fid3.variables["VEGFRA"][0, :, :]
        WRF_var_d3[WRF_VEGFRA_d3 < VEGFRA_percentage] = np.nan
        WRF_var_d3 = WRF_var_d3 * WRF_VEGFRA_d3 / 100

    diff_var = np.ma.masked_where(~mask, proj_WRF_var_ref) - np.ma.masked_where(
        ~mask, WRF_var_d3
    )

    idx = np.isfinite(diff_var) & np.isfinite(diff_TSK)

    coeff = np.polyfit(diff_TSK[idx], diff_var[idx], deg=1)
    x_poly = np.linspace(diff_TSK[idx].min(), diff_TSK[idx].max())
    y_poly = np.polyval(coeff, x_poly)
    if plotting_scatter:
        fig, ax = plt.subplots()
        ax.scatter(diff_TSK[idx], diff_var[idx], s=0.2, c="k")
        # Plot regression line
        a, b = coeff
        ax.plot(
            x_poly,
            y_poly,
            color="b",
            lw=1.5,
            linestyle="--",
            label=f"y = {a:.2f} * x + {b:.2f}",
        )
        ax.legend()
        t1 = "Mean differences: %s K and %s m" % (
            "{:.2f}".format(diff_var[idx].mean()),
            "{:.2f}".format(diff_TSK[idx].mean()),
        )
        plt.text(0, 0, t1, ha="left", wrap=True)
        ax.xaxis.grid(True, which="major")
        ax.yaxis.grid(True, which="major")
        ax.set_xlabel("T diff [K]")
        ax.set_ylabel(f"{name_vars[WRF_var]} diff")
        plt.title("WRF 27km - 3km TSK and %s correlation" % (name_vars[WRF_var]))
        figname = outfolder + "WRF_TSK_%s_correlation.png" % (WRF_var,)
        # plt.show()
        # plt.close()
        nc_fid3.close()
        plt.savefig(figname)

    coeff_all_T = []
    coeff_all2_T = []
    # select range of T_refs
    # TODO: improve
    T_ref_values = range(
        int(model_TSK_d1_topo.min() + 7), int(model_TSK_d1_topo.max() - 3), T_bin_size
    )

    if flag_ini:
        df_coeff = pd.DataFrame(index=T_ref_values)
        flag_ini = False

    if T_bin_flag:
        # Create a mask for model_TSK_d3_topo between 300 and 301 K

        for T_ref in T_ref_values:
            try:
                temp_mask = (model_TSK_d1_topo >= T_ref) & (
                    model_TSK_d1_topo <= T_ref + T_bin_size
                )

                # Apply the mask to differences
                masked_diff_TSK = diff_TSK[temp_mask]
                masked_diff_var = diff_var[temp_mask]

                idx = np.isfinite(masked_diff_var) & np.isfinite(masked_diff_TSK)

                # Apply the overall mask
                diff_TSK_t = masked_diff_TSK[idx]
                diff_var_t = masked_diff_var[idx]

                # Make a writable version of the arrays (if they are masked)
                diff_TSK_t = np.array(diff_TSK_t)
                diff_var_t = np.array(diff_var_t)

                # Identify and exclude outliers using the IQR method
                Q1_diff_TSK = np.percentile(diff_TSK_t, 25)
                Q3_diff_TSK = np.percentile(diff_TSK_t, 75)
                IQR_diff_TSK = Q3_diff_TSK - Q1_diff_TSK
                lower_bound_diff_TSK = Q1_diff_TSK - 1.5 * IQR_diff_TSK
                upper_bound_diff_TSK = Q3_diff_TSK + 1.5 * IQR_diff_TSK

                Q1_diff_var = np.percentile(diff_var_t, 25)
                Q3_diff_var = np.percentile(diff_var_t, 75)
                IQR_diff_var = Q3_diff_var - Q1_diff_var
                lower_bound_diff_var = Q1_diff_var - 1.5 * IQR_diff_var
                upper_bound_diff_var = Q3_diff_var + 1.5 * IQR_diff_var

                # Create masks to exclude outliers
                outlier_mask = (
                    (diff_TSK_t >= lower_bound_diff_TSK)
                    & (diff_TSK_t <= upper_bound_diff_TSK)
                    & (diff_var_t >= lower_bound_diff_var)
                    & (diff_var_t <= upper_bound_diff_var)
                )

                # Apply the mask
                diff_TSK_no_outliers = diff_TSK_t[outlier_mask]
                diff_var_no_outliers = diff_var_t[outlier_mask]
                coeff = np.polyfit(masked_diff_TSK[idx], masked_diff_var[idx], deg=1)
                coeff2 = np.polyfit(diff_TSK_no_outliers, diff_var_no_outliers, deg=1)
                a, b = coeff
                a2, b2 = coeff2
                if plotting_scatter_all:
                    fig, ax = plt.subplots()
                    ax.scatter(
                        masked_diff_TSK[idx], masked_diff_var[idx], s=0.1, c="red"
                    )
                    x_poly = np.linspace(
                        masked_diff_TSK[idx].min(), masked_diff_TSK[idx].max()
                    )
                    y_poly = np.polyval(coeff, x_poly)
                    ax.plot(
                        x_poly,
                        y_poly,
                        color="b",
                        lw=1.5,
                        linestyle="--",
                        label=f"y_all = {a:.2f} * x + {b:.2f}",
                    )
                    # t1 = "Mean differences: %s K and %s m" % (
                    #     "{:.2f}".format(masked_diff_var[idx].mean()),
                    #     "{:.2f}".format(masked_diff_TSK[idx].mean()),
                    # )
                    # plt.text(0, 900, t1, ha="left", wrap=True)

                    # ax.scatter(diff_TSK_no_outliers, diff_var_no_outliers, s=0.2, c="k")
                    # x_poly_b = np.linspace(diff_TSK_no_outliers.min(), diff_TSK_no_outliers.max())
                    # y_poly_b = np.polyval(coeff2, x_poly_b)
                    # ax.set_xlim(-7.5, 7.5)
                    # ax.set_ylim(-10, 10)
                    # ax.plot(
                    #     x_poly_b,
                    #     y_poly_b,
                    #     color="k",
                    #     lw=1.5,
                    #     linestyle="-.",
                    #     label=f"y_perc = {a2:.2f} * x + {b2:.2f}",
                    # )
                    ax.legend()

                    ax.xaxis.grid(True, which="major")
                    ax.yaxis.grid(True, which="major")
                    ax.set_xlabel("TSK diff [K]")
                    ax.set_ylabel(f"{name_vars[WRF_var]} diff")
                    plt.title(
                        "WRF 27km - 3km TSK and %s correlation at %s T_ref"
                        % (name_vars[WRF_var], T_ref)
                    )
                    figname = outfolder + "WRF_TSK_%s_correlation_%s_T_ref.png" % (
                        WRF_var,
                        T_ref,
                    )
                    # plt.show()
                    plt.savefig(figname)
                # plt.show()
                # plt.close()
                coeff_all_T.append(a)
                coeff_all2_T.append(a2)
            except:
                print("Not enough Data for T_ref=%s" % T_ref)
                coeff_all_T.append(np.nan)
    if T_bin_flag:
        df_coeff[name_vars[WRF_var]] = coeff_all_T
    # df_coeff[WRF_var + "_perc"] = coeff_all2_T
if T_bin_flag:
    # Plotting
    ax = df_coeff.plot(marker="o", linestyle="-", figsize=(10, 6), grid=True)

    # Adding labels and title
    ax.set_xlabel("T_ref")
    ax.set_ylabel("Coefficient Values")
    ax.set_title("Coefficient Values for NEE, GPP, and RECO")
    # add legend and name the timeseries
    ax.legend(loc="upper right")
    ax.set_xticks(df_coeff.index)

    figname = outfolder + "WRF_T_ref_coefficients_STD_%s.png" % (STD_VAL,)
    plt.savefig(figname)
    # Show the plot
    # plt.show()
    print(df_coeff)

    # # Create a contour plot
    # fig, ax = plt.subplots()
    # contour_plot = ax.contourf(diff_var[0, :, :], cmap="viridis")

    # # Add colorbar
    # cbar = plt.colorbar(contour_plot, ax=ax)
    # cbar.set_label("[Your Colorbar Label]")

    # # Add labels and title
    # ax.set_xlabel("[Your X-axis Label]")
    # ax.set_ylabel("[Your Y-axis Label]")
    # plt.title("Contour Plot of diff_var")

    # # Show the plot
    # plt.show()
