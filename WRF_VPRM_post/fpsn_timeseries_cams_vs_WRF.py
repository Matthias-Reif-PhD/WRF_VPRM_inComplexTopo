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


def find_nearest_grid_point_and_distance(nc_fid, lat_target, lon_target):
    lat_idx = np.unravel_index(
        np.abs(nc_fid.variables["XLAT"][:] - lat_target).argmin(),
        nc_fid.variables["XLAT"][:].shape,
    )
    lon_idx = np.unravel_index(
        np.abs(nc_fid.variables["XLONG"][:] - lon_target).argmin(),
        nc_fid.variables["XLONG"][:].shape,
    )
    lat_nearest = nc_fid.variables["XLAT"][lat_idx]
    lon_nearest = nc_fid.variables["XLONG"][lon_idx]
    dist_to_CAMS = haversine(lat_target, lon_target, lat_nearest, lon_nearest)
    return dist_to_CAMS


def get_int_var(nc_fid, lat_target, lon_target, WRF_var):
    interpolated_var_dX = griddata(
        (
            nc_fid.variables["XLAT"][:].flatten(),
            nc_fid.variables["XLONG"][:].flatten(),
        ),
        nc_fid.variables[WRF_var][:].flatten(),
        (lat_target, lon_target),
        method="linear",
    )
    return interpolated_var_dX


def proj_on_finer_grid(
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_x
):
    proj_var = griddata(
        (lats_coarse.flatten(), lons_coarse.flatten()),
        var_coarse.flatten(),
        (lats_fine, lons_fine),
        method="linear",
    ).reshape(WRF_var_x.shape)
    return proj_var


def proj_CAMS_on_WRF_grid(
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_x
):
    # Corrected meshgrid order
    lons_coarse_2d, lats_coarse_2d = np.meshgrid(lons_coarse, lats_coarse)
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
    ).reshape(WRF_var_x.shape)

    return proj_var


################################# INPUT ##############################################

# coordinates of FAIR site
lat_target = 47.316564
lon_target = 10.970089
# Select start and end dates and times
start_date = "2020-06-21 00:00:00"
end_date = "2020-06-22 00:00:00"

#######################################################################################
# load CAMS data
CAMS_path = "/home/madse/Build_WRF/DATA/CAMS/ghg-reanalysis_lat43-51_lon5-17_2020-06-01_2020-06-30.nc"
CAMS_data = nc.Dataset(CAMS_path)
times_CAMS = CAMS_data.variables["time"]

CAMS_vars = ["fco2nee", "fco2gpp", "fco2rec", "t2m"]
factor_kgC = 1000000 / 0.044  # conversion from kgCO2/m2/s to mgCO2/m2/s to  mumol/m2/s
factors = [-factor_kgC, factor_kgC, -factor_kgC, 1]


# WRF output data
WRF_path = "/home/madse/Build_WRF/DATA/WRFOUT/Mieming_2020/2020_06_21_4_domains/"
# Use glob to list all files in the directory
file_list_d1 = sorted(glob.glob(os.path.join(WRF_path, "wrfout_d01*")))
WRF_vars = ["FCO2", "FPSN", "RECO", "T2CLM"]  # "T_VEG",
units = [" [mmol m² s⁻¹]", " [mmol m² s⁻¹]", " [mmol m² s⁻¹]", " [K]"]

for WRF_var, CAMS_var, factor, unit in zip(WRF_vars, CAMS_vars, factors, units):
    figname = start_date + "_CAMS_vs_WRFd01_new_reco_" + WRF_var + ".eps"
    figname_diff = start_date + "_diff_CAMS_vs_WRFd01_new_reco_" + WRF_var + ".eps"

    # WRF
    WRF_mean = []
    CAMS_mean = []
    time_steps = []
    i = 0
    # Loop through the files for the 24-hour period
    for nc_f1 in file_list_d1:
        nc_fid1 = nc.Dataset(nc_f1, "r")

        times_variable = nc_fid1.variables["Times"]
        start_date_bytes = times_variable[0, :].tobytes()
        start_date_str = start_date_bytes.decode("utf-8")
        print("Processing WRF Date:", start_date_str)
        print("... ")
        lats_fine = nc_fid1.variables["XLAT"][:]
        lons_fine = nc_fid1.variables["XLONG"][:]
        landmask = nc_fid1.variables["LANDMASK"][:]
        land_mask = landmask == 1
        WRF_var_x = nc_fid1.variables[WRF_var][:]
        WRF_var_x_land = np.ma.masked_where(~land_mask, WRF_var_x)
        # Calculate the average over the land
        WRF_average = np.mean(WRF_var_x)
        print(f"WRF_average: {WRF_average}")
        WRF_average_over_land = np.mean(WRF_var_x_land)
        print(f"WRF_average_over_land: {WRF_average_over_land}")
        WRF_mean.append(WRF_average_over_land.tolist())

        # process CAMS data if times fit
        start_date_nc_f1 = datetime.strptime(start_date_str, "%Y-%m-%d_%H:%M:%S")
        j = 0
        for time_CAMS in times_CAMS:
            date_CAMS = datetime(1900, 1, 1) + timedelta(hours=int(time_CAMS))
            j = j + 1
            if start_date_nc_f1 == date_CAMS:
                lat_CAMS = CAMS_data.variables["latitude"][:]
                lon_CAMS = CAMS_data.variables["longitude"][:]
                var_CAMS = (
                    CAMS_data.variables[CAMS_var][j - 1, :, :].data * factor
                )  # convert unit to mmol m-2 s-1
                CAMS_proj = proj_CAMS_on_WRF_grid(
                    lat_CAMS,
                    lon_CAMS,
                    var_CAMS,
                    lats_fine,
                    lons_fine,
                    WRF_var_x,
                )

                CAMS_var_x_land = np.ma.masked_where(~land_mask, CAMS_proj)

                CAMS_average = np.mean(CAMS_proj)
                print(f"CAMS_average: {CAMS_average}")
                CAMS_average_over_land = np.mean(CAMS_var_x_land)
                print(f"CAMS_average_over_land: {CAMS_average_over_land}")
                CAMS_mean.append(CAMS_average_over_land.tolist())

        time_step = nc_fid1.variables["Times"][:].tobytes().decode("utf-8")
        time_steps.append(time_step)
        i += 1
        # close the NetCDF files when done
        nc_fid1.close()

    # Create a time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        time_steps[1:],
        WRF_mean[1:],
        label="d01 (dx=27km)",
    )
    plt.plot(
        time_steps[3::3],
        CAMS_mean[1:],
        label=CAMS_var + " CAMS (dx=0.75°~60km)",
    )
    plt.xlabel("Time Steps")
    plt.ylabel(WRF_var + " and " + CAMS_var + unit)
    plt.title(WRF_var + " and " + CAMS_var + " average over d01 of WRF")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(figname, format="eps")

    diff_WRF_CAMS = [b - a for a, b in zip(WRF_mean[3::3], CAMS_mean[1:])]

    # Create a time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        time_steps[3::3],
        diff_WRF_CAMS,
        label="CAMS (dx=0.75°~60km) - d01 (dx=27km))",
    )
    plt.xlabel("Time Steps")
    plt.ylabel(WRF_var + " and " + CAMS_var + unit)
    plt.title(WRF_var + " and " + CAMS_var + " differences of average of CAMS - WRF ")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(figname_diff, format="eps")
