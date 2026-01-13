print("========================================================================")
print("Matlab WRF-GHG: IC from CAMS")
print("Michal Galkowski, MPI-BGC Jena")
print("modified by David Ho, MPI-BGC Jena")
print("Translated for python by Noelia Rojas, IFUSP Brazil")
print("over the Amazon domain.")

import numpy as np
import pandas as pd
import netCDF4 as cdf
import xarray as xr
from datetime import datetime, timedelta
import scipy.io
import os, time
import shutil
import sys


if len(sys.argv) < 5:
    print("Usage: python prep_boundary_cond_CO2_BCK.py <res> <sim_start_time> <sim_end_time>")
    res = "3km"
    sim_time = ["2012-07-01 18:00:00", "2012-07-03 00:00:00"]
else:   
    res = sys.argv[1]  
    sim_time = [sys.argv[2]+" "+sys.argv[3], sys.argv[4]+" "+sys.argv[5]]

# Setting important paths to files and directories
wrfbdy_dir_path = f"/scratch/c7071034/WRF_{res}/test/em_real"
CAMS_data_dir_path = "/scratch/c7071034/DATA/CAMS/"
CAMS_interpolation_indices_file_path = (
    "/home/c707/c7071034/Github/WRF_VPRM_pre/settings/interp_indices.txt.npz"
)
IFS_L60_ab_file = (
    "/home/c707/c7071034/Github/WRF_VPRM_pre/settings/ecmwf_coeffs_L60.csv"
)

dates = pd.to_datetime(sim_time[0]).strftime("%Y-%m-%d")
year = pd.to_datetime(sim_time[0]).strftime("%Y")
month = pd.to_datetime(sim_time[0]).strftime("%m")
day = pd.to_datetime(sim_time[0]).strftime("%d")

# To each boundary date, assign a CAMS filename from which it will be read from:
# Ensure month and next month are formatted as two digits
month_int = int(month)
next_month_int = month_int + 1 if month_int < 12 else 1
next_month_str = f"{next_month_int:02d}"
next_year = str(int(year) + 1)

path_CAMS_file = os.path.join(
    CAMS_data_dir_path + f"ghg-reanalysis_CO2_{year}-{month}-01_{year}-{next_month_str}-01/data_mlev.nc"
)
path_CAMS_lnsp_file = os.path.join(
    CAMS_data_dir_path + f"ghg-reanalysis_lnsp_{year}-{month}-01_{year}-{next_month_str}-01.nc"
)
if month == "12":
    path_CAMS_file = os.path.join(
        CAMS_data_dir_path + f"ghg-reanalysis_CO2_{year}-{month}-01_{next_year}-01-01/data_mlev.nc"
    )
    path_CAMS_lnsp_file = os.path.join(
        CAMS_data_dir_path + f"ghg-reanalysis_lnsp_{year}-{month}-01_{next_year}-01-01.nc"
    )

# Prepare wrfbdy parameters shapes
wrfbdy_path = os.path.join(wrfbdy_dir_path, "wrfbdy_d01")
wrfbdy = xr.open_dataset(wrfbdy_path)

# simstart and simend times will be calculated from Times vector element available in wrfbdy_d01
# Define the initial boundary date and interval in hours
boundary_dates = [
    datetime.strptime(d.decode("utf-8"), "%Y-%m-%d_%H:%M:%S")
    for d in wrfbdy["Times"].values
]  ## UTC

bdy_interval_seconds = (boundary_dates[1] - boundary_dates[0]).total_seconds()
bdy_interval_hours = bdy_interval_seconds / 3600

boundary_dates.append(
    boundary_dates[-1] + timedelta(hours=bdy_interval_hours)
)  #### add a time
boundary_date = [d.strftime("%Y%m%d") for d in boundary_dates]

simstart_time = boundary_date[0]
simend_time = boundary_date[-1]

# In case manual setup is needed, execute the following lines:
# Frequency of the boundary is set in WRF namelist.input
# as interval_seconds in the &time_control section
# This value must be equal to that one.
# bdy_interval_seconds = 10800;


# If CAMS product also comes with the a, b coefficients.
##Am = double( ncread( path_CAMS_file( 1, : ), 'hyam' ) );
##Bm = double( ncread( path_CAMS_file( 1, : ), 'hybm' ) );

# Pressures are not available in the wrfbdy_d01 - only perturbation
# geopotential. Will use pressures from wrfinput, with the assumption
# that they do not change (close enough).

wrfinput_path = os.path.join(wrfbdy_dir_path, "wrfinput_d01")
##wrfinput_path = sprintf( '%swrfinput_d01_%4d-%02d-%02d', realoutput_path, year ,month, day );
print("Reading WRF pressures for d01")

wrf_pressure = (
    xr.open_dataset(wrfinput_path)["PB"].values[0]
    + xr.open_dataset(wrfinput_path)["P"].values[0]
)

# Important: boundaries are defined for each tracer with the
# suffix added to the variable name. Suffix codes are:
# _BXS, _BXE, BYS, BYE, denoting 'Boundary X Start',
# 'Boundary X-End', and similar for Y.
# Note that BXS means the WEST boundary (left?), and not south/bottom.
# Dimensions order is not obvious, be careful with that:
# Order: [N_WE] (or SN), [N_VERT_LVLS], [N_BOUNDARY_POINTS = 5 usually] and [TIME]

# Creating two dummy arrays that will be used as templates,
# one for WE boundaries and one for NS, as
# these have different sizes.
dummy_3d_scalar_field = np.zeros(
    (xr.open_dataset(wrfinput_path)["CO2_BIO"][0].shape)
) + (-999.0)
dummy_3d_scalar_field_2 = np.zeros(
    (xr.open_dataset(wrfinput_path)["CO2_BIO_2"][0].shape)
) + (-999.0)
dummy_3d_scalar_field_3 = np.zeros(
    (xr.open_dataset(wrfinput_path)["CO2_BIO_3"][0].shape)
) + (-999.0)
dummy_3d_scalar_field_4 = np.zeros(
    (xr.open_dataset(wrfinput_path)["CO2_BIO_4"][0].shape)
) + (-999.0)
dummy_3d_scalar_field_5 = np.zeros(
    (xr.open_dataset(wrfinput_path)["CO2_BIO_5"][0].shape)
) + (-999.0)
dummy_3d_scalar_field_6 = np.zeros(
    (xr.open_dataset(wrfinput_path)["CO2_BIO_REF"][0].shape)
) + (-999.0)

dummy_4d_X_scalar_field = np.zeros(
    (xr.open_dataset(wrfbdy_path)["CO2_BIO_BXS"].shape)
) + (
    -999.0
)  ### (bdy_width: 5,bottom_top: 50,south_north: 160)
dummy_4d_X_scalar_field_2 = np.zeros(
    (xr.open_dataset(wrfbdy_path)["CO2_BIO_2_BXS"].shape)
) + (
    -999.0
)  ### (bdy_width: 5,bottom_top: 50,south_north: 160)
dummy_4d_X_scalar_field_3 = np.zeros(
    (xr.open_dataset(wrfbdy_path)["CO2_BIO_3_BXS"].shape)
) + (
    -999.0
)  ### (bdy_width: 5,bottom_top: 50,south_north: 160)
dummy_4d_X_scalar_field_4 = np.zeros(
    (xr.open_dataset(wrfbdy_path)["CO2_BIO_4_BXS"].shape)
) + (
    -999.0
)  ### (bdy_width: 5,bottom_top: 50,south_north: 160)
dummy_4d_X_scalar_field_5 = np.zeros(
    (xr.open_dataset(wrfbdy_path)["CO2_BIO_5_BXS"].shape)
) + (
    -999.0
)  ### (bdy_width: 5,bottom_top: 50,south_north: 160)
dummy_4d_X_scalar_field_6 = np.zeros(
    (xr.open_dataset(wrfbdy_path)["CO2_BIO_REF_BXS"].shape)
) + (
    -999.0
)  ### (bdy_width: 5,bottom_top: 50,south_north: 160)


dummy_4d_Y_scalar_field = np.zeros(
    (xr.open_dataset(wrfbdy_path)["CO2_BIO_BYS"].shape)
) + (
    -999.0
)  ### (bdy_width: 5,bottom_top: 50,west_east: 347)
dummy_4d_Y_scalar_field_2 = np.zeros(
    (xr.open_dataset(wrfbdy_path)["CO2_BIO_2_BYS"].shape)
) + (
    -999.0
)  ### (bdy_width: 5,bottom_top: 50,west_east: 347)
dummy_4d_Y_scalar_field_3 = np.zeros(
    (xr.open_dataset(wrfbdy_path)["CO2_BIO_3_BYS"].shape)
) + (
    -999.0
)  ### (bdy_width: 5,bottom_top: 50,west_east: 347)
dummy_4d_Y_scalar_field_4 = np.zeros(
    (xr.open_dataset(wrfbdy_path)["CO2_BIO_4_BYS"].shape)
) + (
    -999.0
)  ### (bdy_width: 5,bottom_top: 50,west_east: 347)
dummy_4d_Y_scalar_field_5 = np.zeros(
    (xr.open_dataset(wrfbdy_path)["CO2_BIO_5_BYS"].shape)
) + (
    -999.0
)  ### (bdy_width: 5,bottom_top: 50,west_east: 347)
dummy_4d_Y_scalar_field_6 = np.zeros(
    (xr.open_dataset(wrfbdy_path)["CO2_BIO_REF_BYS"].shape)
) + (
    -999.0
)  ### (bdy_width: 5,bottom_top: 50,west_east: 347)

n_vertical_levels = dummy_4d_X_scalar_field.shape[2]
n_sn = dummy_4d_X_scalar_field.shape[3]
n_ew = dummy_4d_Y_scalar_field.shape[3]

# Next step is to fill the background fields using CAMS product values
# Precalculated interpolation indices will be needed
print("Loading in the pre-calculated nearest-neighbour interpolation indices.")
interpolation_indices = np.load(CAMS_interpolation_indices_file_path)[
    "cams_indices_d01"
]

# Reading a and b parameters from the model config L137
rawin = np.genfromtxt(
    IFS_L60_ab_file, delimiter=",", skip_header=1, dtype="<U25", encoding=None
)
a = rawin[:, 1]
b = rawin[:, 2]

# Output fields need to be initialized
# NOTE: Initialization gives TOO SMALL time dimension (by 1)
# This will be expanded without warning or notice in the bck loop
# too later calculate tendencies
co2_bck_bxs = dummy_4d_X_scalar_field.copy()
co2_bck_bxe = dummy_4d_X_scalar_field.copy()
co2_bck_bys = dummy_4d_Y_scalar_field.copy()
co2_bck_bye = dummy_4d_Y_scalar_field.copy()

co2_bck_btxs = 0 * dummy_4d_X_scalar_field.copy()
co2_bck_btxe = 0 * dummy_4d_X_scalar_field.copy()
co2_bck_btys = 0 * dummy_4d_Y_scalar_field.copy()
co2_bck_btye = 0 * dummy_4d_Y_scalar_field.copy()

# David:
# Indicator of 25 elements, from wrf time 00:00 ~ 23:00 (current day) + 00:00 (next day) => 25 timesteps
# indicating which wrf_time_idx should fill in with which CAMS_time_idx
# MA = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9];	# <- David: Handing hourly wrfbdy files from 3-hourly CAMS
# MA = [0, 0, 1, 1, 2, 2, 3, 3, 0]   				     	# <- David: Handing 3-hourly wrfbdy files from 6-hourly CAMS
# MA = [0, 1, 2, 3, 0]


def convert_kgkg_to_ppm(co2_kgkg_array):
    """
    Convert CO2 from kg/kg to ppm.

    Args:
        co2_kgkg_array (numpy array): Input array with CO2 in kg/kg.

    Returns:
        numpy array: Converted CO2 values in ppm.
    """
    # Calculate ppm
    M_CO2 = 44.01  # Molecular weight of CO2 (g/mol)
    M_AIR = 28.97  # Molecular weight of dry air (g/mol)
    co2_ppm = (co2_kgkg_array / (1 - co2_kgkg_array)) * (M_AIR / M_CO2) * 1e6
    return co2_ppm


for time_idx in range(len(boundary_dates)):
    current_time = boundary_dates[time_idx]

    print(current_time)
    # Read times from CAMS ml file
    # Times are given as hours since 1900-01-01 00:00:00 UTC
    cams_times = xr.open_dataset(path_CAMS_file)["valid_time"]
    cams_dates = pd.to_datetime(cams_times).strftime("%Y-%m-%d %H:%M:%S")

    # Check if one of the times is the same as current time:
    posw = int(time.mktime(current_time.timetuple()))
    posc = [int(time.mktime(pd.to_datetime(d).timetuple())) for d in cams_times.values]
    reading_from = path_CAMS_file
    cams_time_idx = posc.index(posw) if posw in posc else None

    # Now read appripriate CAMS fields, but only for the selected time.
    # (lev,lat,lon)  == (137, 451, 900)
    cams_co2_kgkg = cdf.Dataset(path_CAMS_file, "r").variables["co2"][
        cams_time_idx, :, :, :
    ]

    cams_co2 = convert_kgkg_to_ppm(cams_co2_kgkg)

    # cams_co2 = ncread( path_CAMS_file, 'co2', nc_start_vector, nc_count_vector  ) * (28.97/44.01)*1e6;

    # Read surface pressure for current time
    nc_file = cdf.Dataset(path_CAMS_lnsp_file, "r")
    cams_lnsp = nc_file.variables["lnsp"][
        cams_time_idx, :, :, :
    ]  ## (time, level, latitude, longitude)
    cams_pressure = np.exp(cams_lnsp)

    # Dummy background fields:
    wrf_init_CO2_BCK = np.zeros(dummy_3d_scalar_field.shape) + (-999.0)

    # Proceed to read the coordinates for each respective boundary
    # It's actually simpler to assign values after interpolating to the
    # whole domain. Not strictly speaking necessary, but works, so...

    # Version 2: Doing bands separately:
    # First: WEST and EAST boundaries (X)

    for lvl_idx in range(n_vertical_levels):
        print(f"YS & YE (BOTTOM and TOP), lvl {lvl_idx+1}/{n_vertical_levels}")
        for lat_idx in [i for i in range(5)] + [i for i in range(n_sn - 5, n_sn)]:
            for lon_idx in range(n_ew):
                # Get CAMS surface pressure
                lat_idx_nearest = int(interpolation_indices[lat_idx, lon_idx, 1])
                lon_idx_nearest = int(interpolation_indices[lat_idx, lon_idx, 0])
                surface_pressure = cams_pressure[0, lat_idx_nearest, lon_idx_nearest]

                # Calculate CAMS vertical pressures for the available levels
                cams_v_pressures = surface_pressure * b.astype(float) + a.astype(float)

                # Get WRF levels
                wrf_v_pressures = np.squeeze(wrf_pressure[:, lat_idx, lon_idx])
                # Find the nearest pressure level in CAMS for each WRF level
                difference = np.abs(cams_v_pressures - wrf_v_pressures[lvl_idx])
                cams_nearest_lvl_idx = min(np.where(difference == min(difference)))[0]

                cams_indices = np.array(
                    [cams_nearest_lvl_idx, lat_idx_nearest, lon_idx_nearest]
                )

                wrf_init_CO2_BCK[lvl_idx, lat_idx, lon_idx] = cams_co2[
                    cams_indices[0], cams_indices[1], cams_indices[2]
                ]

    for lvl_idx in range(n_vertical_levels):
        print(f"XS and XE (LEFT and RIGHT), lvl {lvl_idx+1}/{n_vertical_levels}")
        for lon_idx in [i for i in range(5)] + [i for i in range(n_ew - 5, n_ew)]:
            for lat_idx in range(n_sn):

                # Get CAMS surface pressure
                lat_idx_nearest = int(interpolation_indices[lat_idx, lon_idx, 1])
                lon_idx_nearest = int(interpolation_indices[lat_idx, lon_idx, 0])
                surface_pressure = cams_pressure[0, lat_idx_nearest, lon_idx_nearest]

                # Calculate CAMS vertical pressures for the available levels
                cams_v_pressures = surface_pressure * b.astype(float) + a.astype(float)

                # Get WRF levels
                wrf_v_pressures = np.squeeze(wrf_pressure[:, lat_idx, lon_idx])
                # Find the nearest pressure level in CAMS for each WRF level
                difference = np.abs(cams_v_pressures - wrf_v_pressures[lvl_idx])
                cams_nearest_lvl_idx = min(np.where(difference == min(difference)))[0]

                cams_indices = np.array(
                    [cams_nearest_lvl_idx, lat_idx_nearest, lon_idx_nearest]
                )

                # Assign appropriate values to the boundaries
                wrf_init_CO2_BCK[lvl_idx, lat_idx, lon_idx] = cams_co2[
                    cams_indices[0], cams_indices[1], cams_indices[2]
                ]

    if time_idx < len(boundary_dates) - 1:
        # Full fields now interpolated.
        # Now need to save into respective boundary objects:
        ##  co2_bck_bxs( :, :, :, time_idx ) = permute( wrf_init_CO2_BCK(     1:5  ,      :   , : ), [2 3 1] );
        ##  co2_bck_bxe( :, :, :, time_idx ) = permute( wrf_init_CO2_BCK( end-4:end,      :   , : ), [2 3 1] );
        ##  co2_bck_bys( :, :, :, time_idx ) = permute( wrf_init_CO2_BCK(      :   ,     1:5  , : ), [1 3 2] );
        ##  co2_bck_bye( :, :, :, time_idx ) = permute( wrf_init_CO2_BCK(      :   , end-4:end, : ), [1 3 2] );

        co2_bck_bxs[time_idx, :, :, :] = np.transpose(
            wrf_init_CO2_BCK[:, :, 0:5], axes=(2, 0, 1)
        )
        co2_bck_bxe[time_idx, :, :, :] = np.transpose(
            wrf_init_CO2_BCK[:, :, -5:], axes=(2, 0, 1)
        )
        co2_bck_bys[time_idx, :, :, :] = np.transpose(
            wrf_init_CO2_BCK[:, 0:5, :], axes=(1, 0, 2)
        )
        co2_bck_bye[time_idx, :, :, :] = np.transpose(
            wrf_init_CO2_BCK[:, -5:, :], axes=(1, 0, 2)
        )

    else:
        co2_bck_bxs = np.insert(
            co2_bck_bxs,
            co2_bck_bxs.shape[0],
            np.transpose(wrf_init_CO2_BCK[:, :, 0:5], axes=(2, 0, 1)),
            0,
        )
        co2_bck_bxe = np.insert(
            co2_bck_bxe,
            co2_bck_bxe.shape[0],
            np.transpose(wrf_init_CO2_BCK[:, :, -5:], axes=(2, 0, 1)),
            0,
        )
        co2_bck_bys = np.insert(
            co2_bck_bys,
            co2_bck_bys.shape[0],
            np.transpose(wrf_init_CO2_BCK[:, 0:5, :], axes=(1, 0, 2)),
            0,
        )
        co2_bck_bye = np.insert(
            co2_bck_bye,
            co2_bck_bye.shape[0],
            np.transpose(wrf_init_CO2_BCK[:, -5:, :], axes=(1, 0, 2)),
            0,
        )

# ## Added by David Ho : linear interpolation and overwrite the missing timesteps
# print("Begin linear interpolation for missing timesteps")
# tmax_idx = len(boundary_dates)  # 9
# boundary_dates = [d.strftime("%Y-%m-%d %H:%M:%S") for d in boundary_dates]

# for time_miss in range(0, tmax_idx-1, 2):
#     print(
#         f"overwriting {boundary_dates[time_miss]} with ({boundary_dates[time_miss+1]} + {boundary_dates[time_miss-1]})/2"
#     )

#     co2_bck_bxs[time_miss, :, :, :] = (
#         co2_bck_bxs[time_miss + 1, :, :, :] + co2_bck_bxs[time_miss - 1, :, :, :]
#     ) / 2
#     co2_bck_bxe[time_miss, :, :, :] = (
#         co2_bck_bxe[time_miss + 1, :, :, :] + co2_bck_bxe[time_miss - 1, :, :, :]
#     ) / 2
#     co2_bck_bys[time_miss, :, :, :] = (
#         co2_bck_bys[time_miss + 1, :, :, :] + co2_bck_bys[time_miss - 1, :, :, :]
#     ) / 2
#     co2_bck_bye[time_miss, :, :, :] = (
#         co2_bck_bye[time_miss + 1, :, :, :] + co2_bck_bye[time_miss - 1, :, :, :]
#     ) / 2

# Tendencies can be run separately.
# Here, using slightly modified code from Julia M.
tmax_idx = len(boundary_dates)
print("Assigning tendencies")


for time_idx in range(tmax_idx - 1):
    # Separately for EAST and WEST (X)
    for x in range(n_sn):
        for lvl_idx in range(n_vertical_levels):
            for y in range(co2_bck_bxs.shape[1]):  # Usually 5

                co2_bck_btxs[time_idx, y, lvl_idx, x] = (
                    co2_bck_bxs[time_idx + 1, y, lvl_idx, x]
                    - co2_bck_bxs[time_idx, y, lvl_idx, x]
                ) / bdy_interval_seconds

                co2_bck_btxe[time_idx, y, lvl_idx, x] = (
                    co2_bck_bxe[time_idx + 1, y, lvl_idx, x]
                    - co2_bck_bxe[time_idx, y, lvl_idx, x]
                ) / bdy_interval_seconds

    for x in range(n_ew):
        for lvl_idx in range(n_vertical_levels):
            for y in range(co2_bck_bys.shape[1]):  # Usually 5

                co2_bck_btys[time_idx, y, lvl_idx, x] = (
                    co2_bck_bys[time_idx + 1, y, lvl_idx, x]
                    - co2_bck_bys[time_idx, y, lvl_idx, x]
                ) / bdy_interval_seconds

                co2_bck_btye[time_idx, y, lvl_idx, x] = (
                    co2_bck_bye[time_idx + 1, y, lvl_idx, x]
                    - co2_bck_bye[time_idx, y, lvl_idx, x]
                ) / bdy_interval_seconds

# Diagnostics:
# Finally, write everything out into wrfbdy_d01
print("Writing boundary fields into wrfbdy_d01 (CO2_BCK_BXS, CO2_BCK_BXE, etc.)")

# NOTE:
# Dimension of XXX_bck_bxs objects has been artificially expanded for times
# to calculate tendencies easily (loop above). Now the extra dimension is not
# needed anymore!

tmax_idx = len(boundary_dates)
writable_file_path = wrfbdy_path + "_updated"
shutil.copy(wrfbdy_path, writable_file_path)
ncid = cdf.Dataset(writable_file_path, "r+")

ncid.variables["CO2_BCK_BXS"][:] = co2_bck_bxs[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BCK_BXE"][:] = co2_bck_bxe[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BCK_BYS"][:] = co2_bck_bys[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BCK_BYE"][:] = co2_bck_bye[0 : tmax_idx - 1, :, :, :]

ncid.variables["CO2_BIO_BXS"][:] = co2_bck_bxs[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_BXE"][:] = co2_bck_bxe[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_BYS"][:] = co2_bck_bys[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_BYE"][:] = co2_bck_bye[0 : tmax_idx - 1, :, :, :]

ncid.variables["CO2_BIO_2_BXS"][:] = co2_bck_bxs[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_2_BXE"][:] = co2_bck_bxe[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_2_BYS"][:] = co2_bck_bys[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_2_BYE"][:] = co2_bck_bye[0 : tmax_idx - 1, :, :, :]

ncid.variables["CO2_BIO_3_BXS"][:] = co2_bck_bxs[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_3_BXE"][:] = co2_bck_bxe[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_3_BYS"][:] = co2_bck_bys[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_3_BYE"][:] = co2_bck_bye[0 : tmax_idx - 1, :, :, :]

ncid.variables["CO2_BIO_4_BXS"][:] = co2_bck_bxs[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_4_BXE"][:] = co2_bck_bxe[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_4_BYS"][:] = co2_bck_bys[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_4_BYE"][:] = co2_bck_bye[0 : tmax_idx - 1, :, :, :]

ncid.variables["CO2_BIO_5_BXS"][:] = co2_bck_bxs[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_5_BXE"][:] = co2_bck_bxe[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_5_BYS"][:] = co2_bck_bys[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_5_BYE"][:] = co2_bck_bye[0 : tmax_idx - 1, :, :, :]

ncid.variables["CO2_BIO_REF_BXS"][:] = co2_bck_bxs[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_REF_BXE"][:] = co2_bck_bxe[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_REF_BYS"][:] = co2_bck_bys[0 : tmax_idx - 1, :, :, :]
ncid.variables["CO2_BIO_REF_BYE"][:] = co2_bck_bye[0 : tmax_idx - 1, :, :, :]


print(
    "Writing boundary tendency fields into wrfbdy_d01 (CO2_BCK_BTXS, CO2_BCK_BTXE, etc.)"
)

ncid.variables["CO2_BCK_BTXS"][:] = co2_bck_btxs
ncid.variables["CO2_BCK_BTXE"][:] = co2_bck_btxe
ncid.variables["CO2_BCK_BTYS"][:] = co2_bck_btys
ncid.variables["CO2_BCK_BTYE"][:] = co2_bck_btye

ncid.variables["CO2_BIO_BTXS"][:] = co2_bck_btxs
ncid.variables["CO2_BIO_BTXE"][:] = co2_bck_btxe
ncid.variables["CO2_BIO_BTYS"][:] = co2_bck_btys
ncid.variables["CO2_BIO_BTYE"][:] = co2_bck_btye

ncid.variables["CO2_BIO_2_BTXS"][:] = co2_bck_btxs
ncid.variables["CO2_BIO_2_BTXE"][:] = co2_bck_btxe
ncid.variables["CO2_BIO_2_BTYS"][:] = co2_bck_btys
ncid.variables["CO2_BIO_2_BTYE"][:] = co2_bck_btye

ncid.variables["CO2_BIO_3_BTXS"][:] = co2_bck_btxs
ncid.variables["CO2_BIO_3_BTXE"][:] = co2_bck_btxe
ncid.variables["CO2_BIO_3_BTYS"][:] = co2_bck_btys
ncid.variables["CO2_BIO_3_BTYE"][:] = co2_bck_btye

ncid.variables["CO2_BIO_4_BTXS"][:] = co2_bck_btxs
ncid.variables["CO2_BIO_4_BTXE"][:] = co2_bck_btxe
ncid.variables["CO2_BIO_4_BTYS"][:] = co2_bck_btys
ncid.variables["CO2_BIO_4_BTYE"][:] = co2_bck_btye

ncid.variables["CO2_BIO_5_BTXS"][:] = co2_bck_btxs
ncid.variables["CO2_BIO_5_BTXE"][:] = co2_bck_btxe
ncid.variables["CO2_BIO_5_BTYS"][:] = co2_bck_btys
ncid.variables["CO2_BIO_5_BTYE"][:] = co2_bck_btye

ncid.variables["CO2_BIO_REF_BTXS"][:] = co2_bck_btxs
ncid.variables["CO2_BIO_REF_BTXE"][:] = co2_bck_btxe
ncid.variables["CO2_BIO_REF_BTYS"][:] = co2_bck_btys
ncid.variables["CO2_BIO_REF_BTYE"][:] = co2_bck_btye

ncid.close()
print("Script completed.")
