print("========================================================================")
print("Python WRF-GHG: IC from CAMS")
print("Michal Galkowski, MPI-BGC Jena")
print("modified by David Ho, MPI-BGC Jena")
print("For handling CAMS product from Santiago,")
print("Translated for python by Noelia Rojas, IFUSP Brazil")
print("over the Amazon domain.")

import numpy as np
import pandas as pd
import netCDF4 as cdf
import xarray as xr
from datetime import datetime
import scipy.io
import os
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
wrfinput_dir_path = f"/scratch/c7071034/WRF_{res}/test/em_real"
CAMS_data_dir_path = "/scratch/c7071034/DATA/CAMS/"

# Find wrfinput_d0* files and extract their endings
def find_wrfinput_files(dir_path):
    try:
        files = os.listdir(dir_path)
        wrfinput_files = [
            f for f in files if f.startswith("wrfinput_d0") and "updated" not in f
        ]
        extracted_endings = [f.split("_")[-1] for f in wrfinput_files]

        return extracted_endings
    except Exception as e:
        return {"error": str(e)}


# Execute the function and fill requested_domains
result = find_wrfinput_files(wrfinput_dir_path)
if isinstance(result, dict) and "error" in result:
    print(f"Error: {result['error']}")
else:
    requested_domains = list(result)
    print("Requested domains:", requested_domains)

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

CAMS_interpolation_indices_file_path = (
    "/home/c707/c7071034/Github/WRF_VPRM_pre/settings/interp_indices.txt.npz"
)
IFS_L60_ab_file = (
    "/home/c707/c7071034/Github/WRF_VPRM_pre/settings/ecmwf_coeffs_L60.csv"
)

wrfinput_path = os.path.join(wrfinput_dir_path, "wrfinput_d01")
wrfinput = xr.open_dataset(wrfinput_path)
simstart_time = datetime.strptime(
    wrfinput["Times"].values[0].decode("utf-8"), "%Y-%m-%d_%H:%M:%S"
)  ## UTC
simstart_time = simstart_time.strftime("%Y%m%d %H:%M:%S")


# Reading a and b parameters from the model config L137
rawin = np.genfromtxt(
    IFS_L60_ab_file, delimiter=",", skip_header=1, dtype="<U25", encoding=None
)
a = rawin[:, 1]
b = rawin[:, 2]

print("Getting CAMS latitudes and longitudes from:\n", path_CAMS_file)

cams_lat = xr.open_dataset(path_CAMS_file)["latitude"]
cams_lon = xr.open_dataset(path_CAMS_file)["longitude"]
mesh_lat, mesh_lon = np.meshgrid(cams_lat, cams_lon)
cams_times = xr.open_dataset(path_CAMS_file)["valid_time"]
cams_dates = pd.to_datetime(cams_times.values)
cams_timestamps = cams_dates.astype('int64') // 10**9  # convert ns to seconds
simstart_timestamp = int(datetime.strptime(simstart_time, "%Y%m%d %H:%M:%S").timestamp())

# Prefer exact match, fall back to nearest
match_indices = np.where(cams_timestamps == simstart_timestamp)[0]
if match_indices.size > 0:
    cams_time_idx = match_indices[0]
else:
    print(f"No exact match found for {simstart_time}, using nearest time.")
    cams_time_idx = np.argmin(np.abs(cams_timestamps - simstart_timestamp))

# Important info on differences between sp and lnsp in IFS
# http://www.iup.uni-bremen.de/~hilboll/blog/2018/12/understanding-surface-pressure-in-ecmwf-era5-reanalysis-data/

print("Getting CAMS lnsp (ln of surface pressure) from:\n ", path_CAMS_lnsp_file)

nc_file = cdf.Dataset(path_CAMS_lnsp_file, "r")
# Read the subarray of the 'lnsp' variable
cams_lnsp = nc_file.variables["lnsp"][
    cams_time_idx, :, :, :
]  ## (time, level, latitude, longitude)
cams_pressure = np.exp(cams_lnsp)


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


cams_co2_kgkg = cdf.Dataset(path_CAMS_file, "r").variables["co2"][
    cams_time_idx, :, :, :
]

cams_co2 = convert_kgkg_to_ppm(cams_co2_kgkg)

# Now execute the caluclation per-domain
for domain_idx in range(len(requested_domains)):
    print("Processing domain:", requested_domains[domain_idx])

    # Load CAMS interpolation indices for this domain
    print("Loading in the pre-calculated nearest-neighbour interipolation indices.")
    interpolation_indices = np.load(CAMS_interpolation_indices_file_path)[
        f"cams_indices_{requested_domains[domain_idx]}"
    ]

    # Load wrfinput file for the current domain
    wrfinput_path = os.path.join(
        wrfinput_dir_path, f"wrfinput_{requested_domains[domain_idx]}"
    )
    wrf_xlat = xr.open_dataset(wrfinput_path)["XLAT"].values[0]
    wrf_xlong = xr.open_dataset(wrfinput_path)["XLONG"].values[0]

    # Set offsets as initial values of biogenic tracers
    dummy_3d_scalar_field = xr.open_dataset(wrfinput_path)["CO2_BIO"].values[
        0
    ]  # (lev,sn,ew) == (80, 284, 338)

    n_vertical_levels = dummy_3d_scalar_field.shape[0]
    n_sn = dummy_3d_scalar_field.shape[1]
    n_ew = dummy_3d_scalar_field.shape[2]

    # Proceed to CAMS fields
    # For vertical assignment, nearest neighbour method will be used.
    print("Calculating pressures")

    wrf_pressure = (
        xr.open_dataset(wrfinput_path)["PB"].values[0]
        + xr.open_dataset(wrfinput_path)["P"].values[0]
    )

    wrf_init_CO2_BCK = np.zeros((dummy_3d_scalar_field.shape)) + (-999.0)

    for lat_idx in range(n_sn):
        print(f"Processing latitude band {lat_idx}/{n_sn}")
        for lon_idx in range(n_ew):
            # Get CAMS surface pressure
            surface_pressure = cams_pressure[
                :,
                interpolation_indices[lat_idx, lon_idx, 1].astype(int),
                interpolation_indices[lat_idx, lon_idx, 0].astype(int),
            ]
            cams_v_pressures = surface_pressure * b.astype(float) + a.astype(float)
            # Get WRF levels
            wrf_v_pressures = np.squeeze(wrf_pressure[:, lat_idx, lon_idx])
            for lvl_idx in range(n_vertical_levels):
                difference = np.abs(cams_v_pressures - wrf_v_pressures[lvl_idx])

                # min() added as there was a case where two levels had the same difference
                cams_nearest_lvl_idx = min(np.where(difference == min(difference)))[0]

                # Find the indices of the nearest horizontal grid points for the specified longitude and latitude
                lat_idx_nearest = int(interpolation_indices[lat_idx, lon_idx, 1])
                lon_idx_nearest = int(interpolation_indices[lat_idx, lon_idx, 0])

                #    print(np.array([cams_nearest_lvl_idx, lat_idx_nearest, lon_idx_nearest]))
                # Create a 1D NumPy array containing the indices of the nearest horizontal grid points and the nearest vertical level
                cams_indices = np.array(
                    [cams_nearest_lvl_idx, lat_idx_nearest, lon_idx_nearest]
                )
                wrf_init_CO2_BCK[lvl_idx, lat_idx, lon_idx] = cams_co2[
                    cams_nearest_lvl_idx, lat_idx_nearest, lon_idx_nearest
                ]

    print("Writing values from CAMS for CO2_BCK fields of wrfinput:")
    # had to copy the file to make it writable
    writable_file_path = wrfinput_path + "_updated"
    shutil.copy(wrfinput_path, writable_file_path)

    ncid = cdf.Dataset(writable_file_path, "r+")
    ncid.variables["CO2_BCK"][0] = wrf_init_CO2_BCK
    ncid.variables["CO2_BIO"][0] = wrf_init_CO2_BCK
    ncid.variables["CO2_BIO_2"][0] = wrf_init_CO2_BCK
    ncid.variables["CO2_BIO_3"][0] = wrf_init_CO2_BCK
    ncid.variables["CO2_BIO_4"][0] = wrf_init_CO2_BCK
    ncid.variables["CO2_BIO_5"][0] = wrf_init_CO2_BCK
    ncid.variables["CO2_BIO_REF"][0] = wrf_init_CO2_BCK
    ncid.close()


print("Script completed.")
