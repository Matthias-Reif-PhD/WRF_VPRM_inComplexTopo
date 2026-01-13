import xarray as xr
import glob
import numpy as np

# Path to monthly CAMS NetCDF files
CAMS_data_dir_path = "/scratch/c7071034/DATA/CAMS/"
file_list = sorted(glob.glob(CAMS_data_dir_path + "ghg-reanalysis_surface_2012-*.nc"))

# Collect datasets
datasets = []

for file in file_list:
    ds = xr.open_dataset(file)

    # Sort and drop duplicates in each file
    ds = ds.sortby("valid_time")
    _, index = np.unique(ds["valid_time"], return_index=True)
    ds = ds.isel(valid_time=index)

    datasets.append(ds)

# Concatenate along time
ds_combined = xr.concat(datasets, dim="valid_time")

# 🔁 Final check: sort and drop duplicates globally (across all files)
ds_combined = ds_combined.sortby("valid_time")
_, index = np.unique(ds_combined["valid_time"], return_index=True)
ds_combined = ds_combined.isel(valid_time=index)

# Save to NetCDF
output_path = CAMS_data_dir_path + "ghg-reanalysis_surface_2012_full.nc"
ds_combined.to_netcdf(output_path)

print(f"Saved cleaned dataset with unique timestamps to:\n{output_path}")

# Compute time differences
time_diffs = np.diff(ds_combined["valid_time"].values)

# Convert to hours (assuming datetime64[ns])
time_diffs_hours = np.array([td / np.timedelta64(1, 'h') for td in time_diffs])

# Check if all diffs are exactly 3.0 hours
if np.allclose(time_diffs_hours, 3.0):
    print("✅ All time steps are consistently 3 hours apart.")
else:
    print("❌ Irregular time steps detected!")
    print("Unique step sizes (in hours):", np.unique(time_diffs_hours))