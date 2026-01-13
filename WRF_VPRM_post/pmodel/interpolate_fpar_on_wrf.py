import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from pyproj import Proj, Transformer
import os
from datetime import datetime
import gc

# File paths
# wrf_path = "/home/madse/Downloads/Fluxnet_Data/wrfout_d01_2012-07-01_12:00:00.nc"
# modis_dir = "/home/madse/Downloads/MODIS_data/split_timesteps"
month = 7
wrf_paths = [
    # "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250107_155336_ALPS_3km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250105_193347_ALPS_9km",
    # "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241229_112716_ALPS_27km",
    # "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241227_183215_ALPS_54km",
]

modis_dir = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/split_timesteps"
output_path_base = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/fpar_interpol"

for wrf_path_dx in wrf_paths:
    for day in range(1, 32):
        wrf_path = f"{wrf_path_dx}/wrfout_d01_2012-{month:02d}-{day:02d}_12:00:00"
        # Step 1: Extract WRF time and convert to cftime.DatetimeJulian
        wrf_ds = xr.open_dataset(wrf_path)
        wrf_time_str = (
            wrf_ds["Times"].values.tobytes().decode("utf-8").replace("_", "T")
        )
        wrf_time_dt = np.datetime64(wrf_time_str).astype("datetime64[s]").astype(object)

        # List MODIS files and find closest dates
        modis_files = [
            f
            for f in os.listdir(modis_dir)
            if f.startswith("MODIS_fpar_") and f.endswith(".nc")
        ]
        modis_dates = [
            datetime.strptime(f.split("_")[2].split(".")[0], "%Y%m%d")
            for f in modis_files
        ]
        modis_files_dates = list(zip(modis_files, modis_dates))
        modis_files_dates.sort(key=lambda x: abs((x[1] - wrf_time_dt).total_seconds()))
        closest_files = modis_files_dates[:2]
        closest_files.sort(key=lambda x: x[1])  # Ensure chronological order

        # Load closest MODIS files
        modis_before = xr.open_dataset(os.path.join(modis_dir, closest_files[0][0]))
        modis_after = xr.open_dataset(os.path.join(modis_dir, closest_files[1][0]))
        print(f"Loaded MODIS files: {closest_files[0][0]}, {closest_files[1][0]}")

        # Step 2: Georeference xdim and ydim
        xdim = modis_before["xdim"].values
        ydim = modis_before["ydim"].values

        # define projection information
        proj = Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
        transformer = Transformer.from_proj(proj, Proj(proj="latlong", datum="WGS84"))

        # Create a meshgrid for xdim and ydim
        xgrid, ygrid = np.meshgrid(xdim, ydim)

        # Transform to lat/lon
        modis_lons, modis_lats = transformer.transform(xgrid, ygrid)
        modis_lons = modis_lons.flatten()
        modis_lats = modis_lats.flatten()

        # Step 3: Interpolate MODIS FPAR to WRF time
        fpar_before = modis_before["Fpar_500m"]
        fpar_after = modis_after["Fpar_500m"]
        # set values to nan if they are larger than 1
        fpar_before.values[fpar_before.values > 1] = np.nan
        fpar_after.values[fpar_after.values > 1] = np.nan

        # Time weights for linear interpolation
        time_before = closest_files[0][1]
        time_after = closest_files[1][1]
        weight_before = (time_after - wrf_time_dt).total_seconds() / (
            time_after - time_before
        ).total_seconds()
        weight_after = 1 - weight_before

        # Perform time interpolation
        fpar_interpolated_time = (
            fpar_before * weight_before + fpar_after * weight_after
        ).values.flatten()

        # Step 4: Interpolate MODIS FPAR to WRF grid
        xlat = wrf_ds["XLAT"].values
        xlon = wrf_ds["XLONG"].values
        wrf_coords = np.column_stack([xlat.flatten(), xlon.flatten()])

        fpar_on_wrf_grid = griddata(
            np.column_stack([modis_lats, modis_lons]),
            fpar_interpolated_time,
            wrf_coords,
            method="linear",
        )
        fpar_on_wrf_grid = fpar_on_wrf_grid.reshape(xlat.shape)

        print("FPAR interpolated onto WRF grid.")

        # Ensure fpar_on_wrf_grid is 2D
        # fpar_on_wrf_grid = np.squeeze(fpar_on_wrf_grid)

        # Save to NetCDF
        wrf_path_dx_str = wrf_path_dx.split("_")[-1]
        output_path = (
            f"{output_path_base}/interpolated_fpar_{wrf_path_dx_str}_{wrf_time_str}.nc"
        )

        xr.DataArray(fpar_on_wrf_grid, name="Fpar_500m").to_netcdf(
            output_path, format="NETCDF4_CLASSIC"
        )
        # delete all variables
        del (
            fpar_on_wrf_grid,
            fpar_interpolated_time,
            fpar_before,
            fpar_after,
            modis_before,
            modis_after,
            wrf_ds,
        )
        gc.collect()
        print(f"Interpolated FPAR saved to {output_path}.")
