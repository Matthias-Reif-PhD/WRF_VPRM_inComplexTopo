import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy.interpolate import griddata
from scipy.ndimage import binary_erosion, distance_transform_edt
from datetime import datetime
import glob
import os
from collections import defaultdict
import argparse
import sys


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


def extract_datetime_from_filename(filename):
    """
    Extract datetime from WRF filename assuming format 'wrfout_d0x_YYYY-MM-DD_HH:MM:SS'.
    """
    base_filename = os.path.basename(filename)
    date_str = base_filename.split("_")[-2] + "_" + base_filename.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")


def exctract_dPdT_timeseries(wrf_paths, start_date, end_date, sim_type):
    ############# INPUT ############
    csv_folder = "/scratch/c7071034/DATA/WRFOUT/csv/"
    interp_method = "nearest"  # 'linear', 'nearest', 'cubic'
    temp_gradient = -6.5  # K/km
    STD_TOPO = 200
    res_tag = "_1km"
    ref_sims = ["", "_REF"]  # "_REF" to use REF simulation or "" for tuned values
    val_at5C = 1  # limit value for max dGPPdT between 0-5°, below 0 its set to nan
    ###################################

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").date()
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").date()
    files_d01 = sorted(glob.glob(os.path.join(wrf_paths[1], f"wrfout_d01*")))
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

    df_out_dPdT = pd.DataFrame()
    for ref_sim in ref_sims:
        for wrf_file in file_list:
            # Load the NetCDF file
            wrf_file_d02 = wrf_file.replace("d01", "d02")
            nc_fid1km = nc.Dataset(os.path.join(wrf_paths[0], wrf_file_d02), "r")
            GPP_1km = -nc_fid1km.variables[f"EBIO_GEE{ref_sim}"][0, 0, 10:-10, 10:-10]
            RECO_1km = nc_fid1km.variables[f"EBIO_RES{ref_sim}"][0, 0, 10:-10, 10:-10]
            HGT_1km = nc_fid1km.variables["HGT"][0, 10:-10, 10:-10]
            T2_1km = nc_fid1km.variables["T2"][0, 10:-10, 10:-10] - 273.15
            lats_fine = nc_fid1km.variables["XLAT"][0, 10:-10, 10:-10]
            lons_fine = nc_fid1km.variables["XLONG"][0, 10:-10, 10:-10]
            stdh_topo_1km = nc_fid1km.variables["VAR"][0, 10:-10, 10:-10]
            stdh_mask = stdh_topo_1km >= STD_TOPO

            dGPPdT_ref = -nc_fid1km.variables[f"EBIO_GEE_DPDT{ref_sim}"][
                0, 0, 10:-10, 10:-10
            ]
            dRECOdT_ref = nc_fid1km.variables[f"EBIO_RES_DPDT{ref_sim}"][
                0, 0, 10:-10, 10:-10
            ]

            dGPPdT_ref[T2_1km < 0] = np.nan
            mask_0to5 = (T2_1km >= 0) & (T2_1km <= 5)
            dGPPdT_ref[mask_0to5] = val_at5C

            for wrf_path in wrf_paths[1:]:
                nc_fidcoarsegrid = nc.Dataset(os.path.join(wrf_path, wrf_file), "r")
                GPP_coarsegrid = -nc_fidcoarsegrid.variables[f"EBIO_GEE{ref_sim}"][
                    0, 0, :, :
                ]
                RECO_coarsegrid = nc_fidcoarsegrid.variables[f"EBIO_RES{ref_sim}"][
                    0, 0, :, :
                ]
                HGT_coarsegrid = nc_fidcoarsegrid.variables["HGT"][0]
                T2_coarsegrid = nc_fidcoarsegrid.variables["T2"][0] - 273.15
                veg_type = nc_fidcoarsegrid.variables["IVGTYP"][0, :, :]
                lats_coarsegrid = nc_fidcoarsegrid.variables["XLAT"][0, :, :]
                lons_coarsegrid = nc_fidcoarsegrid.variables["XLONG"][0, :, :]

                proj_GPP_coarsegrid = proj_on_finer_WRF_grid(
                    lats_coarsegrid,
                    lons_coarsegrid,
                    GPP_coarsegrid,
                    lats_fine,
                    lons_fine,
                    GPP_1km,
                    interp_method,
                )
                proj_RECO_coarsegrid = proj_on_finer_WRF_grid(
                    lats_coarsegrid,
                    lons_coarsegrid,
                    RECO_coarsegrid,
                    lats_fine,
                    lons_fine,
                    RECO_1km,
                    interp_method,
                )
                proj_HGT_coarsegrid = proj_on_finer_WRF_grid(
                    lats_coarsegrid,
                    lons_coarsegrid,
                    HGT_coarsegrid,
                    lats_fine,
                    lons_fine,
                    HGT_1km,
                    interp_method,
                )
                proj_T2_coarsegrid = proj_on_finer_WRF_grid(
                    lats_coarsegrid,
                    lons_coarsegrid,
                    T2_coarsegrid,
                    lats_fine,
                    lons_fine,
                    T2_1km,
                    interp_method,
                )
                new_landmask = generate_coastal_mask(
                    veg_type, buffer_km=30.0, grid_spacing_km=50.0
                )  # TODO adopt to resolution
                proj_landmask_coarsegrid = proj_on_finer_WRF_grid(
                    lats_coarsegrid,
                    lons_coarsegrid,
                    new_landmask,
                    lats_fine,
                    lons_fine,
                    HGT_1km,
                    interp_method,
                )

                diff_HGT = proj_HGT_coarsegrid - HGT_1km
                diff_HGT[proj_landmask_coarsegrid * stdh_mask == 0] = np.nan
                conv_factor = 1 / 3600

                dT_calc = diff_HGT / 1000 * temp_gradient
                dT_model = proj_T2_coarsegrid - T2_1km
                dGPP_calc = dGPPdT_ref * conv_factor * dT_calc
                dGPP_model = dGPPdT_ref * conv_factor * dT_model
                dGPP_real = (proj_GPP_coarsegrid - GPP_1km) * conv_factor
                dRECO_calc = dRECOdT_ref * conv_factor * dT_calc
                dRECO_model = dRECOdT_ref * conv_factor * dT_model
                dRECO_real = (proj_RECO_coarsegrid - RECO_1km) * conv_factor

                stdh_mask[T2_1km < 5] = False
                dT_calc[proj_landmask_coarsegrid * stdh_mask == 0] = np.nan
                dT_model[proj_landmask_coarsegrid * stdh_mask == 0] = np.nan
                dGPP_calc[proj_landmask_coarsegrid * stdh_mask == 0] = np.nan
                dGPP_model[proj_landmask_coarsegrid * stdh_mask == 0] = np.nan
                dGPP_real[proj_landmask_coarsegrid * stdh_mask == 0] = np.nan
                dRECO_calc[proj_landmask_coarsegrid * stdh_mask == 0] = np.nan
                dRECO_model[proj_landmask_coarsegrid * stdh_mask == 0] = np.nan
                dRECO_real[proj_landmask_coarsegrid * stdh_mask == 0] = np.nan

                try:
                    dT_calc_mean = np.nanmean(dT_calc)
                    dT_model_mean = np.nanmean(dT_model)
                    dGPP_calc_mean = np.nanmean(dGPP_calc)
                    dGPP_model_mean = np.nanmean(dGPP_model)
                    dGPP_real_mean = np.nanmean(dGPP_real)
                    dRECO_calc_mean = np.nanmean(dRECO_calc)
                    dRECO_model_mean = np.nanmean(dRECO_model)
                    dRECO_real_mean = np.nanmean(dRECO_real)
                except:
                    dT_calc_mean = np.nan
                    dT_model_mean = np.nan
                    dGPP_calc_mean = np.nan
                    dGPP_model_mean = np.nan
                    dGPP_real_mean = np.nan
                    dRECO_calc_mean = np.nan
                    dRECO_model_mean = np.nan
                    dRECO_real_mean = np.nan

                time = extract_datetime_from_filename(wrf_file)
                resolution = wrf_path.split("_")[2]

                # Create a fresh data_row for each resolution
                data_row = {
                    "dT_calc_mean_" + resolution: dT_calc_mean,
                    "dT_model_mean_" + resolution: dT_model_mean,
                    "dGPP_calc_mean_" + resolution: dGPP_calc_mean,
                    "dGPP_model_mean_" + resolution: dGPP_model_mean,
                    "dGPP_real_mean_" + resolution: dGPP_real_mean,
                    "dRECO_calc_mean_" + resolution: dRECO_calc_mean,
                    "dRECO_model_mean_" + resolution: dRECO_model_mean,
                    "dRECO_real_mean_" + resolution: dRECO_real_mean,
                }
                # Add the data_row to the DataFrame
                df_out_dPdT.loc[time, data_row.keys()] = data_row.values()
                # del dT_calc_mean, dT_model_mean, dGPP_calc_mean, dGPP_model_mean, dGPP_real_mean, dRECO_calc_mean, dRECO_model_mean, dRECO_real_mean

                print(
                    f"Processed {wrf_file} at resolution {resolution} with dPdT{res_tag}{ref_sim}"
                )

        # Save the DataFrame to a CSV file
        output_file = os.path.join(
            csv_folder,
            f"dPdT_timeseries_{start_date}_{end_date}{res_tag}{ref_sim}{sim_type}.csv",
        )
        df_out_dPdT.to_csv(output_file, index_label="datetime")
        print(f"Data saved to {output_file}")


def main():

    if len(sys.argv) > 1:  # to run on cluster
        parser = argparse.ArgumentParser(description="Description of your script")
        parser.add_argument(
            "-s", "--start", type=str, help="Format: 2012-01-01 00:00:00"
        )
        parser.add_argument("-e", "--end", type=str, help="Format: 2012-12-31 00:00:00")
        parser.add_argument(
            "-t",
            "--type",
            type=str,
            help="Format: '', '_parm_err' or '_cloudy'",
            default="",
        )
        args = parser.parse_args()
        start_date = args.start
        end_date = args.end
        sim_type = args.type
    else:  # to run locally
        start_date = "2012-01-01 00:00:00"
        end_date = "2012-12-31 00:00:00"
        sim_type = "_cloudy"  # "", "_parm_err" or "_cloudy"

    wrf_paths = [
        f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_1km{sim_type}",  # 1km resolution hat to be included, as dPdT is calculated from 1km to a coarse resolution
        f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_9km{sim_type}",
        f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km{sim_type}",
    ]

    exctract_dPdT_timeseries(wrf_paths, start_date, end_date, sim_type)


if __name__ == "__main__":
    main()
