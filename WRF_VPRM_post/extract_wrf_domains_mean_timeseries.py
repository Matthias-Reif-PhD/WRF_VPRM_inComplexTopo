import netCDF4 as nc
import glob
import os
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import argparse
import sys


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
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_3km
):
    proj_var = griddata(
        (lats_coarse.flatten(), lons_coarse.flatten()),
        var_coarse.flatten(),
        (lats_fine, lons_fine),
        method="linear",
    ).reshape(WRF_var_3km.shape)
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
        points_coarse, var_coarse_reversed.flatten(), points_fine, method="linear"
    ).reshape(WRF_var_1km.shape)

    return proj_var


def extract_datetime_from_filename(filename):
    """
    Extract datetime from WRF filename assuming format 'wrfout_d0x_YYYY-MM-DD_HH:MM:SS'.
    """
    base_filename = os.path.basename(filename)
    date_str = base_filename.split("_")[-2] + "_" + base_filename.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")


def exctract_wrf_domains_mean_timeseries(wrf_paths, start_date, end_date, sim_type):
    ################################# INPUT ##############################################

    run_Pmodel = False  # set to True if you want to run Pmodel and Migliavacca RECO
    csv_folder = "/scratch/c7071034/DATA/WRFOUT/csv/"
    # set standard deviation of topography
    STD_TOPOs = [200]
    STD_TOPO_flags = ["gt"]  # "lt" lower than or "gt" greater than STD_TOPO
    ref_sims = ["", "_REF"]  # "_REF" to use REF simulation or "" for tuned values

    if run_Pmodel:
        migli_path = "/scratch/c7071034/DATA/RECO_Migli/"
        gpp_folder = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/gap_filled/gpp_pmodel/"
        subdaily = "_subdailyC3v2"  # "_subdailyC3v2" or "" to use subdaily GPP

    #######################################################################################
    # load CAMS data
    CAMS_path = "/scratch/c7071034/DATA/CAMS/ghg-reanalysis_surface_2012_full.nc"

    CAMS_data = nc.Dataset(CAMS_path)
    times_CAMS = CAMS_data.variables["valid_time"]

    # convert from accumulated values to instantaneous
    ssr_acc = CAMS_data.variables["ssrd"][:]  # shape: (time, lat, lon)
    ssr_flux = np.full_like(ssr_acc, np.nan, dtype=np.float64)
    ssr_flux[1:] = np.diff(ssr_acc, axis=0)  # 3-hourly, so 10800 seconds
    ssr_flux[ssr_flux < 0] = 0
    CAMS_data.variables["ssrd"] = ssr_flux  # Overwrite in memory

    CAMS_vars = ["fco2gpp", "fco2rec", "t2m", "ssrd"]
    factor_kgC = 1000 / 44.01 * 1000000  # consim_type from kgCO2/m2/s to  mumol/m2/s
    CAMS_factors = [factor_kgC, -factor_kgC, 273.15, 1 / 10800]

    WRF_factors = [-1 / 3600, 1 / 3600, 273.15, 1]
    columns = ["GPP", "RECO", "T2", "SWDOWN"]

    # Convert to datetime (but ignore time part for full-day selection)
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").date()
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").date()

    # Collect all files
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

    timestamps = [extract_datetime_from_filename(f) for f in file_list]
    time_index = pd.to_datetime(timestamps)

    for ref_sim in ref_sims:

        # Use glob to list all files in the directory
        WRF_vars = [
            "EBIO_GEE" + ref_sim,
            "EBIO_RES" + ref_sim,
            "T2",
            "SWDOWN",
        ]
        units = [
            " [mmol m² s⁻¹]",
            " [mmol m² s⁻¹]",
            " [K]",
            "[W m⁻²]",
        ]
        # name_vars = {
        #     "EBIO_GEE" + ref_sim: "WRF GPP",
        #     "EBIO_RES" + ref_sim: "WRF RECO",
        #     "T2": "WRF T2M",
        #     "EBIO_GEE_DPDT" + ref_sim: "WRF dGPPdT",
        #     "EBIO_RES_DPDT" + ref_sim: "WRF dRECOdT",
        # }

        for STD_TOPO in STD_TOPOs:
            for STD_TOPO_flag in STD_TOPO_flags:

                # Initialize empty DataFrames with time as the index
                df_out_3km = pd.DataFrame(index=time_index, columns=columns)
                df_out_9km = pd.DataFrame(index=time_index, columns=columns)
                df_out_27km = pd.DataFrame(index=time_index, columns=columns)
                df_out_54km = pd.DataFrame(index=time_index, columns=columns)
                df_out_1km = pd.DataFrame(index=time_index, columns=columns)
                df_out_cams = pd.DataFrame(index=time_index, columns=columns)
                data_row_3km = {col: 0 for col in df_out_1km.columns}
                data_row_9km = {col: 0 for col in df_out_1km.columns}
                data_row_27km = {col: 0 for col in df_out_1km.columns}
                data_row_54km = {col: 0 for col in df_out_1km.columns}
                data_row_1km = {col: 0 for col in df_out_1km.columns}
                data_row_cams = {col: 0 for col in df_out_1km.columns}
                # define col only for GPP
                if run_Pmodel:
                    df_out_P_3km = pd.DataFrame(index=time_index, columns=["GPP"])
                    df_out_P_9km = pd.DataFrame(index=time_index, columns=["GPP"])
                    df_out_P_27km = pd.DataFrame(index=time_index, columns=["GPP"])
                    df_out_P_54km = pd.DataFrame(index=time_index, columns=["GPP"])
                    df_out_P_1km = pd.DataFrame(index=time_index, columns=["GPP"])
                    data_row_P_3km = {col: 0 for col in df_out_P_1km.columns}
                    data_row_P_9km = {col: 0 for col in df_out_P_1km.columns}
                    data_row_P_27km = {col: 0 for col in df_out_P_1km.columns}
                    data_row_P_54km = {col: 0 for col in df_out_P_1km.columns}
                    data_row_P_1km = {col: 0 for col in df_out_P_1km.columns}
                    # define col only for RECO
                    df_out_M_3km = pd.DataFrame(index=time_index, columns=["RECO"])
                    df_out_M_9km = pd.DataFrame(index=time_index, columns=["RECO"])
                    df_out_M_27km = pd.DataFrame(index=time_index, columns=["RECO"])
                    df_out_M_54km = pd.DataFrame(index=time_index, columns=["RECO"])
                    df_out_M_1km = pd.DataFrame(index=time_index, columns=["RECO"])
                    data_row_M_3km = {col: 0 for col in df_out_M_1km.columns}
                    data_row_M_9km = {col: 0 for col in df_out_M_1km.columns}
                    data_row_M_27km = {col: 0 for col in df_out_M_1km.columns}
                    data_row_M_54km = {col: 0 for col in df_out_M_1km.columns}
                    data_row_M_1km = {col: 0 for col in df_out_M_1km.columns}

                for wrf_file in file_list:
                    time = extract_datetime_from_filename(wrf_file)
                    print("processing ", time)
                    for (
                        WRF_var,
                        CAMS_var,
                        factor,
                        unit,
                        column,
                        WRF_factor,
                    ) in zip(
                        WRF_vars,
                        CAMS_vars,
                        CAMS_factors,
                        units,
                        columns,
                        WRF_factors,
                    ):
                        # WRF
                        i = 0
                        # Loop through the files for the timestep
                        # for nc_f1 in file_list_27km:
                        nc_fid54km = nc.Dataset(
                            os.path.join(wrf_paths[4], wrf_file), "r"
                        )
                        nc_fid27km = nc.Dataset(
                            os.path.join(wrf_paths[3], wrf_file), "r"
                        )
                        nc_fid9km = nc.Dataset(
                            os.path.join(wrf_paths[2], wrf_file), "r"
                        )
                        nc_fid3km = nc.Dataset(
                            os.path.join(wrf_paths[1], wrf_file), "r"
                        )
                        # relace d01 by d02 in the string of wrf_file
                        wrf_file_d02 = wrf_file.replace("d01", "d02")
                        nc_fid1km = nc.Dataset(
                            os.path.join(wrf_paths[0], wrf_file_d02), "r"
                        )

                        times_variable = nc_fid1km.variables["Times"]
                        start_date_bytes = times_variable[0, :].tobytes()
                        start_date_str = start_date_bytes.decode("utf-8")
                        lats_fine = nc_fid1km.variables["XLAT"][0, 10:-10, 10:-10]
                        lons_fine = nc_fid1km.variables["XLONG"][0, 10:-10, 10:-10]
                        landmask = nc_fid1km.variables["LANDMASK"][0, 10:-10, 10:-10]
                        land_mask = landmask == 1

                        if WRF_var == "T2":
                            WRF_var_1km = (
                                nc_fid1km.variables[WRF_var][0, 10:-10, 10:-10]
                                - WRF_factor
                            )
                            WRF_var_3km = (
                                nc_fid3km.variables[WRF_var][0, :, :] - WRF_factor
                            )
                            WRF_var_9km = (
                                nc_fid9km.variables[WRF_var][0, :, :] - WRF_factor
                            )
                            WRF_var_27km = (
                                nc_fid27km.variables[WRF_var][0, :, :] - WRF_factor
                            )
                            WRF_var_54km = (
                                nc_fid54km.variables[WRF_var][0, :, :] - WRF_factor
                            )
                        elif WRF_var == "SWDOWN":
                            WRF_var_1km = nc_fid1km.variables[WRF_var][
                                0, 10:-10, 10:-10
                            ]
                            WRF_var_3km = nc_fid3km.variables[WRF_var][0, :, :]
                            WRF_var_9km = nc_fid9km.variables[WRF_var][0, :, :]
                            WRF_var_27km = nc_fid27km.variables[WRF_var][0, :, :]
                            WRF_var_54km = nc_fid54km.variables[WRF_var][0, :, :]
                        else:
                            WRF_var_1km = (
                                nc_fid1km.variables[WRF_var][0, 0, 10:-10, 10:-10]
                                * WRF_factor
                            )
                            WRF_var_3km = (
                                nc_fid3km.variables[WRF_var][0, 0, :, :] * WRF_factor
                            )
                            WRF_var_9km = (
                                nc_fid9km.variables[WRF_var][0, 0, :, :] * WRF_factor
                            )
                            WRF_var_27km = (
                                nc_fid27km.variables[WRF_var][0, 0, :, :] * WRF_factor
                            )
                            WRF_var_54km = (
                                nc_fid54km.variables[WRF_var][0, 0, :, :] * WRF_factor
                            )

                        lats_3km = nc_fid3km.variables["XLAT"][0, :, :]
                        lons_3km = nc_fid3km.variables["XLONG"][0, :, :]
                        landmask_1 = nc_fid3km.variables["LANDMASK"][0, :, :]
                        WRF_var_3km[landmask_1 == 0] = np.nan
                        proj_WRF_var_3km = proj_on_finer_WRF_grid(
                            lats_3km,
                            lons_3km,
                            WRF_var_3km,
                            lats_fine,
                            lons_fine,
                            WRF_var_1km,
                        )

                        lats_9km = nc_fid9km.variables["XLAT"][0, :, :]
                        lons_9km = nc_fid9km.variables["XLONG"][0, :, :]
                        landmask_2 = nc_fid9km.variables["LANDMASK"][0, :, :]
                        WRF_var_9km[landmask_2 == 0] = np.nan
                        proj_WRF_var_9km = proj_on_finer_WRF_grid(
                            lats_9km,
                            lons_9km,
                            WRF_var_9km,
                            lats_fine,
                            lons_fine,
                            WRF_var_1km,
                        )

                        lats_27km = nc_fid27km.variables["XLAT"][0, :, :]
                        lons_27km = nc_fid27km.variables["XLONG"][0, :, :]
                        landmask_1 = nc_fid27km.variables["LANDMASK"][0, :, :]
                        WRF_var_27km[landmask_1 == 0] = np.nan
                        proj_WRF_var_27km = proj_on_finer_WRF_grid(
                            lats_27km,
                            lons_27km,
                            WRF_var_27km,
                            lats_fine,
                            lons_fine,
                            WRF_var_1km,
                        )

                        lats_54km = nc_fid54km.variables["XLAT"][0, :, :]
                        lons_54km = nc_fid54km.variables["XLONG"][0, :, :]
                        landmask_1 = nc_fid54km.variables["LANDMASK"][0, :, :]

                        WRF_var_54km[landmask_1 == 0] = np.nan
                        proj_WRF_var_54km = proj_on_finer_WRF_grid(
                            lats_54km,
                            lons_54km,
                            WRF_var_54km,
                            lats_fine,
                            lons_fine,
                            WRF_var_1km,
                        )

                        stdh_topo_1km = nc_fid1km.variables["VAR"][0, 10:-10, 10:-10]
                        if STD_TOPO_flag == "gt":
                            stdh_mask = stdh_topo_1km >= STD_TOPO
                        elif STD_TOPO_flag == "lt":
                            stdh_mask = stdh_topo_1km < STD_TOPO
                        mask = land_mask * stdh_mask

                        masked_WRF_var_1km = WRF_var_1km.copy()
                        masked_WRF_var_1km[~(land_mask & stdh_mask)] = np.nan
                        WRF_var_1km_topo_m = np.nanmean(masked_WRF_var_1km)
                        WRF_var_3km_topo = np.nanmean(proj_WRF_var_3km[mask])
                        WRF_var_9km_topo = np.nanmean(proj_WRF_var_9km[mask])
                        WRF_var_27km_topo = np.nanmean(proj_WRF_var_27km[mask])
                        WRF_var_54km_topo = np.nanmean(proj_WRF_var_54km[mask])
                        # process CAMS data if times fit
                        start_date_nc_f1 = datetime.strptime(
                            start_date_str, "%Y-%m-%d_%H:%M:%S"
                        )
                        new_time = start_date_nc_f1
                        j = 0
                        CAMS_topo = np.nan
                        for time_CAMS in times_CAMS:
                            date_CAMS = (
                                datetime(1970, 1, 1)
                                + timedelta(seconds=int(time_CAMS))
                                # - timedelta(hours=1)
                            )
                            j = j + 1
                            if new_time == date_CAMS:
                                lat_CAMS = CAMS_data.variables["latitude"][:]
                                lon_CAMS = CAMS_data.variables["longitude"][:]
                                if WRF_var == "T2":
                                    var_CAMS = (
                                        CAMS_data.variables[CAMS_var][j - 1, :, :].data
                                        - factor
                                    )  # convert unit to mmol m-2 s-1
                                else:
                                    var_CAMS = (
                                        CAMS_data.variables[CAMS_var][j - 1, :, :].data
                                        * factor
                                    )  # convert unit to mmol m-2 s-1
                                CAMS_proj = proj_CAMS_on_WRF_grid(
                                    lat_CAMS,
                                    lon_CAMS,
                                    var_CAMS,
                                    lats_fine,
                                    lons_fine,
                                    WRF_var_1km,
                                )

                                CAMS_topo = np.mean(CAMS_proj[mask])

                        data_row_1km[column] = WRF_var_1km_topo_m
                        data_row_3km[column] = WRF_var_3km_topo
                        data_row_9km[column] = WRF_var_9km_topo
                        data_row_27km[column] = WRF_var_27km_topo
                        data_row_54km[column] = WRF_var_54km_topo
                        data_row_cams[column] = CAMS_topo

                        if run_Pmodel:
                            if column == "GPP":
                                time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
                                nc_fid1km = nc.Dataset(
                                    f"{gpp_folder}gpp_pmodel{subdaily}_1km_"
                                    + time_str
                                    + ".nc",
                                    "r",
                                )
                                gpp_P_1km = nc_fid1km.variables["GPP_Pmodel"][
                                    :, :
                                ].copy()
                                nc_fid3km = nc.Dataset(
                                    f"{gpp_folder}gpp_pmodel{subdaily}_3km_"
                                    + time_str
                                    + ".nc",
                                    "r",
                                )
                                gpp_P_3km = nc_fid3km.variables["GPP_Pmodel"][
                                    :, :
                                ].copy()
                                nc_fid9km = nc.Dataset(
                                    f"{gpp_folder}gpp_pmodel{subdaily}_9km_"
                                    + time_str
                                    + ".nc",
                                    "r",
                                )
                                gpp_P_9km = nc_fid9km.variables["GPP_Pmodel"][
                                    :, :
                                ].copy()
                                nc_fid27km = nc.Dataset(
                                    f"{gpp_folder}gpp_pmodel{subdaily}_27km_"
                                    + time_str
                                    + ".nc",
                                    "r",
                                )
                                gpp_P_27km = nc_fid27km.variables["GPP_Pmodel"][
                                    :, :
                                ].copy()
                                nc_fid54km = nc.Dataset(
                                    f"{gpp_folder}gpp_pmodel{subdaily}_54km_"
                                    + time_str
                                    + ".nc",
                                    "r",
                                )
                                gpp_P_54km = nc_fid54km.variables["GPP_Pmodel"][
                                    :, :
                                ].copy()

                                proj_WRF_P_var_3km = proj_on_finer_WRF_grid(
                                    lats_3km,
                                    lons_3km,
                                    gpp_P_3km,
                                    lats_fine,
                                    lons_fine,
                                    WRF_var_1km,
                                )
                                proj_WRF_P_var_9km = proj_on_finer_WRF_grid(
                                    lats_9km,
                                    lons_9km,
                                    gpp_P_9km,
                                    lats_fine,
                                    lons_fine,
                                    WRF_var_1km,
                                )
                                proj_WRF_P_var_27km = proj_on_finer_WRF_grid(
                                    lats_27km,
                                    lons_27km,
                                    gpp_P_27km,
                                    lats_fine,
                                    lons_fine,
                                    WRF_var_1km,
                                )
                                proj_WRF_P_var_54km = proj_on_finer_WRF_grid(
                                    lats_54km,
                                    lons_54km,
                                    gpp_P_54km,
                                    lats_fine,
                                    lons_fine,
                                    WRF_var_1km,
                                )

                                # data_row_P_1km[column] = np.nanmean(gpp_P_1km[mask])
                                data = np.array(gpp_P_1km, dtype=np.float64)
                                data[~(land_mask & stdh_mask)] = np.nan
                                data_row_P_1km[column] = (
                                    np.nan if np.isnan(data).all() else np.nanmean(data)
                                )

                                data_row_P_3km[column] = np.nanmean(
                                    proj_WRF_P_var_3km[mask]
                                )
                                data_row_P_9km[column] = np.nanmean(
                                    proj_WRF_P_var_9km[mask]
                                )
                                data_row_P_27km[column] = np.nanmean(
                                    proj_WRF_P_var_27km[mask]
                                )
                                data_row_P_54km[column] = np.nanmean(
                                    proj_WRF_P_var_54km[mask]
                                )

                            if column == "RECO":
                                time_str = time.strftime("%Y-%m-%d_%H:%M:%S")

                                nc_fid1km = nc.Dataset(
                                    f"{migli_path}reco_migliavacca{subdaily}_1km_"
                                    + time_str
                                    + ".nc",
                                    "r",
                                )
                                reco_M_1km = nc_fid1km.variables["RECO_Migli"][
                                    :, :
                                ].copy()
                                nc_fid3km = nc.Dataset(
                                    f"{migli_path}reco_migliavacca{subdaily}_3km_"
                                    + time_str
                                    + ".nc",
                                    "r",
                                )
                                reco_M_3km = nc_fid3km.variables["RECO_Migli"][
                                    :, :
                                ].copy()
                                nc_fid9km = nc.Dataset(
                                    f"{migli_path}reco_migliavacca{subdaily}_9km_"
                                    + time_str
                                    + ".nc",
                                    "r",
                                )
                                reco_M_9km = nc_fid9km.variables["RECO_Migli"][
                                    :, :
                                ].copy()
                                nc_fid27km = nc.Dataset(
                                    f"{migli_path}reco_migliavacca{subdaily}_27km_"
                                    + time_str
                                    + ".nc",
                                    "r",
                                )
                                reco_M_27km = nc_fid27km.variables["RECO_Migli"][
                                    :, :
                                ].copy()
                                nc_fid54km = nc.Dataset(
                                    f"{migli_path}reco_migliavacca{subdaily}_54km_"
                                    + time_str
                                    + ".nc",
                                    "r",
                                )
                                reco_M_54km = nc_fid54km.variables["RECO_Migli"][
                                    :, :
                                ].copy()

                                proj_WRF_M_var_3km = proj_on_finer_WRF_grid(
                                    lats_3km,
                                    lons_3km,
                                    reco_M_3km,
                                    lats_fine,
                                    lons_fine,
                                    WRF_var_1km,
                                )
                                proj_WRF_M_var_9km = proj_on_finer_WRF_grid(
                                    lats_9km,
                                    lons_9km,
                                    reco_M_9km,
                                    lats_fine,
                                    lons_fine,
                                    WRF_var_1km,
                                )
                                proj_WRF_M_var_27km = proj_on_finer_WRF_grid(
                                    lats_27km,
                                    lons_27km,
                                    reco_M_27km,
                                    lats_fine,
                                    lons_fine,
                                    WRF_var_1km,
                                )
                                proj_WRF_M_var_54km = proj_on_finer_WRF_grid(
                                    lats_54km,
                                    lons_54km,
                                    reco_M_54km,
                                    lats_fine,
                                    lons_fine,
                                    WRF_var_1km,
                                )

                                data_row_M_1km[column] = np.nanmean(reco_M_1km[mask])
                                data_row_M_3km[column] = np.nanmean(
                                    proj_WRF_M_var_3km[mask]
                                )
                                data_row_M_9km[column] = np.nanmean(
                                    proj_WRF_M_var_9km[mask]
                                )
                                data_row_M_27km[column] = np.nanmean(
                                    proj_WRF_M_var_27km[mask]
                                )
                                data_row_M_54km[column] = np.nanmean(
                                    proj_WRF_M_var_54km[mask]
                                )

                        i += 1

                    df_out_1km.loc[time, :] = data_row_1km
                    df_out_3km.loc[time, :] = data_row_3km
                    df_out_9km.loc[time, :] = data_row_9km
                    df_out_27km.loc[time, :] = data_row_27km
                    df_out_54km.loc[time, :] = data_row_54km
                    df_out_cams.loc[time, :] = data_row_cams
                    if run_Pmodel:
                        df_out_P_1km.loc[time, :] = data_row_P_1km
                        df_out_P_3km.loc[time, :] = data_row_P_3km
                        df_out_P_9km.loc[time, :] = data_row_P_9km
                        df_out_P_27km.loc[time, :] = data_row_P_27km
                        df_out_P_54km.loc[time, :] = data_row_P_54km
                        df_out_M_1km.loc[time, :] = data_row_M_1km
                        df_out_M_3km.loc[time, :] = data_row_M_3km
                        df_out_M_9km.loc[time, :] = data_row_M_9km
                        df_out_M_27km.loc[time, :] = data_row_M_27km
                        df_out_M_54km.loc[time, :] = data_row_M_54km

                # Add suffixes to columns
                df_out_1km = df_out_1km.add_suffix("_1km")
                df_out_3km = df_out_3km.add_suffix("_3km")
                df_out_9km = df_out_9km.add_suffix("_9km")
                df_out_27km = df_out_27km.add_suffix("_27km")
                df_out_54km = df_out_54km.add_suffix("_54km")
                df_out_cams = df_out_cams.add_suffix("_CAMS")
                if run_Pmodel:
                    df_out_P_1km = df_out_P_1km.add_suffix("_pmodel_1km")
                    df_out_P_3km = df_out_P_3km.add_suffix("_pmodel_3km")
                    df_out_P_9km = df_out_P_9km.add_suffix("_pmodel_9km")
                    df_out_P_27km = df_out_P_27km.add_suffix("_pmodel_27km")
                    df_out_P_54km = df_out_P_54km.add_suffix("_pmodel_54km")
                    df_out_M_1km = df_out_M_1km.add_suffix("_migli_1km")
                    df_out_M_3km = df_out_M_3km.add_suffix("_migli_3km")
                    df_out_M_9km = df_out_M_9km.add_suffix("_migli_9km")
                    df_out_M_27km = df_out_M_27km.add_suffix("_migli_27km")
                    df_out_M_54km = df_out_M_54km.add_suffix("_migli_54km")

                # Merge all DataFrames horizontally
                if run_Pmodel:
                    merged_df = pd.concat(
                        [
                            df_out_1km,
                            df_out_3km,
                            df_out_9km,
                            df_out_27km,
                            df_out_54km,
                            df_out_cams,
                            df_out_P_1km,
                            df_out_P_3km,
                            df_out_P_9km,
                            df_out_P_27km,
                            df_out_P_54km,
                            df_out_M_1km,
                            df_out_M_3km,
                            df_out_M_9km,
                            df_out_M_27km,
                            df_out_M_54km,
                        ],
                        axis=1,
                    )
                else:
                    merged_df = pd.concat(
                        [
                            df_out_1km,
                            df_out_3km,
                            df_out_9km,
                            df_out_27km,
                            df_out_54km,
                            df_out_cams,
                        ],
                        axis=1,
                    )

                # Save to CSV
                if run_Pmodel:
                    merged_df.to_csv(
                        f"{csv_folder}timeseries_domain_averaged{ref_sim}{subdaily}_std_topo_{STD_TOPO_flag}_{STD_TOPO}_{start_date}_{end_date}_pmodel.csv",
                        index=True,
                    )
                else:
                    merged_df.to_csv(
                        f"{csv_folder}timeseries_domain_averaged{ref_sim}_std_topo_{STD_TOPO_flag}_{STD_TOPO}{sim_type}_{start_date}_{end_date}.csv",
                        index=True,
                    )


def main():

    if len(sys.argv) > 1:  # to run on cluster
        parser = argparse.ArgumentParser(description="Description of your script")
        parser.add_argument(
            "-s", "--start", type=str, help="Format: 2012-07-01 01:00:00"
        )
        parser.add_argument("-e", "--end", type=str, help="Format: 2012-07-01 01:00:00")
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
        sim_type = ""  # "" or "_cloudy" or "_parm_err"

    wrf_paths = [
        f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_1km{sim_type}",
        f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km{sim_type}",
        f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_9km{sim_type}",
        f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_27km{sim_type}",
        f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km{sim_type}",  # all resolutions have to be used, otherwise code needs to be changed
    ]

    exctract_wrf_domains_mean_timeseries(wrf_paths, start_date, end_date, sim_type)


if __name__ == "__main__":
    main()
