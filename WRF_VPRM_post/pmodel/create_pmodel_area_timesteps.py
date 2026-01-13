import os
import pandas as pd
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d
from datetime import datetime
import glob
from collections import defaultdict
import pandas as pd
import numpy as np
from pyrealm.pmodel import PModel, PModelEnvironment, SubdailyPModel, AcclimationModel
from pyrealm.core.hygro import convert_sh_to_vpd


def extract_datetime_from_filename(filename):
    """
    Extract datetime from WRF filename assuming format 'wrfout_d0x_YYYY-MM-DD_HH:MM:SS'.
    """
    base_filename = os.path.basename(filename)
    date_str = base_filename.split("_")[-2] + "_" + base_filename.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")


def pModel_subdaily_area(
    datetime_subdaily: np.ndarray,
    temp_subdaily: np.ndarray,
    ppfd_subdaily: np.ndarray,
    vpd_subdaily: np.ndarray,
    co2_subdaily: np.ndarray,
    patm_subdaily: np.ndarray,
    fpar_subdaily: np.ndarray,
    days_memory: float,
    window_center_i: int,
    half_width_i: int,
):
    correction_factor = 1 / 3
    gC_to_mumol = 0.0833

    temp_subdaily = np.where(temp_subdaily < -25, np.nan, temp_subdaily)
    vpd_subdaily = np.clip(vpd_subdaily, 0, None)
    ppfd_subdaily = np.clip(ppfd_subdaily, 0, None)

    subdaily_env = PModelEnvironment(
        tc=temp_subdaily,
        vpd=vpd_subdaily,
        co2=co2_subdaily,
        patm=patm_subdaily,
        ppfd=ppfd_subdaily,
        fapar=fpar_subdaily,
    )

    acclim_model = AcclimationModel(
        datetime_subdaily, allow_holdover=True, alpha=1 / days_memory
    )
    acclim_model.set_window(
        window_center=np.timedelta64(window_center_i, "h"),
        half_width=np.timedelta64(half_width_i, "m"),
    )

    pmodel_subdaily = SubdailyPModel(env=subdaily_env, acclim_model=acclim_model)
    pmodel_subdaily_acc = pmodel_subdaily.gpp * gC_to_mumol * correction_factor

    return pmodel_subdaily_acc


def migliavacca_LinGPP(
    T_ref, T0, E0, k_mm, k2, alpha_p, alpha_lai, max_lai, R_lai0, GPP, P, T_A
):
    GPP_gC_per_day = GPP * 12 * 86400 / 10**6
    R_ref = R_lai0 + alpha_lai * max_lai + k2 * GPP_gC_per_day
    f_T = np.exp(E0 * (1 / (T_ref - T0) - 1 / (T_A - T0)))
    f_P = (alpha_p * k_mm + P * (1 - alpha_p)) / (k_mm + P * (1 - alpha_p))

    reco_LinGPP_gC_per_day = R_ref * f_T * f_P
    reco_LinGPP_mumol_per_sec = reco_LinGPP_gC_per_day * 10**6 / (86400 * 12)
    return reco_LinGPP_mumol_per_sec


fixed_boundaries = {  # RLAI and alphaLAI are from literature while the others are tuned with fluxnet data
    "1": {
        "RLAI": 1.02,
        "alphaLAI": 0.42,
        "E0(K)": 121.97746846338332,
        "K (mm)": 6.411872635065828,
        "alpha_p": 0.542458467093071,
        "k2": 0.6871140343559842,
    },
    "2": {
        "RLAI": 1.27,
        "alphaLAI": 0.34,
        "E0(K)": 126.3734980228889,
        "K (mm)": 3.4082270937649963,
        "alpha_p": 0.7126491712262502,
        "k2": 0.2674530834652109,
    },
    "3": {
        "RLAI": 0.78,
        "alphaLAI": 0.44,
        "E0(K)": 66.53200700219477,
        "K (mm)": 6.401451321685523,
        "alpha_p": 0.8484321940207632,
        "k2": 0.46838002848789034,
    },
    "4": {
        "RLAI": 0.42,
        "alphaLAI": 0.57,
        "k2": 0.354,
        "E0(K)": 156.746,
        "alpha_p": 0.850,
        "K (mm)": 0.097,
    },
    "5": {
        "RLAI": 0.42,
        "alphaLAI": 0.57,
        "k2": 0.654,
        "E0(K)": 81.537,
        "alpha_p": 0.474,
        "K (mm)": 0.567,
    },
    "6": {
        "RLAI": 0.25,
        "alphaLAI": 0.40,
        "E0(K)": 137.94375730711272,
        "K (mm)": 3.00018616618101,
        "alpha_p": 1.0,
        "k2": 0.3891423973169539,
    },
    "7": {
        "RLAI": 0.41,
        "alphaLAI": 1.14,
        "E0(K)": 160.82972802277175,
        "K (mm)": 3.4810392498593856,
        "alpha_p": 0.7135894553409096,
        "k2": 0.5358499672960101,
    },
    "8": {
        "RLAI": 0,
        "alphaLAI": 0,
        "E0(K)": 0,
        "K (mm)": 0,
        "alpha_p": 0,
        "k2": 0,
    },
}

# Define the remapping dictionary for CORINE vegetation types
corine_to_vprm = {
    24: 1,  # Coniferous Forest (Evergreen)
    23: 2,  # Broad-leaved Forest (Deciduous)
    25: 3,
    29: 3,  # Mixed Forest and Transitional Woodland-Shrub
    27: 4,
    28: 4,  # Moors and Heathland, Sclerophyllous Vegetation (Shrubland)
    35: 5,
    36: 5,
    37: 5,  # Wetlands: Inland Marshes, Peat Bogs, Salt Marshes
    12: 6,
    13: 6,
    14: 6,
    15: 6,
    16: 6,
    17: 6,
    19: 6,
    20: 6,
    21: 6,
    22: 6,  # Cropland
    18: 7,
    26: 7,  # Grassland: Pastures, Natural Grasslands
    # Others mapped to 8 (gray)
    1: 8,
    2: 8,
    3: 8,
    4: 8,
    5: 8,
    6: 8,
    7: 8,
    8: 8,
    9: 8,
    10: 8,
    11: 8,
    30: 8,
    31: 8,
    32: 8,
    33: 8,
    34: 8,
    38: 8,
    39: 8,
    40: 8,
    41: 8,
    42: 8,
    43: 8,
    44: 8,
}


# Load WRF dataset
# wrf_path = "/home/madse/Downloads/Fluxnet_Data/wrfout_d01_2012-07-01_12:00:00.nc"  # Replace with your file path
wrf_paths = [
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_1km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_9km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_27km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km",
]

modis_path = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/gap_filled/"
migli_path = "/scratch/c7071034/DATA/RECO_Migli"
start_date = "2012-06-12 00:00:00"
end_date = "2012-06-30 00:00:00"

# pmodel parameters
days_mem = 15
half_wdth = 1
window_cent = 12
scaling_factor = 2 / 3
gC_to_mumol = 0.0833  # 1 µg C m⁻² s⁻¹ × (1 µmol C / 12.01 µg C) × (1 µmol CO₂ / 1 µmol C) = 0.0833 µmol CO₂ m⁻² s⁻¹

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

for wrf_path in wrf_paths:
    wrf_path_dx_str = wrf_path.split("_")[-1]

    timesteps = len(file_list)
    # get datetime from first file
    datetimestart = file_list[1].split("_")[2] + " " + file_list[1].split("_")[3]
    domain02 = wrf_path.split("_")[-1] == "1km"  # Check if the file is for domain d02
    if domain02:
        file_list = [f.replace("d01", "d02") for f in file_list]
    else:
        file_list = [f.replace("d02", "d01") for f in file_list]
    wrf_ds = xr.open_dataset(wrf_path + "/" + file_list[0])
    temp = wrf_ds["T2"].to_numpy()
    l, m, n = temp.shape
    fpar_modis_arr = np.zeros((timesteps, m, n))
    ppfd_arr = np.zeros((timesteps, m, n))
    tc_arr = np.zeros((timesteps, m, n))  # Store temperature time series
    co2_arr = np.zeros((timesteps, m, n))  # Store CO₂ time series
    patm_arr = np.zeros((timesteps, m, n))  # Store atmospheric pressure
    vpd_arr = np.zeros((timesteps, m, n))  # Store vapor pressure deficit
    rainc_arr = np.zeros((timesteps, m, n))  # Store rain rate
    lai_wrf_arr = np.zeros((timesteps, m, n))  # Store LAI

    IVGTYP = wrf_ds["IVGTYP"].to_numpy()
    IVGTYP_vprm = np.vectorize(corine_to_vprm.get)(
        IVGTYP[:, :]
    )  # Create a new array for the simplified vegetation categories
    # mask where IVGTYP_vprm is 8
    IVGTYP_mask = np.where(IVGTYP_vprm == 8, True, False)

    t = 0
    # files = files[224:378]  # TODO
    for file in file_list:
        day = int(file.split("_")[2].split("-")[2])
        month = int(file.split("_")[2].split("-")[1])

        if day > 30:
            continue
        wrf_ds = xr.open_dataset(wrf_path + "/" + file)

        # Load variables from WRF dataset
        temp = wrf_ds["T2"].to_numpy() - 273.15  # Convert to Celsius
        patm = wrf_ds["PSFC"].to_numpy()  # Pa
        co2 = wrf_ds["CO2_BIO"].isel(bottom_top=0).to_numpy()  # ppmv
        qvapor = (
            wrf_ds["QVAPOR"].isel(bottom_top=0).to_numpy()
        )  # Water vapor mixing ratio (kg/kg) at the surface level
        psfc = wrf_ds["PSFC"].isel(Time=0).to_numpy()  # Surface pressure (Pa)
        t2 = wrf_ds["T2"].isel(Time=0).to_numpy()  # Temperature at 2m (K)

        # # Calculate actual vapor pressure (ea) in kPa
        # ea = (qvapor * psfc) / (0.622 + qvapor)  # Pa
        # # Calculate saturation vapor pressure (es) in kPa
        # es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3)) * 1000  # convert to Pa
        # # Calculate VPD
        # vpd = np.maximum(0, es - ea)  # Force non-negative VPD

        epsilon = 0.622
        w = qvapor  # your mixing ratio in kg/kg
        q = w / (1 + w)  # now specific humidity in kg/kg

        vpd = (
            convert_sh_to_vpd(sh=q, ta=t2 - 273.15, patm=psfc / 1000) * 1000
        )  # result in Pa
        vpd = np.clip(vpd, 0, np.inf)

        # Load required variables from WRF dataset
        vegfra = wrf_ds["VEGFRA"].to_numpy()  # Vegetation fraction (0 to 1)
        albedo = wrf_ds["ALBEDO"].to_numpy()  # Albedo (0 to 1)
        swdown = wrf_ds["SWDOWN"].to_numpy()  # Downward shortwave radiation (W/m^2)
        ppfd = (
            swdown * 2.30785
        )  # Shortwave radiation (W/m²) × 0.505 -> PAR (W/m²) × 4.57 -> 2.3*x ~ PPFD (umol/m²/s)
        xlat = wrf_ds["XLAT"].to_numpy()  # Latitude (degrees)
        xlon = wrf_ds["XLONG"].to_numpy()  # Longitude (degrees)
        xlat = xlat[0, :, :]
        xlon = xlon[0, :, :]
        # fapar_wrf = (1 - albedo) * (vegfra / 100)  # Calculate fAPAR

        # get modis fpar
        modis_path_in = f"{modis_path}fpar_interpol/interpolated_fpar_{wrf_path_dx_str}_2012-{month:02d}-{day:02d}T12:00:00.nc"
        modis_ds = xr.open_dataset(modis_path_in)
        fpar_modis = modis_ds["FAPAR"].to_numpy()  # fAPAR from MODIS
        # # where modis is nan use values from fapar_wrf
        # fpar_modis = np.where(
        #     np.isnan(fpar_modis), fapar_wrf, fpar_modis
        # )  # TODO: how is this handled in literature? There is no fPAR Data around cities, so these areas could also be masked out, but I would habe to modify the landcover maps...
        # # set fpar_modis to 0 where IVGTYP_vprm is 8
        fpar_modis[IVGTYP_mask] = 0

        # save data in arrays
        tc_arr[t, :, :] = temp[0, :, :]
        co2_arr[t, :, :] = co2[0, :, :]
        patm_arr[t, :, :] = patm[0, :, :]
        vpd_arr[t, :, :] = vpd[0, :, :]

        fpar_modis_arr[t, :, :] = fpar_modis[0, :, :]
        ppfd_arr[t, :, :] = ppfd[0, :, :]

        # get vars for migliavacca_LinGPP
        rainc = wrf_ds["RAINC"].to_numpy()
        rainc_arr[t, :, :] = rainc[0, :, :]
        lai_wrf = wrf_ds["LAI"].to_numpy()
        lai_wrf_arr[t, :, :] = lai_wrf[0, :, :]

        t += 1

    co2_clipped = np.clip(co2_arr, 300, 600)
    vpd_clipped = np.clip(vpd_arr, 1, 4000)
    tc_clipped = np.clip(tc_arr, -5, 45)
    ppfd_arr[ppfd_arr < 0] = 0

    pm_env = PModelEnvironment(
        tc=tc_clipped,
        patm=patm_arr,
        vpd=vpd_clipped,
        co2=co2_clipped,
        fapar=fpar_modis_arr,
        ppfd=ppfd_arr,
    )
    pm_env.summarize()

    # # calculate GPP with acclimation
    datetimes = pd.date_range(
        start=datetimestart, periods=timesteps, freq="h"
    ).to_numpy()

    subdailyC3_arr = pModel_subdaily_area(
        datetimes,
        tc_arr,
        ppfd_arr,
        vpd_arr,
        co2_arr,
        patm_arr,
        fpar_modis_arr,
        days_mem,
        window_cent,
        half_wdth,
    )

    # TODO: Calculation of GPP using fast and slow responses

    t = 0
    for file in file_list:
        subdailyC3_gpp = subdailyC3_arr[t, :, :] * scaling_factor
        # print(t, " ", np.nanmax(subdailyC3_gpp))
        # save data in netcdf
        date_time = file.split("_")[2] + "_" + file.split("_")[3]
        gpp_path_out = f"{modis_path}gpp_pmodel/gpp_pmodel_subdailyC3v2_{wrf_path_dx_str}_{date_time}.nc"
        xr.DataArray(subdailyC3_gpp, name="GPP_Pmodel").to_netcdf(
            gpp_path_out, format="NETCDF4_CLASSIC"
        )
        t += 1
        print(f"Saved GPP data to {gpp_path_out}")

    # Ensure rainc_arr is a NumPy array (3D: time, lat, lon)
    rainc_arr = np.asarray(rainc_arr)
    subdailyC3_arr = np.asarray(subdailyC3_arr)  # Extract GPP data if needed

    if rainc_arr.ndim != 3:
        raise ValueError(
            f"Expected 3D array (time, lat, lon), but got shape {rainc_arr.shape}"
        )

    if subdailyC3_arr.ndim != 3:
        raise ValueError(
            f"Expected 3D array (time, lat, lon), but got shape {subdailyC3_arr.shape}"
        )

    # Apply rolling mean
    rainc_avrg = uniform_filter1d(rainc_arr, size=7 * 24, axis=0, mode="nearest")
    subdailyC3_avrg = uniform_filter1d(
        subdailyC3_arr, size=7 * 24, axis=0, mode="nearest"
    )

    # Compute initial mean for first `window_size` time steps
    initial_mean_rainc = np.mean(rainc_arr[:7, :, :], axis=0)
    initial_mean_subdailyC3 = np.mean(subdailyC3_arr[:1, :, :], axis=0)

    # Fill missing values
    rainc_avrg = np.where(np.isnan(rainc_avrg), initial_mean_rainc, rainc_avrg)
    subdailyC3_avrg = np.where(
        np.isnan(subdailyC3_avrg), initial_mean_subdailyC3, subdailyC3_avrg
    )

    # Ensure no NaNs
    rainc_avrg = np.nan_to_num(rainc_avrg, nan=0)
    subdailyC3_avrg = np.nan_to_num(subdailyC3_avrg, nan=0)

    # settings
    T_ref = 288.15
    T0 = 227.13
    max_lai = np.max(lai_wrf_arr, axis=0)  # Get max LAI for each (lat, lon) location
    # TODO: create max_lai Map from MODIS LAI

    # Convert IVGTYP_vprm to integer type to use it as an index
    IVGTYP_vprm = IVGTYP_vprm.astype(int)

    # Create empty arrays with the same shape as IVGTYP_vprm
    shape = IVGTYP_vprm.shape
    E0_arr = np.zeros(shape)
    K_arr = np.zeros(shape)
    k2_arr = np.zeros(shape)
    alpha_p_arr = np.zeros(shape)
    alphaLAI_arr = np.zeros(shape)
    RLAI_arr = np.zeros(shape)

    # Fill arrays using vegetation type mapping
    for veg_type in range(1, 9):  # Loop through vegetation types 1 to 8
        mask = IVGTYP_vprm == veg_type  # Find locations for current vegetation type
        params = fixed_boundaries[str(veg_type)]  # Retrieve parameters

        E0_arr[mask] = params["E0(K)"]
        K_arr[mask] = params["K (mm)"]
        k2_arr[mask] = params["k2"]
        alpha_p_arr[mask] = params["alpha_p"]
        alphaLAI_arr[mask] = params["alphaLAI"]
        RLAI_arr[mask] = params["RLAI"]

    # Now you can use these arrays in your function call
    Reco_optimized = migliavacca_LinGPP(
        T_ref,
        T0,
        E0_arr,
        K_arr,
        k2_arr,
        alpha_p_arr,
        alphaLAI_arr,
        max_lai,
        RLAI_arr,
        subdailyC3_avrg,
        rainc_avrg,
        tc_arr + 273.15,
    )

    t = 0
    for file in file_list:
        Reco_opt_t = Reco_optimized[t, :, :]
        date_time = file.split("_")[2] + "_" + file.split("_")[3]
        reco_path_out = f"{migli_path}/reco_migliavacca_subdailyC3v2_{wrf_path_dx_str}_{date_time}.nc"
        xr.DataArray(Reco_opt_t, name="RECO_Migli").to_netcdf(
            reco_path_out, format="NETCDF4_CLASSIC"
        )
        t += 1
        print("Reco_optimized written to : ", reco_path_out)
