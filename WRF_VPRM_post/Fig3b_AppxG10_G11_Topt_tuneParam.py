"""
Fig3b_AppxG10_G11_Topt_tuneParam.py
==============================================
Fig. 3b (fig:ToptAlps, panel b), Fig. G10 (fig:VPRMoldPFT) and Fig. G11
(fig:VPRMoldPFT_NNSE). Panel a of Fig. 3 is produced by
Fig3a_Topt_perSiteYear_fit.py.
    ==== run after main_tune_VPRM.py =====
==============================================
Analyze and plot optimized Topt values against mean and max temperatures.
This script processes FLUXNET2015 data to determine the optimum temperature (Topt)
for net ecosystem exchange (NEE) using both mirrored Gaussian and cubic polynomial fits.
It visualizes the relationship between Topt and mean/max temperatures,
highlighting real vs extrapolated Topt values.

needs the file:
"$SCRATCH_PATH"/DATA/Fluxnet2015/Alps/site_info_all_FLUXNET2015.csv

and needs the FLUXNET2015 data for the following sites to the folder:
"$SCRATCH_PATH"/DATA/Fluxnet2015/Alps
including the results from the tuning procedure for Topt and files ending with e.g.:
'old_diff_evo_V23_42.xlsx'

FLX_AT-Neu_FLUXNET2015_FULLSET_2002-2012_1-4
FLX_CH-Cha_FLUXNET2015_FULLSET_2005-2014_2-4
FLX_CH-Dav_FLUXNET2015_FULLSET_1997-2014_1-4
FLX_CH-Fru_FLUXNET2015_FULLSET_2005-2014_2-4
FLX_CH-Lae_FLUXNET2015_FULLSET_2004-2014_1-4
FLX_CH-Oe1_FLUXNET2015_FULLSET_2002-2008_2-4
FLX_CH-Oe2_FLUXNET2015_FULLSET_2004-2014_1-4
FLX_DE-Hai_FLUXNET2015_FULLSET_2000-2012_1-4
FLX_DE-Kli_FLUXNET2015_FULLSET_2004-2014_1-4
FLX_DE-Lkb_FLUXNET2015_FULLSET_2009-2013_1-4
FLX_FR-Fon_FLUXNET2015_FULLSET_2005-2014_1-4
FLX_FR-Gri_FLUXNET2015_FULLSET_2004-2014_1-4
FLX_IT-Isp_FLUXNET2015_FULLSET_2013-2014_1-4
FLX_IT-La2_FLUXNET2015_FULLSET_2000-2002_1-4
FLX_IT-Lav_FLUXNET2015_FULLSET_2003-2014_2-4
FLX_IT-MBo_FLUXNET2015_FULLSET_2003-2013_1-4
FLX_IT-PT1_FLUXNET2015_FULLSET_2002-2004_1-4
FLX_IT-Ren_FLUXNET2015_FULLSET_1998-2013_1-4
FLX_IT-Tor_FLUXNET2015_FULLSET_2008-2014_2-4

==============================================
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.optimize import curve_fit, minimize
import seaborn as sns
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Shared fitted-Topt boxplot panel (also used standalone by Fig3a)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from topt_box import (
    FLIER_PROPS,
    PFT_COLORS,
    build_topt_points,
    draw_topt_box,
    topt_legend_elements,
)

# ==================== Configuration ====================
SCRATCH_PATH = os.getenv("SCRATCH_PATH")
GITHUB_PATH = os.getenv("GITHUB_PATH")
OUTFOLDER = os.getenv("OUTFOLDER")


# Define a mirrored Gaussian function (to model negative GPP)
def mirrored_gaussian(x, a, b, c):
    """
    Mirrored Gaussian function:
    a - amplitude (negative value)
    b - mean (Topt)
    c - standard deviation (spread of the peak)
    """
    return -a * np.exp(-((x - b) ** 2) / (2 * c**2))


# Define a fallback cubic polynomial function
def cubic_polynomial(x, a, b, c, d):
    """
    Cubic polynomial function for extrapolation:
    a, b, c, d - polynomial coefficients
    """
    return a * x**3 + b * x**2 + c * x + d


# Function to calculate RMSE
def calculate_rmse(observed, predicted):
    return np.sqrt(np.mean((observed - predicted) ** 2))


# Function to calculate the minimum of a cubic polynomial using optimization
def find_minimum_of_cubic_poly(coeffs, x_range):
    """
    Use optimization to find the minimum of the cubic polynomial.
    """

    # Define the cubic polynomial function
    def poly_func(x):
        a, b, c, d = coeffs
        return a * x**3 + b * x**2 + c * x + d

    # Convert x_range to a numpy array and calculate its mean
    x_range_mean = np.mean(
        np.array(x_range)
    )  # Calculate the mean of the x_range values

    # Minimize the cubic polynomial to find the minimum
    result = minimize(poly_func, x_range_mean, bounds=[(min(x_range), max(x_range))])
    return result.x[0]  # Return the x value that minimizes the polynomial


# Define paths and parameters
# --- GPP additions for the GMD revision: upward Gaussian (peak = maximum) ---
def gaussian(x, a, b, c):
    """Upward Gaussian -> models the GPP maximum (Topt)."""
    return a * np.exp(-((x - b) ** 2) / (2 * c**2))


def find_maximum_of_cubic_poly(coeffs, x_range):
    """x that maximises the cubic polynomial within the observed range (GPP)."""
    a, b, c, d = coeffs

    def neg_poly_func(x):
        return -(a * x**3 + b * x**2 + c * x + d)

    x_range = np.asarray(x_range)
    res = minimize(
        neg_poly_func, np.mean(x_range), bounds=[(x_range.min(), x_range.max())]
    )
    return res.x[0]


base_path = os.path.join(SCRATCH_PATH, "DATA/Fluxnet2015/Alps/")
plot_path = OUTFOLDER
font_size = 16  # -> ~8 pt printed once the figsizes below are ~2x printed size

# Fig. I16(a) boxplot_PFTs and I16(b) boxplot_siteIDs are placed side by side, so
# they must share one axes rectangle. tight_layout() sized each to its own x tick
# labels (short horizontal class names vs long vertical site IDs), which put the
# five rows at different heights. Fixed margins + no tight bbox instead; `bottom`
# reserves room for the vertical site IDs and the x title (the legend is now a
# separate file -- see save_topt_box_legend -- so no room is reserved for it here).
BOX_MARGINS = dict(left=0.155, right=0.985, top=0.985, bottom=0.115, hspace=0.13)


def save_topt_box_legend(outfile, pft_order, pft_colors, fontsize):
    """Standalone legend for Fig. I16: boxplot_PFTs and boxplot_siteIDs share this
    one file (5 columns), meant to sit below both panels in the manuscript rather
    than inside either PDF."""
    handles = topt_legend_elements(
        central_label="median p50", iqr_label="p25-p75", whisker_label="p10-p90",
    ) + [plt.scatter([], [], c=pft_colors[pft], label=pft) for pft in pft_order]
    fig = plt.figure(figsize=(9.0, 1.3))
    fig.legend(handles=handles, loc="center", ncol=5, fontsize=fontsize,
               frameon=False)
    plt.axis("off")
    plt.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Legend written to {outfile}")
site_info = pd.read_csv(
    os.path.join(SCRATCH_PATH, "DATA/Fluxnet2015/Alps/site_info_all_FLUXNET2015.csv")
)
plot_data = False  # individual site-year plots are saved by Fig3a (with percentile bands)
save_data = True  # Set this to False if you don't want to save the data
plot_boxplot = True  # Set this to False if you don't want to plot the boxplot

# Initialize a dictionary to hold min values for each site and year
min_nee_temp_dict = {}
min_nee_temp_dict2 = {}
min_nee_temp_dict_tmean = {}  # growing-season T_mean per site-year (T >= 3°C days)
default_Topt = {
    "ENF": 20.0,  # Evergreen Needleleaf Forest
    "DBF": 20.0,  # Deciduous Needleleaf Forest
    "MF_": 20.0,  # Mixed Forest
    "SHB": 20.0,  # Shrubland
    "SAV": 20.0,  # Savanna
    "CRO": 22.0,  # Cropland
    "GRA": 18.0,  # Grassland
}

# Iterate over all folders in the base path that start with "FLX_"
for folder in os.listdir(base_path):
    if folder.startswith("FLX_"):
        file_base = "_".join(folder.split("_")[0:4])
        years = "_".join(folder.split("_")[4:6])
        file_path = os.path.join(base_path, folder, f"{file_base}_HH_{years}.csv")

        # Extract site name
        site_name = folder.split("_")[1]
        # get PFT of site
        target_pft = None
        i = 0
        for site_i in site_info["site"]:
            if site_name == site_i:
                target_pft = site_info["pft"][i]
                if target_pft == "EBF":
                    target_pft = "DBF"
                if target_pft == "OSH":  # TODO check OSH
                    target_pft = "SHB"
                if site_name == "AT-Mie":
                    target_pft = "ENF"
            i += 1
        if target_pft is None:
            print(f"No PFT found for {site_name}, skipping.")
            continue
        site_name = target_pft + "_" + site_name

        # Columns to read and converters
        columns_to_copy = [
            "TIMESTAMP_START",
            "TA_F",
            "GPP_NT_VUT_USTAR50",
            "NIGHT",
        ]
        converters = {k: lambda x: float(x) for k in columns_to_copy}

        # Load the data
        df_site = pd.read_csv(file_path, usecols=columns_to_copy, converters=converters)
        df_site["TIMESTAMP_START"] = pd.to_datetime(
            df_site["TIMESTAMP_START"], format="%Y%m%d%H%M"
        )
        df_site["PFT"] = target_pft
        # Clean the data
        df_site["TA_F"] = df_site["TA_F"].replace(-9999.0, np.nan)
        df_site = df_site.dropna(subset=["TA_F"])
        df_site["GPP_NT_VUT_USTAR50"] = df_site["GPP_NT_VUT_USTAR50"].replace(-9999.0, np.nan)
        # check how many nan values are in GPP_NT_VUT_USTAR50
        nan_values = df_site["GPP_NT_VUT_USTAR50"].isna()
        nan_sum = nan_values.sum()
        full_year = len(df_site["GPP_NT_VUT_USTAR50"])
        df_site = df_site.dropna(subset=["GPP_NT_VUT_USTAR50"])
        percent_nan = nan_sum.max() / full_year * 100

        #  find the name of the column with the most missing values
        if percent_nan > 20:
            print(
                f"WARNING: for {site_name} year {year} is skipped, as {percent_nan:2.1f}% are missing"
            )
            continue

        # Set the values to np.nan during nighttime
        night_columns = ["GPP_NT_VUT_USTAR50"]
        df_site.loc[df_site["NIGHT"] == 1, night_columns] = np.nan

        # GPP_NT_VUT_USTAR50 can be slightly negative (night-time partitioning
        # artefact); clip at zero (GPP >= 0) before fitting the upward Gaussian.
        df_site["GPP_NT_VUT_USTAR50"] = df_site["GPP_NT_VUT_USTAR50"].clip(lower=0)

        # Convert units from micromol per m² per second to grams of Carbon per day
        # conversion_factor = 12 * 1e-6 * 60 * 60 * 24  # 12 g C per mol, micromol to mol, per second to per day
        # df_site['GPP_NT_VUT_USTAR50'] *= conversion_factor

        # Resample to daily frequency
        df_site.set_index("TIMESTAMP_START", inplace=True)
        df_daily = (
            df_site.resample("D").agg({"TA_F": "mean", "GPP_NT_VUT_USTAR50": "mean"}).dropna()
        )

        # Extract the year from the timestamp
        df_daily["YEAR"] = df_daily.index.year

        # List of unique years
        years = df_daily["YEAR"].unique()

        # Initialize site entry in the dictionary
        if site_name not in min_nee_temp_dict:
            min_nee_temp_dict[site_name] = {}
            min_nee_temp_dict2[site_name] = {}
        min_nee_temp_dict_tmean.setdefault(site_name, {})

        # Plot for each year and find min NEE temperature
        for year in years:
            # if year == 2012:

            # Filter data for the current year
            df_year = df_daily[df_daily["YEAR"] == year].copy()
            df_year = df_year[df_year["TA_F"] >= 3]

            # Group by each degree of temperature and calculate the mean values
            df_year.loc[:, "TA_F_rounded"] = df_year["TA_F"].round()
            mean_values = df_year.groupby("TA_F_rounded").mean()

            # Initialize variables
            Topt = np.nan
            fitted_curve = None
            rmse_threshold = (
                2.0  # Set the RMSE threshold (you can fine-tune this value)
            )

            # First, try to fit the mirrored Gaussian function
            try:
                # Perform Gaussian fit with mirrored curve (negating amplitude)
                popt, _ = curve_fit(
                    gaussian,
                    mean_values.index,
                    mean_values["GPP_NT_VUT_USTAR50"],
                    p0=[
                        mean_values["GPP_NT_VUT_USTAR50"].max(),
                        mean_values.index[
                            np.argmax(mean_values["GPP_NT_VUT_USTAR50"].values)
                        ],
                        8.0,
                    ],
                    maxfev=10000,
                )
                Topt = popt[
                    1
                ]  # Extract the temperature at the peak of the Gaussian (mean)

                # Generate the fitted curve for plotting or evaluation
                fitted_curve = gaussian(mean_values.index, *popt)

                # Calculate RMSE for the Gaussian fit
                rmse = calculate_rmse(mean_values["GPP_NT_VUT_USTAR50"], fitted_curve)
                print(f"RMSE for Gaussian fit: {rmse}")

                if site_name == "ENF_DE-Lbk" and year == 2012:
                    rmse_threshold = 1

                # If RMSE is above threshold, switch to polynomial fit
                if rmse > rmse_threshold:
                    print(
                        f"RMSE exceeds threshold for {year} at Site {site_name}, switching to polynomial fit."
                    )
                    raise RuntimeError("Gaussian fit RMSE too high")

                # Check if Topt is within the observed range
                if Topt < df_year["TA_F_rounded"].max():
                    print(f"Topt is real for {year} at Site {site_name}")
                    real_Topt_col = "green"
                elif Topt > df_year["TA_F_rounded"].max():
                    print(
                        f"Extrapolated Topt is above observed range for {year} at Site {site_name}"
                    )
                    real_Topt_col = "yellow"  # Indicate extrapolated Topt
                else:
                    Topt = default_Topt.get(
                        site_name.split("_")[0]
                    )  # Fallback to default if Topt cant be found
                    real_Topt_col = "red"

            except RuntimeError as e:
                print(f"Gaussian fit failed for {year} at Site {site_name}: {e}")
                # If RMSE threshold exceeded, use the fallback polynomial fit
                print("Using fallback cubic polynomial fit.")
                try:
                    # Perform cubic polynomial fit
                    popt_poly, _ = curve_fit(
                        cubic_polynomial,
                        mean_values.index,
                        mean_values["GPP_NT_VUT_USTAR50"],
                        maxfev=10000,
                    )

                    # Find the minimum of the cubic polynomial curve (Topt) using optimization
                    Topt = find_maximum_of_cubic_poly(popt_poly, mean_values.index)

                    # Generate the fitted curve for the cubic polynomial
                    fitted_curve = cubic_polynomial(mean_values.index, *popt_poly)

                    # Extend the range and check if Topt is above the highest temperature
                    if Topt > df_year["TA_F_rounded"].max():
                        print(
                            f"Extrapolated Topt is above observed range for {year} at Site {site_name}"
                        )
                        real_Topt_col = "yellow"  # Indicate extrapolated Topt
                    else:
                        real_Topt_col = "green"
                except RuntimeError as e_poly:
                    print(
                        f"Polynomial fit failed for {year} at Site {site_name}: {e_poly}"
                    )
                    Topt = default_Topt.get(
                        site_name.split("_")[0]
                    )  # Fallback to default if Topt cant be found
                    real_Topt_col = "red"

            # dont use CRO as the cutting events disturb the data too much
            if site_name.startswith("CRO"):
                real_Topt_col = "red"
                print(
                    f"WARNING: for {site_name} year {year} CRO is skipped, as cutting events disturb the data too much"
                )
                Topt = default_Topt.get(site_name.split("_")[0])  # Fallback
            # Store the minimum temperature in the dictionary
            if Topt < 5 or Topt > 30:
                Topt = default_Topt.get(
                    site_name.split("_")[0]
                )  # Fallback to default if Topt cant be found
                real_Topt_col = "red"

            min_nee_temp_dict[site_name][year] = Topt
            min_nee_temp_dict2[site_name][year] = (Topt, real_Topt_col)
            min_nee_temp_dict_tmean[site_name][year] = float(df_year["TA_F"].mean())

            # Visualization
            if plot_data:
                plt.figure(figsize=(11.1, 11.1))
                plt.scatter(
                    mean_values.index,
                    mean_values["GPP_NT_VUT_USTAR50"],
                    label=r"grouped GPP per T$_\text{2m}$",
                    color="blue",
                )
                if fitted_curve is not None:
                    plt.plot(
                        mean_values.index,
                        fitted_curve,
                        label="Fitted Curve",
                        color="red",
                    )
                # Topt can be None (default_Topt.get miss) or NaN -> skip the line
                if Topt is not None and np.isfinite(Topt):
                    plt.axvline(
                        Topt,
                        color=real_Topt_col,
                        linestyle="--",
                        label=f"Topt = {Topt:.2f}",
                    )
                plt.xlabel(r"T$_\text{2m}$ [°C]", fontsize=font_size)
                plt.ylabel(r"GPP [$\mu$mol m$^{-2}$ s$^{-1}$]", fontsize=font_size)
                plt.xticks(fontsize=font_size - 2)
                plt.yticks(fontsize=font_size - 2)
                plt.legend(
                    fontsize=font_size,
                    frameon=True,
                    framealpha=0.4,
                )
                plt.grid(True)
                plt.savefig(
                    os.path.join(
                        plot_path,
                        "optimum_temp_" + site_name + "_" + str(year) + ".pdf",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                # plt.show()
                plt.clf()

# === Dump per-site-year GPP-based Topt for the Topt-sensitivity study ===
# Reuses the per-site-year Topt from the loop above (min_nee_temp_dict2); keeps only
# "real" (non-fallback, color != "red") values. The 5-member Topt percentile table is
# derived from this dump by VPRM_tools/derive_topt_members.py (per-site-mean basis,
# V24 = mean as the central member).
_topt_raw_rows = []
for _site, _years in min_nee_temp_dict2.items():
    _pft = _site.split("_")[0]
    for _yr, (_topt, _col) in _years.items():
        if _col != "red" and isinstance(_topt, (int, float)) and np.isfinite(_topt):
            _topt_raw_rows.append([_pft, _site, _yr, float(_topt)])
pd.DataFrame(_topt_raw_rows, columns=["PFT", "site", "year", "Topt"]).to_csv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "VPRM_tools",
        "topt_raw_persiteyear.csv",
    ),
    index=False,
)

# All (T_mean, Topt_GPP) pairs for the Fig 1c regression (non-fallback only)
_gpp_reg_rows = []
for _site, _yd in min_nee_temp_dict2.items():
    _pft = _site.split("_")[0]
    for _yr, (_t, _c) in _yd.items():
        if _c != "red" and isinstance(_t, (int, float)) and np.isfinite(_t):
            _tm = min_nee_temp_dict_tmean.get(_site, {}).get(_yr, np.nan)
            if np.isfinite(_tm):
                _gpp_reg_rows.append({"PFT": _pft, "site": _site, "year": _yr,
                                      "Topt": float(_t), "T_mean": _tm})
df_gpp_topt = pd.DataFrame(_gpp_reg_rows)
df_gpp_topt.to_csv(os.path.join(plot_path, "topt_tmean_pairs.csv"), index=False)

# --- Fitted-Topt panel (top row of the boxplot figures) -----------------------
# Built straight from min_nee_temp_dict2, NOT from df_gpp_topt: the latter also
# requires a finite T_mean and so drops a few site-years, which would make this
# row disagree with the standalone boxplot_Topt_Alps.pdf.
df_topt_pts = build_topt_points(min_nee_temp_dict2)
_perc_csv = ROOT / "VPRM_tools" / "topt_percentiles.csv"
perc = pd.read_csv(_perc_csv).set_index("PFT") if _perc_csv.exists() else None

version = os.getenv("FIGAPPX_VERSION", "V24_SW_05")  # xlsx suffix -> the DATA; override for the Topt-sensitivity members (e.g. V24_SW_05_p10)
iterations = "42"
# Figure filenames carry the version of the data they were built from. They used
# to be pinned to the legacy "V23" tag while the data was already V24_SW_05, which
# left two sets of PDFs side by side and made the manuscript pick up a stale
# Fig. 3(b). See FIGURE_TEX_CHANGES.md for the \includegraphics renames.
plot_version = os.getenv("FIGAPPX_PLOT_VERSION", version)
R2_lt_zero = True  # test so see results for R2_lt_zero - default: True (deletes sites below zero R2)


# European default values
columns = ["ENF", "DBF", "MF", "SHB", "SAV", "CRO", "GRA", "OTH"]

# Define the data (4 rows)
data = [
    [270.2, 271.4, 236.6, 363.0, 682.0, 690.3, 229.1, 0.0],
    [0.1797, 0.1495, 0.2258, 0.0239, 0.0049, 0.1699, 0.0881, 0.0000],
    [0.8800, 0.8233, 0.4321, 0.0000, 0.0000, -0.0144, 0.5843, 0.0000],
    [0.3084, 0.1955, 0.2856, 0.0874, 0.1141, 0.1350, 0.1748, 0.0000],
]

# Create the DataFrame
europe_pars = pd.DataFrame(data, columns=columns)

# Add row labels for the parameters
europe_pars.index = ["RAD0", "alpha", "beta", "lambd"]

for CO2_parametrization in ["old"]:  # "migli","old","new"
    for region in ["Alps"]:  # ,"Europe"
        run_ID = (
            region + "_VPRM_optimized_params_diff_evo_" + version + "_" + iterations
        )
        # Used for every PDF written to plot_path (see plot_version above);
        # run_ID itself still tags the scratch data products so the two never collide.
        run_ID_plot = (
            region
            + "_VPRM_optimized_params_diff_evo_"
            + plot_version
            + "_"
            + iterations
        )
        # base_path = "/home/madse/Downloads/Fluxnet_Data/all_tuned_params/" + run_ID
        # print(f"processing {run_ID}")
        base_path = os.path.join(SCRATCH_PATH, "DATA/Fluxnet2015/Alps/")
        plot_path = OUTFOLDER

        if CO2_parametrization == "migli":
            print(f"for CO2_parametrization of Migliavacca")
        else:
            print(f"for CO2_parametrization of VPRM {CO2_parametrization}")

        folders = [
            f
            for f in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, f))
        ]
        flx_folders = [folder for folder in folders if folder.startswith("FLX_")]

        if not flx_folders:
            print("Warning - There is no input data")
            raise SystemExit(0)

        df_parameters = pd.DataFrame()

        # Loop through each FLX_ folder and append data from XLSX files
        for folder in flx_folders:
            folder_path = os.path.join(base_path, folder)
            files = [
                f
                for f in os.listdir(folder_path)
                if f.endswith(
                    CO2_parametrization
                    + "_diff_evo_"
                    + version
                    + "_"
                    + iterations
                    + ".xlsx"
                )
            ]
            for file in files:
                file_path = os.path.join(folder_path, file)
                data = pd.read_excel(file_path)
                df_parameters = pd.concat([df_parameters, data], axis=0)
        # rename column from df_parameters "PAR0" to "RAD0"
        df_parameters.rename(
            columns={"PAR0": "RAD0"}, inplace=True
        )  # TODO: adopt this in VPRM code

        # --- dump per-site-year lambda/PAR0 for the parameter-spread ensemble (Fig 10) ---
        # Mirrors topt_raw_persiteyear.csv above; lambd is already positive here (the
        # sign flip further down applies only to the aggregated display table).
        pd.DataFrame(
            df_parameters[["PFT", "site_ID", "Year", "RAD0", "lambd"]].dropna().values,
            columns=["PFT", "site", "year", "RAD0", "lambd"],
        ).to_csv(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "VPRM_tools",
                "params_raw_persiteyear.csv",
            ),
            index=False,
        )

        # folders = [
        #     f
        #     for f in os.listdir(base_path)
        #     if os.path.isdir(os.path.join(base_path, f))
        # ]
        # flx_folders = [folder for folder in folders if folder.startswith("FLX_")]

        # if not flx_folders:
        #     print("Warning - There is no input data")
        #     raise SystemExit(0)
        # df_parameters = pd.DataFrame()
        # # Loop through each FLX_ folder and append data from XLSX files
        # for folder in flx_folders:
        #     folder_path = os.path.join(base_path, folder)
        #     files = [f for f in os.listdir(folder_path) if f.endswith(CO2_parametrization+'_diff_evo_'+version+'_'+iterations+'.xlsx')]
        #     for file in files:
        #         file_path = os.path.join(folder_path, file)
        #         data = pd.read_excel(file_path)
        #         df_parameters = pd.concat([df_parameters, data], axis=0)

        df_parameters_nn = df_parameters.copy()
        df_parameters_nn = df_parameters_nn.dropna()

        # Canonical palette (topt_box.PFT_COLORS) + gray for "Others".
        custom_colors = [
            PFT_COLORS[c] for c in ("ENF", "DBF", "MF", "SHB", "SAV", "CRO", "GRA")
        ] + ["#808080"]
        # Full 7-class dict: the former local copy omitted SHB/SAV, which would
        # KeyError below where it is indexed directly rather than via .get().
        pft_colors = PFT_COLORS
        df_parameters_nn = df_parameters_nn[df_parameters_nn["Topt"] < 1]

        print(df_parameters_nn["Topt"] - df_parameters_nn["T_mean"])
        if R2_lt_zero:
            print(
                f"Number of deleted site years due to R2_NEE < 0 = {sum(df_parameters['R2_NEE'] < 0)}"
            )
            df_parameters = df_parameters[df_parameters["R2_NEE"] > 0]
            df_parameters.reset_index(drop=True, inplace=True)
            str_R2_lt_zero = ""
        else:
            print(
                f"Number of deleted site years due to R2_NEE > 0 = {sum(df_parameters['R2_NEE'] < 0)}"
            )
            df_parameters = df_parameters[df_parameters["R2_NEE"] < 0]
            df_parameters.reset_index(drop=True, inplace=True)
            str_R2_lt_zero = "_R2_lt_zero"

        # ~2/3 height so this prints at the same height as Fig. 3(a),
        # whose saved PDF has aspect 0.66 (417.8 x 277.0 pt).
        plt.figure(figsize=(6.25, 4.08))
        _df_reg = df_gpp_topt[df_gpp_topt["PFT"] != "CRO"]
        _reg_colors = [pft_colors.get(p, "#999999") for p in _df_reg["PFT"]]
        plt.scatter(
            _df_reg["T_mean"],
            _df_reg["Topt"],
            alpha=1,
            c=_reg_colors,
        )
        coefficients = np.polyfit(_df_reg["T_mean"], _df_reg["Topt"], 1)
        poly = np.poly1d(coefficients)
        equation_regression = f"y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}"
        print(equation_regression)
        plt.plot(
            _df_reg["T_mean"],
            poly(_df_reg["T_mean"]),
            color="red",
            label=f"y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}",
        )
        plt.xlabel(r"$T_{\mathrm{mean}}$ [°C]", fontsize=font_size)
        plt.ylabel(r"$T_{\mathrm{opt}}$ (GPP) [°C]", fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(True)

        # Legend from the classes actually plotted, not from the palette: CRO is
        # filtered out of _df_reg (cutting events) and SHB/SAV have no Alpine
        # sites, so keying off the 7-class palette listed classes with no points.
        _reg_pfts = set(_df_reg["PFT"])
        for pft, color in pft_colors.items():
            if pft not in _reg_pfts:
                continue
            plt.scatter(
                [], [], c=color, label=pft
            )  # Create an empty scatter plot for each PFT label

        plt.legend(
            fontsize=font_size - 3,
            frameon=True,
            framealpha=0.4,
            loc="lower right",  # upper left sat on top of the scatter
        )
        plt.tight_layout()
        plt.savefig(
            plot_path
            + "/regression_Topt_vs_Tmean_"
            + CO2_parametrization
            + "_"
            + run_ID_plot
            + str_R2_lt_zero
            + ".pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(6.25, 6.25))
        plt.scatter(
            df_parameters["T_max"].dropna(),
            df_parameters["Topt"].dropna(),
            alpha=1,
            c=df_parameters["PFT"].map(pft_colors),
        )
        coefficients = np.polyfit(df_parameters["T_max"], df_parameters["Topt"], 1)
        poly = np.poly1d(coefficients)
        plt.plot(
            df_parameters["T_max"],
            poly(df_parameters["T_max"]),
            color="red",
            label=f"y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}",
        )
        equation_regression = f"y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}"
        equation_normal = "y = x"
        plt.xlabel(r"$T_{\mathrm{max}}$", fontsize=font_size)
        plt.ylabel(r"$T_{\mathrm{opt}}$", fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        plt.grid(True)

        for pft, color in pft_colors.items():
            plt.scatter(
                [], [], c=color, label=pft
            )  # Create an empty scatter plot for each PFT label

        plt.legend(
            fontsize=font_size,
            frameon=True,
            framealpha=0.4,
        )

        plt.tight_layout()
        plt.savefig(
            base_path
            + "/regression_Topt_vs_Tmax_"
            + CO2_parametrization
            + "_"
            + run_ID
            + str_R2_lt_zero
            + ".pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        if CO2_parametrization == "new":
            parameters_to_plot = [
                "Topt",
                "RAD0",
                "lambd",
                "alpha1",
                "alpha2",
                "beta",
                "T_crit",
                "T_mult",
                "gamma",
                "theta1",
                "theta2",
                "theta3",
            ]
            labels_plot = {
                "Topt": r"T$_\text{opt}$ [°C]",
                "RAD0": r"RAD$_0$",
                "lambd": r"$\lambda$",
                "alpha1": r"$\alpha_1$",
                "alpha2": r"$\alpha_2$",
                "beta": r"$\beta$",
                "T_crit": r"T$_\text{crit}$ [°C]",
                "T_mult": r"T$_\text{mult}$",
                "gamma": r"$\gamma$",
                "theta1": r"$\theta_1$",
                "theta2": r"$\theta_2$",
                "theta3": r"$\theta_3$",
            }
        elif CO2_parametrization == "old":
            parameters_to_plot = [
                "RAD0",
                "lambd",
                "alpha",
                "beta",
            ]
            labels_plot = {
                "RAD0": r"RAD$_0$ $[\mu \text{mol m}^{\text{-2}} \text{s}^{-1}]$",
                "lambd": r"$\lambda$ [-]",
                "alpha": r"$\alpha$ $[\frac{\mu \text{mol m}^{\text{-2}} \text{s}^{-1}}{K}]$",
                "beta": r"$\beta$ $[\mu \text{mol m}^{\text{-2}} \text{s}^{-1}]$",
            }
        elif CO2_parametrization == "migli":
            parameters_to_plot = [
                "k2",
                "E0(K)",
                "alpha_p",
                "K (mm)",
                "days_memory",
                "window_center",
                "half_width",
            ]
            labels_plot = {
                "k2": "k2",
                "E0(K)": "E0(K)",
                "alpha_p": r"$\alpha_p$",
                "K (mm)": "K (mm)",
                "days_memory": "days memory",
                "window_center": "window center",
                "half_width": "half width",
            }
        # Define the color palette and the PFT color mapping

        df_parameters.sort_values(by="PFT", inplace=True)
        # Create a list of colors for the boxplot based on the sorted PFTs
        pft_order = list(df_parameters["PFT"].unique())
        colors = [pft_colors[pft] for pft in pft_order]

        # One extra row on top for the fitted Topt (the xlsx Topt is prescribed
        # per class, so its spread only exists in the GPP-vs-T fits).
        if CO2_parametrization == "new":
            fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(9.4, 15.7))
        elif CO2_parametrization == "old":
            fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6.3, 11.8))
        elif CO2_parametrization == "migli":
            fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(6.3, 25.2))

        axes = axes.flatten()

        # --- top row: fitted Topt per vegetation class ---
        # bxp positions must be 0-based to line up with seaborn's categorical axis.
        draw_topt_box(
            axes[0],
            df_topt_pts,
            order=pft_order,
            group_col="PFT",
            perc=perc,
            colors=colors,
            positions=range(len(pft_order)),
            show_counts=False,  # site counts belong in the caption, not the ticks
            xlabel=None,  # x axis is only titled once, on the bottom row
            label_fontsize=font_size,
            tick_fontsize=font_size,
            show_legend=False,  # documented in the legend strip below instead
        )
        # every row shares the same categories -> label only the bottom one
        axes[0].tick_params(axis="x", labelbottom=False)

        _last = len(parameters_to_plot)  # index of the bottom row
        for i, parameter in enumerate(parameters_to_plot, start=1):
            label = labels_plot.get(parameter, parameter)
            # No swarmplot: it plotted every site-year, which showed far more
            # points here than the top Topt row (now outliers-only, see
            # draw_topt_box) -- sns.boxplot's default fliers keep both consistent.
            sns.boxplot(
                x="PFT",
                y=parameter,
                data=df_parameters,
                ax=axes[i],
                palette=colors,
                order=pft_order,
                flierprops=FLIER_PROPS,
            )
            # axes[i].set_title(f'{parameter} by vegetation class',fontsize=font_size+2, weight='bold')
            # all rows share the same categories -> title the x axis only at the bottom
            axes[i].set_xlabel(
                "vegetation class" if i == _last else "", fontsize=font_size
            )
            axes[i].set_ylabel(label, fontsize=font_size - 2)
            axes[i].tick_params(axis="both", which="major", labelsize=font_size)
            if i == _last:
                axes[i].tick_params(axis="x", rotation=0)  # class names are short
            else:
                axes[i].tick_params(axis="x", labelbottom=False)

        for _ax in axes[len(parameters_to_plot) + 1:]:
            _ax.set_visible(False)

        fig.subplots_adjust(**BOX_MARGINS)
        plt.savefig(
            plot_path
            + "/boxplot_PFTs_"
            + CO2_parametrization
            + "_"
            + run_ID_plot
            + str_R2_lt_zero
            + ".pdf",
            dpi=300,
        )
        plt.close()

        # One legend for both boxplot_PFTs and boxplot_siteIDs (see
        # save_topt_box_legend) -- place it below the pair in the manuscript.
        save_topt_box_legend(
            plot_path + "/legend_topt_boxplots.pdf",
            pft_order, pft_colors, font_size - 2,
        )

        # Create a dictionary mapping site_ID to PFT
        site_to_pft = df_parameters.set_index("site_ID")["PFT"].to_dict()

        # Create a list of colors for each site based on the PFT
        site_order = list(df_parameters["site_ID"].unique())
        site_colors = [pft_colors[site_to_pft[site]] for site in site_order]

        # Topt point cloud keys are PFT-prefixed ("ENF_CH-Dav"); the parameter
        # frame uses the bare site_ID. Align them on the bare name so the top
        # row shares the x categories of the parameter rows below it.
        df_topt_sites = df_topt_pts.assign(
            site_ID=df_topt_pts["site"].str.split("_", n=1).str[1]
        )

        if CO2_parametrization == "new":
            fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(9.4, 15.7))
        elif CO2_parametrization == "old":
            fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6.3, 11.8))
        elif CO2_parametrization == "migli":
            fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(6.3, 25.2))
        axes = axes.flatten()

        # --- top row: fitted Topt per site (raw year values; no per-site
        # percentile table exists). Sites without a fitted Topt -- the CRO sites,
        # excluded because cutting events distort the GPP-vs-T fit -- keep their
        # x slot and stay blank so all rows remain column-aligned.
        draw_topt_box(
            axes[0],
            df_topt_sites,
            order=site_order,
            group_col="site_ID",
            perc=None,
            colors=site_colors,
            positions=range(len(site_order)),
            show_counts=False,
            xlabel=None,  # x axis is only titled once, on the bottom row
            label_fontsize=font_size,
            tick_fontsize=font_size,
            show_legend=False,
        )
        axes[0].tick_params(axis="x", labelbottom=False)

        _last = len(parameters_to_plot)  # index of the bottom row
        for i, parameter in enumerate(parameters_to_plot, start=1):
            label = labels_plot.get(parameter, parameter)
            sns.boxplot(
                x="site_ID",
                y=parameter,
                data=df_parameters,
                ax=axes[i],
                palette=site_colors,
                order=site_order,
                flierprops=FLIER_PROPS,
            )
            # all rows share the same categories -> title the x axis only at the bottom
            axes[i].set_xlabel("site_ID" if i == _last else "", fontsize=font_size)
            axes[i].set_ylabel(label, fontsize=font_size - 2)
            axes[i].tick_params(axis="both", which="major", labelsize=font_size)
            if i == _last:
                axes[i].tick_params(axis="x", rotation=90)  # site IDs are long
            else:
                axes[i].tick_params(axis="x", labelbottom=False)
            if CO2_parametrization == "new":
                axes[i].tick_params(axis="x", which="major", labelsize=font_size)

        for _ax in axes[len(parameters_to_plot) + 1:]:
            _ax.set_visible(False)

        # No legend here: the strip under boxplot_PFTs is worded generically
        # and serves both panels of Fig. I16.

        fig.subplots_adjust(**BOX_MARGINS)
        plt.savefig(
            plot_path
            + "/boxplot_siteIDs_"
            + CO2_parametrization
            + "_"
            + run_ID_plot
            + str_R2_lt_zero
            + ".pdf",
            dpi=300,
        )
        plt.close()

        grouped = df_parameters.groupby("PFT")
        dfs_to_concat = []
        for parameter in parameters_to_plot:
            for pft, group_data in grouped:
                mean = group_data[parameter].mean()
                median = group_data[parameter].median()
                # Create a DataFrame with the new row
                new_row = pd.DataFrame(
                    {
                        "PFT": [pft],
                        "Parameter": [parameter],
                        "Mean": [mean],
                        "Median": [median],
                    }
                )
                # Append the new DataFrame to the list
                dfs_to_concat.append(new_row)

        mean_median_df = pd.concat(dfs_to_concat, ignore_index=True)
        mean_median_df.to_excel(
            base_path
            + "/mean_median_params_"
            + CO2_parametrization
            + "_"
            + run_ID
            + str_R2_lt_zero
            + ".xlsx",
            index=False,
        )

        # Pivoting the DataFrame
        pivoted_mean = mean_median_df.pivot(
            index="Parameter", columns="PFT", values="Mean"
        )
        pivoted_median = mean_median_df.pivot(
            index="Parameter", columns="PFT", values="Median"
        )

        # Adding a column with the mean of all PFTs
        # pivoted_mean['Mean_All_PFTs'] = df_parameters[parameters_to_plot].mean()
        # pivoted_median['Median_All_PFTs'] = df_parameters[parameters_to_plot].median()
        # add europe_pars["SAV","SHB","OTH"] to pivoted_mean =
        pivoted_mean[["SAV", "SHB", "OTH"]] = europe_pars[["SAV", "SHB", "OTH"]]
        pivoted_median[["SAV", "SHB", "OTH"]] = europe_pars[["SAV", "SHB", "OTH"]]
        pivoted_mean = pivoted_mean[
            ["ENF", "DBF", "MF", "SHB", "SAV", "CRO", "GRA", "OTH"]
        ]
        pivoted_median = pivoted_median[
            ["ENF", "DBF", "MF", "SHB", "SAV", "CRO", "GRA", "OTH"]
        ]
        pivoted_mean = pivoted_mean.reindex(["RAD0", "lambd", "alpha", "beta"])
        pivoted_median = pivoted_median.reindex(["RAD0", "lambd", "alpha", "beta"])
        pivoted_mean.loc["lambd"] = pivoted_mean.loc["lambd"] * -1
        pivoted_median.loc["lambd"] = pivoted_median.loc["lambd"] * -1
        # Exporting to CSV
        # save values with precision of 3 digits
        pivoted_mean.to_csv(
            base_path
            + "/"
            + region
            + "_parameters_mean_"
            + CO2_parametrization
            + "_"
            + run_ID
            + str_R2_lt_zero
            + ".csv",
            index=False,
            float_format="%.3f",
        )
        pivoted_median.to_csv(
            base_path
            + "/"
            + region
            + "_parameters_median_"
            + CO2_parametrization
            + "_"
            + run_ID
            + str_R2_lt_zero
            + ".csv",
            index=False,
            float_format="%.3f",
        )

        parameters_to_plot = [
            "R2_GPP",
            "RMSE_GPP",
            "MAE_GPP",
            "R2_Reco",
            "RMSE_Reco",
            "MAE_Reco",
            "R2_NEE",
            "RMSE_NEE",
            "MAE_NEE",
        ]
        grouped = df_parameters.groupby("PFT")
        dfs_to_concat = []
        for parameter in parameters_to_plot:
            for pft, group_data in grouped:
                mean = group_data[parameter].mean()
                median = group_data[parameter].median()
                # Create a DataFrame with the new row
                new_row = pd.DataFrame(
                    {
                        "PFT": [pft],
                        "Parameter": [parameter],
                        "Mean": [mean],
                        "Median": [median],
                    }
                )
                # Append the new DataFrame to the list
                dfs_to_concat.append(new_row)

        mean_median_df = pd.concat(dfs_to_concat, ignore_index=True)
        mean_median_df.to_excel(
            base_path
            + "/mean_median_R2_RMSE_"
            + CO2_parametrization
            + "_"
            + run_ID
            + str_R2_lt_zero
            + ".xlsx",
            index=False,
        )

        parameters_to_plot = ["NNSE_GPP", "NNSE_Reco", "NNSE_NEE"]
        labels_plot = {
            "NNSE_GPP": "NNSE GPP",
            "NNSE_Reco": r"NNSE R$_\text{eco}$",
            "NNSE_NEE": "NNSE NEE",
        }
        # parameters_to_plot = ['R2_GPP', 'RMSE_GPP', 'MAE_GPP', 'R2_Reco', 'RMSE_Reco', 'MAE_Reco', 'R2_NEE', 'RMSE_NEE', 'MAE_NEE']

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8.3, 4.15))
        axes = axes.flatten()

        _mid = len(parameters_to_plot) // 2  # panels sit side by side -> label the middle one
        for i, parameter in enumerate(parameters_to_plot):
            label = labels_plot.get(parameter, parameter)
            # No swarmplot (matches boxplot_PFTs/boxplot_siteIDs, see topt_box.py):
            # a raw per-site-year dot cloud on top of the box was busier than the
            # fitted-Topt row it now sits alongside; sns.boxplot's default fliers
            # via FLIER_PROPS keep the outlier style consistent with those panels.
            sns.boxplot(
                x="PFT",
                y=parameter,
                data=df_parameters,
                ax=axes[i],
                palette=colors,
                order=pft_order,
                flierprops=FLIER_PROPS,
            )
            # all panels share the same categories -> title the x axis only once
            axes[i].set_xlabel(
                "vegetation class" if i == _mid else "", fontsize=font_size
            )
            axes[i].set_ylabel(label, fontsize=font_size - 2)
            # class names are short (matches boxplot_PFTs), but these 5 categories
            # sit in a third of that plot's width (3 side-by-side panels here vs.
            # its single wide column) -- a smaller x tick label keeps them
            # horizontal without overlapping.
            axes[i].tick_params(axis="x", rotation=0, labelsize=font_size - 5)
            axes[i].tick_params(axis="y", which="major", labelsize=font_size)
            if "R2" in parameter:
                axes[i].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(
            plot_path
            + "/boxplot_NNSE_"
            + CO2_parametrization
            + "_"
            + run_ID_plot
            + str_R2_lt_zero
            + ".pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
