import matplotlib

matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit, minimize
import seaborn as sns


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
base_path = "/scratch/c7071034/DATA/Fluxnet2015/Alps/"
plot_path = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"
font_size = 20
site_info = pd.read_csv(
    "/scratch/c7071034/DATA/Fluxnet2015/Alps/site_info_all_FLUXNET2015.csv"
)
plot_data = True  # Set this to False if you don't want to plot the data
save_data = True  # Set this to False if you don't want to save the data
plot_boxplot = True  # Set this to False if you don't want to plot the boxplot

# Initialize a dictionary to hold min values for each site and year
min_nee_temp_dict = {}
min_nee_temp_dict2 = {}
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
        site_name = target_pft + "_" + site_name

        # Columns to read and converters
        columns_to_copy = [
            "TIMESTAMP_START",
            "TA_F",
            "NEE_VUT_REF",
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
        df_site["NEE_VUT_REF"] = df_site["NEE_VUT_REF"].replace(-9999.0, np.nan)
        # check how many nan values are in NEE_VUT_REF
        nan_values = df_site["NEE_VUT_REF"].isna()
        nan_sum = nan_values.sum()
        full_year = len(df_site["NEE_VUT_REF"])
        df_site = df_site.dropna(subset=["NEE_VUT_REF"])
        percent_nan = nan_sum.max() / full_year * 100

        #  find the name of the column with the most missing values
        if percent_nan > 20:
            print(
                f"WARNING: for {site_name} year {year} is skipped, as {percent_nan:2.1f}% are missing"
            )
            continue

        # Set the values to np.nan during nighttime
        night_columns = ["NEE_VUT_REF"]
        df_site.loc[df_site["NIGHT"] == 1, night_columns] = np.nan

        # Convert units from micromol per m² per second to grams of Carbon per day
        # conversion_factor = 12 * 1e-6 * 60 * 60 * 24  # 12 g C per mol, micromol to mol, per second to per day
        # df_site['NEE_VUT_REF'] *= conversion_factor

        # Resample to daily frequency
        df_site.set_index("TIMESTAMP_START", inplace=True)
        df_daily = (
            df_site.resample("D").agg({"TA_F": "mean", "NEE_VUT_REF": "mean"}).dropna()
        )

        # Extract the year from the timestamp
        df_daily["YEAR"] = df_daily.index.year

        # List of unique years
        years = df_daily["YEAR"].unique()

        # Initialize site entry in the dictionary
        if site_name not in min_nee_temp_dict:
            min_nee_temp_dict[site_name] = {}
            min_nee_temp_dict2[site_name] = {}

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
                    mirrored_gaussian,
                    mean_values.index,
                    mean_values["NEE_VUT_REF"],
                    maxfev=10000,
                )
                Topt = popt[
                    1
                ]  # Extract the temperature at the peak of the Gaussian (mean)

                # Generate the fitted curve for plotting or evaluation
                fitted_curve = mirrored_gaussian(mean_values.index, *popt)

                # Calculate RMSE for the Gaussian fit
                rmse = calculate_rmse(mean_values["NEE_VUT_REF"], fitted_curve)
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
                        mean_values["NEE_VUT_REF"],
                        maxfev=10000,
                    )

                    # Find the minimum of the cubic polynomial curve (Topt) using optimization
                    Topt = find_minimum_of_cubic_poly(popt_poly, mean_values.index)

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

            # Visualization
            if plot_data:
                plt.figure(figsize=(8, 8))
                plt.scatter(
                    mean_values.index,
                    mean_values["NEE_VUT_REF"],
                    label=r"grouped NEE per T$_\text{2m}$",
                    color="blue",
                )
                if fitted_curve is not None:
                    plt.plot(
                        mean_values.index,
                        fitted_curve,
                        label="Fitted Curve",
                        color="red",
                    )
                if not np.isnan(Topt):
                    plt.axvline(
                        Topt,
                        color=real_Topt_col,
                        linestyle="--",
                        label=f"Topt = {Topt:.2f}",
                    )
                plt.xlabel(r"T$_\text{2m}$ [°C]", fontsize=font_size)
                plt.ylabel(r"NEE [$\mu$mol m$^{-2}$ s$^{-1}$]", fontsize=font_size)
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

version = "V23"
iterations = "42"
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
        # base_path = "/home/madse/Downloads/Fluxnet_Data/all_tuned_params/" + run_ID
        # print(f"processing {run_ID}")
        base_path = "/scratch/c7071034/DATA/Fluxnet2015/Alps/"
        plot_path = "/home/c707/c7071034/Github/WRF_VPRM_post/plots"

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

        custom_colors = [
            "#006400",
            "#228B22",
            "#8FBC8F",
            "#A0522D",
            "#FFD700",
            "#FFA07A",
            "#7CFC00",
            "#808080",
        ]  # Added gray for "Others"
        pft_colors = {
            "ENF": "#006400",
            "DBF": "#228B22",
            "MF": "#8FBC8F",
            "CRO": "#FFA07A",
            "GRA": "#7CFC00",
        }
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

        plt.figure(figsize=(8, 8))
        plt.scatter(
            df_parameters["T_mean"],
            df_parameters["Topt"],
            alpha=1,
            c=df_parameters["PFT"].map(pft_colors),
        )
        coefficients = np.polyfit(df_parameters["T_mean"], df_parameters["Topt"], 1)
        poly = np.poly1d(coefficients)
        equation_regression = f"y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}"
        print(equation_regression)
        equation_normal = "y = x"
        plt.plot(
            df_parameters["T_mean"],
            poly(df_parameters["T_mean"]),
            color="red",
            label=f"y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}",
        )
        plt.xlabel(r"$T_{\mathrm{mean}}$ [°C]", fontsize=font_size)
        plt.ylabel(r"$T_{\mathrm{opt}}$ [°C]", fontsize=font_size)
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
            plot_path
            + "/regression_Topt_vs_Tmean_"
            + CO2_parametrization
            + "_"
            + run_ID
            + str_R2_lt_zero
            + ".pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(8, 8))
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
        pft_order = df_parameters["PFT"].unique()
        colors = [pft_colors[pft] for pft in pft_order]

        if CO2_parametrization == "new":
            fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))
        elif CO2_parametrization == "old":
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 15))
        elif CO2_parametrization == "migli":
            fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(10, 35))

        axes = axes.flatten()
        for i, parameter in enumerate(parameters_to_plot):
            label = labels_plot.get(parameter, parameter)
            sns.boxplot(
                x="PFT", y=parameter, data=df_parameters, ax=axes[i], palette=colors
            )
            sns.swarmplot(
                x="PFT",
                y=parameter,
                data=df_parameters,
                color="0.25",
                alpha=0.5,
                ax=axes[i],
            )
            # axes[i].set_title(f'{parameter} by PFT',fontsize=font_size+2, weight='bold')
            axes[i].set_xlabel("PFT", fontsize=font_size)
            axes[i].set_ylabel(label, fontsize=font_size - 2)
            axes[i].tick_params(axis="x", rotation=45)
            axes[i].tick_params(axis="both", which="major", labelsize=font_size)

        handles = []
        for pft, color in pft_colors.items():
            handles.append(plt.scatter([], [], c=color, label=pft))

        # plt.legend(handles=handles, bbox_to_anchor=(0.5, -0.27), loc='upper center', ncol=len(pft_colors)//2, fontsize=font_size)

        plt.tight_layout()
        plt.savefig(
            plot_path
            + "/boxplot_PFTs_"
            + CO2_parametrization
            + "_"
            + run_ID
            + str_R2_lt_zero
            + ".pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create a dictionary mapping site_ID to PFT
        site_to_pft = df_parameters.set_index("site_ID")["PFT"].to_dict()

        # Create a list of colors for each site based on the PFT
        site_colors = [
            pft_colors[site_to_pft[site]] for site in df_parameters["site_ID"].unique()
        ]

        if CO2_parametrization == "new":
            fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))
        elif CO2_parametrization == "old":
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 15))
        elif CO2_parametrization == "migli":
            fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(10, 35))
        axes = axes.flatten()

        for i, parameter in enumerate(parameters_to_plot):
            label = labels_plot.get(parameter, parameter)
            sns.boxplot(
                x="site_ID",
                y=parameter,
                data=df_parameters,
                ax=axes[i],
                palette=site_colors,
            )
            axes[i].set_xlabel("site_ID", fontsize=font_size)
            axes[i].set_ylabel(label, fontsize=font_size - 2)
            axes[i].tick_params(axis="x", rotation=45)
            axes[i].tick_params(axis="both", which="major", labelsize=font_size)
            if CO2_parametrization == "new":
                axes[i].tick_params(axis="x", which="major", labelsize=font_size)

        # Create legend handles
        handles = []
        for pft, color in pft_colors.items():
            handles.append(plt.scatter([], [], c=color, label=pft))

        axes[0].legend(
            handles=handles,
            loc="upper right",
            ncol=len(pft_colors),
            fontsize=font_size - 2,
            frameon=True,
            framealpha=0.4,
        )

        plt.tight_layout()
        plt.savefig(
            plot_path
            + "/boxplot_siteIDs_"
            + CO2_parametrization
            + "_"
            + run_ID
            + str_R2_lt_zero
            + ".pdf",
            dpi=300,
            bbox_inches="tight",
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

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        axes = axes.flatten()

        for i, parameter in enumerate(parameters_to_plot):
            label = labels_plot.get(parameter, parameter)
            sns.boxplot(
                x="PFT", y=parameter, data=df_parameters, ax=axes[i], palette=colors
            )
            sns.swarmplot(
                x="PFT",
                y=parameter,
                data=df_parameters,
                color="0.25",
                alpha=0.5,
                ax=axes[i],
            )
            axes[i].set_xlabel("PFT", fontsize=font_size)
            axes[i].set_ylabel(label, fontsize=font_size - 2)
            axes[i].tick_params(axis="x", rotation=45)
            axes[i].tick_params(axis="both", which="major", labelsize=font_size)
            if "R2" in parameter:
                axes[i].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(
            plot_path
            + "/boxplot_NNSE_"
            + CO2_parametrization
            + "_"
            + run_ID
            + str_R2_lt_zero
            + ".pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
