import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit, minimize


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
site_info = pd.read_csv(
    "/scratch/c7071034/DATA/Fluxnet2015/Alps/site_info_all_FLUXNET2015.csv"
)
plot_data = False  # Set this to False if you don't want to plot the data
save_data = True  # Set this to False if you don't want to save the data
plot_boxplot = True  # Set this to False if you don't want to plot the boxplot
font_size = 20
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

                # plt.title(f'Optimal Temperature (Topt) for {site_name} in {year}')
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


# Filter the dictionary to exclude entries where real_Topt_col is "red"
filtered_dict = {
    site: {year: Topt for year, (Topt, color) in year_data.items() if color != "red"}
    for site, year_data in min_nee_temp_dict2.items()
}

# Convert the filtered dictionary to a DataFrame
filtered_df = pd.DataFrame(filtered_dict).T

# Sort the DataFrame by site and year
filtered_df = filtered_df.sort_index(axis=0, ascending=True)

# Optionally save the filtered DataFrame to a CSV file
if save_data:
    filtered_df.to_csv(os.path.join(plot_path, "filtered_Topt_values.csv"))

# Display the filtered DataFrame
print(filtered_df)


counts = {}
for site, years in min_nee_temp_dict2.items():
    vals = [val[0] for val in years.values()]  # take first element of tuple
    non_nan_count = sum(isinstance(v, float) and not np.isnan(v) for v in vals)
    counts[site] = non_nan_count

print(counts)
# print sum of all counts
print("Total number of valid Topt entries:", sum(counts.values()))
if plot_boxplot:
    min_nee_temp_df_long = filtered_df.reset_index().melt(
        id_vars="index", var_name="Year", value_name="Temperature"
    )
    min_nee_temp_df_long.rename(columns={"index": "Site"}, inplace=True)
    # Ensure Year is string for safe concatenation/labeling
    min_nee_temp_df_long["Year"] = min_nee_temp_df_long["Year"].astype(str)

    font_size = 23
    plt.figure(figsize=(8, 8))

    # Collect unique sites and data
    sites = min_nee_temp_df_long["Site"].unique()
    data = [
        min_nee_temp_df_long.loc[min_nee_temp_df_long["Site"] == site, "Temperature"]
        .dropna()
        .values
        for site in sites
    ]

    # Boxplot with mean line
    plt.boxplot(
        data,
        vert=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops=dict(facecolor="lightblue", color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),  # median black
        meanprops=dict(color="green", linestyle="-", linewidth=3),  # dashed mean
    )

    # Align y-ticks with site names
    plt.yticks(range(1, len(sites) + 1), sites)
    plt.xlabel(r"$T_{\mathrm{opt}}$ [°C]", fontsize=font_size)
    plt.ylabel("Site", fontsize=font_size)
    plt.tick_params(labelsize=16)
    # plt.title('optimum Temperatures from FLUXNET2015 TA_F and NEE_VUT_REF',fontsize=font_size)
    plt.suptitle("")  # Remove the default 'Boxplot grouped by Site' title
    plt.grid(True)

    # Save the plot to an EPS file
    plt.savefig(
        os.path.join(plot_path, "boxplot_Topt_Alps.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
