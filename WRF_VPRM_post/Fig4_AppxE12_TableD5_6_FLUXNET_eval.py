"""
Fig4_FLUXNET_eval.py

Refactored from `plot_FLX_ts_1km.py` with minimal changes to behaviour.
- Organised into functions: main(), load_csvs(), load_cams(), process_location()
- Removed stray tokens and small cleanup only.
- Keeps plotting and metric calculations unchanged.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timedelta
import pytz
import numpy as np
import pandas as pd
import matplotlib

# use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from permetrics import RegressionMetric
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc


# ---- Helper functions (kept from original, lightly wrapped) ----


def find_file_paths(base_dir, location):
    """Find CSV file for FLUXNET site by pattern."""
    pattern = f"FLX_{location}_FLUXNET2015_FULLSET*/FLX_{location}_FLUXNET2015_FULLSET_HH*.csv"
    search_pattern = os.path.join(base_dir, pattern)
    matches = glob.glob(search_pattern)
    if matches:
        return matches[0]
    else:
        print(f"No file found for location: {location}")
        return None


def read_FLUXNET_site(start_date, end_date, location, base_dir, var_flx):
    """Read and preprocess FLUXNET site CSV for given variable.

    Returns resampled hourly DataFrame with column var_flx and column 'NIGHT'.
    """
    start_date_obj = start_date.replace(tzinfo=pytz.UTC)
    end_date_obj = end_date.replace(tzinfo=pytz.UTC)

    file_path = find_file_paths(base_dir, location)
    if file_path is None:
        return pd.DataFrame()

    df_FLX_site = pd.read_csv(file_path, sep=",")

    # Drop second row (units row) if present (index 0 is header row)
    if len(df_FLX_site) > 0:
        # In the dataset the first data row may contain units; drop row 0 if it's non-datetime
        try:
            _ = pd.to_datetime(
                df_FLX_site.loc[0, "TIMESTAMP_START"], format="%Y%m%d%H%M"
            )
        except Exception:
            # safest: drop first row that contained units
            df_FLX_site = df_FLX_site.drop(index=0)

    # Parse timestamps
    df_FLX_site["TIMESTAMP_START"] = pd.to_datetime(
        df_FLX_site["TIMESTAMP_START"], format="%Y%m%d%H%M", errors="coerce"
    )

    if df_FLX_site["TIMESTAMP_START"].isna().any():
        print("Some 'TIMESTAMP_START' values could not be parsed:")
        print(df_FLX_site[df_FLX_site["TIMESTAMP_START"].isna()].head())

    # Adjust timezone handling: previous code subtracted 1 hour then localized to UTC
    df_FLX_site["TIMESTAMP_START"] = df_FLX_site["TIMESTAMP_START"] - pd.Timedelta(
        hours=1
    )
    df_FLX_site["TIMESTAMP_START"] = df_FLX_site["TIMESTAMP_START"].dt.tz_localize(
        "UTC"
    )

    # Filter by requested date range
    df_FLX_site = df_FLX_site[
        (df_FLX_site["TIMESTAMP_START"] >= start_date_obj)
        & (df_FLX_site["TIMESTAMP_START"] <= end_date_obj)
    ]

    if df_FLX_site.empty:
        print(f"No data in the specified date range for {var_flx} in {location}.")
        return pd.DataFrame()

    # Select relevant columns and clean
    if var_flx not in df_FLX_site.columns:
        print(f"Requested variable {var_flx} not in FLUXNET file columns.")
        return pd.DataFrame()

    df_FLX_site = df_FLX_site[["TIMESTAMP_START", var_flx, "NIGHT"]].copy()
    df_FLX_site = df_FLX_site.mask(df_FLX_site == -9999, np.nan)

    # For GPP variables ensure non-negative daytime values
    if var_flx.startswith("GPP"):
        df_FLX_site[var_flx] = np.where(
            df_FLX_site["NIGHT"] == 1, 0, df_FLX_site[var_flx]
        )
        df_FLX_site[var_flx] = df_FLX_site[var_flx].clip(lower=0)

    # Index and resample to hourly means
    df_FLX_site["TIMESTAMP_START"] = pd.to_datetime(
        df_FLX_site["TIMESTAMP_START"]
    )  # drop tz
    df_FLX_site.set_index("TIMESTAMP_START", inplace=True)
    df_FLX_site_resampled = df_FLX_site.resample("1h").mean()
    df_FLX_site_resampled.reset_index(inplace=True)

    return df_FLX_site_resampled


# ---- CAMS loading / interpolation ----


def get_int_var(lat_target, lon_target, lats, lons, var_CAMS):
    interpolator = RegularGridInterpolator((lats, lons), var_CAMS)
    return interpolator((lat_target, lon_target))


def load_CAMS(path_CAMS_file):
    ds = nc.Dataset(path_CAMS_file)
    times_CAMS = ds.variables["valid_time"]
    gppbfas = ds.variables["gppbfas"][:]
    rec_bfas = ds.variables["recbfas"][:]
    lat_CAMS = ds.variables["latitude"][:]
    lon_CAMS = ds.variables["longitude"][:]
    return ds, times_CAMS, gppbfas, rec_bfas, lat_CAMS, lon_CAMS


# ---- Main processing function (single location) ----


def process_location(
    location,
    fluxtypes,
    plot_labels,
    units,
    dataframes,
    start_date,
    end_date,
    base_dir_FLX,
    df_CAMS_hourly_all,
    resolution_colors,
    sim_type,
    outfolder,
    timespan,
):
    consolidated_metrics_all = pd.DataFrame()

    for fluxtype, unit, plot_label in zip(fluxtypes, units, plot_labels):
        consolidated_metrics_df = pd.DataFrame(
            index=["MAE", "RMSE", "R2"]
        )  # Metrics as rows

        variables = {
            f"SITE_{fluxtype}": "solid",
            f"ALPS_{fluxtype}": "dashdot",
            f"REF_{fluxtype}": "dashed",
        }
        WRF_plot_labels = {
            f"SITE_{fluxtype}": plot_label + " SITE",
            f"ALPS_{fluxtype}": plot_label + " ALPS",
            f"REF_{fluxtype}": plot_label + " REF",
        }

        # choose var_flx and CAMS var
        if fluxtype == "RECO_WRF":
            var_flx = "RECO_NT_VUT_USTAR50"
            var_CAMS_plot = "fco2rec"
        elif fluxtype == "GPP_WRF":
            var_flx = "GPP_NT_VUT_USTAR50"
            var_CAMS_plot = "fco2gpp"
        elif fluxtype == "NEE_WRF":
            var_flx = "NEE_VUT_USTAR50"
            var_CAMS_plot = "fco2nee"
        elif fluxtype == "T2_WRF":
            var_flx = "TA_F"
            var_CAMS_plot = "t2m"
        else:
            print(f"Unknown fluxtype: {fluxtype}")
            continue

        # read FLUXNET site data
        df_FLX_site = read_FLUXNET_site(
            start_date, end_date, location, base_dir_FLX, var_flx
        )

        if df_FLX_site.empty:
            print(f"Skipping {location} for {fluxtype} due to empty FLUXNET data.")
            continue

        plt.figure(figsize=(10, 6))
        for resolution, df in dataframes.items():
            # build df_loc depending on fluxtype (kept logic identical)
            if fluxtype == "NEE_WRF":
                df_loc_gee = df.filter(regex=f"^{location}_(.*_GPP_WRF)$")
                df_loc_res = df.filter(regex=f"^{location}_(.*_RECO_WRF)$")
                df_loc_gpp_pmodel = df.filter(regex=f"^{location}_(.*_GPP_Pmodel)$")
                df_loc_reco_migli = df.filter(regex=f"^{location}_(.*_RECO_Migli)$")
                df_loc = df_loc_res.add(df_loc_gee, fill_value=0)
                df_loc = df_loc.add(df_loc_gpp_pmodel, fill_value=0)
                df_loc = df_loc.add(df_loc_reco_migli, fill_value=0)

                nee_columns = {}
                nee_columns_P = {}
                for col in df_loc.columns:
                    if "_GPP_WRF" in col:
                        prefix = col.replace("_GPP_WRF", "")
                        if f"{prefix}_RECO_WRF" in df_loc.columns:
                            nee_columns[f"{prefix}_NEE_WRF"] = (
                                f"{prefix}_RECO_WRF",
                                f"{prefix}_GPP_WRF",
                            )
                for col in df_loc.columns:
                    if "_GPP_Pmodel" in col:
                        prefix = col.replace("_GPP_Pmodel", "")
                        if f"{prefix}_RECO_Migli" in df_loc.columns:
                            nee_columns_P[f"{prefix}_NEE_PMigli"] = (
                                f"{prefix}_RECO_Migli",
                                f"{prefix}_GPP_Pmodel",
                            )

                for nee_col, (res_col, gee_col) in nee_columns.items():
                    df_loc[nee_col] = df_loc[res_col] + df_loc[gee_col]
                df_loc = df_loc.drop(
                    columns=df_loc.filter(regex="(_GPP_WRF|_RECO_WRF)$").columns
                )

                for nee_col_P, (reco_col, gpp_col) in nee_columns_P.items():
                    df_loc[nee_col_P] = df_loc[reco_col] - df_loc[gpp_col]
                df_loc = df_loc.drop(
                    columns=df_loc.filter(regex="(_GPP_Pmodel|_RECO_Migli)$").columns
                )

            elif fluxtype == "T2_WRF":
                df_loc = df.filter(regex=f"^{location}_REF(.*_{fluxtype})$")
                df_loc = df_loc - 273.15
            elif fluxtype == "GPP_WRF":
                df_loc = -df.filter(regex=f"^{location}_(.*_{fluxtype})$")
                df_loc_pmodel = df.filter(regex=f"^{location}_(.*ALPS_GPP_Pmodel)$")
                df_loc_pmodel = df_loc_pmodel.fillna(0)
                df_loc = df_loc.add(df_loc_pmodel, fill_value=0)
            elif fluxtype == "RECO_WRF":
                df_loc = df.filter(regex=f"^{location}_(.*_{fluxtype})$")
                df_loc_migli = df.filter(regex=f"^{location}_(.*ALPS_RECO_Migli)$")
                df_loc = df_loc.add(df_loc_migli, fill_value=0)
            else:
                df_loc = pd.DataFrame()

            print(f"{resolution}: {df_loc.columns.tolist()}")

            for column in df_loc.columns:
                variable = (
                    column.split("_")[-3]
                    + "_"
                    + column.split("_")[-2]
                    + "_"
                    + column.split("_")[-1]
                )
                if variable in variables:
                    df_loc["hour"] = df_loc.index.hour

                    # Ensure TIMESTAMP_START in df_FLX_site is datetime and set as index
                    if "TIMESTAMP_START" in df_FLX_site.columns:
                        df_FLX_site["TIMESTAMP_START"] = pd.to_datetime(
                            df_FLX_site["TIMESTAMP_START"]
                        ).dt.tz_localize(None)
                        df_FLX_site = df_FLX_site.set_index("TIMESTAMP_START")

                    df_FLX_site = df_FLX_site[df_FLX_site.index.isin(df_loc.index)]
                    df_FLX_site = df_FLX_site.loc[df_loc.index]

                    hourly_avg = df_loc.groupby("hour")[column].mean()

                    # mask out nan values and build obs/mod arrays
                    if df_loc[column].isna().any():
                        mask = ~df_loc[column].isna()
                        obs = df_FLX_site[var_flx][mask.tolist()].to_numpy()
                        mod = df_loc[column][mask].to_numpy()
                    else:
                        obs = df_FLX_site[var_flx].to_numpy()
                        mod = df_loc[column].to_numpy()

                    # Build evaluator from obs/mod
                    evaluator = RegressionMetric(obs.tolist(), mod.tolist())

                    # Retrieve standard metrics and add MAE
                    metrics = evaluator.get_metrics_by_list_names(["MAE", "RMSE", "R2"])

                    # Explicit bias/amplitude metrics
                    diff = mod - obs
                    metrics["MB"] = np.nanmean(diff)
                    metrics["NMB"] = (
                        np.nansum(diff) / np.nansum(obs)
                        if np.nansum(obs) != 0
                        else np.nan
                    )
                    metrics["STD_RATIO"] = (
                        np.nanstd(mod) / np.nanstd(obs)
                        if np.nanstd(obs) != 0
                        else np.nan
                    )

                    for metric, value in metrics.items():
                        consolidated_metrics_df.loc[
                            metric, column + "_" + resolution
                        ] = value

                    linestyle = variables[variable]
                    par_plot_label = WRF_plot_labels[variable]
                    if "T2" in variable:
                        par_plot_label = r"T$_\text{2m}$"
                        linestyle = "solid"
                    color_i = resolution_colors[resolution]

                    if (
                        variable == "std_GPP_Pmodel"
                        or variable == "std_RECO_Migli"
                        or variable == "std_NEE_PMigli"
                    ):
                        color_i = "purple"

                    plt.plot(
                        hourly_avg,
                        label=rf"WRF {par_plot_label} - MB={metrics['MB']:.2f} MAE={metrics['MAE']:.2f}",
                        linestyle=linestyle,
                        color=color_i,
                    )

        # CAMS plotting and metrics
        if location in df_CAMS_hourly_all:
            df_CAMS_hourly = df_CAMS_hourly_all[location]
            df_CAMS_hourly["hour"] = df_CAMS_hourly.index.hour

            # Ensure timezone-naive indices for merging
            df_FLX_site.index = df_FLX_site.index.tz_localize(None)
            df_CAMS_hourly.index = df_CAMS_hourly.index.tz_localize(None)

            df_merged = pd.merge(
                df_FLX_site,
                df_CAMS_hourly,
                left_index=True,
                right_index=True,
                suffixes=("_FLX", "_CAMS"),
            )
            cols_to_fix = [
                "fco2gpp",
                "fco2gpp_bfas",
                "fco2rec",
                "fco2rec_bfas",
                "fco2nee",
                "fco2nee_bfas",
                "t2m",
            ]
            df_merged[cols_to_fix] = df_merged[cols_to_fix].apply(
                pd.to_numeric, errors="coerce"
            )
            df_merged[cols_to_fix] = df_merged[cols_to_fix].interpolate(method="linear")

            cams_l = df_merged[var_CAMS_plot]
            # build obs/mod arrays for CAMS comparison
            obs_cams = df_FLX_site[var_flx].to_numpy()
            mod_cams = cams_l.to_numpy()
            evaluator = RegressionMetric(obs_cams.tolist(), mod_cams.tolist())
            metrics = evaluator.get_metrics_by_list_names(["MAE", "R2"])
            diff_cams = mod_cams - obs_cams
            metrics["MB"] = np.nanmean(diff_cams)
            metrics["NMB"] = (
                np.nansum(diff_cams) / np.nansum(obs_cams)
                if np.nansum(obs_cams) != 0
                else np.nan
            )
            metrics["STD_RATIO"] = (
                np.nanstd(mod_cams) / np.nanstd(obs_cams)
                if np.nanstd(obs_cams) != 0
                else np.nan
            )
            for metric, value in metrics.items():
                consolidated_metrics_df.loc[
                    metric, var_CAMS_plot + "_" + resolution
                ] = value

            if var_CAMS_plot == "t2m":
                hourly_avg_CAMS = df_CAMS_hourly.groupby("hour")[var_CAMS_plot].mean()
                r_text = r"T$_\text{2m}$"
                plt.plot(
                    hourly_avg_CAMS,
                    label=rf"CAMS {r_text} - MB={metrics['MB']:.2f} MAE={metrics['MAE']:.2f} ",
                    color="orange",
                )
            else:
                hourly_avg_CAMS = df_CAMS_hourly.groupby("hour")[var_CAMS_plot].mean()
                if var_CAMS_plot == "fco2gpp":
                    r_text = "GPP"
                elif var_CAMS_plot == "fco2rec":
                    r_text = r"R$_\text{eco}$"
                elif var_CAMS_plot == "fco2nee":
                    r_text = "NEE"
                plt.plot(
                    hourly_avg_CAMS,
                    label=rf"CAMS {r_text} - MB={metrics['MB']:.2f} MAE={metrics['MAE']:.2f}",
                    color="orange",
                    linestyle="dashed",
                )

                cams_bfas = df_merged[var_CAMS_plot + "_bfas"]
                obs_bfas = df_FLX_site[var_flx].to_numpy()
                mod_bfas = cams_bfas.to_numpy()
                evaluator_bfas = RegressionMetric(obs_bfas.tolist(), mod_bfas.tolist())
                metrics_bfas = evaluator_bfas.get_metrics_by_list_names(["MAE", "R2"])
                diff_bfas = mod_bfas - obs_bfas
                metrics_bfas["MB"] = np.nanmean(diff_bfas)
                metrics_bfas["NMB"] = (
                    np.nansum(diff_bfas) / np.nansum(obs_bfas)
                    if np.nansum(obs_bfas) != 0
                    else np.nan
                )
                metrics_bfas["STD_RATIO"] = (
                    np.nanstd(mod_bfas) / np.nanstd(obs_bfas)
                    if np.nanstd(obs_bfas) != 0
                    else np.nan
                )
                hourly_avg_CAMS_bfas = df_CAMS_hourly.groupby("hour")[
                    var_CAMS_plot + "_bfas"
                ].mean()
                plt.plot(
                    hourly_avg_CAMS_bfas,
                    label=rf"CAMS  {r_text} (BFAS) - MB={metrics_bfas['MB']:.2f} MAE={metrics_bfas['MAE']:.2f}",
                    color="orange",
                )

        # FLUXNET hourly average plotting
        df_FLX_site["hour"] = df_FLX_site.index.hour
        hourly_avg_FLX = df_FLX_site.groupby("hour")[var_flx].mean()
        if var_flx == "TA_F":
            r_text = r"T$_\text{2m}$"
        elif var_flx == "GPP_NT_VUT_USTAR50":
            r_text = "GPP"
        elif var_flx == "RECO_NT_VUT_USTAR50":
            r_text = r"R$_\text{eco}$"
        elif var_flx == "NEE_VUT_USTAR50":
            r_text = "NEE"
        plt.plot(
            hourly_avg_FLX,
            label=rf"FLUXNET {r_text}",
            linestyle="solid",
            color="black",
        )

        plt.xlabel("UTC [h]", fontsize=20)
        if unit == "[°C]":
            plt.ylabel(rf"{r_text} [°C]", fontsize=20)
        else:
            plt.ylabel(rf"{r_text} [$\mu$mol m⁻² s⁻¹]", fontsize=20)
        plt.tick_params(labelsize=20)
        plt.legend(loc="upper left", fontsize=17, frameon=True, framealpha=0.6)

        plt.xticks([0, 6, 12, 18, 24])
        plt.grid()
        plt.tight_layout()

        # Save
        save_sim_type = sim_type if sim_type != "" else "_clear_sky"
        plt.savefig(
            f"{outfolder}/{location}_{fluxtype}_comparison_hourly_{resolution}{save_sim_type}_{timespan}.pdf",
            bbox_inches="tight",
        )
        plt.close()

        consolidated_metrics_all = pd.concat(
            [consolidated_metrics_all, consolidated_metrics_df], axis=1
        )

        print(
            f"finished: {location}_{fluxtype}_comparison_hourly_{resolution}_{timespan}"
        )

    return consolidated_metrics_all


def write_latex_table_from_metrics(
    consolidated_metrics_total,
    model_lat_lon,
    outfile="flux_evaluation_1km.tex",
):
    """
    Write LaTeX table from consolidated_metrics_total and append
    a row with column-wise averages across sites.
            "\\hline\\hline\n"
    Assumes consolidated_metrics_total:
    - rows: ["MAE", "R2"]
    - columns: <SITE>_<PARAM>_<FLUX>_<RES>
    """

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    vegfrac_lookup = {}
    for d in model_lat_lon:
        site = d["name"].split("_")[0]
        vegfrac_lookup[site] = int(round(d["veg_frac"] * 100))

    # ------------------------------------------------------------------
    # Site and PFT-type definition
    # ------------------------------------------------------------------
    sites = [
        ("AT-Neu", "GRA"),
        ("CH-Dav", "ENF"),
        ("IT-Lav", "ENF"),
        ("IT-MBo", "GRA"),
        ("IT-Ren", "ENF"),
    ]

    fluxes = {
        "NEE_WRF": "NEE",
        "GPP_WRF": "GPP",
        "RECO_WRF": r"R$_{eco}$",
    }

    params = ["SITE", "ALPS", "REF"]

    def fmt(val, bold=False):
        s = f"{val:.2f}"
        return f"\\textbf{{{s}}}" if bold else s

    # ------------------------------------------------------------------
    # Accumulators for mean row (include MAE)
    # ------------------------------------------------------------------
    t2m_rmse_all = []
    t2m_r2_all = []
    t2m_mae_all = []

    flux_rmse_all = {flux: {p: [] for p in params} for flux in fluxes}
    flux_r2_all = {flux: {p: [] for p in params} for flux in fluxes}
    flux_mae_all = {flux: {p: [] for p in params} for flux in fluxes}

    # ------------------------------------------------------------------
    # Write table
    # ------------------------------------------------------------------
    with open(outfile, "w") as f:
        f.write(
            "\\begin{tabular}{ll|c|ccc|ccc|ccc}\n"
            "\\hline\n"
            "Site & PFT fraction & $T_\\text{2m}$ "
            "& NEE SITE & NEE ALPS & NEE REF "
            "& GPP SITE & GPP ALPS & GPP REF "
            "& RECO SITE & RECO ALPS & RECO REF \\\\\n"
            "\\hline\\hline\n"
        )

        # --------------------------------------------------------------
        # Per-site rows
        # --------------------------------------------------------------
        for site, pft_type in sites:
            veg_pct = vegfrac_lookup[site]
            pft_str = f"{veg_pct}\\% {pft_type}"

            # --- T2m (REF only)
            col_t2m = f"{site}_REF_T2_WRF_1km"
            t2m_rmse = consolidated_metrics_total.loc["RMSE", col_t2m]
            t2m_r2 = consolidated_metrics_total.loc["R2", col_t2m]

            t2m_rmse_all.append(t2m_rmse)
            t2m_r2_all.append(t2m_r2)

            # display RMSE (R2)
            row = f"{site} & {pft_str} & {t2m_rmse:.2f} ({t2m_r2:.2f})"

            # --- Fluxes
            for flux in fluxes:
                rmse_vals = []
                r2_vals = []

                for p in params:
                    col = f"{site}_{p}_{flux}_1km"
                    rm = consolidated_metrics_total.loc["RMSE", col]
                    r2v = consolidated_metrics_total.loc["R2", col]

                    rmse_vals.append(rm)
                    r2_vals.append(r2v)

                    flux_rmse_all[flux][p].append(rm)
                    flux_r2_all[flux][p].append(r2v)

                rmse_min = min(rmse_vals)
                r2_max = max(r2_vals)

                for i, p in enumerate(params):
                    rm = rmse_vals[i]
                    r2v = r2_vals[i]
                    rm_s = fmt(rm, rm == rmse_min)
                    r2_s = fmt(r2v, r2v == r2_max)
                    # show RMSE (R2) only
                    row += f" & {rm_s} ({r2_s})"

            f.write(row + " \\\\\n")
        f.write("\hline \n")

        # --------------------------------------------------------------
        # Mean row (across sites)
        # --------------------------------------------------------------
        # --- Determine extrema of mean values (for bolding)
        t2m_rmse_mean = np.mean(t2m_rmse_all)
        t2m_r2_mean = np.mean(t2m_r2_all)  # single column → no comparison

        flux_rmse_mean = {
            flux: {p: np.mean(flux_rmse_all[flux][p]) for p in params}
            for flux in fluxes
        }
        flux_r2_mean = {
            flux: {p: np.mean(flux_r2_all[flux][p]) for p in params} for flux in fluxes
        }

        row = "Mean & -- " f"& {t2m_rmse_mean:.2f} ({t2m_r2_mean:.2f})"

        for flux in fluxes:
            rmse_min = min(flux_rmse_mean[flux].values())
            r2_max = max(flux_r2_mean[flux].values())

            for p in params:
                rm = flux_rmse_mean[flux][p]
                r2v = flux_r2_mean[flux][p]
                rm_s = fmt(rm, rm == rmse_min)
                r2_s = fmt(r2v, r2v == r2_max)
                row += f" & {rm_s} ({r2_s})"

        f.write(row + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    print(f"LaTeX table written to {outfile}")


def compute_fluxnet_site_means(
    locations, start_date_2012, end_date_2012, base_dir_FLX, df_example
):
    """
    Compute mean values for FLUXNET variables per location.

    Returns DataFrame with columns for yearly means and _WRFmatch means
    (filtered to only timestamps in df_example). Includes a 'Mean' row with averages across sites.
    """
    var_flx_list = [
        "TA_F",
        "SW_IN_F",
        "GPP_NT_VUT_USTAR50",
        "RECO_NT_VUT_USTAR50",
        "NEE_VUT_USTAR50",
    ]
    # Create columns for both yearly and WRF-matched values
    all_columns = var_flx_list + [f"{var}_WRFmatch" for var in var_flx_list]
    df_FLX_site_means = pd.DataFrame(
        index=list(locations) + ["Mean"], columns=all_columns
    )

    for location in locations:
        for var_flx in var_flx_list:
            df_FLX_site = read_FLUXNET_site(
                start_date_2012, end_date_2012, location, base_dir_FLX, var_flx
            )
            if not df_FLX_site.empty and var_flx in df_FLX_site.columns:
                df_FLX_site_means.loc[location, var_flx] = df_FLX_site[var_flx].mean()
            else:
                df_FLX_site_means.loc[location, var_flx] = np.nan

            if not df_FLX_site.empty and var_flx in df_FLX_site.columns:
                # Set TIMESTAMP_START as index and ensure timezone-naive for comparison
                df_FLX_site.set_index("TIMESTAMP_START", inplace=True)
                df_FLX_site.index = df_FLX_site.index.tz_localize(None)
                # Filter to only timesteps that are in df_example (also needs to be timezone-naive)
                df_example_index_naive = df_example.index.tz_localize(None)
                df_FLX_site_filtered = df_FLX_site[
                    df_FLX_site.index.isin(df_example_index_naive)
                ]
                if not df_FLX_site_filtered.empty:
                    df_FLX_site_means.loc[location, f"{var_flx}_WRFmatch"] = (
                        df_FLX_site_filtered[var_flx].mean()
                    )
                else:
                    df_FLX_site_means.loc[location, f"{var_flx}_WRFmatch"] = np.nan
            else:
                df_FLX_site_means.loc[location, f"{var_flx}_WRFmatch"] = np.nan

    # Compute mean row across all sites (only for the locations, not including Mean row itself)
    locations_list = [loc for loc in df_FLX_site_means.index if loc != "Mean"]
    for col in all_columns:
        df_FLX_site_means.loc["Mean", col] = (
            df_FLX_site_means.loc[locations_list, col].astype(float).mean()
        )

    return df_FLX_site_means


def write_latex_table_fluxnet_means(
    df_FLX_site_means, outfile="fluxnet_site_means.tex"
):
    """
    Write LaTeX table with FLUXNET site mean values.

    Format: yearly_mean [WRFmatch_mean]
    """
    var_flx_list = [
        "TA_F",
        "SW_IN_F",
        "GPP_NT_VUT_USTAR50",
        "RECO_NT_VUT_USTAR50",
        "NEE_VUT_USTAR50",
    ]
    var_names = [
        r"T$_\text{2m}$ [°C]",
        r"SW$_\text{in}$ [W m$^{-2}$]",
        r"GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
        r"R$_\text{eco}$ [$\mu$mol m$^{-2}$ s$^{-1}$]",
        r"NEE [$\mu$mol m$^{-2}$ s$^{-1}$]",
    ]

    with open(outfile, "w") as f:
        f.write(
            "\\begin{tabular}{l|" + "c" * len(var_names) + "}\n"
            "\\hline\n"
            "Site & " + " & ".join(var_names) + " \\\\\n"
            "\\hline\\hline\n"
        )

        for site in df_FLX_site_means.index:
            if site == "Mean":
                f.write("\\hline\n")  # Add horizontal line before mean row

            row = f"{site}"

            # Add variable values
            for var_flx in var_flx_list:
                yearly_val = df_FLX_site_means.loc[site, var_flx]
                wrf_match_val = df_FLX_site_means.loc[site, f"{var_flx}_WRFmatch"]

                if pd.isna(yearly_val):
                    cell_str = "--"
                else:
                    cell_str = f"{yearly_val:.2f}"
                    if not pd.isna(wrf_match_val):
                        cell_str += f" ({wrf_match_val:.2f})"

                row += f" & {cell_str}"

            f.write(row + " \\\\\n")

        f.write("\\hline\n\\end{tabular}\n")

    print(f"LaTeX FLUXNET means table written to {outfile}")


def write_latex_table_from_metrics_bias(
    consolidated_metrics_total,
    model_lat_lon,
    outfile="flux_evaluation_1km_bias.tex",
):
    """
    Write LaTeX table summarising bias metrics per site.

    Each cell formatted as: MB (MAE)
    """

    vegfrac_lookup = {}
    for d in model_lat_lon:
        site = d["name"].split("_")[0]
        vegfrac_lookup[site] = int(round(d["veg_frac"] * 100))

    sites = [
        ("AT-Neu", "GRA"),
        ("CH-Dav", "ENF"),
        ("IT-Lav", "ENF"),
        ("IT-MBo", "GRA"),
        ("IT-Ren", "ENF"),
    ]

    fluxes = {
        "NEE_WRF": "NEE",
        "GPP_WRF": "GPP",
        "RECO_WRF": r"R$_{eco}$",
    }

    params = ["SITE", "ALPS", "REF"]

    def fmt(val, bold=False):
        s = f"{val:.2f}"
        return f"\\textbf{{{s}}}" if bold else s

    # Build bias table: show NMB (STD_RATIO). Also compute mean row and bold best values
    with open(outfile, "w") as f:
        f.write(
            "\\begin{tabular}{ll|c|ccc|ccc|ccc}\n"
            "\\hline\n"
            "Site & PFT fraction & $T_\\text{2m}$ "
            "& NEE SITE & NEE ALPS & NEE REF "
            "& GPP SITE & GPP ALPS & GPP REF "
            "& RECO SITE & RECO ALPS & RECO REF \\\\\n"
            "\\hline\\hline\n"
        )

        # accumulators to compute mean across sites (store MB, NMB, STD)
        mean_vals = {"T2": {"MB": [], "MAE": []}}
        for flux in fluxes:
            mean_vals[flux] = {p: {"MB": [], "MAE": []} for p in params}

        # per-site rows
        for site, pft_type in sites:
            veg_pct = vegfrac_lookup[site]
            pft_str = f"{veg_pct}\\% {pft_type}"

            # T2m: include MB, NMB and STD_RATIO
            col_t2m = f"{site}_REF_T2_WRF_1km"
            mb_t2m = consolidated_metrics_total.loc["MB", col_t2m]
            nmb_t2m = consolidated_metrics_total.loc["MAE", col_t2m]
            mean_vals["T2"]["MB"].append(mb_t2m)
            mean_vals["T2"]["MAE"].append(nmb_t2m)

            row = f"{site} & {pft_str} & {mb_t2m:.2f} ({nmb_t2m:.2f})"

            # For each flux, collect values for this site, determine best among params and format
            for flux in fluxes:
                mb_list = []
                mae_list = []
                for p in params:
                    col = f"{site}_{p}_{flux}_1km"
                    mb = consolidated_metrics_total.loc["MB", col]
                    mae = consolidated_metrics_total.loc["MAE", col]
                    mb_list.append(mb)
                    mae_list.append(mae)
                    # store for mean computation
                    mean_vals[flux][p]["MB"].append(mb)
                    mean_vals[flux][p]["MAE"].append(mae)

                # determine best indices: MAE closest to zero
                mb_best_idx = (
                    int(np.nanargmin(np.abs(np.array(mb_list))))
                    if len(mb_list) > 0
                    else 0
                )
                mae_best_idx = (
                    int(np.nanargmin(np.abs(np.array(mae_list))))
                    if len(mae_list) > 0
                    else 0
                )

                # format each param value: MB (NMB) [STD], bold NMB/STD if best
                for i, (mb_val, mae_val) in enumerate(zip(mb_list, mae_list)):
                    mb_s = fmt(mb_val, bold=(i == mb_best_idx))
                    mae_s = fmt(mae_val, bold=(i == mae_best_idx))
                    row += f" & {mb_s} ({mae_s})"

            f.write(row + " \\\\\n")

        # Mean row across sites
        f.write("\\hline\n")
        mean_row = "Mean & -- "
        # mean for T2m (MB (MAE))
        mean_mb_t2 = (
            np.mean(mean_vals["T2"]["MB"]) if len(mean_vals["T2"]["MB"]) > 0 else np.nan
        )
        mean_mae_t2 = (
            np.mean(mean_vals["T2"]["MAE"])
            if len(mean_vals["T2"]["MAE"]) > 0
            else np.nan
        )

        mean_row += f"& {mean_mb_t2:.2f} ({mean_mae_t2:.2f})"

        # for each flux compute mean per param and bold best among params (NMB closest to zero, STD closest to 1)
        for flux in fluxes:
            # compute mean arrays per param
            mb_means = [np.mean(mean_vals[flux][p]["MB"]) for p in params]
            mae_means = [np.mean(mean_vals[flux][p]["MAE"]) for p in params]
            # best indices
            mb_best_idx = int(np.nanargmin(np.abs(np.array(mb_means))))
            mae_best_idx = int(np.nanargmin(np.abs(np.array(mae_means))))
            # format
            for i, p in enumerate(params):
                mb_s = fmt(mb_means[i], bold=(i == mb_best_idx))
                mae_s = fmt(mae_means[i], bold=(i == mae_best_idx))
                mean_row += f" & {mb_s} ({mae_s})"

        f.write(mean_row + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    print(f"LaTeX bias table written to {outfile}")


def main():
    # Parameters copied from original script
    timespan = "2012-01-01 00:00:00_2012-12-31 00:00:00"
    sim_type = "_all"
    radius = 30
    csv_dir = "/scratch/c7071034/DATA/WRFOUT/csv"
    outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots"
    base_dir_FLX = "/scratch/c7071034/DATA/Fluxnet2015/Alps"
    fluxtypes = ["T2_WRF", "NEE_WRF", "GPP_WRF", "RECO_WRF"]
    plot_labels = [r"T$_\text{2m}$", "NEE", "GPP", r"R$_\text{eco}$"]
    units = [
        "[°C]",
        r"[$\mu$mol m$^{-2}$ s$^{-1}$]",
        r"[$\mu$mol m$^{-2}$ s$^{-1}$]",
        r"[$\mu$mol m$^{-2}$ s$^{-1}$]",
    ]
    res_dx = "1km"

    # Find CSV files
    if sim_type == "_all":
        csv_files = glob.glob(
            f"{csv_dir}/wrf_FLUXNET_sites_{res_dx}*_{timespan}_r{radius}.csv"
        )
    else:
        csv_files = glob.glob(
            f"{csv_dir}/wrf_FLUXNET_sites_{res_dx}{sim_type}_{timespan}_r{radius}.csv"
        )

    csv_files_sorted = sorted(
        csv_files,
        key=lambda x: int(x.split(f"_sites_")[1].split("km")[0]),
        reverse=True,
    )

    # read CSVs into dict
    dataframes = {}
    for csv_file in csv_files_sorted:
        resolution = res_dx
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        dataframes[resolution] = df

    # use last loaded df to determine date range
    df_example = df
    start_date = df_example.index[0]
    end_date = df_example.index[-1]

    # locations (kept same logic)
    if res_dx == "1km":
        locations = ["CH-Dav", "IT-Lav", "IT-Ren", "AT-Neu", "IT-MBo"]
        locations_hgt = {
            "CH-Dav": 1639,
            "IT-Lav": 1353,
            "IT-Ren": 1730,
            "AT-Neu": 970,
            "IT-MBo": 1550,
        }
    else:
        locations = [
            "IT-Tor",
            "IT-Lav",
            "AT-Neu",
            "CH-Cha",
            "CH-Dav",
            "CH-Fru",
            "CH-Lae",
            "CH-Oe1",
            "CH-Oe2",
            "DE-Lkb",
            "IT-Isp",
            "IT-La2",
            "IT-MBo",
            "IT-PT1",
            "IT-Ren",
        ]
        locations_hgt = {
            "AT-Neu": 970,
            "CH-Cha": 393,
            "CH-Dav": 1639,
            "CH-Fru": 982,
            "CH-Lae": 689,
            "CH-Oe1": 450,
            "CH-Oe2": 452,
            "DE-Lkb": 1308,
            "IT-Isp": 210,
            "IT-La2": 1350,
            "IT-Lav": 1353,
            "IT-MBo": 1550,
            "IT-PT1": 60,
            "IT-Ren": 1730,
            "IT-Tor": 2160,
        }

    locations_d03 = [
        "IT-Tor",
        "IT-Lav",
        "AT-Neu",
        "CH-Cha",
        "CH-Dav",
        "CH-Fru",
        "CH-Lae",
        "CH-Oe1",
        "CH-Oe2",
        "DE-Lkb",
        "IT-Isp",
        "IT-La2",
        "IT-MBo",
        "IT-PT1",
        "IT-Ren",
        "DE-Hai",
        "DE-Kli",
        "FR-Fon",
        "FR-Gri",
    ]

    # Get average values of each location for the entire year and WRF-matched timesteps
    start_date_2012 = pd.Timestamp(timespan.split("_")[0]) - pd.Timedelta(hours=1)
    end_date_2012 = pd.Timestamp(timespan.split("_")[1]) - pd.Timedelta(hours=1)

    df_FLX_site_means = compute_fluxnet_site_means(
        locations_d03, start_date_2012, end_date_2012, base_dir_FLX, df_example
    )
    # Write FLUXNET site means table
    write_latex_table_fluxnet_means(
        df_FLX_site_means,
        outfile=f"{outfolder}/fluxnet_site_means_r{radius}.tex",
    )

    # pft_site_match loading (kept identical)
    if sim_type == "_all":
        pft_site_match = pd.read_csv(
            f"{csv_dir}/distances_{res_dx}_{timespan}_r{radius}.csv"
        )
    else:
        pft_site_match = pd.read_csv(
            f"{csv_dir}/distances_{res_dx}{sim_type}_{timespan}_r{radius}.csv"
        )

    model_lat_lon = []
    for i in range(len(locations)):
        model_lat_lon.append(
            {
                "name": pft_site_match["name"][i],
                "lat": pft_site_match["lat_wrf"][i],
                "lon": pft_site_match["lon_wrf"][i],
                "veg_frac": pft_site_match["veg_frac_idx"][i],
            }
        )

    # load CAMS data once
    CAMS_data_dir_path = "/scratch/c7071034/DATA/CAMS/"
    path_CAMS_file = os.path.join(
        CAMS_data_dir_path + "ghg-reanalysis_surface_2012_full.nc"
    )
    CAMS_data, times_CAMS, gppbfas, rec_bfas, lat_CAMS, lon_CAMS = load_CAMS(
        path_CAMS_file
    )

    # Build CAMS hourly series per location
    df_CAMS_hourly_all = {}
    CAMS_vars = ["fco2gpp", "fco2rec", "fco2nee", "t2m"]
    factor_kgC = 1000000 / 0.04401
    factors = [factor_kgC, -factor_kgC, -factor_kgC, 1]

    for location_ll in model_lat_lon:
        lat_target, lon_target = location_ll["lat"], location_ll["lon"]
        data_rows = []
        j = 0
        for time_CAMS in times_CAMS:
            date_CAMS = datetime(1970, 1, 1) + timedelta(seconds=int(time_CAMS))
            formatted_time = date_CAMS.strftime("%Y-%m-%d %H:%M:%S")
            row = {"time": formatted_time}
            for CAMS_var, factor in zip(CAMS_vars, factors):
                var_CAMS = CAMS_data.variables[CAMS_var][j, :, :].data * factor
                row[CAMS_var] = get_int_var(
                    lat_target, lon_target, lat_CAMS, lon_CAMS, var_CAMS
                )
                if CAMS_var == "fco2gpp":
                    var_CAMS_b = var_CAMS * gppbfas[j, :, :].data
                    row["fco2gpp_bfas"] = get_int_var(
                        lat_target, lon_target, lat_CAMS, lon_CAMS, var_CAMS_b
                    )
                if CAMS_var == "fco2rec":
                    var_CAMS_b = var_CAMS * rec_bfas[j, :, :].data
                    row["fco2rec_bfas"] = get_int_var(
                        lat_target, lon_target, lat_CAMS, lon_CAMS, var_CAMS_b
                    )
                if CAMS_var == "fco2nee":
                    row["fco2nee_bfas"] = row.get("fco2nee")
                    row["fco2nee"] = row.get("fco2rec") - row.get("fco2gpp")
            j += 1
            data_rows.append(row)

        df_CAMS = pd.DataFrame(data_rows)
        df_CAMS["time"] = pd.to_datetime(df_CAMS["time"])
        df_CAMS.set_index("time", inplace=True)
        df_CAMS = df_CAMS.astype(float)
        df_CAMS = df_CAMS[~df_CAMS.index.duplicated(keep="first")]
        df_CAMS_hourly = df_CAMS.resample("h").interpolate(method="linear")
        df_CAMS_hourly["t2m"] -= 273.15
        colname = location_ll["name"].split("_")[:1]
        df_CAMS_hourly_all[colname[0]] = df_CAMS_hourly

    # resolution colors
    resolution_colors = {
        "1km": "blue",
        "3km": "darkgrey",
        "9km": "purple",
        "27km": "red",
        "54km": "green",
    }

    # Run processing for each location and build consolidated metrics
    consolidated_metrics_total = pd.DataFrame()
    for location in locations:
        consolidated_metrics_all = process_location(
            location,
            fluxtypes,
            plot_labels,
            units,
            dataframes,
            start_date,
            end_date,
            base_dir_FLX,
            df_CAMS_hourly_all,
            resolution_colors,
            sim_type,
            outfolder,
            timespan,
        )
        consolidated_metrics_total = pd.concat(
            [consolidated_metrics_total, consolidated_metrics_all], axis=1
        )

    write_latex_table_from_metrics(
        consolidated_metrics_total,
        model_lat_lon,
        outfile=f"{outfolder}/flux_evaluation_1km_r{radius}.tex",
    )
    # Also write bias table (MB (NMB) [STD_RATIO])
    write_latex_table_from_metrics_bias(
        consolidated_metrics_total,
        model_lat_lon,
        outfile=f"{outfolder}/flux_evaluation_1km_bias_r{radius}.tex",
    )

    print("All done.")


if __name__ == "__main__":
    main()
