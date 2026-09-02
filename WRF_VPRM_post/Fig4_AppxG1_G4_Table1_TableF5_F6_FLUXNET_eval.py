"""
Fig4_AppxG1_G4_Table1_TableF5_F6_FLUXNET_eval.py
==============================================
Fig. 4 (fig:IT-Mbo_interp, IT-MBo site) and Figs. G1-G4 (fig:AT-Neu_interp,
fig:IT-Ren_interp, fig:CH-Dav_interp, fig:IT-Lav_interp, the other 4 sites),
plus Table 1 (tab:filtered_sites_alps_bias), Table F6
(tab:filtered_sites_alps_rmse) and Table F5 (tab:filtered_sites_alps_means).
Refactored from `plot_FLX_ts_1km.py` with minimal changes to behaviour.
- Organised into functions: main(), load_csvs(), load_cams(), process_location()
- Removed stray tokens and small cleanup only.
- Keeps plotting and metric calculations unchanged.

you have to download the FLUXNET2015 data for the following sites to the folder:
"$SCRATCH_PATH"/DATA/Fluxnet2015/Alps

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

if you set fluxnet_site_means = True in main() you can generate Table D7
==============================================
"""

from __future__ import annotations

import glob
import os
import math
from decimal import Decimal, ROUND_HALF_UP
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path


def fmt2(v, nd=2):
    """Format with round-half-AWAY-FROM-ZERO (not Python's half-to-even).

    -0.855 -> "-0.86". Decimal(str(...)) is deliberate: going through the repr
    avoids the binary-float artefact that would make Decimal(-0.855) give -0.85.
    """
    v = float(v)
    if math.isnan(v):
        return "--"
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(str(v)).quantize(q, rounding=ROUND_HALF_UP))

# Load environment variables from .env file
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from permetrics import RegressionMetric
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc

# ==================== Configuration ====================
SCRATCH_PATH = os.getenv("SCRATCH_PATH")
GITHUB_PATH = os.getenv("GITHUB_PATH")
OUTFOLDER = os.getenv("OUTFOLDER")
CSVFOLDER = os.getenv("CSVFOLDER")
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


# ---- Figure geometry ----
# Paper convention: draw at 2x the printed size and use 2x fonts, so text lands
# at ~8 pt on the page (body text is ~9.5 pt). The site/fluxtype panels are
# included at 0.23\linewidth and the legend strip at 0.30\linewidth
# (\linewidth = 500 pt = 6.944 in), hence 2 x 0.23 x 6.944 = 3.2 in wide.
# The four panels sit 4-across (0.23 each), so at 1.6 in printed they target ~7 pt
# rather than the 8 pt used by the wider figures -- enough for the long
# "GPP [umol m-2 s-1]" labels to still fit.
# Taller than the old 10:6 aspect on purpose: only the *width* sets the scale
# factor (LaTeX fits to width), so extra height is free and is what lets the long
# "GPP [umol m-2 s-1]" y label fit without being clipped.
PANEL_FIGSIZE = (3.2, 2.6)  # scale 0.51 -> 14 pt lands at ~7.1 pt printed
LEGEND_FIGSIZE = (4.2, 0.95)  # legend strip, included at 0.30\linewidth
FS_LABEL = 14  # -> ~7.1 pt printed
FS_TICK = 13  # -> ~6.6 pt printed
FS_LEGEND = 14  # legend strip is wider, so it keeps ~7 pt

# ---- Percentile / channel helpers (recompute-aware) ----
ALPS_TAGS = ["topt_p10", "topt_p25", "topt_p50", "topt_p75", "topt_p90"]
MEDIAN_TAG = "topt_p50"
PCTS = (10, 25, 50, 75, 90)


def _metrics_from(obs, mod):
    """RMSE/R2/MAE + bias/amplitude metrics from aligned obs/mod arrays."""
    evaluator = RegressionMetric(obs.tolist(), mod.tolist())
    m = evaluator.get_metrics_by_list_names(["MAE", "RMSE", "R2"])
    diff = mod - obs
    m["MB"] = np.nanmean(diff)
    m["NMB"] = np.nansum(diff) / np.nansum(obs) if np.nansum(obs) != 0 else np.nan
    m["STD_RATIO"] = (
        np.nanstd(mod) / np.nanstd(obs) if np.nanstd(obs) != 0 else np.nan
    )
    m["mean_fluxnet"] = np.nanmean(obs)
    return m


def _channel_series(df_src, location, channel, base):
    """Model time series for one channel/flux.

    ALPS/SITE come from the offline-recomputed columns (`*_recalc`, already in
    umol m-2 s-1 with the correct sign / NEE = RECO-GPP). REF (-> DEFAULT) is rebuilt
    from the online EBIO `*_WRF` columns (GPP stored negative). Returns a Series
    or None."""
    if base == "T2":
        col = f"{location}_{channel}_T2_WRF"
        return (df_src[col] - 273.15) if col in df_src.columns else None
    rcol = f"{location}_{channel}_{base}_recalc"
    if rcol in df_src.columns:
        return df_src[rcol]
    # online channel (REF): reconstruct from EBIO *_WRF columns
    if base == "GPP":
        c = f"{location}_{channel}_GPP_WRF"
        return -df_src[c] if c in df_src.columns else None
    if base == "RECO":
        c = f"{location}_{channel}_RECO_WRF"
        return df_src[c] if c in df_src.columns else None
    if base == "NEE":
        g, r = f"{location}_{channel}_GPP_WRF", f"{location}_{channel}_RECO_WRF"
        if g in df_src.columns and r in df_src.columns:
            return df_src[r] + df_src[g]  # EBIO sum = RECO - |GPP|
        return None
    return None


def load_site_csvs(tag, csv_folder, res_dx, sim_suffixes, timespan, radius):
    """Pooled (clear + cloudy, or whichever sim_suffixes are passed) site-level
    df for one Topt tag, at one resolution. Module-level (not a closure inside
    main()) so other scripts that need the same per-tag site csvs --
    e.g. mk_bias_percentile_sensitivity.py -- load them the same way rather
    than re-implementing the csv paths."""
    parts = []
    for s in sim_suffixes:
        f = (
            f"{csv_folder}/wrf_FLUXNET_sites_{res_dx}{s}_recalc_{tag}"
            f"_{timespan}_r{radius}.csv"
        )
        if os.path.exists(f):
            parts.append(pd.read_csv(f, index_col=0, parse_dates=True))
        else:
            print(f"  WARNING missing: {os.path.basename(f)}")
    return pd.concat(parts).sort_index()


def save_legend_only_flx(color, outfile):
    """Standalone legend for the site/fluxtype comparison figures, for placement
    below the figure grid in the manuscript. process_location() draws no
    in-panel legend; this is the only place these entries are documented.

    Covers the GPP/RECO/NEE channel set (SITE solid, ALPS dashdot + Topt
    percentile band, DEFAULT dashed) plus FLUXNET (black solid). T2's panels
    only ever show WRF (REF, solid) + FLUXNET, both already covered here
    (WRF/REF and ALPS share the same "model" role visually, but keep WRF
    explicit since T2 has no SITE/DEFAULT/Topt-band split).
    """
    handles = [
        Line2D([0], [0], color=color, lw=2, linestyle="-"),
        Line2D([0], [0], color=color, lw=2, linestyle="dashdot"),
        Line2D([0], [0], color=color, lw=2, linestyle="--"),
        Line2D([0], [0], color="black", lw=2, linestyle="-"),
        Patch(facecolor=color, alpha=0.35, linewidth=0),
        Patch(facecolor=color, alpha=0.15, linewidth=0),
    ]
    labels = [
        "SITE",
        "ALPS",
        "DEFAULT",
        "FLUXNET",
        r"ALPS p25-p75 (ens.)",
        r"ALPS p10-p90 (ens.)",
    ]

    # 3 columns -> 6 entries in 2 rows. The strip is wider than at ncol=2, so its
    # \includegraphics width in the manuscript has to grow to keep the printed
    # text at ~8 pt -- see FIGURE_TEX_CHANGES.md.
    fig = plt.figure(figsize=LEGEND_FIGSIZE)
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=3,
        fontsize=FS_LEGEND,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    plt.axis("off")
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    print(f"Legend written to {outfile}")


# ---- Main processing function (single location) ----


def process_location(
    location,
    fluxtypes,
    plot_labels,
    units,
    dataframes,
    ALPS_dfs,
    start_date,
    end_date,
    base_dir_FLX,
    df_CAMS_hourly_all,
    resolution_colors,
    sim_type,
    outfolder,
    timespan,
    resolution="1km",
    make_plots=True,
):
    """`make_plots=False` skips every plt.figure/plot/savefig call and returns
    only the metrics -- for reusing this against a non-p50 Topt tag without
    overwriting the p50 comparison PDFs, whose filenames don't encode the tag
    (see mk_bias_percentile_sensitivity.py). Metric computation itself is
    unconditional either way."""
    consolidated_metrics_all = pd.DataFrame()
    df0 = dataframes[resolution]  # representative df for this resolution/tag
    color_i = resolution_colors[resolution]

    for fluxtype, unit, plot_label in zip(fluxtypes, units, plot_labels):
        consolidated_metrics_df = pd.DataFrame(index=["MAE", "RMSE", "R2"])
        base = fluxtype.replace("_WRF", "")  # GPP / RECO / NEE / T2

        # choose var_flx and CAMS var
        if fluxtype == "RECO_WRF":
            var_flx, var_CAMS_plot, r_text = "RECO_NT_VUT_USTAR50", "fco2rec", r"R$_\text{eco}$"
        elif fluxtype == "GPP_WRF":
            var_flx, var_CAMS_plot, r_text = "GPP_NT_VUT_USTAR50", "fco2gpp", "GPP"
        elif fluxtype == "NEE_WRF":
            var_flx, var_CAMS_plot, r_text = "NEE_VUT_USTAR50", "fco2nee", "NEE"
        elif fluxtype == "T2_WRF":
            var_flx, var_CAMS_plot, r_text = "TA_F", "t2m", r"T$_\text{2m}$"
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
        if "TIMESTAMP_START" in df_FLX_site.columns:
            df_FLX_site["TIMESTAMP_START"] = pd.to_datetime(
                df_FLX_site["TIMESTAMP_START"]
            ).dt.tz_localize(None)
            df_FLX_site = df_FLX_site.set_index("TIMESTAMP_START")

        # Drawn at 2x the printed size (0.23\linewidth = 1.60in -> 3.2in) so the
        # 16/14 pt fonts below land at ~8/7 pt on the page. See PANEL_FIGSIZE.
        if make_plots:
            plt.figure(figsize=PANEL_FIGSIZE)

        def _align(ms):
            flx = df_FLX_site[df_FLX_site.index.isin(ms.index)]
            flx = flx.loc[ms.index]
            mask = ~ms.isna()
            return flx[var_flx][mask.values].to_numpy(), ms[mask].to_numpy()

        # channel -> (linestyle, legend label, draw_band)
        if fluxtype == "T2_WRF":
            channels = [("REF", "solid", "WRF", False)]
        else:
            channels = [
                ("SITE", "solid", "SITE", False),
                ("ALPS", "dashdot", "ALPS", True),
                ("REF", "dashed", "DEFAULT", False),
            ]

        for channel, linestyle, label, draw_band in channels:
            ms = _channel_series(df0, location, channel, base)
            if ms is None:
                continue
            obs, mod = _align(ms)
            for metric, value in _metrics_from(obs, mod).items():
                consolidated_metrics_df.loc[
                    metric, f"{location}_{channel}_{fluxtype}_{resolution}"
                ] = value

            if not make_plots:
                continue

            if draw_band:
                # per-hour percentile band across the Topt tags (metrics use p50)
                hcols = []
                for dft in ALPS_dfs:
                    mst = _channel_series(dft, location, channel, base)
                    if mst is not None:
                        hcols.append(mst.groupby(mst.index.hour).mean())
                M = pd.concat(hcols, axis=1)
                hrs = M.index.values
                P = {p: np.nanpercentile(M.values, p, axis=1) for p in PCTS}
                if np.nanmax(np.abs(P[90] - P[10])) > 1e-12:
                    plt.fill_between(hrs, P[10], P[90], color=color_i, alpha=0.15, linewidth=0)
                    plt.fill_between(hrs, P[25], P[75], color=color_i, alpha=0.35, linewidth=0)
                plt.plot(hrs, P[50], linestyle=linestyle, color=color_i, label=label)
            else:
                hourly_avg = ms.groupby(ms.index.hour).mean()
                plt.plot(
                    hourly_avg.index,
                    hourly_avg.values,
                    linestyle=linestyle,
                    color=color_i,
                    label=label,
                )

        # CAMS plotting and metrics
        if location in df_CAMS_hourly_all:
            df_CAMS_hourly = df_CAMS_hourly_all[location]
            df_CAMS_hourly["hour"] = df_CAMS_hourly.index.hour

            # Ensure timezone-naive indices for merging
            if df_FLX_site.index.tz is not None:
                df_FLX_site.index = df_FLX_site.index.tz_localize(None)
            if df_CAMS_hourly.index.tz is not None:
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
                    label="CAMS",
                    color="orange",
                    linestyle="dashed",
                )
            else:
                hourly_avg_CAMS = df_CAMS_hourly.groupby("hour")[var_CAMS_plot].mean()
                if var_CAMS_plot == "fco2gpp":
                    plt.plot(
                        hourly_avg_CAMS,
                        label=rf"CAMS",
                        color="orange",
                        linestyle="dashed",
                    )
                else:
                    plt.plot(
                        hourly_avg_CAMS,
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

                if var_CAMS_plot == "fco2gpp":
                    plt.plot(
                        hourly_avg_CAMS_bfas,
                        label=rf"CAMS (BFAS)",
                        color="orange",
                    )
                else:
                    plt.plot(
                        hourly_avg_CAMS_bfas,
                        color="orange",
                    )

        # FLUXNET hourly average plotting
        if make_plots:
            df_FLX_site["hour"] = df_FLX_site.index.hour
            hourly_avg_FLX = df_FLX_site.groupby("hour")[var_flx].mean()

            if var_flx == "TA_F":
                r_text = r"T$_\text{2m}$"
                plt.plot(
                    hourly_avg_FLX,
                    label=rf"FLUXNET",
                    linestyle="solid",
                    color="black",
                )
            elif var_flx == "RECO_NT_VUT_USTAR50":
                r_text = r"R$_\text{eco}$"
                plt.plot(
                    hourly_avg_FLX,
                    label=rf"FLUXNET",
                    linestyle="solid",
                    color="black",
                )
            elif var_flx == "NEE_VUT_USTAR50":
                r_text = "NEE"
                plt.plot(
                    hourly_avg_FLX,
                    label=rf"FLUXNET",
                    linestyle="solid",
                    color="black",
                )
            elif var_flx == "GPP_NT_VUT_USTAR50":
                r_text = "GPP"
                plt.plot(
                    hourly_avg_FLX,
                    label=rf"FLUXNET",
                    linestyle="solid",
                    color="black",
                )

            plt.xlabel("UTC [h]", fontsize=FS_LABEL)
            if unit == "[°C]":
                plt.ylabel(rf"{r_text} [°C]", fontsize=FS_LABEL)
            else:
                plt.ylabel(rf"{r_text} [$\mu$mol m⁻² s⁻¹]", fontsize=FS_LABEL)
            plt.tick_params(labelsize=FS_TICK)
            # No in-panel legend: legend_FLX_comparison.pdf (save_legend_only_flx,
            # written once in main()) carries the full SITE/ALPS/DEFAULT/FLUXNET +
            # Topt-band legend for placement below the site/fluxtype figure grid.

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
        "GPP_WRF": "GPP",
        "RECO_WRF": r"R$_{eco}$",
        "NEE_WRF": "NEE",
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
            " & & $T_\\text{2m}$ & \\multicolumn{3}{c|}{GPP} & \\multicolumn{3}{c|}{$R_\\text{eco}$} & \\multicolumn{3}{c}{NEE} \\\\\n"
            "Site & vegetation class & WRF "
            "& SITE & ALPS & DEFAULT "
            "& SITE & ALPS & DEFAULT "
            "& SITE & ALPS & DEFAULT \\\\\n"
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


# Grouping the manuscript's hand-written table used (tab:filtered_sites_alps_means):
# (i) inside domain d03 -> footnote 1, (ii) remaining Alpine sites -> no
# footnote, (iii) outside but close to domain d01 -> footnote 2. Site order
# within each group matches the manuscript row-for-row. CH-Oe1/IT-Isp/IT-La2/
# IT-PT1 (in group ii) have no 2012 measurements and are dropped at emit time
# (see the all-NaN check in write_latex_table_fluxnet_means), not listed here
# as a separate exclusion set, so this stays a single source of truth for the
# 19-site membership.
_SITE_GROUPS = [
    (["AT-Neu", "IT-MBo", "CH-Dav", "IT-Lav", "IT-Ren"], 1),
    (["IT-Tor", "CH-Cha", "CH-Fru", "CH-Lae", "CH-Oe1", "CH-Oe2", "DE-Lkb", "IT-Isp", "IT-La2", "IT-PT1"], None),
    (["DE-Hai", "DE-Kli", "FR-Fon", "FR-Gri"], 2),
]


def write_latex_table_fluxnet_means(
    df_FLX_site_means, outfile="fluxnet_site_means.tex"
):
    """
    Write LaTeX table with FLUXNET site mean values, grouped and footnoted to
    match the manuscript's hand-written tab:filtered_sites_alps_means (see
    _SITE_GROUPS), with a \\hline between groups and \\hline\\hline before the
    trailing Mean row (computed by compute_fluxnet_site_means over all 19
    sites, NaN-site-excluding, unaffected by which rows are dropped here).

    Format per cell: yearly_mean (WRFmatch_mean) -- WRFmatch is the same
    24-representative-day sample as the rest of the paper's tables.

    Sites with no 2012 measurements (all five yearly values NaN) are dropped
    from the emitted rows -- the exclusion list is written as a trailing
    LaTeX comment so the caption can state it explicitly, rather than being
    hand-maintained in the .tex.
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

    def _row_cells(site):
        cells = []
        for var_flx in var_flx_list:
            yearly_val = df_FLX_site_means.loc[site, var_flx]
            wrf_match_val = df_FLX_site_means.loc[site, f"{var_flx}_WRFmatch"]
            if pd.isna(yearly_val):
                cells.append("--")
            else:
                cell_str = fmt2(yearly_val)
                if not pd.isna(wrf_match_val):
                    cell_str += f" ({fmt2(wrf_match_val)})"
                cells.append(cell_str)
        return cells

    dropped = []
    with open(outfile, "w") as f:
        f.write(
            "\\begin{tabular}{l|" + "c" * len(var_names) + "}\n"
            "\\hline\n"
            "Site & " + " & ".join(var_names) + " \\\\\n"
            "\\hline\\hline\n"
        )

        for gi, (sites, footnote) in enumerate(_SITE_GROUPS):
            if gi > 0:
                f.write("\\hline\n")
            for site in sites:
                if site not in df_FLX_site_means.index:
                    continue
                yearly_vals = [df_FLX_site_means.loc[site, v] for v in var_flx_list]
                if all(pd.isna(v) for v in yearly_vals):
                    dropped.append(site)
                    continue
                label = site + (f" \\footnotemark[{footnote}]" if footnote else "")
                f.write(label + " & " + " & ".join(_row_cells(site)) + " \\\\\n")

        f.write("\\hline\\hline\n")
        f.write("Mean & " + " & ".join(_row_cells("Mean")) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write(f"% excluded (no 2012 measurements): {', '.join(dropped)}\n")

    n_emitted = sum(len(sites) for sites, _ in _SITE_GROUPS) - len(dropped)
    print(f"LaTeX FLUXNET means table written to {outfile} "
          f"({n_emitted} sites, excluded: {dropped})")


def verify_fluxnet_site_means_table(outfile, bias_outfile):
    """Check 12: 15 data rows (19 sites minus the 4 with no 2012
    measurements), none entirely '--'; the 24-day (WRFmatch, in-parens)
    T2m/GPP/R_eco/NEE values for the five d03 sites match
    flux_evaluation_1km_bias_r30.tex's FLUXNET-mean ($\\overline{FLX}$)
    column cell for cell.

    IT-MBo T2m is allowed to differ by <= 0.01: the bias table's _align()
    additionally drops any hour where the MODEL series is NaN before
    averaging FLUXNET obs, while this table's WRFmatch mean only requires
    the observation itself to exist at that hour -- a real, understood
    pairing-rule difference (documented in the caption), not a bug.
    """
    lines = open(outfile).read().splitlines()
    data_rows = [
        l for l in lines
        if " & " in l and not l.startswith("Site") and not l.startswith("Mean")
    ]
    assert len(data_rows) == 15, f"expected 15 data rows, got {len(data_rows)}"
    for l in data_rows:
        cells = [c.strip() for c in l.rstrip().rstrip("\\").rstrip().split("&")[1:]]
        assert not all(c == "--" for c in cells), f"row is entirely empty: {l}"

    def _cell24(cell):
        if cell == "--" or "(" not in cell:
            return None
        return float(cell.split("(")[1].rstrip(") \\"))

    d03_sites = ["AT-Neu", "IT-MBo", "CH-Dav", "IT-Lav", "IT-Ren"]
    site_row = {}
    for l in data_rows:
        name = l.split("&")[0].strip().split(" ")[0]  # strip trailing \footnotemark
        if name in d03_sites:
            cells = [c.strip() for c in l.rstrip().rstrip("\\").rstrip().split("&")[1:]]
            site_row[name] = {
                "T2m": _cell24(cells[0]), "GPP": _cell24(cells[2]),
                "RECO": _cell24(cells[3]), "NEE": _cell24(cells[4]),
            }

    bias_lines = open(bias_outfile).read().splitlines()
    bias_row = {}
    for l in bias_lines:
        for s in d03_sites:
            if l.startswith(s + " &"):
                cells = [c.strip() for c in l.rstrip().rstrip("\\").rstrip().split("&")]
                # [0]=site [1]=vegclass [2]=T2m FLXmean [3]=T2m MB(MAE)
                # [4]=GPP FLXmean [5..7]=GPP SITE/ALPS/DEFAULT
                # [8]=R_eco FLXmean [9..11]=R_eco SITE/ALPS/DEFAULT
                # [12]=NEE FLXmean [13..15]=NEE SITE/ALPS/DEFAULT
                bias_row[s] = {
                    "T2m": float(cells[2]), "GPP": float(cells[4]),
                    "RECO": float(cells[8]), "NEE": float(cells[12]),
                }

    for s in d03_sites:
        assert s in site_row, f"{s} missing from {outfile}"
        assert s in bias_row, f"{s} missing from {bias_outfile}"
        for var in ("T2m", "GPP", "RECO", "NEE"):
            got, exp = site_row[s][var], bias_row[s][var]
            tol = 0.010001 if (s == "IT-MBo" and var == "T2m") else 1e-6
            assert abs(got - exp) <= tol, (
                f"{s} {var} 24-day: {outfile} has {got}, {bias_outfile} has "
                f"{exp} (diff {abs(got - exp):.4f}, tol {tol})"
            )

    # 19-site (15 non-empty) mean row: 24-day and yearly values both checked
    mean_line = next(l for l in lines if l.startswith("Mean &"))
    mean_cells = [c.strip() for c in mean_line.rstrip().rstrip("\\").rstrip().split("&")[1:]]
    exp_yearly = [7.25, 143.69, 4.06, 3.20, -0.79]
    exp_24day = [7.41, 138.65, 3.92, 3.17, -0.66]
    for cell, exp_y, exp_d, label in zip(mean_cells, exp_yearly, exp_24day,
                                          ("T2m", "SWin", "GPP", "RECO", "NEE")):
        got_y = float(cell.split(" (")[0])
        got_d = _cell24(cell)
        assert abs(got_y - exp_y) <= 0.010001, (
            f"Mean {label} yearly: {outfile} has {got_y}, expected {exp_y}"
        )
        assert abs(got_d - exp_d) <= 0.010001, (
            f"Mean {label} 24-day: {outfile} has {got_d}, expected {exp_d}"
        )

    print(f"  [ok] {outfile}: 15 data rows, none entirely empty, 24-day "
          f"T2m/GPP/R_eco/NEE match {bias_outfile} for the 5 d03 sites "
          "(IT-MBo T2m within 0.01, understood pairing-rule difference), "
          "Mean row matches expected yearly/24-day values")


def write_latex_table_from_metrics_bias(
    consolidated_metrics_total,
    model_lat_lon,
    outfile="flux_evaluation_1km_bias.tex",
    res_dx="1km",
    params=None,
):
    """
    Write LaTeX table summarising bias metrics per site.

    Each cell formatted as: MB (MAE)

    res_dx selects which resolution's metric columns to read (they are named
    `{site}_{channel}_{flux}_{res}`), so the same writer produces the 1 km
    table and its 9/54 km counterparts (Follow-up 15e) in an identical layout.

    params defaults to SITE/ALPS/REF. At 9 and 54 km pass ["ALPS", "REF"]:
    SITE parameters are 1 km per-site fits and the coarse domains carry no
    per-site parameterisation, so a SITE column there would be fabricated.
    """
    if params is None:
        params = ["SITE", "ALPS", "REF"]

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
        "GPP_WRF": "GPP",
        "RECO_WRF": r"R$_{eco}$",
        "NEE_WRF": "NEE",
    }

    def fmt(val, bold=False):
        s = f"{val:.2f}"
        return f"\\textbf{{{s}}}" if bold else s

    # Column layout follows `params`, so dropping SITE at 9/54 km narrows the
    # table rather than leaving an empty column behind.
    label = {"SITE": "SITE", "ALPS": "ALPS", "REF": "DEFAULT"}
    nflux = len(params) + 1  # + the FLUXNET-mean column
    flux_hdr = " & ".join(
        ["$\\overline{\\text{FLX}}$"] + [label[p] for p in params]
    )

    # Build bias table: show MB (MAE). Also compute mean row and bold best values
    with open(outfile, "w") as f:
        f.write(
            "\\begin{tabular}{ll|cc|" + "|".join(["c" * nflux] * 3) + "}\n"
            "\\hline\n"
            " & & \\multicolumn{2}{c|}{$T_\\text{2m}$}"
            f" & \\multicolumn{{{nflux}}}{{c|}}{{GPP}}"
            f" & \\multicolumn{{{nflux}}}{{c|}}{{$R_\\text{{eco}}$}}"
            f" & \\multicolumn{{{nflux}}}{{c}}{{NEE}} \\\\\n"
            "Site & vegetation class & $\\overline{\\text{FLX}}$ & WRF "
            f"& {flux_hdr} & {flux_hdr} & {flux_hdr} \\\\\n"
            "\\hline\\hline\n"
        )

        # accumulators to compute mean across sites
        mean_vals = {"T2": {"mean_fluxnet": [], "MB": [], "MAE": []}}
        for flux in fluxes:
            mean_vals[flux] = {
                p: {"mean_fluxnet": [], "MB": [], "MAE": []} for p in params
            }

        # per-site rows
        for site, pft_type in sites:
            veg_pct = vegfrac_lookup[site]
            pft_str = f"{veg_pct}\\% {pft_type}"

            # T2m: include mean_fluxnet, MB and MAE
            col_t2m = f"{site}_REF_T2_WRF_{res_dx}"
            mean_t2m = consolidated_metrics_total.loc["mean_fluxnet", col_t2m]
            mb_t2m = consolidated_metrics_total.loc["MB", col_t2m]
            mae_t2m = consolidated_metrics_total.loc["MAE", col_t2m]
            mean_vals["T2"]["mean_fluxnet"].append(mean_t2m)
            mean_vals["T2"]["MB"].append(mb_t2m)
            mean_vals["T2"]["MAE"].append(mae_t2m)

            row = f"{site} & {pft_str} & {mean_t2m:.2f} & {mb_t2m:.2f} ({mae_t2m:.2f})"

            # For each flux, collect values for this site, determine best among params and format
            for flux in fluxes:
                mean_list = []
                mb_list = []
                mae_list = []

                # First get FLUXNET mean (using SITE param as reference)
                col_fl = f"{site}_{params[0]}_{flux}_{res_dx}"
                mean_fl = consolidated_metrics_total.loc["mean_fluxnet", col_fl]

                # Now loop through all params to get MB and MAE
                for p in params:
                    col = f"{site}_{p}_{flux}_{res_dx}"
                    mb = consolidated_metrics_total.loc["MB", col]
                    mae = consolidated_metrics_total.loc["MAE", col]
                    mb_list.append(mb)
                    mae_list.append(mae)
                    # store for mean computation
                    mean_vals[flux][p]["mean_fluxnet"].append(mean_fl)
                    mean_vals[flux][p]["MB"].append(mb)
                    mean_vals[flux][p]["MAE"].append(mae)

                # determine best indices: closest to zero
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

                # Add FLUXNET mean first
                row += f" & {mean_fl:.2f}"

                # format each param value: MB (MAE)
                for i, (mb_val, mae_val) in enumerate(zip(mb_list, mae_list)):
                    mb_s = fmt(mb_val, bold=(i == mb_best_idx))
                    mae_s = fmt(mae_val, bold=(i == mae_best_idx))
                    row += f" & {mb_s} ({mae_s})"

            f.write(row + " \\\\\n")

        # Mean row across sites
        f.write("\\hline\n")
        mean_row = "Mean & -- "

        # mean for T2m
        mean_mean_t2 = (
            np.mean(mean_vals["T2"]["mean_fluxnet"])
            if len(mean_vals["T2"]["mean_fluxnet"]) > 0
            else np.nan
        )
        mean_mb_t2 = (
            np.mean(mean_vals["T2"]["MB"]) if len(mean_vals["T2"]["MB"]) > 0 else np.nan
        )
        mean_mae_t2 = (
            np.mean(mean_vals["T2"]["MAE"])
            if len(mean_vals["T2"]["MAE"]) > 0
            else np.nan
        )

        mean_row += f"& {mean_mean_t2:.2f} & {mean_mb_t2:.2f} ({mean_mae_t2:.2f})"

        # for each flux compute mean per param and bold best among params
        for flux in fluxes:
            # compute mean arrays per param
            mean_means = [np.mean(mean_vals[flux][p]["mean_fluxnet"]) for p in params]
            mb_means = [np.mean(mean_vals[flux][p]["MB"]) for p in params]
            mae_means = [np.mean(mean_vals[flux][p]["MAE"]) for p in params]

            # best indices
            mb_best_idx = int(np.nanargmin(np.abs(np.array(mb_means))))
            mae_best_idx = int(np.nanargmin(np.abs(np.array(mae_means))))

            # Add FLUXNET mean (use first param's mean as they should all be the same)
            mean_row += f" & {mean_means[0]:.2f}"

            # format each param
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
    sim_type = "_all"  # pools clear+cloudy (the only variant used in the paper)
    radius = 30
    fluxnet_site_means = os.getenv("FLUXNET_SITE_MEANS", "1") not in (
        "0",
        "",
    )  # Table D7; set FLUXNET_SITE_MEANS=0 to skip (it takes a few minutes)
    outfolder = OUTFOLDER
    base_dir_FLX = os.path.join(SCRATCH_PATH, "DATA/Fluxnet2015/Alps")
    fluxtypes = ["T2_WRF", "NEE_WRF", "GPP_WRF", "RECO_WRF"]
    plot_labels = [r"T$_\text{2m}$", "NEE", "GPP", r"R$_\text{eco}$"]
    units = [
        "[°C]",
        r"[$\mu$mol m$^{-2}$ s$^{-1}$]",
        r"[$\mu$mol m$^{-2}$ s$^{-1}$]",
        r"[$\mu$mol m$^{-2}$ s$^{-1}$]",
    ]
    # Resolution under evaluation. Default 1km reproduces the published run
    # bit-for-bit; EVAL_RES=9km / 54km emits the coarse-domain counterparts of
    # Table 1 (Follow-up 15e) so the coarse runs are shown against the towers
    # and not only against the 1 km run.
    res_dx = os.getenv("EVAL_RES", "1km")
    is_coarse = res_dx != "1km"
    # SITE parameters are 1 km per-site fits; the coarse domains carry no
    # per-site parameterisation, so that column would be fabricated there.
    eval_params = ["ALPS", "REF"] if is_coarse else ["SITE", "ALPS", "REF"]
    if is_coarse:
        # Table D7 (site means) is resolution-independent -- it is FLUXNET
        # observations plus the 24-day sampling mask, already emitted by the
        # 1 km run. Re-emitting it here would just overwrite it with itself.
        fluxnet_site_means = False

    # ALPS = offline recompute over the Topt percentile sweep (one CSV per tag);
    # tables use the p50 (median) tag, the per-site plots draw the band. Each
    # recalc CSV also carries the online REF (-> DEFAULT) `*_WRF` columns.
    # The FLUXNET evaluation pools clear + cloudy days ("_all"); otherwise a
    # single sim type.
    sim_suffixes = ["", "_cloudy"] if sim_type == "_all" else [sim_type]

    # Only the p50 member was recomputed at 9/54 km, and the coarse table
    # quotes the p50 ALPS channel exactly as Table 1 does -- so the ensemble
    # band is a 1 km product and the coarse runs carry a single member.
    alps_tags = [MEDIAN_TAG] if is_coarse else ALPS_TAGS
    ALPS_dfs = [
        load_site_csvs(tag, CSVFOLDER, res_dx, sim_suffixes, timespan, radius)
        for tag in alps_tags
    ]
    dataframes = {res_dx: ALPS_dfs[alps_tags.index(MEDIAN_TAG)]}  # p50 representative

    df_example = dataframes[res_dx]
    start_date = df_example.index[0]
    end_date = df_example.index[-1]

    # locations (kept same logic). The coarse-domain evaluation deliberately
    # uses the SAME five d03 towers as the 1 km table, so the three tables are
    # directly comparable row for row; the 15-site branch below is unused.
    if res_dx in ("1km", "9km", "54km"):
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

    locations_d01 = [
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

    # pft_site_match loading (kept identical)
    if sim_type == "_all":
        pft_site_match = pd.read_csv(
            f"{CSVFOLDER}distances_{res_dx}_{timespan}_r{radius}.csv"
        )
    else:
        pft_site_match = pd.read_csv(
            f"{CSVFOLDER}distances_{res_dx}{sim_type}_{timespan}_r{radius}.csv"
        )

    # Match by site name, not by row position. The old positional read
    # (`for i in range(len(locations))`) only happened to be correct at 1 km,
    # where the distances CSV lists exactly the five d03 sites in the same
    # order as `locations`. The coarse-domain CSVs carry all 15 d01 sites in a
    # different order, so position would silently pick the wrong rows.
    _by_site = {n.split("_")[0]: n for n in pft_site_match["name"]}
    _rows = pft_site_match.set_index("name")
    model_lat_lon = []
    for site in locations:
        if site not in _by_site:
            raise KeyError(
                f"{site} has no row in the {res_dx} distances CSV -- rerun "
                f"extract_SITE_timeseries.py --res {res_dx}"
            )
        r = _rows.loc[_by_site[site]]
        model_lat_lon.append(
            {
                "name": _by_site[site],
                "lat": r["lat_wrf"],
                "lon": r["lon_wrf"],
                "veg_frac": r["veg_frac_idx"],
            }
        )

    # CAMS disabled for RC2 revision (removed from paper).
    # Uncomment the block below to re-enable CAMS loading and plotting.
    # CAMS_data_dir_path = os.path.join(SCRATCH_PATH, "DATA/CAMS/")
    # path_CAMS_file = os.path.join(
    #     CAMS_data_dir_path + "ghg-reanalysis_surface_2012_full.nc"
    # )
    # CAMS_data, times_CAMS, gppbfas, rec_bfas, lat_CAMS, lon_CAMS = load_CAMS(
    #     path_CAMS_file
    # )
    # df_CAMS_hourly_all = {}
    # CAMS_vars = ["fco2gpp", "fco2rec", "fco2nee", "t2m"]
    # factor_kgC = 1000000 / 0.04401
    # factors = [factor_kgC, -factor_kgC, -factor_kgC, 1]
    # for location_ll in model_lat_lon:
    #     lat_target, lon_target = location_ll["lat"], location_ll["lon"]
    #     data_rows = []
    #     j = 0
    #     for time_CAMS in times_CAMS:
    #         date_CAMS = datetime(1970, 1, 1) + timedelta(seconds=int(time_CAMS))
    #         formatted_time = date_CAMS.strftime("%Y-%m-%d %H:%M:%S")
    #         row = {"time": formatted_time}
    #         for CAMS_var, factor in zip(CAMS_vars, factors):
    #             var_CAMS = CAMS_data.variables[CAMS_var][j, :, :].data * factor
    #             row[CAMS_var] = get_int_var(
    #                 lat_target, lon_target, lat_CAMS, lon_CAMS, var_CAMS
    #             )
    #             if CAMS_var == "fco2gpp":
    #                 var_CAMS_b = var_CAMS * gppbfas[j, :, :].data
    #                 row["fco2gpp_bfas"] = get_int_var(
    #                     lat_target, lon_target, lat_CAMS, lon_CAMS, var_CAMS_b
    #                 )
    #             if CAMS_var == "fco2rec":
    #                 var_CAMS_b = var_CAMS * rec_bfas[j, :, :].data
    #                 row["fco2rec_bfas"] = get_int_var(
    #                     lat_target, lon_target, lat_CAMS, lon_CAMS, var_CAMS_b
    #                 )
    #             if CAMS_var == "fco2nee":
    #                 row["fco2nee_bfas"] = row.get("fco2nee")
    #                 row["fco2nee"] = row.get("fco2rec") - row.get("fco2gpp")
    #         j += 1
    #         data_rows.append(row)
    #     df_CAMS = pd.DataFrame(data_rows)
    #     df_CAMS["time"] = pd.to_datetime(df_CAMS["time"])
    #     df_CAMS.set_index("time", inplace=True)
    #     df_CAMS = df_CAMS.astype(float)
    #     df_CAMS = df_CAMS[~df_CAMS.index.duplicated(keep="first")]
    #     df_CAMS_hourly = df_CAMS.resample("h").interpolate(method="linear")
    #     df_CAMS_hourly["t2m"] -= 273.15
    #     colname = location_ll["name"].split("_")[:1]
    #     df_CAMS_hourly_all[colname[0]] = df_CAMS_hourly
    df_CAMS_hourly_all = {}

    # resolution colors
    resolution_colors = {
        "1km": "blue",
        "3km": "darkgrey",
        "9km": "purple",
        "27km": "red",
        "54km": "green",
    }

    # Standalone legend for the site/fluxtype figures (no in-panel legend --
    # see process_location); one file, for placement below the figure grid.
    save_legend_only_flx(
        resolution_colors[res_dx], f"{outfolder}/legend_FLX_comparison.pdf"
    )

    # Run processing for each location and build consolidated metrics
    consolidated_metrics_total = pd.DataFrame()
    for location in locations:
        consolidated_metrics_all = process_location(
            location,
            fluxtypes,
            plot_labels,
            units,
            dataframes,
            ALPS_dfs,
            start_date,
            end_date,
            base_dir_FLX,
            df_CAMS_hourly_all,
            resolution_colors,
            sim_type,
            outfolder,
            timespan,
            resolution=res_dx,
        )
        consolidated_metrics_total = pd.concat(
            [consolidated_metrics_total, consolidated_metrics_all], axis=1
        )

    if not is_coarse:
        # The full-metrics table (RMSE/R2/...) stays a 1 km product; the
        # coarse-domain question is specifically about MB/MAE against the towers.
        write_latex_table_from_metrics(
            consolidated_metrics_total,
            model_lat_lon,
            outfile=f"{outfolder}/flux_evaluation_1km_r{radius}.tex",
        )
    # Also write bias table (MB (MAE))
    write_latex_table_from_metrics_bias(
        consolidated_metrics_total,
        model_lat_lon,
        outfile=f"{outfolder}/flux_evaluation_{res_dx}_bias_r{radius}.tex",
        res_dx=res_dx,
        params=eval_params,
    )

    if fluxnet_site_means:
        # Get average values of each location for the entire year and WRF-matched timesteps
        start_date_2012 = pd.Timestamp(timespan.split("_")[0]) - pd.Timedelta(hours=1)
        end_date_2012 = pd.Timestamp(timespan.split("_")[1]) - pd.Timedelta(hours=1)

        df_FLX_site_means = compute_fluxnet_site_means(
            locations_d01, start_date_2012, end_date_2012, base_dir_FLX, df_example
        )
        # Write FLUXNET site means table
        _fsm_outfile = f"{outfolder}fluxnet_site_means_r{radius}.tex"
        write_latex_table_fluxnet_means(
            df_FLX_site_means,
            outfile=_fsm_outfile,
        )
        verify_fluxnet_site_means_table(
            _fsm_outfile, f"{outfolder}/flux_evaluation_1km_bias_r{radius}.tex"
        )

    print("All done.")


if __name__ == "__main__":
    main()
