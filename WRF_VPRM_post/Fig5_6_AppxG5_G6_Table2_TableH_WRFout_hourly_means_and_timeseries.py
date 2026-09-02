"""
Fig5_6_AppxG5_G6_Table2_TableH_WRFout_hourly_means_and_timeseries.py
==============================================
Fig. 5 (fig:timeseries_values), Fig. 6 (fig:timeseries_diffs), Figs. G5-G6
(fig:timeseries_seasonal_clear/cloudy), Table 2 (tab:hourly_2012year) and
Tables H1-H5, H7, H9 (seasonal decomposition, tab:par_members). NOT Fig. 7 --
that is produced by Fig7_AppxH6_H8_effects_seasonal.py from the tables this
script ships.
Generate figures and tables for WRF-VPRM output analysis:
- Time series plots of domain-averaged variables at different resolutions
- Hourly mean diurnal cycles
- Domain-averaged tables of means and differences
==============================================
"""

from __future__ import annotations

import matplotlib

# use non-interactive backend for headless environments
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import numpy as np
import pandas as pd
import os
import math
import re
import zlib
import glob
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from dotenv import load_dotenv
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def fmt(v, nd=2):
    """Format with round-half-AWAY-FROM-ZERO (not Python's half-to-even).

    -0.855 -> "-0.86", -0.995 -> "-1.00". Decimal(str(...)) is deliberate: going
    through the repr avoids the binary-float artefact that would make
    Decimal(-0.855) round to -0.85.
    """
    if v is None:
        return "--"
    v = float(v)
    if math.isnan(v):
        return "--"
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(str(v)).quantize(q, rounding=ROUND_HALF_UP))

# ALPS is the offline recompute over a Topt percentile sweep; the spread across
# these tags is drawn as nested bands (dark p25-p75, light p10-p90, p50 median
# line). DEFAULT (RC1's renamed "DF") has no percentile recompute -> single line.
ALPS_TAGS = ["topt_p10", "topt_p25", "topt_p50", "topt_p75", "topt_p90"]
MEDIAN_TAG = "topt_p50"
PCTS = (10, 25, 50, 75, 90)

# ---- Figure geometry ----
# Paper convention: draw a figure at ~2x the size it is printed at, so the fonts
# below land at ~7 pt on the page. The Fig. 5/7 panels sit 5-across at
# 0.18\linewidth = 90 pt = 1.25 in printed (\linewidth = 500 pt), hence 2.5 in
# wide here (scale 0.52: 14 pt -> 7.2 pt printed). Previously these were 10 in
# wide with 30 pt fonts, which printed at ~3.9 pt.
# The full-width seasonal figures (plot_timeseries_by_resolution /
# plot_timeseries_differences, 0.60\linewidth) already print at ~7 pt and keep
# their (12, 6) size.
# Taller than the old 10:6 aspect on purpose: LaTeX fits to width, so only the
# width sets the scale factor and extra height is free -- it is what lets the long
# "GPP [umol m-2 s-1]" / "$\Delta_res$..." y labels fit without being clipped.
HOURLY_FIGSIZE = (2.5, 3.1)
# Fixed axes rectangle shared by plot_hourly_averages / plot_hourly_differences.
# These panels are tiled 5-across in the manuscript, so they must all be the same
# size with the axes in the same place; tight_layout + bbox_inches="tight" sized
# each one to its own y-label width instead (MediaBoxes ranged 158-172 pt against
# a nominal 180), which shifted every box and varied the printed font by ~8 %.
# `left` is set by the widest label, "$\Delta_\text{res}$R$_\text{eco}$ [umol m-2 s-1]"
# (measured via matplotlib's own mathtext renderer -- no TeX on this host -- against
# every column x with/without prefix combination; left=0.30 left only ~5.7 pt of
# clearance for the tightest case and the S-down stacked-fraction label actually
# clipped, both fixed here).
HOURLY_MARGINS = dict(left=0.33, right=0.97, bottom=0.20, top=0.97)
LEGEND_FIGSIZE = (7.0, 0.9)  # legend_hourly; bbox_inches="tight" crops to content
LEGEND_DIFF_FIGSIZE = (5.5, 0.9)  # legend_hourly_diff; ditto
FS_LABEL = 14  # -> ~7.2 pt printed
FS_TICK = 13  # -> ~6.7 pt printed
FS_LEGEND = 13

# Load environment variables from .env file
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ==================== Configuration ====================
SCRATCH_PATH = os.getenv("SCRATCH_PATH")
GITHUB_PATH = os.getenv("GITHUB_PATH")
OUTFOLDER = os.getenv("OUTFOLDER")

# Variable labels for scientific LaTeX-style plotting
var_labels = {
    "GPP": "GPP",
    "RECO": r"R$_{\text{eco}}$",
    "NEE": "NEE",
    "T2": r"T$_{\text{2m}}$",
    "SWDOWN": r"S$_\downarrow$",
}


def compute_nee(df: pd.DataFrame, resolutions: list) -> None:
    for res in resolutions:
        df[f"NEE_{res}"] = -df[f"GPP_{res}"] + df[f"RECO_{res}"]


def preprocess_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df["Unnamed: 0"], format="%Y-%m-%d %H:%M:%S")
    df.set_index("datetime", inplace=True)
    df["hour"] = df.index.hour
    return df


def group_hourly_average(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=["number"]).columns
    return df[numeric_columns].groupby("hour").mean()


def _pctl_over_tags(dfs: list, col: str):
    """Stack `col` across the tag dataframes (aligned on the shared index) and
    return (index, {pct: array}) of per-row percentiles, or (None, None)."""
    series = [d[col] for d in dfs if col in d.columns]
    if not series:
        return None, None
    M = pd.concat(series, axis=1)
    return M.index, {p: np.nanpercentile(M.values, p, axis=1) for p in PCTS}


def _has_spread(P) -> bool:
    return P is not None and np.nanmax(np.abs(P[90] - P[10])) > 1e-12


def _plot_segmented_band(idx, P, color, collect_xticks=False):
    """Plot the p50 line + nested bands over the per-day-segmented x axis used by
    the domain timeseries figures. Returns (xticks, xticklabels)."""
    dfp = pd.DataFrame(P, index=idx)
    xticks, xticklabels = [], []
    current_x = 0
    grouped = dfp.groupby(dfp.index.date)
    valid_days = [(d, g) for d, g in grouped if not g[50].dropna().eq(0).all()]
    for i, (date, g) in enumerate(valid_days):
        x = np.arange(len(g)) + current_x
        if np.nanmax(np.abs(g[90].values - g[10].values)) > 1e-12:
            plt.fill_between(
                x, g[10].values, g[90].values, color=color, alpha=0.15, linewidth=0
            )
            plt.fill_between(
                x, g[25].values, g[75].values, color=color, alpha=0.35, linewidth=0
            )
        plt.plot(x, g[50].values, linestyle="-", linewidth=1.5, color=color)
        if collect_xticks:
            xticks.append(current_x)
            xticklabels.append(str(date))
        if i < len(valid_days) - 1:
            plt.axvspan(
                current_x + len(g), current_x + len(g) + 1, color="lightgray", alpha=0.5
            )
        current_x += len(g) + 1
    return xticks, xticklabels


def _plot_segmented_line(series, color, linestyle):
    """Plot a single (DEFAULT) series over the same per-day-segmented x axis."""
    s = series.dropna()
    grouped = s.groupby(s.index.date)
    valid_days = [(d, g) for d, g in grouped if not g.dropna().eq(0).all()]
    current_x = 0
    for date, g in valid_days:
        x = np.arange(len(g)) + current_x
        plt.plot(x, g.values, linestyle=linestyle, linewidth=1.5, color=color)
        current_x += len(g) + 1


# --- Seasonal composites (DJF/MAM/JJA/SON; 3 monthly days each) ---------------
SEASONS = [("DJF", [12, 1, 2]), ("MAM", [3, 4, 5]), ("JJA", [6, 7, 8]),
           ("SON", [9, 10, 11])]
ALL_MONTHS = [m for _, months in SEASONS for m in months]  # union == 1..12
_SEG = 25  # 24 diurnal hours + 1 gap between season segments


def _season_diurnal(df, col, months):
    """24-value diurnal mean of `col` over the rows whose month is in `months`."""
    if col not in df.columns:
        return None
    s = df[df.index.month.isin(months)][col].dropna()
    if s.empty:
        return None
    return s.groupby(s.index.hour).mean()  # index = hour 0..23


def _season_pctl(dfs, col, months):
    """Per-hour percentiles across the tag dfs for one season -> (hours, {p:arr})."""
    cols = [_season_diurnal(d, col, months) for d in dfs]
    cols = [c for c in cols if c is not None]
    if not cols:
        return None, None
    M = pd.concat(cols, axis=1)
    return M.index.values, {p: np.nanpercentile(M.values, p, axis=1) for p in PCTS}


def _seasonal_band(dfs, col, color):
    """Plot p50 line + nested bands for each season segment, concatenated.
    Returns (xticks-at-segment-centres, season labels)."""
    xticks, labels = [], []
    for i, (name, months) in enumerate(SEASONS):
        hours, P = _season_pctl(dfs, col, months)
        x0 = i * _SEG
        labels.append(name)
        xticks.append(x0 + 12)
        if hours is None:
            continue
        x = np.arange(len(hours)) + x0
        if np.nanmax(np.abs(P[90] - P[10])) > 1e-12:
            plt.fill_between(x, P[10], P[90], color=color, alpha=0.15, linewidth=0)
            plt.fill_between(x, P[25], P[75], color=color, alpha=0.35, linewidth=0)
        plt.plot(x, P[50], linestyle="-", linewidth=1.5, color=color)
        if i < len(SEASONS) - 1:
            plt.axvspan(x0 + 24, x0 + 25, color="lightgray", alpha=0.4)
    return xticks, labels


def _seasonal_line(df, col, color, linestyle):
    """Plot a single (DEFAULT) seasonal-composite series over the same segments."""
    for i, (name, months) in enumerate(SEASONS):
        s = _season_diurnal(df, col, months)
        if s is None:
            continue
        x = np.arange(len(s)) + i * _SEG
        plt.plot(x, s.values, linestyle=linestyle, linewidth=1.5, color=color)


def _seasonal_diff_band(dfs, series_col, base_col, color):
    """Seasonal nested bands of the (res - 1km) difference across tags."""
    xticks, labels = [], []
    for i, (name, months) in enumerate(SEASONS):
        cols = []
        for d in dfs:
            if series_col in d.columns and base_col in d.columns:
                diff = (d[series_col] - d[base_col])
                diff = diff[diff.index.month.isin(months)].dropna()
                if not diff.empty:
                    cols.append(diff.groupby(diff.index.hour).mean())
        x0 = i * _SEG
        labels.append(name)
        xticks.append(x0 + 12)
        if not cols:
            continue
        M = pd.concat(cols, axis=1)
        P = {p: np.nanpercentile(M.values, p, axis=1) for p in PCTS}
        x = np.arange(len(M.index)) + x0
        if np.nanmax(np.abs(P[90] - P[10])) > 1e-12:
            plt.fill_between(x, P[10], P[90], color=color, alpha=0.15, linewidth=0)
            plt.fill_between(x, P[25], P[75], color=color, alpha=0.35, linewidth=0)
        plt.plot(x, P[50], linestyle="-", linewidth=1.5, color=color)
        if i < len(SEASONS) - 1:
            plt.axvspan(x0 + 24, x0 + 25, color="lightgray", alpha=0.4)
    return xticks, labels


def get_legend_elements(resolutions, resolution_colors):
    """Legend entries for legend_hourly.pdf.

    Shared by the main-text hourly panels (plot_hourly_averages /
    plot_hourly_differences), which do not draw CAMS -- so CAMS is skipped here
    too; drawing its entry would advertise a curve absent from those panels.
    """
    handles = []
    labels = []

    for res in resolutions:
        if res == "CAMS":
            continue
        color = resolution_colors[res]

        # ALPS (solid, p50 median)
        handles.append(Line2D([0], [0], color=color, lw=2, linestyle="-"))
        labels.append(f"{res}, ALPS")

        # DEFAULT (dashed)
        handles.append(Line2D([0], [0], color=color, lw=2, linestyle="--"))
        labels.append(f"{res}, DEFAULT")

    # Topt percentile bands (ALPS spread).
    handles.append(Patch(facecolor="gray", alpha=0.35, linewidth=0))
    labels.append(r"ALPS p25-p75 (ens.)")
    handles.append(Patch(facecolor="gray", alpha=0.15, linewidth=0))
    labels.append(r"ALPS p10-p90 (ens.)")

    return handles, labels


def save_legend_only(
    resolutions,
    resolution_colors,
    OUTFOLDER,
    column,
):

    handles, labels = get_legend_elements(resolutions, resolution_colors)

    fig = plt.figure(figsize=LEGEND_FIGSIZE)
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=4,  # 8 entries -> 2 rows
        fontsize=FS_LABEL,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
    )

    plt.axis("off")
    plt.savefig(
        f"{OUTFOLDER}legend_hourly.pdf",
        bbox_inches="tight",
    )
    plt.close()


def save_legend_only_diff(
    resolutions_diff: list,
    resolution_colors: dict,
    OUTFOLDER: str,
    column: str,
):

    handles = []
    labels = []

    for res in resolutions_diff:
        color = resolution_colors[res]

        # ALPS (solid, p50 median)
        handles.append(Line2D([0], [0], color=color, lw=2, linestyle="-"))
        labels.append(f"{res}-1km, ALPS")

        # DEFAULT (dashed)
        handles.append(Line2D([0], [0], color=color, lw=2, linestyle="--"))
        labels.append(f"{res}-1km, DEFAULT")

    # Topt percentile bands (ALPS spread) -- plot_hourly_differences draws these
    # nested fill_betweens (alpha 0.15 / 0.35) same as plot_hourly_averages, so
    # this legend needs to document them too.
    handles.append(Patch(facecolor="gray", alpha=0.35, linewidth=0))
    labels.append(r"ALPS p25-p75 (ens.)")
    handles.append(Patch(facecolor="gray", alpha=0.15, linewidth=0))
    labels.append(r"ALPS p10-p90 (ens.)")

    fig = plt.figure(figsize=LEGEND_DIFF_FIGSIZE)
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=3,  # 6 entries -> 2 rows
        fontsize=FS_LABEL,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
    )

    plt.axis("off")
    plt.savefig(
        f"{OUTFOLDER}legend_hourly_diff.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_timeseries_differences(
    dfs: list,
    df_ref: pd.DataFrame,
    column: str,
    unit: str,
    resolutions: list,
    OUTFOLDER: str,
    ref_sim: bool,
    resolution_colors: dict,
    STD_TOPO: int,
    start_date: str,
    end_date: str,
    sim_type: str,
):
    plt.figure(figsize=(12, 6))
    xticks, xticklabels = [], []

    for res in resolutions:
        if res in ["CAMS", "1km"]:
            continue
        color = resolution_colors[res]
        series_col = f"{column}_{res}"
        baseline_col = f"{column}_1km"

        # ALPS: seasonal nested band of the (res - 1km) difference across tags
        xticks, xticklabels = _seasonal_diff_band(dfs, series_col, baseline_col, color)
        plt.plot([], [], label=f"{res}-1km, ALPS", linestyle="-", color=color)

        # ----- DEFAULT reference (single dashed line): per-season (res - 1km) -----
        if ref_sim and series_col in df_ref and baseline_col in df_ref:
            for i, (name, months) in enumerate(SEASONS):
                sc = _season_diurnal(df_ref, series_col, months)
                bc = _season_diurnal(df_ref, baseline_col, months)
                if sc is None or bc is None:
                    continue
                x = np.arange(len(sc)) + i * _SEG
                plt.plot(x, (sc - bc).values, linestyle="--", linewidth=1.5, color=color)
            plt.plot([], [], label=f"{res}-1km, DEFAULT", linestyle="--", color=color)

    plt.xticks(xticks, xticklabels)
    plt.xlabel("Season (diurnal cycle, UTC)", fontsize=20)
    plt.ylabel(r"$\Delta_\text{res}$" + f"{var_labels[column]} {unit}", fontsize=20)
    plt.tick_params(axis="x", labelsize=18)
    plt.tick_params(axis="y", labelsize=18)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=18)
    plt.tight_layout()

    plt.savefig(
        f"{OUTFOLDER}timeseries_diff_of_resolutions_{column}_domain_averaged_std_topo_{STD_TOPO}{sim_type}_{start_date}_{end_date}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_timeseries_by_resolution(
    dfs: list,
    df_ref: pd.DataFrame,
    column: str,
    unit: str,
    resolutions: list,
    OUTFOLDER: str,
    ref_sim: bool,
    resolution_colors: dict,
    STD_TOPO: int,
    start_date: str,
    end_date: str,
    sim_type: str,
):
    plt.figure(figsize=(12, 6))
    xticks, xticklabels = [], []

    for res in resolutions:
        if res == "CAMS":  # CAMS is a single external product, drawn separately below
            continue
        color = resolution_colors[res]
        series_col = f"{column}_{res}"

        # ALPS seasonal percentile band over tags
        xticks, xticklabels = _seasonal_band(dfs, series_col, color)
        plt.plot([], [], label=f"{res}, ALPS", linestyle="-", color=color)

        # ----- DEFAULT reference (single dashed line) -----
        if ref_sim and column not in ("T2", "SWDOWN") and series_col in df_ref:
            _seasonal_line(df_ref, series_col, color, "--")
            plt.plot([], [], label=f"{res}, DEFAULT", linestyle="--", color=color)

    # ----- CAMS (single solid line, no ALPS/DEFAULT split, no percentile band) -----
    # Restored per the RC1/RC2 author comments (AC_RC1:153, AC_RC2:206), which cite
    # this seasonal appendix figure as the evidence that CAMS does not reproduce a
    # realistic seasonal cycle over the Alpine domain. Values come from the p50 df
    # (same convention as Table 1's CAMS column); CAMS itself has no Topt tag, so
    # every tag df carries an identical copy of it.
    if "CAMS" in resolutions:
        cams_col = f"{column}_CAMS"
        p50 = dfs[ALPS_TAGS.index(MEDIAN_TAG)]
        if cams_col in p50.columns:
            cams_color = resolution_colors["CAMS"]
            _seasonal_line(p50, cams_col, cams_color, "-")
            plt.plot([], [], label="CAMS", linestyle="-", color=cams_color)

    # Topt percentile bands (ALPS spread, drawn per-resolution by _seasonal_band
    # above as alpha 0.15/0.35 fill_betweens) -- document them in the legend, but
    # only for GPP/NEE: the flux variables where the Topt spread is the point.
    handles, labels = plt.gca().get_legend_handles_labels()
    if column in ("GPP", "NEE"):
        handles += [
            Patch(facecolor="gray", alpha=0.35, linewidth=0),
            Patch(facecolor="gray", alpha=0.15, linewidth=0),
        ]
        labels += [
            r"ALPS p25-p75 (ens.)",
            r"ALPS p10-p90 (ens.)",
        ]

    plt.xticks(xticks, xticklabels)
    plt.xlabel("Season (diurnal cycle, UTC)", fontsize=20)
    plt.ylabel(f"{var_labels[column]} {unit}", fontsize=20)
    plt.tick_params(axis="x", labelsize=18)
    plt.tick_params(axis="y", labelsize=18)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(handles=handles, labels=labels, fontsize=18)
    plt.tight_layout()
    plt.savefig(
        f"{OUTFOLDER}timeseries_{column}_domain_averaged_std_topo_{STD_TOPO}{sim_type}_{start_date}_{end_date}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_hourly_averages(
    hourly_list: list,
    hourly_avg_ref: pd.DataFrame,
    column: str,
    unit: str,
    resolutions: list,
    OUTFOLDER: str,
    ref_sim: bool,
    resolution_colors: dict,
    STD_TOPO: int,
    start_date: str,
    end_date: str,
    sim_type: str,
):
    plt.figure(figsize=HOURLY_FIGSIZE)
    for res in resolutions:
        # CAMS is not shown in the main-text hourly panels (nor in legend_hourly,
        # see get_legend_elements) -- skip it outright rather than only muting its
        # band, which is what used to leave an undocumented orange curve here.
        if res == "CAMS":
            continue
        color = resolution_colors[res]
        col = f"{column}_{res}"
        idx, P = _pctl_over_tags(hourly_list, col)
        if idx is None:
            continue
        if res != "CAMS" and _has_spread(P):
            plt.fill_between(idx, P[10], P[90], color=color, alpha=0.15, linewidth=0)
            plt.fill_between(idx, P[25], P[75], color=color, alpha=0.35, linewidth=0)
        plt.plot(idx, P[50], linestyle="-", color=color)

        if ref_sim and res != "CAMS":
            if column != "T2" and column != "SWDOWN":
                series_ref = hourly_avg_ref[col].dropna()
                plt.plot(
                    series_ref.index,
                    series_ref,
                    linestyle="--",
                    color=color,
                )

    plt.xlabel("UTC [h]", fontsize=FS_LABEL)
    plt.ylabel(f"{var_labels[column]} {unit}", fontsize=FS_LABEL)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks([0, 6, 12, 18, 24])
    plt.tick_params(axis="x", labelsize=FS_TICK)
    plt.tick_params(axis="y", labelsize=FS_TICK)
    plt.subplots_adjust(**HOURLY_MARGINS)
    plt.savefig(
        f"{OUTFOLDER}timeseries_hourly_{column}_domain_averaged_std_topo_{STD_TOPO}{sim_type}_{start_date}_{end_date}.pdf"
    )
    plt.close()


# Fig. 6(i) (RECO, cloudy 54km-diff): the auto-ticks span -0.05..-0.15 at 0.025
# spacing (5 ticks, 3-decimal labels e.g. "-0.150") -- wide enough that the
# y-label was pushed off the canvas entirely even after the HOURLY_MARGINS
# widening above. Pin 3 round ticks and drop the tick-label font a touch for
# this one panel rather than widening the (shared) margin further for
# everyone else's sake.
_YTICK_OVERRIDE = {
    ("RECO", "_cloudy"): ([-0.05, -0.10, -0.15], FS_TICK - 2),
}


def plot_hourly_differences(
    hourly_list: list,
    hourly_avg_ref: pd.DataFrame,
    column: str,
    unit: str,
    resolutions_diff: list,
    OUTFOLDER: str,
    ref_sim: bool,
    resolution_colors: dict,
    STD_TOPO: int,
    start_date: str,
    end_date: str,
    sim_type: str,
):
    plt.figure(figsize=HOURLY_FIGSIZE)
    for res in resolutions_diff:
        color = resolution_colors[res]
        # ALPS: per-hour percentile band of (res - 1km) across tags
        diffs = [h[f"{column}_{res}"] - h[f"{column}_1km"] for h in hourly_list]
        M = pd.concat(diffs, axis=1)
        P = {p: np.nanpercentile(M.values, p, axis=1) for p in PCTS}
        idx = M.index
        if _has_spread(P):
            plt.fill_between(idx, P[10], P[90], color=color, alpha=0.15, linewidth=0)
            plt.fill_between(idx, P[25], P[75], color=color, alpha=0.35, linewidth=0)
        plt.plot(idx, P[50], linestyle="-", color=color)

        if ref_sim:
            diff_ref = (
                hourly_avg_ref[f"{column}_{res}"] - hourly_avg_ref[f"{column}_1km"]
            )
            plt.plot(diff_ref.index, diff_ref, linestyle="--", color=color)

    plt.xlabel("UTC [h]", fontsize=FS_LABEL)
    plt.ylabel(r"$\Delta_\text{res}$" + f"{var_labels[column]} {unit}", fontsize=FS_LABEL)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks([0, 6, 12, 18, 24])
    plt.tick_params(axis="x", labelsize=FS_TICK)
    ax = plt.gca()
    override = _YTICK_OVERRIDE.get((column, sim_type))
    if override:
        yticks, y_fs = override
        ax.set_yticks(yticks)
        ax.minorticks_off()
        ax.tick_params(axis="y", labelsize=y_fs)
    else:
        ax.tick_params(axis="y", labelsize=FS_TICK)
    plt.subplots_adjust(**HOURLY_MARGINS)
    plt.savefig(
        f"{OUTFOLDER}timeseries_hourly_diff_of_54km_{column}_domain_averaged_std_topo_{STD_TOPO}{sim_type}_{start_date}_{end_date}.pdf"
    )
    plt.close()


def compute_hourly_means_and_differences_reshaped(
    hourly_avg: pd.DataFrame,
    hourly_avg_ref: pd.DataFrame,
    columns: list,
    resolutions: list,
    ref_sim: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records_means = []
    records_diffs = []
    baseline = "1km"

    for col in columns:
        for res in resolutions:
            key = f"{col}_{res}"
            base_key = f"{col}_{baseline}"

            if key in hourly_avg.columns:
                mean_opt = hourly_avg[key].mean()
                records_means.append(
                    {
                        "variable": col,
                        "type": "ALPS",
                        "resolution": res,
                        "mean": mean_opt,
                    }
                )

                if res != baseline and base_key in hourly_avg.columns:
                    diff_opt = (hourly_avg[key] - hourly_avg[base_key]).mean()
                    records_diffs.append(
                        {
                            "variable": col,
                            "type": "ALPS_DIFF",
                            "resolution": f"{res}-{baseline}",
                            "mean": diff_opt,
                        }
                    )

            if ref_sim and res != "CAMS" and key in hourly_avg_ref.columns:
                mean_ref = hourly_avg_ref[key].mean()
                records_means.append(
                    {
                        "variable": col,
                        "type": "DF",
                        "resolution": res,
                        "mean": mean_ref,
                    }
                )

                if res != baseline and base_key in hourly_avg_ref.columns:
                    diff_ref = (hourly_avg_ref[key] - hourly_avg_ref[base_key]).mean()
                    records_diffs.append(
                        {
                            "variable": col,
                            "type": "DF_DIFF",
                            "resolution": f"{res}-{baseline}",
                            "mean": diff_ref,
                        }
                    )

    df_means = pd.DataFrame(records_means).pivot(
        index="variable", columns=["type", "resolution"], values="mean"
    )
    df_diffs = pd.DataFrame(records_diffs).pivot(
        index="variable", columns=["type", "resolution"], values="mean"
    )

    df_pct = pd.DataFrame()
    for col in df_diffs.columns:
        diff_type, diff_res = col
        if diff_type == "ALPS_DIFF":
            base_res = diff_res.split("-")[1]
            ref_col = ("ALPS", base_res)
        elif diff_type == "DF_DIFF":
            base_res = diff_res.split("-")[1]
            ref_col = ("DF", base_res)
        else:
            continue

        if ref_col in df_means.columns:
            pct_col = (f"{diff_type}_PCT", diff_res)
            df_pct[pct_col] = (df_diffs[col] / df_means[ref_col]) * 100

    return df_means, df_diffs, df_pct


def load_domain_csv(path, resolutions):
    """One domain-averaged timeseries csv -> datetime-indexed df with NEE added."""
    d = pd.read_csv(path)
    compute_nee(d, resolutions)
    return preprocess_datetime(d)


def load_regime(sim, csv_folder, STD_TOPO, start_date, end_date, resolutions, columns,
                 ref_sim=True):
    """ALPS tag dfs + DEFAULT reference + the derived table frames, one regime.

    ALPS = offline recompute over the Topt percentile sweep (one df per tag);
    DEFAULT = online reference (single). Tables use the p50 (median) tag.
    Module-level (not a closure inside main()) so other scripts that need the
    same per-tag/DEFAULT csvs -- e.g. Fig7_AppxH6_H8_effects_seasonal.py -- load them the
    same way rather than re-implementing the csv paths and column setup.
    """
    alps = [
        load_domain_csv(
            f"{csv_folder}timeseries_domain_averaged_recalc_{tag}_std_topo_gt_{STD_TOPO}{sim}_{start_date}_{end_date}.csv",
            resolutions,
        )
        for tag in ALPS_TAGS
    ]
    ref = (
        load_domain_csv(
            f"{csv_folder}timeseries_domain_averaged_REF_std_topo_gt_{STD_TOPO}{sim}_{start_date}_{end_date}.csv",
            resolutions,
        )
        if ref_sim
        else None
    )
    h_list = [group_hourly_average(d) for d in alps]
    h_avg = h_list[ALPS_TAGS.index(MEDIAN_TAG)]  # p50 median (table)
    h_ref = group_hourly_average(ref) if ref_sim else None
    m, d_, p = compute_hourly_means_and_differences_reshaped(
        h_avg, h_ref, columns, resolutions, ref_sim
    )
    d_.columns = pd.MultiIndex.from_tuples(d_.columns)
    p.columns = pd.MultiIndex.from_tuples(p.columns)
    return dict(
        alps=alps, ref=ref, hourly_list=h_list, hourly_avg=h_avg,
        hourly_avg_ref=h_ref, means=m, diffs=d_, pct=p,
    )


# Units deliberately omitted from these row labels -- the manuscript caption
# states them, and repeating them made Table 1 too wide (see FIGURE_TEX_CHANGES.md).
# Do not "helpfully" restore them here; edit the caption instead if units are needed.
TABLE1_VARIABLES = [
    ("T2", r"$T_\text{2m}$"),
    ("SWDOWN", r"S$\downarrow$"),
    ("GPP", r"GPP"),
    ("RECO", r"R$_{\text{eco}}$"),
    ("NEE", r"NEE"),
]
# T2 and S-down do not depend on the parameter set, so their ALPS/DEFAULT pair
# collapses to one merged cell and they lead each block; the ALPS/DEFAULT
# sub-header is emitted after them, immediately before the flux rows it labels
# (Follow-up 16f). The two lists partition TABLE1_VARIABLES.
TABLE1_MET = TABLE1_VARIABLES[:2]
TABLE1_FLUXES = TABLE1_VARIABLES[2:]

# Follow-up 16i: below this reference magnitude the relative bracket is dropped
# and only the absolute difference is printed. A percentage of a near-zero
# denominator says nothing about the model -- the winter domain means put GPP at
# 0.01-0.2 umol m-2 s-1 and produced [245%], [317%], [2400%] cells. S-down
# (50-260 W m-2) is never near zero and so always keeps its percentage; T2 never
# has one at all (Follow-up 12d), a Celsius ratio depending on where the zero of
# the scale sits.
PCT_MIN_REF = 0.5  # umol m-2 s-1


def _signed(v):
    """fmt(v) with a forced sign, and never a negative zero (Follow-up 16j)."""
    s = fmt(v)
    if s not in ("--",) and float(s) == 0.0:
        s = s.lstrip("-")
    return s if s.startswith("-") else "+" + s


def _paren(dval, ref, var):
    """The '(delta [pct%])' parenthetical shared by all four tables.

    `ref` is the denominator of the relative bracket: the 1 km value of the same
    parameter set for a resolution difference, the 1 km DEFAULT for the parameter
    difference, the 1 km ALPS for a CAMS difference. The bracket is suppressed
    for T2 always and for a flux whose |ref| < PCT_MIN_REF.
    """
    s = _signed(dval)
    if var != "T2" and abs(ref) >= PCT_MIN_REF:
        s += f" [{fmt(abs(dval / ref * 100), 0)}\\%]"
    return f"({s})"


_T1_GROUP_HDR = (
    "Sample & Var & \\multicolumn{2}{c|}{1~km} & \\multicolumn{2}{c|}{9~km}"
    " & \\multicolumn{2}{c}{54~km} \\\\\n"
)
# The manuscript prints which difference each column carries, so the table can
# be read without the caption; the numbers are unaffected.
_T1_SUB_HDR_PLAIN = " & & ALPS & DEFAULT & ALPS & DEFAULT & ALPS & DEFAULT \\\\\n"
_T1_SUB_HDR = (
    " & & ALPS $(\\Delta_\\text{par})$ & DEFAULT"
    " & ALPS $(\\Delta_\\text{res}^\\text{9km})$ & DEFAULT $(\\Delta_\\text{res}^\\text{9km})$"
    " & ALPS $(\\Delta_\\text{res}^\\text{54km})$ & DEFAULT $(\\Delta_\\text{res}^\\text{54km})$ \\\\\n"
)


def _t1_cells(var, means, diff_mean, diff_pct=None):
    """Raw numbers for one Table-1 row: (alps_1km, ref_1km, wrf).

    `wrf` is [(val, dval) x4] in the order 9 km ALPS, 9 km DEFAULT, 54 km ALPS,
    54 km DEFAULT. 'DF' is the internal MultiIndex data key for the default
    parameter set; the column is *labelled* DEFAULT but the key must not be
    renamed. Percentages are no longer carried through here -- every bracket is
    derived in `_paren` from the delta and its own reference, so there is one
    definition of the denominator rather than two.
    """
    alps_1km = means.loc[var, ("ALPS", "1km")]
    ref_1km = means.loc[var, ("DF", "1km")]
    wrf = [
        (
            means.loc[var, (sim, res)],
            diff_mean.loc[var, (f"{sim}_DIFF", f"{res}-1km")],
        )
        for res in ("9km", "54km")
        for sim in ("ALPS", "DF")
    ]
    return alps_1km, ref_1km, wrf


def _season_cells(p50, ref, months, var):
    """`_t1_cells` for one season of one regime, from the raw timeseries."""
    a1 = _smean(p50, f"{var}_1km", months)
    d1 = _smean(ref, f"{var}_1km", months)
    wrf = []
    for res in ("9km", "54km"):
        for df, base in ((p50, a1), (ref, d1)):
            v = _smean(df, f"{var}_{res}", months)
            wrf.append((v, v - base))
    return a1, d1, wrf


def _mean_cells(a, b):
    """Element-wise mean of two `_t1_cells` triples, with the deltas re-derived
    from the averaged levels (delta = coarse - 1 km of the same parameter set).

    Each regime is an equal-size composite, so the 24-day value is the plain
    average of the two and this reproduces what a recompute on the pooled
    timeseries gives.
    """
    alps_1km = (a[0] + b[0]) / 2.0
    ref_1km = (a[1] + b[1]) / 2.0
    base = [alps_1km, ref_1km, alps_1km, ref_1km]
    wrf = []
    for i in range(4):
        val = (a[2][i][0] + b[2][i][0]) / 2.0
        wrf.append((val, val - base[i]))
    return alps_1km, ref_1km, wrf


def _t1_met_row(sample, var, label, alps_1km, ref_1km, wrf):
    """A T2 / S-down row: one merged cell per grid spacing (Follow-up 16f).

    The merge is only legitimate because the meteorology is identical in the two
    parameter sets -- VPRM is diagnostic and does not feed back on WRF -- so that
    is asserted here rather than assumed. If it ever fails the merge is wrong and
    this must stop the run, not quietly print one of the two.
    """
    assert abs(alps_1km - ref_1km) < 1e-9, (
        f"{sample or '(cont.)'} {var} 1km: ALPS {alps_1km!r} != DEFAULT {ref_1km!r} "
        "-- cannot merge the ALPS/DEFAULT pair"
    )
    row = f"{sample} & {label} & \\multicolumn{{2}}{{c|}}{{{fmt(alps_1km)}}}"
    for j, (res, close) in enumerate((("9km", "c|"), ("54km", "c"))):
        (va, da), (vd, dd) = wrf[2 * j], wrf[2 * j + 1]
        assert abs(va - vd) < 1e-9 and abs(da - dd) < 1e-9, (
            f"{sample or '(cont.)'} {var} {res}: ALPS {va!r} ({da!r}) != "
            f"DEFAULT {vd!r} ({dd!r}) -- cannot merge the ALPS/DEFAULT pair"
        )
        cell = f"{fmt(va)} {_paren(da, alps_1km, var)}"
        row += f" & \\multicolumn{{2}}{{{close}}}{{{cell}}}"
    return row + " \\\\\n"


def _t1_flux_row(var, label, alps_1km, ref_1km, wrf, bold_nee=True):
    """A GPP / R_eco / NEE row.

    The 1 km ALPS cell carries the parameter effect Delta_par = DEFAULT - ALPS
    (Follow-up 16a; sign flipped so that every Delta_par^s = x_s - x_ALPS, ALPS
    subtracted throughout -- Eq. 2 of the manuscript), in the same parenthesis
    form as the coarse columns and computed from the UNROUNDED means -- that is
    what makes it the same number as Delta_par in Table_seasonal_effects_*. It
    replaces the old |ALPS|/|DEFAULT| ratio, which was the one cell of the
    table reporting a ratio where every other reported a difference. DEFAULT
    keeps no bracket.
    """
    row = f" & {label} & {fmt(alps_1km)} {_paren(ref_1km - alps_1km, ref_1km, var)}"
    row += f" & {fmt(ref_1km)}"
    for i, (val, dval) in enumerate(wrf):
        paren = _paren(dval, alps_1km if i % 2 == 0 else ref_1km, var)
        if var == "NEE" and bold_nee:
            paren = f"\\textbf{{{paren}}}"
        row += f" & {fmt(val)} {paren}"
    return row + " \\\\\n"


def _t1_block(f, sample, cells, bold_nee=True, annotate_hdr=True):
    """One block: the two merged meteorology rows, the ALPS/DEFAULT sub-header
    under a short \\cmidrule, then the three flux rows. `cells` maps a variable
    name to its `_t1_cells` triple. The block name sits in column 1 of the first
    row only. No rule below the sub-header, so the only rules crossing column 1
    are the full-width ones that separate the blocks."""
    for i, (var, label) in enumerate(TABLE1_MET):
        f.write(_t1_met_row(sample if i == 0 else "", var, label, *cells[var]))
    f.write("\\cmidrule(lr){3-8}\n")
    f.write(_T1_SUB_HDR if annotate_hdr else _T1_SUB_HDR_PLAIN)
    for var, label in TABLE1_FLUXES:
        f.write(_t1_flux_row(var, label, *cells[var], bold_nee=bold_nee))


def write_domain_average_table(
    clear,
    cloudy,
    outfile="domain_averaged_2012.tex",
):
    """Write the combined domain-averaged Table 1 as a bare tabular.

    Three blocks -- clear sky, clouds and rain, and their 24-day average -- each
    carrying all five variables (Follow-up 16g; the 24-day block used to carry
    NEE alone). Emits ONLY \\begin{tabular} ... \\end{tabular}; the manuscript
    supplies the table float, caption and label. CAMS is no longer a column here
    -- it has its own table (`write_cams_table`, Follow-up 16h).

    Parameters
    ----------
    clear, cloudy : (means, diff_mean, diff_pct) triples, one per regime.
    """
    blocks = []
    for sample, src in (("Clear sky", clear), ("Clouds and rain", cloudy)):
        blocks.append((sample, {v: _t1_cells(v, *src) for v, _ in TABLE1_VARIABLES}))
    blocks.append(
        (
            "24-day average",
            {v: _mean_cells(blocks[0][1][v], blocks[1][1][v])
             for v, _ in TABLE1_VARIABLES},
        )
    )

    with open(outfile, "w") as f:
        f.write("\\begin{tabular}{ll|rr|rr|rr}\n\\toprule\n")
        f.write(_T1_GROUP_HDR)
        f.write("\\midrule\n")
        for i, (sample, cells) in enumerate(blocks):
            if i:
                f.write("\\midrule\n")
            _t1_block(f, sample, cells)
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"LaTeX table written to {outfile}")


def _numeric_cells(line):
    """Body-cell strings of a LaTeX row, stripped of bolding and the row label."""
    body = line.rstrip().rstrip("\\").rstrip()
    return [c.strip() for c in body.split("&")[1:]]


_MULTICOL_RE = re.compile(r"^\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$")


def _unwrap(cell):
    """A cell's content, with any \\multicolumn wrapper and \\textbf stripped."""
    c = cell.strip()
    m = _MULTICOL_RE.match(c)
    if m:
        c = m.group(2).strip()
    return c.replace("\\textbf{", "").strip().rstrip("}").strip()


def _lead_number(cell):
    """The leading value of a cell like '-1.83 \\textbf{(-0.10 [6\\%])}'."""
    return _unwrap(cell).split(" ")[0]


_BLOCK_NAMES = ("Clear sky", "Clouds and rain", "24-day average")
# A 1 km ALPS cell in the old ratio convention: a bare value followed by a
# square bracket, with no parenthesis. Must not survive anywhere (Follow-up 16a).
_RATIO_CELL_RE = re.compile(r"^-?\d+\.\d+ \[\d+\\%\]$")


def _read_blocks(path, block_names=_BLOCK_NAMES):
    """Parse one of the `Sample & Var` tables into {block: {var: [cells]}}.

    The block name sits in column 1 of its first row only, so it is carried
    forward until the next one appears.
    """
    label_to_var = {lbl: var for var, lbl in TABLE1_VARIABLES}
    out = {b: {} for b in block_names}
    block = None
    for l in open(path).read().splitlines():
        if not l.endswith("\\\\"):
            continue
        head = l.split("&")[0].strip()
        if head in block_names:
            block = head
        if block is None:
            continue
        lbl = l.split("&")[1].strip() if l.count("&") >= 1 else ""
        if lbl in label_to_var:
            out[block][label_to_var[lbl]] = _numeric_cells(l)[1:]
    return out


def verify_table1(path):
    """Assert the invariants the manuscript relies on.

    - three blocks, five variables each (Follow-up 16g: the 24-day block used to
      carry NEE alone);
    - every NEE row bolds exactly its four WRF delta parentheticals;
    - the 24-day block is the mean of the two regime blocks, every variable and
      every column;
    - the 1 km ALPS cell is a Delta_par parenthetical, never the old ratio.
    """
    blocks = _read_blocks(path)
    for b in _BLOCK_NAMES:
        got = sorted(blocks[b])
        assert got == sorted(v for v, _ in TABLE1_VARIABLES), (
            f"block '{b}' has variables {got}, expected all five"
        )
    print(f"  [ok] Table 1: {len(_BLOCK_NAMES)} blocks x "
          f"{len(TABLE1_VARIABLES)} variables, 24-day block included")

    for b in _BLOCK_NAMES:
        n = " & ".join(blocks[b]["NEE"]).count("\\textbf{")
        assert n == 4, f"NEE row of '{b}' has {n} \\textbf groups, expected 4"
    print("  [ok] Table 1: every NEE row bolds exactly its four WRF deltas")

    # The 24-day block is computed from UNROUNDED values, so comparing against
    # the mean of the two *printed* values can differ by one display unit (0.01)
    # through double rounding -- true -0.9915 prints -0.99, while
    # (-1.75 + -0.24)/2 = -0.995 would print -1.00. Anything beyond that is a bug.
    doubled = []
    for var, _lbl in TABLE1_VARIABLES:
        c, d, a = (blocks[b][var] for b in _BLOCK_NAMES)
        for j, (cc, dd, gg) in enumerate(zip(c, d, a)):
            exp = (float(_lead_number(cc)) + float(_lead_number(dd))) / 2.0
            delta = abs(float(_lead_number(gg)) - exp)
            assert delta <= 0.010001, (
                f"24-day {var} col {j}: printed {_lead_number(gg)}, "
                f"mean(clear={_lead_number(cc)}, cloudy={_lead_number(dd)})="
                f"{fmt(exp)} (off by {delta:.4f}, more than one display unit)"
            )
            if delta > 1e-9:
                doubled.append(f"{var}/col{j}")
    total = sum(len(blocks[_BLOCK_NAMES[0]][v]) for v, _ in TABLE1_VARIABLES)
    print("  [ok] Table 1: 24-day block = mean of the two regimes, all 5 variables")
    if doubled:
        # Expected, and about half the cells: the mean of two 2-decimal numbers
        # lands on an exact .xx5 tie half the time, which fmt's round-half-away
        # pushes up, while the true unrounded mean sits just below it. The block
        # is computed from the unrounded means, so the printed value is the right
        # one; this only bounds how far the two can drift.
        print(f"       ({len(doubled)}/{total} cells differ by 0.01 through double "
              f"rounding, all within one display unit)")


def verify_no_ratio_cells(paths):
    """Follow-up 16a/16b: no 1 km cell anywhere still reports |ALPS|/|DEFAULT|.

    The 1 km ALPS cell of a flux row must be `value (signed-delta [pct%])`; the
    merged meteorology cells and the DEFAULT cell carry no bracket at all. A
    bare `value [N%]` is the old ratio and must not appear in any cell.
    """
    checked = 0
    for path in paths:
        for l in open(path).read().splitlines():
            if not l.endswith("\\\\"):
                continue
            for cell in _numeric_cells(l):
                assert not _RATIO_CELL_RE.match(_unwrap(cell)), (
                    f"ratio-style 1 km cell survives in {os.path.basename(path)}: {cell}"
                )
                checked += 1
    print(f"  [ok] no |ALPS|/|DEFAULT| ratio cell in any of {len(paths)} tables "
          f"({checked} cells checked)")


def _cell_delta(cell):
    """The signed delta printed inside a cell's parenthesis, as a string."""
    m = re.search(r"\(([+-]\d+\.\d+)", cell)
    return m.group(1) if m else None


def verify_cross_tables(table1, effects_clear, effects_cloudy, effects_all):
    """Follow-up 16c: the reason for the change.

    Table 1's 1 km Delta_par must be, character for character, the Delta_par of
    the matching Year row of the seasonal-effects tables, and its 24-day 9 km/
    54 km ALPS deltas the matching Delta_res^9/Delta_res^54. Exact equality on
    the printed 2-decimal strings, no tolerance.
    """
    blocks = _read_blocks(table1)
    # Column index inside the seasonal-effects rows. GPP's block is 6 wide in
    # the per-regime tables and 7 wide in the pooled one (sigma^Delta_IQR).
    def _effects_row(path, name, gpp_cols):
        line = next(l for l in open(path).read().splitlines()
                    if l.startswith(name + " &"))
        cells = _numeric_cells(line)
        off = [0, gpp_cols, gpp_cols + 3]
        return {v: (cells[o], cells[o + 1], cells[o + 2])  # (Dres9, Dres54, Dpar)
                for v, o in zip(("GPP", "RECO", "NEE"), off)}

    # Flux-row cell order: 0 = 1 km ALPS, 1 = 1 km DEFAULT, 2/3 = 9 km ALPS/
    # DEFAULT, 4/5 = 54 km ALPS/DEFAULT.
    pairings = [
        ("clear-sky 1 km Delta_par", "Clear sky", 0,
         _effects_row(effects_clear, "Year", 6), 2),
        ("cloudy 1 km Delta_par", "Clouds and rain", 0,
         _effects_row(effects_cloudy, "Year", 6), 2),
        ("24-day 1 km Delta_par", "24-day average", 0,
         _effects_row(effects_all, "Year", 7), 2),
        ("24-day 9 km Delta_res", "24-day average", 2,
         _effects_row(effects_all, "Year", 7), 0),
        ("24-day 54 km Delta_res", "24-day average", 4,
         _effects_row(effects_all, "Year", 7), 1),
    ]
    bad = []
    for what, block, col, ref, ri in pairings:
        for var in ("GPP", "RECO", "NEE"):
            got = _cell_delta(blocks[block][var][col])
            exp = ref[var][ri]
            ok = got == exp
            print(f"    [{'ok' if ok else 'FAIL'}] {what:26s} {var:5s} "
                  f"Table 1 {got}  vs  seasonal-effects {exp}")
            if not ok:
                bad.append((what, var, got, exp))
    assert not bad, (
        "cross-table mismatch (Table 1 vs Table_seasonal_effects_*): " + repr(bad)
    )
    print("  [ok] all cross-table cells agree exactly on the printed strings")


def verify_rounding_residuals(paths):
    """Follow-up 16d: report, do not hide.

    Subtracting the two *printed* 1 km values need not reproduce the printed
    Delta_par, which is computed from the unrounded means. List every cell where
    it does not; those get one caption sentence rather than being hand-matched
    away. Returns the list of (table, block, var, alps, default, printed, naive).
    """
    residuals = []
    for path in paths:
        name = os.path.basename(path)
        blocks = _read_blocks(path, _block_names_of(path))
        for block, vars_ in blocks.items():
            for var in ("GPP", "RECO", "NEE"):
                if var not in vars_:
                    continue
                alps_cell, def_cell = vars_[var][0], vars_[var][1]
                printed = _cell_delta(alps_cell)
                if printed is None:
                    continue
                naive = float(_lead_number(def_cell)) - float(_lead_number(alps_cell))
                if fmt(naive) != fmt(float(printed)):
                    residuals.append((name, block, var, _lead_number(alps_cell),
                                      _lead_number(def_cell), printed, fmt(naive)))
    if residuals:
        print(f"  [note] {len(residuals)} cells where DEFAULT - ALPS on the PRINTED "
              "1 km values does not reproduce the printed Delta_par:")
        for name, block, var, a, d, p, n in residuals:
            print(f"         {name:44s} {block:16s} {var:5s} "
                  f"{a} - {d} = {n}, printed {p}")
    else:
        print("  [ok] every printed Delta_par equals the difference of the two "
              "printed 1 km values")
    return residuals


def _block_names_of(path):
    """Block names of a `Sample & Var` table: seasons for J1/J2, samples else."""
    if "seasonal_domain_averaged" in os.path.basename(path):
        return tuple(n for n, _ in SEASONS)
    return _BLOCK_NAMES


def verify_cams_table(path, regimes):
    """Follow-up 16h: every cell of the CAMS table recomputed from the raw
    timeseries, with the same definition it had as a column of Table 1 / J1 / J2
    -- the CAMS value and its difference to the 1 km ALPS run of the same sample
    and season -- so the move cannot have silently changed a number."""
    p50 = {s: regimes[s]["alps"][ALPS_TAGS.index(MEDIAN_TAG)] for s in ("", "_cloudy")}
    blocks = _read_blocks(path)
    sims = {"Clear sky": [""], "Clouds and rain": ["_cloudy"],
            "24-day average": ["", "_cloudy"]}
    n = 0
    for block, ss in sims.items():
        for var, _lbl in TABLE1_VARIABLES:
            cells = blocks[block][var]
            assert len(cells) == len(_CAMS_COLS), (
                f"{block}/{var}: {len(cells)} columns, expected {len(_CAMS_COLS)}"
            )
            for (cname, months), cell in zip(_CAMS_COLS, cells):
                cams = np.mean([_smean(p50[s], f"{var}_CAMS", months) for s in ss])
                ref = np.mean([_smean(p50[s], f"{var}_1km", months) for s in ss])
                exp = f"{fmt(cams)} {_paren(cams - ref, ref, var)}"
                assert cell == exp, (
                    f"CAMS {block}/{var}/{cname}: on-disk {cell!r} != recomputed "
                    f"{exp!r}"
                )
                n += 1
    print(f"  [ok] CAMS table: all {n} cells reproduce the CAMS value and its "
          "difference to 1 km ALPS, as carried in Table 1 / J1 / J2 before the move")


def report_seasonal_par_ranges(regimes):
    """Follow-up 16e: the parameter effect by season in the new convention.

    The appendix quoted "ALPS reaches 150-160 % of DEFAULT GPP and 174-231 % of
    R_eco in every season", read off J1/J2's old ratio column. Print the min and
    max across the four seasons, per regime, for GPP and R_eco, as absolute
    Delta_par and as the relative percentage |Delta_par| / |DEFAULT|, so that
    sentence can be rewritten from these numbers rather than by subtracting 100
    from the old ones.
    """
    print("\nFollow-up 16e -- seasonal parameter effect in the new convention")
    print("  (Delta_par = DEFAULT - ALPS at 1 km; rel% = |Delta_par|/|DEFAULT|)")
    for sim, label in (("", "clear sky"), ("_cloudy", "clouds and rain")):
        p50 = regimes[sim]["alps"][ALPS_TAGS.index(MEDIAN_TAG)]
        ref = regimes[sim]["ref"]
        print(f"  {label}:")
        for var in ("GPP", "RECO", "NEE"):
            per_season = []
            for name, months in SEASONS:
                a1, d1, _ = _season_cells(p50, ref, months, var)
                per_season.append((name, d1 - a1, abs((d1 - a1) / d1) * 100, d1))
            cells = "  ".join(
                f"{n} {_signed(d)}"
                + (f" [{fmt(p, 0)}%]" if abs(b) >= PCT_MIN_REF else " [--]")
                for n, d, p, b in per_season
            )
            print(f"    {var:5s} {cells}")
            shown = [t for t in per_season if abs(t[3]) >= PCT_MIN_REF]
            dmin = min(per_season, key=lambda t: t[1])
            dmax = max(per_season, key=lambda t: t[1])
            rng = (f"rel {fmt(min(t[2] for t in shown), 0)}-"
                   f"{fmt(max(t[2] for t in shown), 0)}%" if shown else "rel --")
            weak = [t[0] for t in per_season if abs(t[3]) < PCT_MIN_REF]
            print(f"          range: Delta_par {_signed(dmin[1])} ({dmin[0]}) to "
                  f"{_signed(dmax[1])} ({dmax[0]}); {rng}"
                  + (f"  [-- = suppressed, |DEFAULT| < {PCT_MIN_REF}: "
                     f"{', '.join(weak)}]" if weak else ""))


# ---- Natural table width -----------------------------------------------------
# There is no TeX on this host, so this is a metric-model ESTIMATE, not a
# compiler measurement: Computer Modern Roman (cmr) advance widths in em at
# 7 pt (\scriptsize in a 10 pt document), plus LaTeX's 2*\tabcolsep per column
# and \arrayrulewidth per vertical rule. Calibrate the offset against a real
# \showthe\wd before trusting an absolute number; what it is reliable for is
# comparing two tables of the same shape.
_CM_EM = {
    "0": .5, "1": .5, "2": .5, "3": .5, "4": .5, "5": .5, "6": .5, "7": .5,
    "8": .5, "9": .5, ".": .278, ",": .278, "-": .333, "+": .778, "(": .389,
    ")": .389, "[": .278, "]": .278, "%": .833, " ": .333, "~": .333, "/": .5,
    "A": .75, "B": .708, "C": .722, "D": .764, "E": .681, "F": .653, "G": .785,
    "H": .75, "I": .361, "J": .514, "K": .778, "L": .625, "M": .917, "N": .75,
    "O": .778, "P": .681, "Q": .778, "R": .736, "S": .556, "T": .722, "U": .75,
    "V": .75, "W": 1.028, "X": .75, "Y": .75, "Z": .611,
    "a": .5, "b": .556, "c": .444, "d": .556, "e": .444, "f": .306, "g": .5,
    "h": .556, "i": .278, "j": .306, "k": .528, "l": .278, "m": .833, "n": .556,
    "o": .5, "p": .556, "q": .528, "r": .392, "s": .394, "t": .389, "u": .556,
    "v": .528, "w": .722, "x": .528, "y": .528, "z": .444,
}
_SCRIPT_RATIO = 0.7  # \scriptstyle relative size, for the subscripted labels
_FS_SCRIPTSIZE = 7.0  # pt
_ARRAYRULE = 0.4  # pt
# copernicus.cls does not leave \tabcolsep at the 6 pt article default. Rather
# than guess, this is back-solved from the four hand-built drafts, whose widths
# were measured with a real \showthe\wd: the raw model (6 pt) came out
# +36.0/+37.6/+36.8/+29.8 pt high on tables of 8/8/8/7 columns, i.e. a flat
# ~4.5 pt per column, which is 2*(6 - 3.75). With this value the model
# reproduces all four measurements to within 2 pt -- see _DRAFT_WIDTHS.
_TABCOLSEP = 3.75  # pt

# The compiler-measured natural widths of the hand-built drafts, at \scriptsize.
# Kept as the calibration reference for estimate_table_width: the model is
# trustworthy for comparing tables of this shape, not as an absolute measurement.
_DRAFT_WIDTHS = {
    "Table_1_domain_averaged_2012.tex": 469.1,
    "Table_seasonal_domain_averaged_2012.tex": 446.1,
    "Table_seasonal_domain_averaged_2012_cloudy.tex": 439.9,
    "Table_CAMS_domain_averaged_2012.tex": 455.5,
}

# Labels whose set width is not their source length. Widths in em, built from
# the same cmr metrics: base glyph + subscript at _SCRIPT_RATIO, and cmsy's
# \downarrow at 0.5 em.
_LABEL_EM = {
    r"$T_\text{2m}$": .722 + _SCRIPT_RATIO * (.5 + .833),
    r"S$\downarrow$": .556 + .5,
    r"R$_{\text{eco}}$": .736 + _SCRIPT_RATIO * (.444 + .444 + .5),
    r"R$_\text{eco}$": .736 + _SCRIPT_RATIO * (.444 + .444 + .5),
}


def _text_width_pt(s):
    """Estimated set width of one already-stripped cell string, in pt."""
    if s in _LABEL_EM:
        return _LABEL_EM[s] * _FS_SCRIPTSIZE
    em = sum(_CM_EM.get(c, .5) for c in s)
    return em * _FS_SCRIPTSIZE


def estimate_table_width(path):
    """Estimated natural width of a bare tabular at \\scriptsize, in pt.

    Merged (\\multicolumn{2}) cells are charged to the pair they span, which is
    how TeX resolves them here: they are narrower than the two flux cells below,
    so they never set the column width and the approximation costs nothing.
    """
    lines = open(path).read().splitlines()
    spec = next(l for l in lines if l.startswith("\\begin{tabular}"))
    spec = spec.split("{", 2)[2].rstrip("}")
    ncol = sum(1 for c in spec if c in "lrc")
    nrule = spec.count("|")

    widths = [0.0] * ncol
    for l in lines:
        if not l.endswith("\\\\"):
            continue
        cells = [c.strip() for c in l.rstrip("\\").rstrip().split("&")]
        i = 0
        for cell in cells:
            m = _MULTICOL_RE.match(cell)
            span = int(m.group(1)) if m else 1
            w = _text_width_pt(_unwrap(cell)) / span
            for k in range(i, min(i + span, ncol)):
                widths[k] = max(widths[k], w)
            i += span
    return sum(widths) + 2 * _TABCOLSEP * ncol + _ARRAYRULE * nrule


def report_table_widths(paths, textblock_pt=503.6):
    """Estimated natural width of each table against the text block and against
    the measured width of the hand-built draft of the same shape."""
    print(f"\nEstimated natural width at \\scriptsize (text block {textblock_pt} pt)")
    print("  est'd      vs text block   draft   vs draft   table")
    for p in paths:
        name = os.path.basename(p)
        w = estimate_table_width(p)
        draft = _DRAFT_WIDTHS.get(name)
        flag = "OVERFULL" if w > textblock_pt else "fits"
        d = f"{draft:7.1f}  {w - draft:+7.1f}" if draft else f"{'--':>7s}  {'--':>7s}"
        print(f"  {w:7.1f} pt  {flag:8s}     {d}   {name}")
    print("  (metric model, not a compilation -- no TeX on this host; calibrated "
          "on the four\n   draft measurements, so read the 'vs draft' column, not "
          "the absolute number)")


def verify_no_t2_percent(files):
    """Check (Follow-up 12d): no T2 row in any of the three domain-averaged
    tables (Table 1, or either seasonal domain-averaged table) contains a
    percentage bracket. A ratio of Celsius values depends on where the zero
    of the scale sits and is not a physical quantity -- the near-zero winter
    domain mean is exactly what sent it to 600-1800% before this fix."""
    t2_label = r"$T_\text{2m}$"
    total_checked = 0
    for path in files:
        lines = open(path).read().splitlines()
        t2_lines = [l for l in lines if t2_label in l]
        assert t2_lines, f"no T2 rows found in {path} -- label pattern may have changed"
        for l in t2_lines:
            assert "%" not in l, f"T2 row still has a % bracket in {path}: {l}"
        total_checked += len(t2_lines)
    print(f"  [ok] no '%' character in any T2 row across {len(files)} domain-averaged "
          f"tables ({total_checked} rows checked)")


def verify_seasonal_vs_annual(regimes, tol=0.005):
    """Check 6: the four seasonal means average to the annual (Table 1) value.

    Reported per regime for every variable and column. Returns a list of
    (regime, variable, column, seasonal_mean, annual) mismatches.
    """
    bad = []
    for sim, R in regimes.items():
        p50 = R["alps"][ALPS_TAGS.index(MEDIAN_TAG)]
        ref = R["ref"]
        for var, _ in TABLE1_VARIABLES:
            for res in ("1km", "9km", "54km"):
                for key, df in (("ALPS", p50), ("DF", ref)):
                    seas = [_smean(df, f"{var}_{res}", m) for _, m in SEASONS]
                    got = float(np.mean(seas))
                    annual = float(R["means"].loc[var, (key, res)])
                    if abs(got - annual) > tol:
                        bad.append((sim or "clear", var, f"{key}/{res}", got, annual))
    return bad


def _smean(df, col, months):
    """24-h mean of the seasonal diurnal cycle of `col` (same aggregation as the
    annual table's hourly_avg.mean())."""
    s = _season_diurnal(df, col, months)
    return float(s.mean()) if s is not None else float("nan")


def write_seasonal_domain_average_table(ALPS_dfs, df_ref, outfile):
    """The domain-averaged table per meteorological season, one file per regime
    (clear/cloudy) -- Tables J1 and J2 of the appendix.

    Same layout and same conventions as Table 1 (`write_domain_average_table`),
    so the two read as one family: `Sample & Var` label columns with the season
    in column 1 of its block's first row, the 1/9/54 km group header once at the
    top, T2 and S-down merged across the ALPS/DEFAULT pair ahead of the
    sub-header, and the 1 km ALPS cell carrying Delta_par = DEFAULT - ALPS
    rather than the old |ALPS|/|DEFAULT| ratio (Follow-up 16b). CAMS has moved
    to its own table (`write_cams_table`).
    """
    p50 = ALPS_dfs[ALPS_TAGS.index(MEDIAN_TAG)]
    with open(outfile, "w") as f:
        f.write("\\begin{tabular}{ll|rr|rr|rr}\n\\toprule\n")
        f.write(_T1_GROUP_HDR.replace("Sample", "Season"))
        f.write("\\midrule\n")
        for i, (name, months) in enumerate(SEASONS):
            if i:
                f.write("\\midrule\n")
            cells = {
                var: _season_cells(p50, df_ref, months, var)
                for var, _ in TABLE1_VARIABLES
            }
            # No bolding here: the appendix tables carry no "bold marks the NEE
            # resolution difference" convention, only Table 1 does.
            _t1_block(f, name, cells, bold_nee=False, annotate_hdr=False)
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX table written to {outfile}")


_CAMS_COLS = [(name, months) for name, months in SEASONS] + [("Year", ALL_MONTHS)]


def write_cams_table(regimes, outfile):
    """CAMS on its own (Follow-up 16h), pulled out of Table 1 and J1/J2.

    Same `Sample & Var` shape as the other three, but the columns are the four
    seasons plus the year and there is nothing to split, so no ALPS/DEFAULT
    sub-header and no \\cmidrule. Each cell is the CAMS value with its
    difference to the 1 km ALPS run of the same sample and season; the 24-day
    block is the mean of the two regimes, CAMS and reference alike.

    Putting every winter cell side by side is what made the near-zero-denominator
    pathology impossible to miss ([245%], [317%], [2400%]); those brackets are
    suppressed by `_paren` below PCT_MIN_REF.

    Parameters
    ----------
    regimes : {"": clear, "_cloudy": cloudy} as built in main(), each carrying
        the ALPS tag dfs under "alps".
    """
    p50 = {s: regimes[s]["alps"][ALPS_TAGS.index(MEDIAN_TAG)] for s in ("", "_cloudy")}

    def _pair(sim, var, months):
        """(CAMS value, 1 km ALPS reference) for one regime, season, variable."""
        d = p50[sim]
        return (
            _smean(d, f"{var}_CAMS", months),
            _smean(d, f"{var}_1km", months),
        )

    samples = [("Clear sky", ""), ("Clouds and rain", "_cloudy"), ("24-day average", None)]
    with open(outfile, "w") as f:
        f.write("\\begin{tabular}{ll|rrrrr}\n\\toprule\n")
        f.write("Sample & Var & " + " & ".join(n for n, _ in _CAMS_COLS) + " \\\\\n")
        f.write("\\midrule\n")
        for bi, (sample, sim) in enumerate(samples):
            if bi:
                f.write("\\midrule\n")
            for vi, (var, label) in enumerate(TABLE1_VARIABLES):
                row = f"{sample if vi == 0 else ''} & {label}"
                for _name, months in _CAMS_COLS:
                    if sim is None:
                        pairs = [_pair(s, var, months) for s in ("", "_cloudy")]
                        cams = sum(p[0] for p in pairs) / 2.0
                        ref = sum(p[1] for p in pairs) / 2.0
                    else:
                        cams, ref = _pair(sim, var, months)
                    row += f" & {fmt(cams)} {_paren(cams - ref, ref, var)}"
                f.write(row + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX table written to {outfile}")


_SEASONAL_EFFECTS_FLUXES = [("GPP", "GPP"), ("RECO", r"$R_\text{eco}$"), ("NEE", "NEE")]


def _member_tagmeans(member_dfs, var, months):
    """Per-member seasonal mean of var_1km, in ALPS_TAGS order (index 0=p10,
    1=p25, 2=p50, 3=p75, 4=p90). Single source of truth for the per-member
    level values -- used by the level-spread row, the p50-deviation table,
    and the p50-rank report, so all three read the exact same numbers."""
    return [_smean(d, f"{var}_1km", months) for d in member_dfs]


def _member_dres(member_dfs, var, months):
    """Per-member Delta_res (54km - 1km) seasonal mean, in ALPS_TAGS order
    (index 2 = p50). Single source of truth for the dres-spread row, the
    dres p50-deviation table, and the p50-rank report."""
    return [_smean(d, f"{var}_54km", months) - _smean(d, f"{var}_1km", months)
            for d in member_dfs]


def _seasonal_effects_header(include_dres_iqr=False):
    """Fixed tabular header for the seasonal-effects tables (shared by the
    per-regime and the pooled/combined writer). Every flux carries Delta_res
    at both 9 km and 54 km (coarse - 1km, p50) plus Delta_par (DEFAULT - ALPS,
    1km, p50; ALPS subtracted, Eq. 2). GPP and NEE additionally carry the two
    Topt-percentile-member variants, Delta_par^p25/p75 -- RECO has no Topt
    dependence (alpha/beta unvaried across members) so its two slots are
    omitted rather than published as structural zeros. GPP carries sigma_IQR
    (the 1km-level member spread) always, plus sigma^Delta_IQR (the Delta_res
    member spread, Follow-up 9b) as a further column when include_dres_iqr --
    only the combined/'_all' table gets that. R_eco and NEE carry no spread
    column at all (Follow-up 11): R_eco's is structurally zero and NEE's is
    identical to GPP's (NEE = RECO - GPP, RECO invariant), both asserted in
    _verify_seasonal_effects_rows rather than published redundantly. No
    sigma_env anywhere (dropped project-wide, per Follow-up 11)."""
    gpp_cols = 7 if include_dres_iqr else 6
    colspec = "r" * gpp_cols + "|rrr|rrrrr"
    dres = r"$\Delta_\text{res}^{9}$ & $\Delta_\text{res}^{54}$"
    par_pct = r"$\Delta_\text{par}^{p25}$ & $\Delta_\text{par}^{p75}$"
    gpp_header = f"{dres} & $\\Delta_\\text{{par}}$ & {par_pct} & $\\sigma_\\text{{IQR}}$"
    if include_dres_iqr:
        gpp_header += r" & $\sigma^{\Delta}_\text{IQR}$"
    reco_header = f"{dres} & $\\Delta_\\text{{par}}$"
    nee_header = f"{dres} & $\\Delta_\\text{{par}}$ & {par_pct}"
    return (
        f"\\begin{{tabular}}{{l|{colspec}}}\n\\toprule\n"
        f"& \\multicolumn{{{gpp_cols}}}{{c|}}{{GPP}} & \\multicolumn{{3}}{{c|}}"
        "{$R_\\text{eco}$} & \\multicolumn{5}{c}{NEE} \\\\\n"
        f"Season & {gpp_header} & {reco_header} & {nee_header}"
        " \\\\\n\\midrule\n"
    )


def _seasonal_effects_row(name, months, p50, ref, member_dfs, include_dres_iqr=False):
    """One formatted row: resolution effects at 9 km and 54 km (coarse-1km,
    p50) and the parameter effect Delta_par = DEFAULT - ALPS (1km, p50) for
    all three fluxes -- Delta_par^s = x_s - x_ALPS, ALPS subtracted throughout
    (Eq. 2); plain Delta_par is the DEFAULT member (s = DEFAULT). GPP and NEE
    also carry the two Topt-percentile-member variants, Delta_par^p25 =
    x(p25 tag) - x(p50 tag) and Delta_par^p75 = x(p75 tag) - x(p50 tag) (RECO
    omitted -- no Topt dependence). GPP also carries its ensemble-spread
    statistic(s): sigma_IQR (1km level; the darker band in Figs. 5/7) and,
    when include_dres_iqr, sigma^Delta_IQR (Delta_res member spread, at
    54 km). All as 24-h seasonal means [umol m-2 s-1].

    With exactly 5 members, numpy's linear interpolation places percentile 25
    at sorted-index 1.0 and percentile 75 at sorted-index 3.0 -- i.e.
    sigma_IQR is the gap between the 2nd- and 4th-ranked member BY FLUX
    VALUE, not necessarily the p25-tagged and p75-tagged members: the flux
    response to T_opt is not monotonic, so the p50 (tag) member need not sit
    at rank 2 (see Table_member_deviations_* and the p50-rank report in
    main() for whether it actually does here) -- and neither need
    Delta_par^p25/p75, for the same reason.
    """
    row = name
    for var, _lbl in _SEASONAL_EFFECTS_FLUXES:
        dres9 = _smean(p50, f"{var}_9km", months) - _smean(p50, f"{var}_1km", months)
        dres54 = _smean(p50, f"{var}_54km", months) - _smean(p50, f"{var}_1km", months)
        dpar = _smean(ref, f"{var}_1km", months) - _smean(p50, f"{var}_1km", months)
        vals = [dres9, dres54, dpar]
        tagmeans = _member_tagmeans(member_dfs, var, months) if var != "RECO" else None
        if tagmeans is not None:
            vals += [tagmeans[1] - tagmeans[2], tagmeans[3] - tagmeans[2]]
        for v in vals:
            s = fmt(v)
            row += f" & {s if s.startswith('-') else '+' + s}"
        if var == "GPP":
            iqr = np.nanpercentile(tagmeans, 75) - np.nanpercentile(tagmeans, 25)
            row += f" & {fmt(iqr)}"
            if include_dres_iqr:
                dres_m = _member_dres(member_dfs, var, months)
                iqr_dres = np.nanpercentile(dres_m, 75) - np.nanpercentile(dres_m, 25)
                row += f" & {fmt(iqr_dres)}"
    return row + " \\\\\n"


def write_seasonal_effects_table(ALPS_dfs, df_ref, outfile, rows=None):
    """Seasonal effects table for one regime (clear or cloudy), plus a
    trailing 'Year' row (the annual mean over this regime's own 12
    representative days). 7 numeric columns: GPP gets Delta_res/Delta_par/
    sigma_IQR, R_eco and NEE get Delta_res/Delta_par only."""
    if rows is None:
        rows = list(SEASONS) + [("Year", ALL_MONTHS)]
    p50 = ALPS_dfs[ALPS_TAGS.index(MEDIAN_TAG)]
    with open(outfile, "w") as f:
        f.write(_seasonal_effects_header(include_dres_iqr=False))
        for name, months in rows:
            f.write(_seasonal_effects_row(name, months, p50, df_ref, ALPS_dfs, include_dres_iqr=False))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX table written to {outfile}")


def write_seasonal_effects_table_combined(alps_clear, ref_clear, alps_cloudy, ref_cloudy, outfile):
    """Seasonal effects table pooling all 24 representative days (12 clear +
    12 cloudy), plus a trailing 'Year' row (the annual mean over all 24 days).
    8 numeric columns: GPP gets Delta_res/Delta_par/sigma_IQR/sigma^Delta_IQR
    (the last one merged in from Follow-up 9b, no longer a separate table),
    R_eco and NEE get Delta_res/Delta_par only.

    Delta_res/Delta_par pool linearly: each regime contributes an equal number
    of days per season (and, for the Year row, per year), so recomputing on
    the concatenated raw timeseries reproduces the mean of the two regimes'
    values exactly. sigma_IQR/sigma^Delta_IQR do NOT pool linearly -- they are
    percentiles of the 5 members' seasonal means, so each member's clear+cloudy
    days are concatenated FIRST, the seasonal mean taken per (pooled) member,
    and only then are the percentiles taken across the 5 pooled member-means.
    """
    p50_pooled = pd.concat([
        alps_clear[ALPS_TAGS.index(MEDIAN_TAG)], alps_cloudy[ALPS_TAGS.index(MEDIAN_TAG)]
    ])
    ref_pooled = pd.concat([ref_clear, ref_cloudy])
    pooled_members = [pd.concat([c, d]) for c, d in zip(alps_clear, alps_cloudy)]
    rows = list(SEASONS) + [("Year", ALL_MONTHS)]
    with open(outfile, "w") as f:
        f.write(_seasonal_effects_header(include_dres_iqr=True))
        for name, months in rows:
            f.write(_seasonal_effects_row(name, months, p50_pooled, ref_pooled, pooled_members, include_dres_iqr=True))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX table written to {outfile}")




# ---------------------------------------------------------------------------
# Paper view of the seasonal-effects tables.
#
# The canonical writer above emits every column the analysis produces, in
# analysis order, and the verifiers read that file. The manuscript prints a
# subset in a different order: per flux Delta_par^p25, Delta_par^p75,
# Delta_par, Delta_res^9km, Delta_res^54km (R_eco only the last three, having
# no T_opt dependence), and no sigma columns. Emitting that view here is what
# lets the archived workflow regenerate the file the manuscript \input{}s,
# rather than a differently-shaped table a reader has to reconcile by hand.
# The canonical table keeps its own name (*_full.tex) and stays the verified
# artefact; _assert_paper_view_subset checks the published view against it.
_PAPER_VIEW_ORDER = {"GPP": [3, 4, 2, 0, 1], "RECO": [2, 0, 1], "NEE": [3, 4, 2, 0, 1]}


def _paper_view_header():
    dres = r"$\Delta_\text{res}^\text{9km}$ & $\Delta_\text{res}^\text{54km}$"
    par_pct = r"$\Delta_\text{par}^{p25}$ & $\Delta_\text{par}^{p75}$"
    full = f"{par_pct} & $\\Delta_\\text{{par}}$ & {dres}"
    reco = f"$\\Delta_\\text{{par}}$ & {dres}"
    return (
        "\\begin{tabular}{l|rrrrr|rrr|rrrrr}\n\\toprule\n"
        "& \\multicolumn{5}{c|}{GPP} & \\multicolumn{3}{c|}{R$_\\text{eco}$}"
        " & \\multicolumn{5}{c}{NEE} \\\\\n"
        f"Season & {full} & {reco} & {full}"
        " \\\\\n\\midrule\n"
    )


def _paper_view_row(name, months, p50, ref, member_dfs):
    """Same numbers as _seasonal_effects_row, in the manuscript's column order
    and without the sigma columns."""
    row = name
    for var, _lbl in _SEASONAL_EFFECTS_FLUXES:
        dres9 = _smean(p50, f"{var}_9km", months) - _smean(p50, f"{var}_1km", months)
        dres54 = _smean(p50, f"{var}_54km", months) - _smean(p50, f"{var}_1km", months)
        dpar = _smean(ref, f"{var}_1km", months) - _smean(p50, f"{var}_1km", months)
        vals = [dres9, dres54, dpar]
        if var != "RECO":
            tm = _member_tagmeans(member_dfs, var, months)
            vals += [tm[1] - tm[2], tm[3] - tm[2]]
        for i in _PAPER_VIEW_ORDER[var]:
            s = fmt(vals[i])
            if s.lstrip('-').strip('0.') == '':   # -0.00 and 0.00 print alike
                s = s.lstrip('-')
            row += f" & {s if s.startswith('-') else '+' + s}"
    return row + " \\\\\n"


def write_seasonal_effects_table_paper(p50, ref, member_dfs, outfile, last_label):
    rows = list(SEASONS) + [(last_label, ALL_MONTHS)]
    with open(outfile, "w") as f:
        f.write(_paper_view_header())
        for i, (name, months) in enumerate(rows):
            if i == len(rows) - 1:      # separate the pooled row from the seasons
                f.write("\\midrule\n")
            f.write(_paper_view_row(name, months, p50, ref, member_dfs))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX table (paper view) written to {outfile}")


def _assert_paper_view_subset(paper_file, full_file):
    """Every number in the published view must appear in the verified canonical
    table for the same row, so the reordering cannot silently alter a value."""
    def cells(path):
        out = []
        for line in open(path):
            line = line.strip()
            if "&" not in line or line.startswith("\\") or "multicolumn" in line:
                continue
            if line.startswith("Season"):   # header: labels differ by design
                continue
            parts = [c.strip() for c in line.rstrip("\\\\").split("&")]
            if len(parts) < 4:
                continue
            out.append((parts[0], sorted(c for c in parts[1:] if c)))
        return out
    pv, fu = cells(paper_file), cells(full_file)
    assert len(pv) == len(fu), f"paper view has {len(pv)} rows, canonical {len(fu)}"
    for (n1, c1), (n2, c2) in zip(pv, fu):
        missing = [c for c in c1 if c not in c2]
        assert not missing, f"row {n1}: {missing} not in canonical row {n2}"
    print(f"  [ok] paper view {paper_file.split('/')[-1]} is a subset of the verified table")


def _verify_seasonal_effects_rows(p50, ref, member_dfs, rows, outfile, include_dres_iqr=False):
    """Shared row-level assertions for the seasonal-effects tables (per-regime
    and pooled/combined alike): Delta_res (9/54 km)/Delta_par/Delta_par^p25/p75
    match what's on disk for GPP and NEE (RECO: Delta_res/Delta_par only, no
    Topt dependence); if `rows` includes a 'Year' row, each of those matches
    the mean of the four seasonal rows' on-disk values (each season is an
    equal-size day-group -- 3 days for the per-regime tables, 6 for the pooled
    one -- so this holds exactly to rounding; see
    write_seasonal_effects_table_combined's docstring); GPP's on-disk
    sigma_IQR (and sigma^Delta_IQR when include_dres_iqr) matches
    recomputation, sigma_IQR matches the exact-index shortcut valid for n=5;
    R_eco's member spread is exactly zero (alpha/beta unvaried, not
    published); NEE's member spread equals GPP's exactly (NEE = RECO - GPP,
    RECO invariant across members, not published) -- both checked here since
    that equality is the whole justification for Follow-up 11 dropping their
    columns, not assumed. No sigma_env anywhere (dropped project-wide)."""
    fluxes = ["GPP", "RECO", "NEE"]
    gpp_cols = 7 if include_dres_iqr else 6
    offsets = [0, gpp_cols, gpp_cols + 3]  # start index of each flux's block
    season_names = dict(SEASONS)
    lines = open(outfile).read().splitlines()
    season_vals = {var: {k: [] for k in ("dres9", "dres54", "dpar", "dpar25", "dpar75")}
                   for var in fluxes}
    for name, months in rows:
        disk_row = next((l for l in lines if l.startswith(name + " &")), None)
        assert disk_row is not None, f"row '{name}' not found in {outfile}"
        cells = _numeric_cells(disk_row)

        gpp_iqr_recomp = None
        for vi, var in enumerate(fluxes):
            off = offsets[vi]
            dres9 = _smean(p50, f"{var}_9km", months) - _smean(p50, f"{var}_1km", months)
            dres54 = _smean(p50, f"{var}_54km", months) - _smean(p50, f"{var}_1km", months)
            dpar = _smean(ref, f"{var}_1km", months) - _smean(p50, f"{var}_1km", months)
            got = {"dres9": cells[off], "dres54": cells[off + 1], "dpar": cells[off + 2]}
            exp = {"dres9": dres9, "dres54": dres54, "dpar": dpar}

            tagmeans_unsorted = _member_tagmeans(member_dfs, var, months)
            if var != "RECO":
                got["dpar25"], got["dpar75"] = cells[off + 3], cells[off + 4]
                exp["dpar25"] = tagmeans_unsorted[1] - tagmeans_unsorted[2]
                exp["dpar75"] = tagmeans_unsorted[3] - tagmeans_unsorted[2]

            for k, e in exp.items():
                exp_s = fmt(e)
                exp_s = exp_s if exp_s.startswith("-") else "+" + exp_s
                assert got[k] == exp_s, (
                    f"{name} {var} {k}: on-disk {got[k]} != recomputed {exp_s} "
                    "-- Delta_res/Delta_par changed"
                )

            tagmeans = sorted(tagmeans_unsorted)
            assert len(tagmeans) == 5, f"expected 5 members, got {len(tagmeans)}"
            iqr = np.nanpercentile(tagmeans, 75) - np.nanpercentile(tagmeans, 25)
            assert abs(iqr - (tagmeans[3] - tagmeans[1])) < 1e-9, (
                f"{name} {var}: sigma_IQR != sorted[3]-sorted[1] "
                f"({iqr} vs {tagmeans[3] - tagmeans[1]})"
            )
            sigma_off = off + (5 if var != "RECO" else 3)
            if var == "RECO":
                assert abs(iqr) < 1e-9, (
                    f"{name} RECO: expected zero member spread (alpha/beta not "
                    f"varied), got IQR={iqr}"
                )
            elif var == "GPP":
                gpp_iqr_recomp = iqr
                got_gpp_iqr = cells[sigma_off]
                assert got_gpp_iqr == fmt(iqr), (
                    f"{name} GPP sigma_IQR: on-disk {got_gpp_iqr} != recomputed "
                    f"{fmt(iqr)}"
                )
                if include_dres_iqr:
                    dres_m = sorted(_member_dres(member_dfs, var, months))
                    iqr_dres = np.nanpercentile(dres_m, 75) - np.nanpercentile(dres_m, 25)
                    got_gpp_iqr_dres = cells[sigma_off + 1]
                    assert got_gpp_iqr_dres == fmt(iqr_dres), (
                        f"{name} GPP sigma^Delta_IQR: on-disk {got_gpp_iqr_dres} "
                        f"!= recomputed {fmt(iqr_dres)}"
                    )
            elif var == "NEE":
                assert abs(iqr - gpp_iqr_recomp) < 1e-9, (
                    f"{name} NEE member spread {iqr} != GPP member spread "
                    f"{gpp_iqr_recomp} -- expected identical (RECO invariant "
                    "across members)"
                )

            if name == "Year":
                for k, g in got.items():
                    exp_mean = float(np.mean(season_vals[var][k]))
                    got_f = float(g)
                    assert abs(got_f - exp_mean) <= 0.010001, (
                        f"Year {var} {k}: on-disk {got_f}, mean of 4 seasons "
                        f"{exp_mean:.4f} (off by {abs(got_f - exp_mean):.4f})"
                    )
            elif name in season_names:
                for k, g in got.items():
                    season_vals[var][k].append(float(g))


def verify_seasonal_effects_spread(ALPS_dfs, df_ref, outfile, rows=None):
    """Check 7: GPP's sigma_IQR matches the exact-index shortcut valid for
    n=5 and the on-disk value; R_eco's member spread is exactly zero; NEE's
    equals GPP's; Delta_res/Delta_par are unchanged from what's on disk; and
    the Year row equals the mean of the four seasonal rows."""
    if rows is None:
        rows = list(SEASONS) + [("Year", ALL_MONTHS)]
    p50 = ALPS_dfs[ALPS_TAGS.index(MEDIAN_TAG)]
    _verify_seasonal_effects_rows(p50, df_ref, ALPS_dfs, rows, outfile, include_dres_iqr=False)
    print(
        "  [ok] seasonal effects: GPP sigma_IQR = sorted[3]-sorted[1] for n=5 "
        "and matches disk, R_eco member spread exactly zero, NEE = GPP member "
        "spread, Delta_res/Delta_par unchanged, Year row = mean of 4 seasons"
    )


def verify_seasonal_effects_spread_combined(alps_clear, ref_clear, alps_cloudy, ref_cloudy, outfile):
    """Checks for the pooled/combined seasonal-effects table:
    1. All of _verify_seasonal_effects_rows's per-row checks (including the
       Year-row-equals-mean-of-seasons check), run against the pooled data.
    2. Linear-pooling check: Delta_res/Delta_par at every row (4 seasons +
       Year) equal the mean of the two regime tables' own values for that row.
    3. Nonlinearity evidence: sigma_IQR does NOT in general equal the mean of
       the two regimes' sigma_IQR -- report the per-cell differences, and
       assert at least one is non-negligible, as a regression guard against
       the combined table ever being reimplemented as a naive average of the
       two regime tables.
    """
    p50_pooled = pd.concat([
        alps_clear[ALPS_TAGS.index(MEDIAN_TAG)], alps_cloudy[ALPS_TAGS.index(MEDIAN_TAG)]
    ])
    ref_pooled = pd.concat([ref_clear, ref_cloudy])
    pooled_members = [pd.concat([c, d]) for c, d in zip(alps_clear, alps_cloudy)]
    rows = list(SEASONS) + [("Year", ALL_MONTHS)]

    _verify_seasonal_effects_rows(p50_pooled, ref_pooled, pooled_members, rows, outfile, include_dres_iqr=True)

    fluxes = ["GPP", "RECO", "NEE"]
    p50_clear = alps_clear[ALPS_TAGS.index(MEDIAN_TAG)]
    p50_cloudy = alps_cloudy[ALPS_TAGS.index(MEDIAN_TAG)]
    max_iqr_diff = 0.0
    diff_report = []
    for name, months in rows:
        for var in fluxes:
            dres_c = _smean(p50_clear, f"{var}_54km", months) - _smean(p50_clear, f"{var}_1km", months)
            dpar_c = _smean(p50_clear, f"{var}_1km", months) - _smean(ref_clear, f"{var}_1km", months)
            dres_d = _smean(p50_cloudy, f"{var}_54km", months) - _smean(p50_cloudy, f"{var}_1km", months)
            dpar_d = _smean(p50_cloudy, f"{var}_1km", months) - _smean(ref_cloudy, f"{var}_1km", months)
            dres_pooled = _smean(p50_pooled, f"{var}_54km", months) - _smean(p50_pooled, f"{var}_1km", months)
            dpar_pooled = _smean(p50_pooled, f"{var}_1km", months) - _smean(ref_pooled, f"{var}_1km", months)
            assert abs(dres_pooled - (dres_c + dres_d) / 2) <= 0.010001, (
                f"{name} {var} Delta_res: pooled {dres_pooled:.4f} != mean of "
                f"regimes {(dres_c + dres_d) / 2:.4f}"
            )
            assert abs(dpar_pooled - (dpar_c + dpar_d) / 2) <= 0.010001, (
                f"{name} {var} Delta_par: pooled {dpar_pooled:.4f} != mean of "
                f"regimes {(dpar_c + dpar_d) / 2:.4f}"
            )

            tm_c = sorted(_member_tagmeans(alps_clear, var, months))
            tm_d = sorted(_member_tagmeans(alps_cloudy, var, months))
            tm_pooled = sorted(_member_tagmeans(pooled_members, var, months))
            iqr_c = np.nanpercentile(tm_c, 75) - np.nanpercentile(tm_c, 25)
            iqr_d = np.nanpercentile(tm_d, 75) - np.nanpercentile(tm_d, 25)
            iqr_pooled = np.nanpercentile(tm_pooled, 75) - np.nanpercentile(tm_pooled, 25)
            diff = iqr_pooled - (iqr_c + iqr_d) / 2
            max_iqr_diff = max(max_iqr_diff, abs(diff))
            diff_report.append((name, var, iqr_c, iqr_d, iqr_pooled, diff))

    print("  sigma_IQR: pooled vs mean(clear, cloudy) -- evidence pooling used members, not summary stats")
    print(f"  {'row':5s} {'var':5s} {'IQR_clear':>10s} {'IQR_cloudy':>10s} {'IQR_pooled':>10s} {'IQR_pooled-mean(regimes)':>26s}")
    for name, var, ic, idd, ip, diff in diff_report:
        print(f"  {name:5s} {var:5s} {ic:10.4f} {idd:10.4f} {ip:10.4f} {diff:+26.4f}")
    assert max_iqr_diff > 1e-6, (
        "sigma_IQR pooled values match mean(clear, cloudy) everywhere -- "
        "suspicious: expected genuine nonlinearity from member-level pooling"
    )
    print(
        f"  [ok] combined seasonal effects: per-row checks pass, Delta_res/Delta_par "
        f"pool linearly (match mean of regimes), sigma_IQR does not "
        f"(max |diff| = {max_iqr_diff:.4f})"
    )


def print_dres_iqr_ratio(outfile):
    """Ratio of GPP's two spread definitions, both now published as the last
    two columns of GPP's block in the same combined table (Follow-up 11
    merged what was Table_seasonal_effects_dres_2012_all.tex, Follow-up 9b,
    into this file -- that standalone table is no longer produced). Read
    straight off disk, not recomputed, so the ratio matches what the paper
    would actually cite."""
    lines = open(outfile).read().splitlines()
    rows = list(SEASONS) + [("Year", ALL_MONTHS)]
    print("  sigma^Delta_IQR(GPP) / sigma_IQR(GPP) -- ratio of the two spread definitions")
    print(f"  {'row':5s} {'IQR_dres':>10s} {'IQR_level':>10s} {'ratio':>8s}")
    for name, _months in rows:
        row = next((l for l in lines if l.startswith(name + " &")), None)
        assert row is not None, f"row '{name}' not found in {outfile}"
        cells = _numeric_cells(row)
        iqr_level, iqr_dres = float(cells[5]), float(cells[6])
        ratio = iqr_dres / iqr_level if iqr_level not in (0, float("nan")) else float("nan")
        print(f"  {name:5s} {iqr_dres:10.4f} {iqr_level:10.4f} {ratio:8.3f}")


def _within_between_daily(p50_clear, p50_cloudy):
    """Pooled p50 ALPS data (24 days: 12 clear + 12 cloudy), collapsed to one
    24h mean per calendar day. Shared by the writer and the verifier so both
    start from the same daily series."""
    p50_pooled = pd.concat([p50_clear, p50_cloudy])
    daily = p50_pooled.groupby(p50_pooled.index.date).mean(numeric_only=True)
    daily.index = pd.to_datetime(daily.index)
    return daily


def write_within_vs_between_season_table(p50_clear, p50_cloudy, outfile):
    """RC2 promise (Follow-up 9e): show that the day-to-day spread within a
    season is smaller than the spread between seasons' means. Per flux
    (1km ALPS, 24h daily means, 6 days per season = 3 clear + 3 cloudy):
    within-season std/range across the 6 days, the seasonal mean itself, and
    the between-season std/range of the four seasonal means.
    """
    daily = _within_between_daily(p50_clear, p50_cloudy)
    lines = [
        "\\begin{tabular}{l|rrr}\n\\toprule\n",
        "Season & mean & within-season std & within-season range \\\\\n\\midrule\n",
    ]
    for var, lbl in _SEASONAL_EFFECTS_FLUXES:
        col = f"{var}_1km"
        lines.append(f"\\multicolumn{{4}}{{l}}{{{lbl}}} \\\\\n")
        season_means = []
        for name, months in SEASONS:
            vals = daily.loc[daily.index.month.isin(months), col].dropna()
            assert len(vals) == 6, f"{name} {var}: expected 6 days, got {len(vals)}"
            season_means.append(float(vals.mean()))
            lines.append(
                f"{name} & {fmt(vals.mean())} & {fmt(vals.std())} "
                f"& {fmt(vals.max() - vals.min())} \\\\\n"
            )
        lines.append(
            f"\\emph{{between-season}} & -- & {fmt(np.std(season_means, ddof=1))} "
            f"& {fmt(max(season_means) - min(season_means))} \\\\\n\\midrule\n"
        )
    lines[-1] = lines[-1].replace("\\midrule\n", "")  # no trailing rule before \bottomrule
    lines.append("\\bottomrule\n\\end{tabular}\n")
    with open(outfile, "w") as f:
        f.writelines(lines)
    print(f"LaTeX table written to {outfile}")


def verify_within_vs_between_season_table(p50_clear, p50_cloudy, outfile):
    """Check 9: every season has exactly 6 daily values; the on-disk mean/
    within-season std/range match an independent recomputation from the
    pooled daily series; and the within-vs-between comparison is printed
    plainly per flux (no algebraic shortcut available here -- std/range are
    already atomic, so this is purely an on-disk-matches-recomputation check,
    not a linearity proof like the seasonal-effects tables)."""
    daily = _within_between_daily(p50_clear, p50_cloudy)
    lines = open(outfile).read().splitlines()

    print("  within-season vs between-season spread (1km ALPS, 24h daily means):")
    for var, lbl in _SEASONAL_EFFECTS_FLUXES:
        col = f"{var}_1km"
        season_means = []
        season_stds = []
        for name, months in SEASONS:
            vals = daily.loc[daily.index.month.isin(months), col].dropna()
            assert len(vals) == 6, f"{name} {var}: expected 6 days, got {len(vals)}"
            mean_v, std_v, rng_v = float(vals.mean()), float(vals.std()), float(vals.max() - vals.min())
            season_means.append(mean_v)
            season_stds.append(std_v)

            # each season label appears once per flux block (GPP/RECO/NEE, in
            # that fixed order); pick the occurrence matching this var.
            cells_all = [_numeric_cells(l) for l in lines if l.startswith(name + " &")]
            var_idx = [v for v, _ in _SEASONAL_EFFECTS_FLUXES].index(var)
            assert len(cells_all) == len(_SEASONAL_EFFECTS_FLUXES), (
                f"row '{name}' found {len(cells_all)} times in {outfile}, "
                f"expected {len(_SEASONAL_EFFECTS_FLUXES)} (one per flux block)"
            )
            cells = cells_all[var_idx]
            for got, exp, lbl2 in ((cells[0], mean_v, "mean"), (cells[1], std_v, "std"), (cells[2], rng_v, "range")):
                assert got == fmt(exp), f"{name} {var} {lbl2}: on-disk {got} != recomputed {fmt(exp)}"

        between_std = float(np.std(season_means, ddof=1))
        between_range = max(season_means) - min(season_means)

        between_cells_all = [
            _numeric_cells(l) for l in lines if l.startswith("\\emph{between-season} &")
        ]
        assert len(between_cells_all) == len(_SEASONAL_EFFECTS_FLUXES), (
            f"'between-season' row found {len(between_cells_all)} times in {outfile}, "
            f"expected {len(_SEASONAL_EFFECTS_FLUXES)}"
        )
        between_cells = between_cells_all[var_idx]  # ["--", std, range]
        assert between_cells[1] == fmt(between_std), (
            f"{var} between-season std: on-disk {between_cells[1]} != recomputed {fmt(between_std)}"
        )
        assert between_cells[2] == fmt(between_range), (
            f"{var} between-season range: on-disk {between_cells[2]} != recomputed {fmt(between_range)}"
        )

        mean_within_std = float(np.mean(season_stds))
        smaller = "smaller" if mean_within_std < between_std else "NOT smaller"
        print(f"    {lbl}: mean within-season std = {fmt(mean_within_std)}, "
              f"between-season std = {fmt(between_std)}, between-season range = {fmt(between_range)} "
              f"-- within-season spread is {smaller} than between-season spread")

    print("  [ok] within-vs-between season table: 6 days per season, on-disk values match recomputation")


# =====================================================================
# Follow-up 10: like-for-like comparison of Delta_par against the ensemble.
# Delta_par = x(DEFAULT) - x(p50) is a SIGNED displacement of the reported
# run relative to one other parameter set; sigma_IQR/sigma_env are WIDTHS
# across all 5 members. Not the same shape. The right yardstick is the
# signed displacement of the reported (p50) run if the calibration were
# swapped for each of the other four equally defensible members:
# x(p50) - x(pX) for X in {p10, p25, p75, p90}. Two variants -- on the 1km
# level (read against Delta_par) and on Delta_res per member (read against
# Delta_res) -- both built from the exact tagmeans/dres_m the spread columns
# already use (_member_tagmeans / _member_dres), not re-derived.
# =====================================================================
_DEVIATION_TAGS = [("p10", 0), ("p25", 1), ("p75", 3), ("p90", 4)]  # (tag, index into ALPS_TAGS-ordered tagmeans); p50 itself is index 2
_DEVIATION_INNER = ("p25", "p75")  # the two percentile-adjacent comparisons, checked against sigma_IQR


def _deviation_header():
    """Fixed tabular header for the p50-deviation tables: four signed
    x(p50)-x(pX) columns per flux, X in {p10,p25,p75,p90}."""
    return (
        "\\begin{tabular}{l|rrrr|rrrr|rrrr}\n\\toprule\n"
        "& \\multicolumn{4}{c|}{GPP} & \\multicolumn{4}{c|}{$R_\\text{eco}$}"
        " & \\multicolumn{4}{c}{NEE} \\\\\n"
        "Season"
        + (
            " & $x(p50){-}x(p10)$ & $x(p50){-}x(p25)$"
            " & $x(p50){-}x(p75)$ & $x(p50){-}x(p90)$"
        )
        * 3
        + " \\\\\n\\midrule\n"
    )


def _deviation_row(name, months, member_dfs, get_tagmeans):
    """One formatted row: the four signed x(p50)-x(pX) deviations per flux."""
    row = name
    for var, _lbl in _SEASONAL_EFFECTS_FLUXES:
        tm = get_tagmeans(member_dfs, var, months)
        p50 = tm[2]
        for _tag, idx in _DEVIATION_TAGS:
            d = p50 - tm[idx]
            s = fmt(d)
            row += f" & {s if s.startswith('-') else '+' + s}"
    return row + " \\\\\n"


def _member_tagmeans_res(member_dfs, var, months, res):
    """Per-member seasonal mean of var_{res} (res e.g. '1km'/'54km'), in
    ALPS_TAGS order (index 2 = p50). Generalization of _member_tagmeans to an
    arbitrary resolution, needed for Table_par_members' Delta_par^54 column
    (Delta_par evaluated at the 54 km grid rather than at 1 km)."""
    return [_smean(d, f"{var}_{res}", months) for d in member_dfs]


_PAR_MEMBERS_FLUXES = [v for v, _ in _SEASONAL_EFFECTS_FLUXES if v != "RECO"]  # GPP, NEE


def _par_members_header():
    """Header for Table_par_members (tab:par_members, Appendix H9, was the
    unfulfilled 'Table_par_members_2012_all.tex' \\input): one row per
    (season, calibration). Delta_par at 1 and 54 km and Delta_res at 54 km,
    for GPP and NEE -- R_eco carries no Topt dependence (identical across
    members) and is reported once, in Table 1/tab:effects_all, not repeated
    here, per the manuscript caption."""
    return (
        "\\begin{tabular}{ll|rrr|rrr}\n\\toprule\n"
        "& & \\multicolumn{3}{c|}{GPP} & \\multicolumn{3}{c}{NEE} \\\\\n"
        "Season & Set"
        + (" & $\\Delta_\\text{par}^\\text{1 km}$ & $\\Delta_\\text{par}^\\text{54 km}$"
           " & $\\Delta_\\text{res}^\\text{54 km}$") * 2
        + " \\\\\n\\midrule\n"
    )


def _par_members_block(name, months, member_dfs, ref_df, p50_idx, tag_labels):
    """One season block: 5 rows, one per calibration member (p10..p90) in
    ALPS_TAGS order, season name in column 1 of the first row only (same
    single-label-per-block convention as write_domain_average_table's
    _t1_block). Delta_par^{res}(M) = x(M, res) - x(p50, res) (member M
    against the reported ALPS/p50 set, at that same resolution); Delta_res^54
    (M) = x(M, 54km) - x(M, 1km) (that member's own resolution effect) --
    together these reproduce e.g. NEE(p25,54km) - NEE(p50,1km) =
    Delta_par^1(p25) + Delta_res^54(p25) exactly, the decomposition quoted in
    the Discussion for the p25 calibration."""
    block = []
    tm = {v: {res: (_member_tagmeans(member_dfs, v, months) if res == "1km"
                    else _member_tagmeans_res(member_dfs, v, months, res))
              for res in ("1km", "54km")}
          for v in _PAR_MEMBERS_FLUXES}
    ref = {v: {res: _smean(ref_df, f"{v}_{res}", months) for res in ("1km", "54km")}
           for v in _PAR_MEMBERS_FLUXES}
    for i, tag in enumerate(tag_labels):
        lab = "p50 (ALPS)" if i == p50_idx else tag
        row = (name if i == 0 else "") + f" & {lab}"
        for v in _PAR_MEMBERS_FLUXES:
            # Manuscript convention (Eq. dpar): Delta_par = x_DEFAULT - x_s,
            # taken against *that* calibration rather than against ALPS/p50,
            # as the table caption states. The p50 row then reproduces the
            # plain Delta_par of tab:effects_all, and the Discussion
            # decomposition Delta_par^1(p25) is the p25-minus-p50 row
            # difference.
            dpar1 = ref[v]["1km"] - tm[v]["1km"][i]
            dpar54 = ref[v]["54km"] - tm[v]["54km"][i]
            dres54 = tm[v]["54km"][i] - tm[v]["1km"][i]
            for val in (dpar1, dpar54, dres54):
                s = fmt(val)
                row += f" & {s if s.startswith('-') else '+' + s}"
        block.append(row + " \\\\\n")
    return block


def write_par_members_table(alps_clear, alps_cloudy, ref_clear, ref_cloudy, outfile):
    """Table_par_members_2012_all.tex (tab:par_members, Appendix H9): the two
    effect sizes (Delta_par at 1 and 54 km, Delta_res at 54 km) under each of
    the five Topt-percentile calibrations, per season and over all 24 sampled
    days, with the 12 clear-sky and 12 cloudy/rainy days pooled -- matching
    the pooling convention of write_seasonal_effects_table_combined. GPP and
    NEE only; R_eco is identical across members (alpha/beta unvaried) and is
    reported once elsewhere, per the manuscript caption."""
    pooled_members = [pd.concat([c, d]) for c, d in zip(alps_clear, alps_cloudy)]
    ref_pooled = pd.concat([ref_clear, ref_cloudy])
    p50_idx = ALPS_TAGS.index(MEDIAN_TAG)
    tag_labels = [t.replace("topt_", "") for t in ALPS_TAGS]
    rows = list(SEASONS) + [("24-day average", ALL_MONTHS)]
    with open(outfile, "w") as f:
        f.write(_par_members_header())
        for i, (name, months) in enumerate(rows):
            if i:
                f.write("\\midrule\n")
            for line in _par_members_block(name, months, pooled_members, ref_pooled, p50_idx, tag_labels):
                f.write(line)
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX table written to {outfile}")


def write_member_deviations_table(alps_clear, alps_cloudy, outfile, get_tagmeans):
    """10a: emit the four signed p50-anchored deviations per season (+ Year),
    per flux, pooling all 24 representative days. `get_tagmeans` selects the
    level variant (_member_tagmeans) or the Delta_res variant (_member_dres)."""
    pooled_members = [pd.concat([c, d]) for c, d in zip(alps_clear, alps_cloudy)]
    rows = list(SEASONS) + [("Year", ALL_MONTHS)]
    with open(outfile, "w") as f:
        f.write(_deviation_header())
        for name, months in rows:
            f.write(_deviation_row(name, months, pooled_members, get_tagmeans))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX table written to {outfile}")


def verify_member_deviations_table(alps_clear, alps_cloudy, dev_outfile, get_tagmeans,
                                    spread_outfile, gpp_iqr_idx):
    """Check 10: on-disk deviations match recomputation from the same
    per-member means the spread columns use; all four deviations exactly
    zero for R_eco; NEE's member spread equals GPP's exactly (RECO invariant
    across members, not published for NEE either since Follow-up 11); GPP's
    sigma_IQR (or sigma^Delta_IQR, for the dres variant) matches the
    already-verified value at `gpp_iqr_idx` in `spread_outfile` (cross-file
    validation for the one flux that is actually published); and the
    algebraic bound holds:

        max(|4 deviations|)       >= sigma_env / 2
        max(|2 inner deviations|) >= sigma_IQR / 2

    (proof for the first: sigma_env = max(tm) - min(tm) over all 5 tagged
    values including p50; p50 splits [min, max] into two segments summing to
    sigma_env, so the larger segment is >= sigma_env/2, and that segment's
    endpoint -- unless p50 itself is the min or max -- is one of the four
    tagged deviations.) sigma_env is no longer published anywhere (Follow-up
    11 dropped it project-wide), so it is recomputed here directly from the
    member means as max(tm)-min(tm) -- the same quantity the old published
    column held, just no longer cross-file-validated since there is no
    longer a second file to validate it against. A violation of either bound
    means the deviation table's own four values are internally inconsistent
    with each other, not a property of the underlying flux response -- stop
    and report rather than treating it as a real finding.
    """
    pooled_members = [pd.concat([c, d]) for c, d in zip(alps_clear, alps_cloudy)]
    rows = list(SEASONS) + [("Year", ALL_MONTHS)]
    fluxes = [v for v, _ in _SEASONAL_EFFECTS_FLUXES]

    dev_lines = open(dev_outfile).read().splitlines()
    spread_lines = open(spread_outfile).read().splitlines()

    for name, months in rows:
        dev_row = next((l for l in dev_lines if l.startswith(name + " &")), None)
        assert dev_row is not None, f"row '{name}' not found in {dev_outfile}"
        dev_cells = _numeric_cells(dev_row)
        spread_row = next((l for l in spread_lines if l.startswith(name + " &")), None)
        assert spread_row is not None, f"row '{name}' not found in {spread_outfile}"
        spread_cells = _numeric_cells(spread_row)
        gpp_iqr_disk = float(spread_cells[gpp_iqr_idx])

        gpp_iqr_recomp = None
        for vi, var in enumerate(fluxes):
            tm = get_tagmeans(pooled_members, var, months)
            p50 = tm[2]
            devs = {tag: p50 - tm[idx] for tag, idx in _DEVIATION_TAGS}

            got = dev_cells[4 * vi: 4 * vi + 4]
            for (tag, _idx), g in zip(_DEVIATION_TAGS, got):
                exp = fmt(devs[tag])
                exp = exp if exp.startswith("-") else "+" + exp
                assert g == exp, (
                    f"{name} {var} x(p50)-x({tag}): on-disk {g} != recomputed {exp} "
                    f"in {dev_outfile}"
                )

            iqr = np.nanpercentile(tm, 75) - np.nanpercentile(tm, 25)
            env = max(tm) - min(tm)

            if var == "RECO":
                for tag, d in devs.items():
                    assert abs(d) < 1e-9, (
                        f"{name} RECO x(p50)-x({tag}): expected exactly zero "
                        f"(alpha/beta not varied), got {d}"
                    )
                assert abs(iqr) < 1e-9, f"{name} RECO member spread not zero ({iqr})"
            elif var == "GPP":
                gpp_iqr_recomp = iqr
                assert fmt(iqr) == fmt(gpp_iqr_disk), (
                    f"{name} GPP spread: on-disk {fmt(gpp_iqr_disk)} in "
                    f"{spread_outfile} != recomputed {fmt(iqr)}"
                )
            elif var == "NEE":
                assert abs(iqr - gpp_iqr_recomp) < 1e-9, (
                    f"{name} NEE member spread {iqr} != GPP member spread "
                    f"{gpp_iqr_recomp} -- expected identical (RECO invariant "
                    "across members)"
                )

            max4 = max(abs(v) for v in devs.values())
            max_inner = max(abs(devs[t]) for t in _DEVIATION_INNER)
            assert max4 >= env / 2 - 0.010001, (
                f"{name} {var}: max|deviation| {max4:.4f} < sigma_env/2 {env / 2:.4f} "
                "(sigma_env recomputed, no longer published) -- deviation table "
                "internally inconsistent"
            )
            assert max_inner >= iqr / 2 - 0.010001, (
                f"{name} {var}: max|inner deviation| {max_inner:.4f} < sigma_IQR/2 "
                f"{iqr / 2:.4f} -- deviation table internally inconsistent"
            )

    print(f"  [ok] {dev_outfile}: on-disk deviations match recomputation, "
          f"R_eco deviations exactly zero, NEE spread == GPP spread, GPP "
          f"spread matches {spread_outfile}, algebraic bounds hold")


def report_p50_rank(alps_clear, alps_cloudy, get_tagmeans, label):
    """10b: rank of the p50 (tag) member among the five sorted member means
    (0=lowest flux, 4=highest, 2=middle), per season+Year and flux. Printed
    only, no .tex. The flux response to T_opt is not monotonic, so p50 need
    not sit at rank 2 -- if it doesn't, the percentile order does not carry
    over to the flux, which matters for how Sect. 3.2 describes the band."""
    pooled_members = [pd.concat([c, d]) for c, d in zip(alps_clear, alps_cloudy)]
    rows = list(SEASONS) + [("Year", ALL_MONTHS)]
    print(f"  p50 rank among the 5 sorted member means ({label}; 0=lowest, 4=highest, 2=middle):")
    any_offcenter = False
    for name, months in rows:
        for var, _lbl in _SEASONAL_EFFECTS_FLUXES:
            tm = get_tagmeans(pooled_members, var, months)
            p50_val = tm[2]
            srt = sorted(tm)
            degenerate = (srt[-1] - srt[0]) < 1e-9
            rank = srt.index(p50_val)
            if degenerate:
                print(f"    {name:5s} {var:5s} rank={rank}  (degenerate: all 5 members equal, spread=0)")
                continue
            flag = "" if rank == 2 else "  <-- NOT rank 2"
            if rank != 2:
                any_offcenter = True
            print(f"    {name:5s} {var:5s} rank={rank}{flag}")
    if any_offcenter:
        print(f"  [{label}] p50 is NOT rank 2 in at least one season/flux -- "
              "the percentile order does not carry over to the flux here")
    else:
        print(f"  [{label}] p50 sits at rank 2 (the middle) everywhere it is not degenerate")


# ==================== PDF content-stream verification ====================
# Dependency-free: FlateDecode content streams are inflated with zlib; small
# resource dictionaries (ExtGState alpha, etc.) matplotlib leaves uncompressed
# and can be grepped straight out of the raw file bytes.
# NB: matplotlib wraps content-stream lines (".. 1.5 w 1\n0.6470588235 0 RG"),
# so this must tolerate arbitrary whitespace -- with literal spaces the check
# silently passed while the orange CAMS stroke was in fact present.
_CAMS_ORANGE_RE = re.compile(rb"1\s+0\.647\d*\s+0\s+(RG|rg)")


def _pdf_content_streams(path):
    """Concatenated, inflated content-stream bytes of a PDF (for colour ops)."""
    data = open(path, "rb").read()
    chunks = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        raw = m.group(1)
        try:
            chunks.append(zlib.decompress(raw))
        except zlib.error:
            chunks.append(raw)
    return b"\n".join(chunks), data


def _pdf_has_cams_orange(path):
    streams, _ = _pdf_content_streams(path)
    return bool(_CAMS_ORANGE_RE.search(streams))


def _pdf_has_alpha(path, value):
    """True if an ExtGState with fill alpha `value` (e.g. 0.15) is defined.

    matplotlib writes these resource dicts uncompressed, so the raw bytes
    already contain the literal '/ca <value>'.
    """
    _, raw = _pdf_content_streams(path)
    return f"/ca {value}".encode() in raw


def _pdftotext(path):
    return subprocess.run(
        ["pdftotext", "-layout", path, "-"], capture_output=True, check=True
    ).stdout.decode("utf-8", errors="replace")


def verify_cams_restoration(OUTFOLDER, STD_TOPO, start_date, end_date, columns):
    """Assert the CAMS-restoration + DEFAULT-label + Topt-band invariants.

    - CAMS orange present in all ten seasonal appendix figures (5 vars x 2 regimes).
    - CAMS orange absent from legend_hourly.pdf and every timeseries_hourly_* file
      (main-text hourly panels + the 54 km-diff panels).
    - pdftotext of the GPP/RECO/NEE seasonal figures contains "DEFAULT", not the
      stale bare "DEF " label.
    - The Topt band fills (alpha 0.15 / 0.35) are still drawn on GPP and NEE.
    """
    seasonal_files = {
        (col, sim): (
            f"{OUTFOLDER}timeseries_{col}_domain_averaged_std_topo_{STD_TOPO}{sim}"
            f"_{start_date}_{end_date}.pdf"
        )
        for col in columns
        for sim in ("", "_cloudy")
    }

    missing_cams = [
        k for k, p in seasonal_files.items() if not _pdf_has_cams_orange(p)
    ]
    assert not missing_cams, f"CAMS orange missing from seasonal figures: {missing_cams}"
    print(f"  [ok] CAMS orange present in all {len(seasonal_files)} seasonal appendix figures")

    hourly_files = sorted(
        glob.glob(f"{OUTFOLDER}timeseries_hourly_*.pdf")
    ) + [f"{OUTFOLDER}legend_hourly.pdf"]
    leaked_cams = [p for p in hourly_files if _pdf_has_cams_orange(p)]
    assert not leaked_cams, f"CAMS orange leaked into main-text files: {leaked_cams}"
    print(f"  [ok] CAMS orange absent from legend_hourly.pdf and all "
          f"{len(hourly_files) - 1} timeseries_hourly_* files")

    label_files = [
        seasonal_files[(col, sim)]
        for col in ("GPP", "RECO", "NEE")
        for sim in ("", "_cloudy")
        if (col, sim) in seasonal_files
    ]
    bad_label = []
    for p in label_files:
        txt = _pdftotext(p)
        if "DEFAULT" not in txt or "DEF " in txt:
            bad_label.append(p)
    assert not bad_label, f"stale/missing DEFAULT label in: {bad_label}"
    print(f"  [ok] {len(label_files)} GPP/RECO/NEE seasonal figures say DEFAULT, not DEF")

    band_files = [
        seasonal_files[(col, sim)]
        for col in ("GPP", "NEE")
        for sim in ("", "_cloudy")
        if (col, sim) in seasonal_files
    ]
    bad_bands = [
        p for p in band_files
        if not (_pdf_has_alpha(p, 0.15) and _pdf_has_alpha(p, 0.35))
    ]
    assert not bad_bands, f"Topt band fills (alpha 0.15/0.35) missing in: {bad_bands}"
    print(f"  [ok] Topt band fills (alpha 0.15 and 0.35) present in all "
          f"{len(band_files)} GPP/NEE seasonal figures")

    return seasonal_files


def main(tables_only=False):
    """Regenerate the tables and, unless `tables_only`, all figures.

    `--tables-only` exists because the table block runs in seconds while the
    figure block takes minutes, and the table assertions are the part that gets
    iterated on. It changes nothing about what the tables contain.
    """
    csv_folder = os.path.join(GITHUB_PATH, "WRF_VPRM_inComplexTopo/WRF_VPRM_post/csv/")
    start_date, end_date = "2012-01-01 00:00:00", "2012-12-31 00:00:00"
    STD_TOPO = 200
    ref_sim = True
    sim_type = os.getenv("SIM_TYPE", "")  # "" (clear) or "_cloudy"

    columns = ["GPP", "RECO", "NEE", "T2", "SWDOWN"]
    units = [
        " [µmol m⁻² s⁻¹]",
        " [µmol m⁻² s⁻¹]",
        " [µmol m⁻² s⁻¹]",
        " [°C]",
        " [W m⁻²]",
    ]
    resolutions = ["1km", "9km", "54km", "CAMS"]
    resolutions_diff = ["54km", "9km"]

    # Table 1 spans BOTH regimes (clear + cloudy + their 24-day average), so load
    # both regardless of SIM_TYPE; the figures below still use SIM_TYPE only.
    regimes = {
        s: load_regime(s, csv_folder, STD_TOPO, start_date, end_date, resolutions,
                        columns, ref_sim)
        for s in ("", "_cloudy")
    }
    R = regimes[sim_type]

    hourly_list = R["hourly_list"]
    hourly_avg = R["hourly_avg"]
    hourly_avg_ref = R["hourly_avg_ref"]
    df_means, df_diffs, df_pct = R["means"], R["diffs"], R["pct"]

    table1_file = f"{OUTFOLDER}Table_1_domain_averaged_2012.tex"
    write_domain_average_table(
        (regimes[""]["means"], regimes[""]["diffs"], regimes[""]["pct"]),
        (regimes["_cloudy"]["means"], regimes["_cloudy"]["diffs"],
         regimes["_cloudy"]["pct"]),
        outfile=table1_file,
    )
    verify_table1(table1_file)

    # Seasonal (DJF/MAM/JJA/SON) tables. write_seasonal_domain_average_table now
    # carries its own CAMS column and both parameter sets' deltas, so it needs
    # BOTH regimes regardless of SIM_TYPE, same as Table 1 above -- one file per
    # regime (clear: no suffix, matching the file already in the manuscript;
    # cloudy: "_cloudy" suffix, the new counterpart). write_seasonal_effects_table
    # now lives in the same loop (it used to be a single SIM_TYPE-gated call, so a
    # default run never produced a cloudy counterpart -- a real gap, since Sect.
    # 3.2 discusses both regimes' bands but could only cite a clear-sky spread).
    for _sim in ("", "_cloudy"):
        write_seasonal_domain_average_table(
            regimes[_sim]["alps"],
            regimes[_sim]["ref"],
            outfile=f"{OUTFOLDER}Table_seasonal_domain_averaged_2012{_sim}.tex",
        )

        # canonical (all columns, analysis order) -> verified below;
        # paper view (manuscript column set/order) -> the name the .tex \input{}s
        _t7_out = f"{OUTFOLDER}Table_seasonal_effects_2012{_sim}_full.tex"
        _t7_paper = f"{OUTFOLDER}Table_seasonal_effects_2012{_sim}.tex"
        write_seasonal_effects_table(regimes[_sim]["alps"], regimes[_sim]["ref"], outfile=_t7_out)
        write_seasonal_effects_table_paper(
            regimes[_sim]["alps"][ALPS_TAGS.index(MEDIAN_TAG)], regimes[_sim]["ref"],
            regimes[_sim]["alps"], _t7_paper, "12-day average")
        _assert_paper_view_subset(_t7_paper, _t7_out)

        # check 7: sigma_IQR <= sigma_env, sigma_IQR = sorted[3]-sorted[1] for
        # n=5, R_eco spread exactly zero, Delta_res/Delta_par unchanged, Year
        # row = mean of the 4 seasons.
        verify_seasonal_effects_spread(regimes[_sim]["alps"], regimes[_sim]["ref"], _t7_out)

    # Combined seasonal-effects table, pooling all 24 representative days (12
    # clear + 12 cloudy). Delta_res/Delta_par pool linearly (recomputed from the
    # concatenated raw timeseries); sigma_IQR/sigma^Delta_IQR do not, and are
    # recomputed from each member's pooled seasonal mean -- see the function's
    # docstring. Written here, ahead of the domain-averaged checks, because
    # verify_cross_tables reads its Year row (Follow-up 16c).
    _t7_all = f"{OUTFOLDER}Table_seasonal_effects_2012_all_full.tex"
    write_seasonal_effects_table_combined(
        regimes[""]["alps"], regimes[""]["ref"],
        regimes["_cloudy"]["alps"], regimes["_cloudy"]["ref"],
        outfile=_t7_all,
    )
    verify_seasonal_effects_spread_combined(
        regimes[""]["alps"], regimes[""]["ref"],
        regimes["_cloudy"]["alps"], regimes["_cloudy"]["ref"],
        _t7_all,
    )

    # Follow-up 16h: CAMS out of Table 1 and J1/J2 into one table of its own.
    cams_file = f"{OUTFOLDER}Table_CAMS_domain_averaged_2012.tex"
    write_cams_table(regimes, cams_file)
    verify_cams_table(cams_file, regimes)

    _domain_tables = [
        table1_file,
        f"{OUTFOLDER}Table_seasonal_domain_averaged_2012.tex",
        f"{OUTFOLDER}Table_seasonal_domain_averaged_2012_cloudy.tex",
    ]

    # Follow-up 12d: T2 is never a percentage anywhere -- Celsius ratios are
    # not a physical quantity, and the winter near-zero domain mean sent this
    # to 600-1800% before the fix in _paren.
    verify_no_t2_percent(_domain_tables + [cams_file])

    # Follow-up 16a/16b: the ratio convention is gone from every 1 km cell.
    verify_no_ratio_cells(_domain_tables)

    # Follow-up 16c: Table 1's 1 km Delta_par and its 24-day 54 km Delta_res are
    # the same printed numbers as the seasonal-effects tables' Year rows. This is
    # the reason for the change, so it is asserted, not assumed.
    verify_cross_tables(
        table1_file,
        # the canonical tables, not the paper views: this check reads columns
        # by position, and the published views reorder them by design
        f"{OUTFOLDER}Table_seasonal_effects_2012_full.tex",
        f"{OUTFOLDER}Table_seasonal_effects_2012_cloudy_full.tex",
        f"{OUTFOLDER}Table_seasonal_effects_2012_all_full.tex",
    )

    # Follow-up 16d: where subtracting the two printed 1 km values does not
    # reproduce the printed Delta_par, say so -- one caption sentence, not a
    # hand-matched number.
    verify_rounding_residuals(_domain_tables)

    # Follow-up 16e: the seasonal Delta_par ranges the appendix text needs.
    report_seasonal_par_ranges(regimes)

    report_table_widths(_domain_tables + [cams_file])

    # Ratio of GPP's two spread definitions, both now columns of _t7_all
    # (Follow-up 11 merged sigma^Delta_IQR in directly -- the standalone
    # Table_seasonal_effects_dres_2012_all.tex from Follow-up 9b is no
    # longer produced).
    print_dres_iqr_ratio(_t7_all)

    # RC2 promise (9e): within-season (6 days) vs between-season (4 seasonal
    # means) variability, 1km ALPS p50.
    _p50_clear = regimes[""]["alps"][ALPS_TAGS.index(MEDIAN_TAG)]
    _p50_cloudy = regimes["_cloudy"]["alps"][ALPS_TAGS.index(MEDIAN_TAG)]
    _t_wvb = f"{OUTFOLDER}Table_within_vs_between_season_2012.tex"
    write_within_vs_between_season_table(_p50_clear, _p50_cloudy, _t_wvb)
    verify_within_vs_between_season_table(_p50_clear, _p50_cloudy, _t_wvb)

    # Follow-up 10: like-for-like comparison of Delta_par against the
    # ensemble -- p50-anchored signed deviations, not a width. 10a: level
    # (read against Delta_par) and Delta_res (read against Delta_res)
    # variants. 10b: where p50 actually sits in the sorted member ranking.
    # Both variants' reference sigma_IQR now live in the one merged table
    # _t7_all: GPP's 1km-level sigma_IQR at column index 5, GPP's
    # sigma^Delta_IQR (Delta_res member spread) at column index 6 (shifted
    # from 2/3 now that Delta_res^9/Delta_par^p25/p75 sit ahead of them).
    _t_dev_level = f"{OUTFOLDER}Table_member_deviations_2012_all.tex"
    write_member_deviations_table(regimes[""]["alps"], regimes["_cloudy"]["alps"], _t_dev_level, _member_tagmeans)
    verify_member_deviations_table(
        regimes[""]["alps"], regimes["_cloudy"]["alps"], _t_dev_level, _member_tagmeans,
        _t7_all, 5,
    )

    _t_dev_dres = f"{OUTFOLDER}Table_member_deviations_dres_2012_all.tex"
    write_member_deviations_table(regimes[""]["alps"], regimes["_cloudy"]["alps"], _t_dev_dres, _member_dres)
    verify_member_deviations_table(
        regimes[""]["alps"], regimes["_cloudy"]["alps"], _t_dev_dres, _member_dres,
        _t7_all, 6,
    )

    report_p50_rank(regimes[""]["alps"], regimes["_cloudy"]["alps"], _member_tagmeans, "level")
    report_p50_rank(regimes[""]["alps"], regimes["_cloudy"]["alps"], _member_dres, "Delta_res")

    # Table_par_members_2012_all.tex (tab:par_members, Appendix H9): the
    # manuscript's actual \input for this table never had a producer -- this
    # is that producer, not a Follow-up-numbered addition.
    # combined paper view, pooling both regimes, with the manuscript's
    # "24-day average" label for the pooled row
    _t7_all_paper = f"{OUTFOLDER}Table_seasonal_effects_2012_all.tex"
    write_seasonal_effects_table_paper(
        pd.concat([regimes[""]["alps"][ALPS_TAGS.index(MEDIAN_TAG)],
                   regimes["_cloudy"]["alps"][ALPS_TAGS.index(MEDIAN_TAG)]]),
        pd.concat([regimes[""]["ref"], regimes["_cloudy"]["ref"]]),
        [pd.concat([c, d]) for c, d in zip(regimes[""]["alps"],
                                           regimes["_cloudy"]["alps"])],
        _t7_all_paper, "24-day average")
    _assert_paper_view_subset(_t7_all_paper, _t7_all)

    _t_par_members = f"{OUTFOLDER}Table_par_members_2012_all.tex"
    write_par_members_table(regimes[""]["alps"], regimes["_cloudy"]["alps"],
                            regimes[""]["ref"], regimes["_cloudy"]["ref"], _t_par_members)

    # check 6: seasonal means must average back to the annual Table-1 means
    _bad = verify_seasonal_vs_annual(regimes)
    if _bad:
        print(
            f"  [WARN] seasonal vs annual mismatch in {len(_bad)} of "
            f"{2 * len(TABLE1_VARIABLES) * 3 * 2} column checks "
            "(the four seasons are equally weighted, the annual mean is "
            "day-weighted, so they only agree when the regime days are evenly "
            "spread over the seasons):"
        )
        for regime, var, col, got, annual in _bad[:10]:
            print(
                f"         {regime:7s} {var:6s} {col:9s} "
                f"seasonal={got:8.3f}  annual={annual:8.3f}  d={got - annual:+.3f}"
            )
        if len(_bad) > 10:
            print(f"         ... and {len(_bad) - 10} more")
    else:
        print("  [ok] seasonal means average to the annual Table-1 means")

    if tables_only:
        print("\n--tables-only: figures not regenerated.")
        return

    resolution_colors = {
        "1km": "black",
        "9km": "blue",
        "54km": "red",
        "CAMS": "orange",
    }

    # Seasonal appendix figures (plot_timeseries_by_resolution): both regimes,
    # regardless of SIM_TYPE, so one run refreshes all ten files with the CAMS
    # trace restored and the DEFAULT label current in both the clear and the
    # (previously stale) cloudy batch.
    for _sim in ("", "_cloudy"):
        Rp = regimes[_sim]
        for column, unit in zip(columns, units):
            plot_timeseries_by_resolution(
                Rp["alps"],
                Rp["ref"],
                column,
                unit,
                resolutions,
                OUTFOLDER,
                ref_sim,
                resolution_colors,
                STD_TOPO,
                start_date,
                end_date,
                _sim,
            )

    # Fig. 5 (clear sky) and Fig. 7 (cloudy) are the same panels for the two
    # regimes, so loop over both here -- as the seasonal block above already does
    # -- rather than depending on SIM_TYPE. With the old single-regime version one
    # run refreshed only one of the two figures and the other silently kept an
    # older build.
    for _sim in ("", "_cloudy"):
        _R = regimes[_sim]
        _alps, _ref = _R["alps"], _R["ref"]
        _hourly_list, _hourly_ref = _R["hourly_list"], _R["hourly_avg_ref"]

        for column, unit in zip(columns, units):
            plot_timeseries_differences(
                _alps,
                _ref,
                column,
                unit,
                resolutions,
                OUTFOLDER,
                ref_sim,
                resolution_colors,
                STD_TOPO,
                start_date,
                end_date,
                _sim,
            )

            plot_hourly_averages(
                _hourly_list,
                _hourly_ref,
                column,
                unit,
                resolutions,
                OUTFOLDER,
                ref_sim,
                resolution_colors,
                STD_TOPO,
                start_date,
                end_date,
                _sim,
            )

            plot_hourly_differences(
                _hourly_list,
                _hourly_ref,
                column,
                unit,
                resolutions_diff,
                OUTFOLDER,
                ref_sim,
                resolution_colors,
                STD_TOPO,
                start_date,
                end_date,
                _sim,
            )
            print(f"Completed plots for {column}{_sim or ' (clear)'}")

    # The legend strips do not depend on the regime -> write them once.
    save_legend_only(resolutions, resolution_colors, OUTFOLDER, "NEE")
    save_legend_only_diff(resolutions_diff, resolution_colors, OUTFOLDER, "NEE")

    seasonal_files = verify_cams_restoration(
        OUTFOLDER, STD_TOPO, start_date, end_date, columns
    )
    print("\nRegenerated files:")
    for p in sorted(set(seasonal_files.values())) + [f"{OUTFOLDER}legend_hourly.pdf"]:
        print(f"  {os.path.getsize(p):>8d}  {p}")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument(
        "--tables-only",
        action="store_true",
        help="write and verify the LaTeX tables, skip the (slow) figure block",
    )
    main(tables_only=_ap.parse_args().tables_only)
