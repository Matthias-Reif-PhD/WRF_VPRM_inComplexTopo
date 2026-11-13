"""
Fig10_dFldT_hourly_mean.py
==============================================
Fig. 10 (fig:dPdT_Tcontribution).
Plots hourly mean diurnal cycles of dT, dGPP, and dRECO
for different WRF resolutions and simulation types (OPT and REF).
Utilizes combined data from both cloudy and clear-sky simulations if specified.
==============================================
"""

import matplotlib

# use non-interactive backend for headless environments
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

########### Settings ###########
CSVFOLDER = os.getenv("CSVFOLDER", "./csv/")
OUTFOLDER = os.getenv("OUTFOLDER", "./plots/")
start_date = "2012-01-01 00:00:00"
end_date = "2012-12-31 00:00:00"
ref_tag = "_1km"
sim_type = "_cloudy"  # "" or "_cloudy"
plot_cloudy_and_clear = True  # merges clear+cloudy ("all") — the variant used in the paper
STD_TOPO = 200
resolutions = ["54km", "9km"]
variable_groups = {
    "dT": "T [°C]",
    "dGPP": r"GPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "dRECO": r"R$_\text{eco}$ [$\mu$mol m$^{-2}$ s$^{-1}$]",
}
###############################42
# ALPS is now the offline recompute over a Topt percentile sweep; the spread
# across these tags is drawn as nested bands (dark p25-p75, light p10-p90, p50
# median line). DEFAULT (the default/reference, RC1's renamed "DF") has no percentile
# recompute and stays a single online line.
ALPS_TAGS = ["topt_p10", "topt_p25", "topt_p50", "topt_p75", "topt_p90"]
MEDIAN_TAG = "topt_p50"
sim_suffixes = ["", "_cloudy"] if plot_cloudy_and_clear else [sim_type]


def load_merged(tag_suffix):
    """Load + merge the requested sim types for one dPdT CSV family."""
    parts = []
    for s in sim_suffixes:
        f = os.path.join(
            CSVFOLDER,
            f"dPdT_timeseries_{start_date}_{end_date}{ref_tag}{tag_suffix}{s}.csv",
        )
        if os.path.exists(f):
            parts.append(pd.read_csv(f, index_col="datetime", parse_dates=True))
        else:
            print(f"  WARNING missing: {os.path.basename(f)}")
    if not parts:
        raise FileNotFoundError(f"no dPdT CSV for suffix '{tag_suffix}'")
    return pd.concat(parts).sort_index()


# ALPS percentile sweep (one merged df per tag) + DEFAULT reference (online _REF)
ALPS_dfs = [load_merged(f"_recalc_{t}") for t in ALPS_TAGS]
df = ALPS_dfs[ALPS_TAGS.index(MEDIAN_TAG)]  # p50 median (used by percent plot)
df_ref = load_merged("_REF")


resolution_colors = {
    "54km": "red",
    "9km": "blue",
}
resolution_colors_light = {
    "54km": mcolors.to_rgba("salmon"),
    "9km": mcolors.to_rgba("lightskyblue"),
}


def _hourly_mean(df, col):
    """Diurnal (per-hour-of-day) mean of one column; (hours, values) or (None, None)."""
    if col not in df.columns:
        return None, None
    s = df[col].dropna()
    if s.empty:
        return None, None
    g = s.groupby(s.index.hour).mean()
    return g.index.values, g.values


def _hourly_percentiles(dfs, col, pcts=(10, 25, 50, 75, 90)):
    """Per-hour percentiles across the tag dfs for one column."""
    series = []
    for d in dfs:
        h, v = _hourly_mean(d, col)
        if h is not None:
            series.append(pd.Series(v, index=h))
    if not series:
        return None, None
    M = pd.concat(series, axis=1)
    hours = M.index.values
    return hours, {p: np.nanpercentile(M.values, p, axis=1) for p in pcts}


# --- Percentile-band plot: ALPS as nested bands (p10-p90 light, p25-p75 dark,
#     p50 median line); DEFAULT as a single online reference line. ---
def plot_combined(ALPS_dfs, df_ref, variable_groups):
    for var_prefix, ylabel in variable_groups.items():
        plt.figure(figsize=(6.0, 4.2))  # 2x the 0.43\linewidth printed width

        for res in resolutions:
            # which quantities to draw and their base styling
            # Styling follows Fig. 6 (save_legend_only_diff in the Fig5_6 script):
            # ALPS solid, DEFAULT dashed, with colour lightness -- not a third and
            # fourth linestyle -- carrying Delta_res vs Delta_dT. The former
            # ":"/"-." pair made four linestyles compete with the two colours.
            #
            # zorder: the res loop is OUTER and kinds INNER, so draw order is
            # 54km Delta_res -> 54km Delta_dT -> 9km Delta_res -> 9km Delta_dT, and
            # the 9km Delta_res bands and lines used to paint over the 54km
            # Delta_dT curves. The temperature contribution is what this figure is
            # about, so it is pinned above everything rather than reordering the
            # loops: bands 1, Delta_res lines 2, Delta_dT lines 3.
            if var_prefix == "dT":
                kinds = [("model", "-", resolution_colors, 3, 2.4)]  # dT: single line
            else:
                kinds = [
                    ("real", "-", resolution_colors, 2, 2.0),  # Delta_res
                    ("model", "-", resolution_colors_light, 3, 2.4),  # Delta_dT
                ]

            for kind, style, cdict, z, lw in kinds:
                col = f"{var_prefix}_{kind}_mean_{res}"
                base = cdict.get(res, "gray")

                # ALPS percentile band
                hours, P = _hourly_percentiles(ALPS_dfs, col)
                if hours is not None:
                    if np.nanmax(np.abs(P[90] - P[10])) > 1e-9:
                        plt.fill_between(
                            hours, P[10], P[90], color=base, alpha=0.15,
                            linewidth=0, zorder=1,
                        )
                        plt.fill_between(
                            hours, P[25], P[75], color=base, alpha=0.35,
                            linewidth=0, zorder=1,
                        )
                    plt.plot(hours, P[50], linestyle=style, color=base, lw=lw,
                             zorder=z)

                # DEFAULT single reference line (online _REF); dashed in every
                # family, per Fig. 6's ALPS-solid/DEFAULT-dashed convention.
                ref_color = cdict.get(res, "gray")
                rh, rv = _hourly_mean(df_ref, col)
                if rh is not None:
                    plt.plot(rh, rv, linestyle="--", color=ref_color, lw=lw,
                             zorder=z)

        plt.xticks(np.arange(0, 24, 3))  # 3-hourly; data is hour 0..23
        plt.xlabel("UTC [h]", fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.tick_params(labelsize=16)
        plt.grid(True)
        plt.tight_layout()
        st = "all" if plot_cloudy_and_clear else sim_type
        plotname = f"{OUTFOLDER}hourly_avg_{var_prefix}_combined_std_topo_{STD_TOPO}{st}_{start_date}_{end_date}.pdf"
        print(f"saved plot: {plotname}")
        plt.savefig(plotname, bbox_inches="tight")
        plt.close()


def get_all_legend_elements_combined(
    resolutions, resolution_colors, resolution_colors_light
):
    """Generate comprehensive legend elements for all plot_combined variants (8 entries).

    Wording and styles follow Fig. 6 (`save_legend_only_diff` in the Fig5_6
    script): the "{res}-1km, ALPS" / "{res}-1km, DEFAULT" naming, and ALPS solid
    against DEFAULT dashed. The Delta prefix is Fig. 10's own addition, needed
    because it draws two families where Fig. 6 draws one; those two are separated
    by colour lightness, matching the line weights plot_combined uses.
    """
    handles = []
    labels = []

    # Delta_res entries (dark colors: red and blue)
    for res in resolutions:
        color = resolution_colors[res]
        # ALPS (solid, central member)
        handles.append(Line2D([0], [0], color=color, lw=2.0, linestyle="-"))
        labels.append(r"$\Delta_{\text{res}}$ " + res + "-1km, ALPS")
        # DEFAULT (dashed)
        handles.append(Line2D([0], [0], color=color, lw=2.0, linestyle="--"))
        labels.append(r"$\Delta_{\text{res}}$ " + res + "-1km, DEFAULT")

    # Delta_dT entries (light colors: salmon and lightskyblue), drawn in the
    # foreground and a touch heavier, as in plot_combined
    for res in resolutions:
        color = resolution_colors_light[res]
        # ALPS (solid, central member)
        handles.append(Line2D([0], [0], color=color, lw=2.4, linestyle="-"))
        labels.append(r"$\Delta_{\partial\text{T}}$ " + res + "-1km, ALPS")
        # DEFAULT (dashed)
        handles.append(Line2D([0], [0], color=color, lw=2.4, linestyle="--"))
        labels.append(r"$\Delta_{\partial\text{T}}$ " + res + "-1km, DEFAULT")

    # Topt percentile bands (ALPS spread)
    handles.append(Patch(facecolor="gray", alpha=0.35, linewidth=0))
    labels.append(r"ALPS p25-p75 (ens.)")
    handles.append(Patch(facecolor="gray", alpha=0.15, linewidth=0))
    labels.append(r"ALPS p10-p90 (ens.)")

    return handles, labels


def save_all_legend_only_combined(
    resolutions,
    resolution_colors,
    resolution_colors_light,
    OUTFOLDER,
):
    """Save comprehensive legend for all plot_combined variants in a separate figure."""
    handles, labels = get_all_legend_elements_combined(
        resolutions, resolution_colors, resolution_colors_light
    )

    # Five columns, one per series group, two rows of ALPS-then-DEFAULT:
    #   Delta_res 54km | Delta_res 9km | Delta_dT 54km | Delta_dT 9km | percentiles
    # matplotlib fills a legend COLUMN-major, so with 10 handles and ncol=5 each
    # column takes the next consecutive pair -- which is exactly the order
    # get_all_legend_elements_combined builds them in. Keep the two in step: a
    # reordering there silently regroups the columns here.
    # At ncol=4 the 10 entries wrapped into 3 ragged rows that split the pairs
    # across columns (9km-ALPS sat under 54km-DEFAULT), which is what made the
    # groups hard to read.
    fig = plt.figure(figsize=(11.0, 0.9))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=5,
        fontsize=16,
        frameon=False,
        handlelength=3,
        columnspacing=1.8,
    )

    plt.axis("off")
    plt.savefig(
        f"{OUTFOLDER}legend_combined_all.pdf",
        bbox_inches="tight",
    )
    plt.close()


# Removed: get_legend_elements_combined / save_legend_only_combined (per-variable
# legend_combined_{var}.pdf). Never called -- the manuscript uses the single
# legend_combined_all.pdf from save_all_legend_only_combined above -- and they
# still encoded the pre-round-2 ":"/"-." linestyles, so leaving them in place
# would have contradicted the figure they claimed to describe.


def plot_gpp_percent_explained(
    df, df_ref, variable_groups, resolutions, OUTFOLDER, STD_TOPO, start_date, end_date
):
    for var_prefix, ylabel in variable_groups.items():
        plt.figure(figsize=(6.0, 4.2))  # 2x the 0.43\linewidth printed width
        width = 0.4
        hours = np.arange(24)

        for i, res in enumerate(resolutions):
            col_real = f"{var_prefix}_real_mean_{res}"
            col_model = f"{var_prefix}_model_mean_{res}"
            col_real_ref = f"{var_prefix}_real_mean_{res}"
            col_model_ref = f"{var_prefix}_model_mean_{res}"

            if col_real not in df.columns or col_model not in df.columns:
                print(f"Missing OPT columns for {res}. Skipping...")
                continue

            if (
                col_real_ref not in df_ref.columns
                or col_model_ref not in df_ref.columns
            ):
                print(f"Missing DF columns for {res}. Skipping...")
                continue

            # --- OPT ---
            real_series = df[col_real].dropna()
            model_series = df[col_model].dropna()
            common_index = real_series.index.intersection(model_series.index)
            real_series = real_series.loc[common_index]
            model_series = model_series.loc[common_index]
            real_hourly = real_series.groupby(real_series.index.hour).mean()
            model_hourly = model_series.groupby(model_series.index.hour).mean()
            percent_explained = (model_hourly / real_hourly) * 100
            mean_percent = percent_explained.mean()
            plt.bar(
                hours + i * width - width / 2,
                percent_explained,
                width=width,
                color=resolution_colors.get(res, "gray"),
                alpha=0.7,
                label=f"{res} OPT (avg: {mean_percent:.1f}%)",
            )

            # --- REF ---
            real_series_ref = df_ref[col_real_ref].dropna()
            model_series_ref = df_ref[col_model_ref].dropna()
            common_index_ref = real_series_ref.index.intersection(
                model_series_ref.index
            )
            real_series_ref = real_series_ref.loc[common_index_ref]
            model_series_ref = model_series_ref.loc[common_index_ref]
            real_hourly_ref = real_series_ref.groupby(real_series_ref.index.hour).mean()
            model_hourly_ref = model_series_ref.groupby(
                model_series_ref.index.hour
            ).mean()
            percent_explained_ref = (model_hourly_ref / real_hourly_ref) * 100
            mean_percent_ref = percent_explained_ref.mean()
            plt.bar(
                hours + i * width - width / 2 + width / 2,
                percent_explained_ref,
                width=width,
                color=resolution_colors.get(res, "gray"),
                alpha=0.4,
                label=f"{res} DEFAULT (avg: {mean_percent_ref:.1f}%)",
            )

        plt.axhline(100, color="black", linestyle="--", linewidth=1)
        plt.xticks(hours)
        plt.tick_params(labelsize=14)
        plt.xlabel("UTC [h]")
        plt.ylabel(f"{ylabel} explained by model [%]")
        plt.ylim(-100, 100)
        plt.legend(frameon=True, framealpha=0.4)
        plt.grid(True)
        plt.tight_layout()
        if plot_cloudy_and_clear:
            sim_type = "all"
        outpath = os.path.join(
            OUTFOLDER,
            f"hourly_percent_{var_prefix}_model_real_std_topo_{STD_TOPO}{sim_type}_{start_date}_{end_date}.pdf",
        )
        print(f"saved percent GPP explained plot: {outpath}")
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()


plot_combined(ALPS_dfs, df_ref, variable_groups)

# Save comprehensive legend with all 8 entries
save_all_legend_only_combined(
    resolutions,
    resolution_colors,
    resolution_colors_light,
    OUTFOLDER,
)

variable_subset = {
    "dGPP": "ΔGPP ",
    "dRECO": "ΔRECO ",
}
plot_gpp_percent_explained(
    df, df_ref, variable_subset, resolutions, OUTFOLDER, STD_TOPO, start_date, end_date
)
