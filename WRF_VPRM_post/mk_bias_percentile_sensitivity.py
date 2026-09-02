"""
mk_bias_percentile_sensitivity.py
==============================================
Exploratory follow-up to Table 1 (`flux_evaluation_1km_bias_r30.tex`,
`tab:filtered_sites_alps_bias`): that table's "ALPS" column compares the
FLUXNET towers against only the p50 (median Topt-tag) member. This script
builds the same MB (MAE) bias table for the other four percentile members
(p10, p25, p75, p90) and prints a side-by-side sensitivity report, so we can
judge whether the Topt-percentile choice moves the tower bias enough to
matter -- mirroring the already-established finding that it barely moves
model skill at the domain scale (Table 1) -- before deciding whether a table
belongs in the manuscript.

Reuses Fig4_AppxG1_G4_Table1_TableF5_F6_FLUXNET_eval.py's own data loading
(`load_site_csvs`) and table writer (`write_latex_table_from_metrics_bias`)
unchanged, and calls its `process_location(..., make_plots=False)` so this
NEVER touches the existing `{site}_{fluxtype}_comparison_hourly_1km...pdf`
plots or `legend_FLX_comparison.pdf` -- those file names don't encode which
Topt tag was used, so regenerating them here with a non-p50 tag would
silently corrupt the ones the main text actually embeds.

Writes exactly four NEW tables (p50's own table is left untouched, not
regenerated here) plus two plots:
  flux_evaluation_1km_bias_r30_p10.tex
  flux_evaluation_1km_bias_r30_p25.tex
  flux_evaluation_1km_bias_r30_p75.tex
  flux_evaluation_1km_bias_r30_p90.tex
  mb_vs_percentile_bias_r30.pdf
  mae_vs_percentile_bias_r30.pdf

Both plots show, per site (color) and flux (panel), the ALPS channel across
the five Topt tags (solid, marker) alongside that site's SITE (thick dashed)
and DEFAULT (thick dash-dot) values. DEFAULT is exactly tag-independent (the
single online run); SITE is read from the same per-tag data rather than
assumed flat -- it turns out to carry a small residual tag dependence (RECO's
SITE column is exactly flat, GPP/NEE's drift by a few tenths, well under
ALPS's spread), so plotting it this way is the honest picture rather than an
approximation.

These are exploratory, pre-decision outputs -- NOT added to ship_tables.py's
SHIP list. Ship by hand once/if a table or figure is actually added to the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from Fig4_AppxG1_G4_Table1_TableF5_F6_FLUXNET_eval import (
    ALPS_TAGS,
    CSVFOLDER,
    MEDIAN_TAG,
    OUTFOLDER,
    SCRATCH_PATH,
    load_site_csvs,
    process_location,
    write_latex_table_from_metrics_bias,
)

# Same constants main() uses, 1 km only (the other four members were never
# recomputed at 9/54 km -- see Fig4's main() comment on alps_tags).
TIMESPAN = "2012-01-01 00:00:00_2012-12-31 00:00:00"
SIM_TYPE = "_all"
SIM_SUFFIXES = ["", "_cloudy"]
RADIUS = 30
RES_DX = "1km"
BASE_DIR_FLX = f"{SCRATCH_PATH}/DATA/Fluxnet2015/Alps"
FLUXTYPES = ["T2_WRF", "NEE_WRF", "GPP_WRF", "RECO_WRF"]
PLOT_LABELS = [r"T$_\text{2m}$", "NEE", "GPP", r"R$_\text{eco}$"]
UNITS = [
    "[°C]",
    r"[$\mu$mol m$^{-2}$ s$^{-1}$]",
    r"[$\mu$mol m$^{-2}$ s$^{-1}$]",
    r"[$\mu$mol m$^{-2}$ s$^{-1}$]",
]
RESOLUTION_COLORS = {"1km": "blue", "3km": "darkgrey", "9km": "purple",
                     "27km": "red", "54km": "green"}
LOCATIONS = ["CH-Dav", "IT-Lav", "IT-Ren", "AT-Neu", "IT-MBo"]  # the 5 d03 towers


def _build_model_lat_lon():
    """Same site -> {name, lat, lon, veg_frac} lookup main() builds, from the
    1 km distances csv (tag-independent -- towers don't move between tags)."""
    pft_site_match = pd.read_csv(f"{CSVFOLDER}distances_{RES_DX}_{TIMESPAN}_r{RADIUS}.csv")
    by_site = {n.split("_")[0]: n for n in pft_site_match["name"]}
    rows = pft_site_match.set_index("name")
    out = []
    for site in LOCATIONS:
        r = rows.loc[by_site[site]]
        out.append({"name": by_site[site], "lat": r["lat_wrf"], "lon": r["lon_wrf"],
                    "veg_frac": r["veg_frac_idx"]})
    return out


def _metrics_for_tag(tag, model_lat_lon):
    """consolidated_metrics_total for one Topt tag, metrics only (make_plots=False)."""
    df0 = load_site_csvs(tag, CSVFOLDER, RES_DX, SIM_SUFFIXES, TIMESPAN, RADIUS)
    dataframes = {RES_DX: df0}
    consolidated_metrics_total = pd.DataFrame()
    for location in LOCATIONS:
        consolidated_metrics_all = process_location(
            location,
            FLUXTYPES,
            PLOT_LABELS,
            UNITS,
            dataframes,
            [df0],  # ALPS_dfs: only used to draw the percentile band, skipped (make_plots=False)
            df0.index[0],
            df0.index[-1],
            BASE_DIR_FLX,
            {},  # df_CAMS_hourly_all -- CAMS disabled, same as main()
            RESOLUTION_COLORS,
            SIM_TYPE,
            OUTFOLDER,
            TIMESPAN,
            resolution=RES_DX,
            make_plots=False,
        )
        consolidated_metrics_total = pd.concat(
            [consolidated_metrics_total, consolidated_metrics_all], axis=1
        )
    return consolidated_metrics_total


_SENSITIVITY_FLUXES = [("GPP_WRF", "GPP"), ("RECO_WRF", r"R_eco"), ("NEE_WRF", "NEE")]


def report_bias_percentile_sensitivity(metrics_by_tag, sites):
    """Print MB (MAE) for the ALPS channel across all five Topt tags, per site
    and cross-site mean, for GPP/RECO/NEE -- the number that answers whether
    the percentile choice moves the tower bias enough to matter. T2m and the
    SITE/REF channels are tag-independent (T2m/REF come from the single
    DEFAULT run; SITE is the 1 km per-site fit, untouched by the ALPS Topt
    sweep) so they aren't part of this comparison."""
    tags = list(metrics_by_tag)
    short = {t: t.replace("topt_", "") for t in tags}
    print("\nPercentile sensitivity of the ALPS-channel tower bias, MB (MAE) "
          f"[umol m-2 s-1] -- tags: {', '.join(short[t] for t in tags)}")
    for flux, label in _SENSITIVITY_FLUXES:
        print(f"\n  {label}:")
        mb_by_tag = {t: [] for t in tags}
        mae_by_tag = {t: [] for t in tags}
        for site in sites:
            row = []
            for t in tags:
                col = f"{site}_ALPS_{flux}_{RES_DX}"
                mb = metrics_by_tag[t].loc["MB", col]
                mae = metrics_by_tag[t].loc["MAE", col]
                mb_by_tag[t].append(mb)
                mae_by_tag[t].append(mae)
                row.append((t, mb, mae))
            mbs = [r[1] for r in row]
            rng = max(mbs) - min(mbs)
            cells = "  ".join(f"{short[t]:>4s} {mb:+.2f} ({mae:.2f})" for t, mb, mae in row)
            print(f"    {site:8s} {cells}   MB range={rng:.2f}")
        mean_mb = {t: float(np.mean(mb_by_tag[t])) for t in tags}
        mean_mae = {t: float(np.mean(mae_by_tag[t])) for t in tags}
        mean_cells = "  ".join(f"{short[t]:>4s} {mean_mb[t]:+.2f} ({mean_mae[t]:.2f})" for t in tags)
        mean_rng = max(mean_mb.values()) - min(mean_mb.values())
        print(f"    {'Mean':8s} {mean_cells}   MB range={mean_rng:.2f}")


# Channel line style: every individual-site line is thin (only the
# cross-site Mean is thick, drawn separately below). ALPS gets a marker at
# every percentile tag -- that's the whole point of the sweep. SITE/DEFAULT
# are read from the same per-tag data (see module docstring: SITE isn't
# perfectly flat) but conceptually aren't a percentile series, so they get a
# single marker at p50 rather than one per tag (added separately below). Each
# channel gets its own marker shape (square/triangle/circle) so the three are
# distinguishable even in black and white.
_CHANNEL_STYLE = {
    "ALPS": dict(linestyle="-", marker="s", markersize=4, linewidth=1.0),
    "SITE": dict(linestyle="--", linewidth=1.0),
    "REF": dict(linestyle="-.", linewidth=1.0),
}
_CHANNEL_MARKER = {"ALPS": "s", "SITE": "^", "REF": "o"}
_CHANNEL_LABEL = {"ALPS": "ALPS", "SITE": "SITE", "REF": "DEFAULT"}


def _tag_series(metrics_by_tag, tags, site, flux, channel, metric):
    return [metrics_by_tag[t].loc[metric, f"{site}_{channel}_{flux}_{RES_DX}"] for t in tags]


def plot_metric_vs_percentile(metrics_by_tag, sites, metric, ylabel, outfile):
    """`metric` ("MB" or "MAE") vs. Topt percentile, one panel per flux
    (GPP/RECO/NEE). Each site gets one color: ALPS is the percentile-
    dependent solid/marker line (the point of this whole exercise) plus a
    bold cross-site Mean; that site's SITE (dashed) and DEFAULT (dash-dot)
    values are drawn alongside in the same color, read from the same
    per-tag data rather than assumed constant -- see module docstring for
    why SITE isn't perfectly flat. Exploratory (not yet decided for the
    paper)."""
    tags = list(metrics_by_tag)
    pcts = [int(t.replace("topt_p", "")) for t in tags]
    site_colors = plt.cm.tab10(np.linspace(0, 1, len(sites)))

    p50_idx = tags.index(MEDIAN_TAG)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True)
    for ax, (flux, label) in zip(axes, _SENSITIVITY_FLUXES):
        for site, color in zip(sites, site_colors):
            for channel in ("SITE", "REF", "ALPS"):
                vals = _tag_series(metrics_by_tag, tags, site, flux, channel, metric)
                ax.plot(pcts, vals, color=color, **_CHANNEL_STYLE[channel])
                if channel != "ALPS":
                    ax.plot([50], [vals[p50_idx]], marker=_CHANNEL_MARKER[channel],
                            markersize=4, color=color)
        mean_alps = [
            float(np.mean([metrics_by_tag[t].loc[metric, f"{s}_ALPS_{flux}_{RES_DX}"] for s in sites]))
            for t in tags
        ]
        ax.plot(pcts, mean_alps, marker="s", markersize=5, color="black", linewidth=2.5)
        if metric == "MB":
            ax.axhline(0, color="0.6", lw=0.8, linestyle="--", zorder=0)
        ax.axvline(50, color="0.85", lw=0.8, linestyle=":", zorder=0)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Topt percentile (ALPS member)", fontsize=9)
        ax.set_xticks(pcts)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel, fontsize=9)

    # Opaque legend boxes: the SITE/DEFAULT reference lines are thick and
    # span the full x-range, so a frameless legend anywhere would sit on top
    # of one of them.
    _legend_box = dict(frameon=True, facecolor="white", edgecolor="0.75", framealpha=1.0)

    site_handles = [Line2D([0], [0], color=c, lw=1.0, marker="s", markersize=4) for c in site_colors]
    site_handles.append(Line2D([0], [0], color="black", lw=2.5, marker="s", markersize=6))
    site_lg = axes[-1].legend(site_handles, list(sites) + ["Mean"], fontsize=6.5, loc="best",
                              title="Site (ALPS)", title_fontsize=6.5, **_legend_box)
    site_lg.get_frame().set_linewidth(0.4)

    channel_handles = [
        Line2D([0], [0], color="0.3", marker=_CHANNEL_MARKER[c], markersize=4,
               linestyle=_CHANNEL_STYLE[c]["linestyle"], linewidth=_CHANNEL_STYLE[c]["linewidth"])
        for c in ("ALPS", "SITE", "REF")
    ]
    chan_lg = axes[0].legend(channel_handles, [_CHANNEL_LABEL[c] for c in ("ALPS", "SITE", "REF")],
                             fontsize=6.5, loc="best", title="Channel", title_fontsize=6.5,
                             **_legend_box)
    chan_lg.get_frame().set_linewidth(0.4)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot written to {outfile}")


def main():
    model_lat_lon = _build_model_lat_lon()

    metrics_by_tag = {}
    for tag in ALPS_TAGS:
        print(f"processing tag {tag} ...")
        metrics_by_tag[tag] = _metrics_for_tag(tag, model_lat_lon)
        if tag != MEDIAN_TAG:
            suffix = tag.replace("topt_", "")
            outfile = f"{OUTFOLDER}flux_evaluation_1km_bias_r30_{suffix}.tex"
            write_latex_table_from_metrics_bias(
                metrics_by_tag[tag],
                model_lat_lon,
                outfile=outfile,
                res_dx=RES_DX,
                params=["SITE", "ALPS", "REF"],
            )

    report_bias_percentile_sensitivity(metrics_by_tag, LOCATIONS)
    plot_metric_vs_percentile(metrics_by_tag, LOCATIONS, "MB", r"MB [$\mu$mol m$^{-2}$ s$^{-1}$]",
                              f"{OUTFOLDER}mb_vs_percentile_bias_r30.pdf")
    plot_metric_vs_percentile(metrics_by_tag, LOCATIONS, "MAE", r"MAE [$\mu$mol m$^{-2}$ s$^{-1}$]",
                              f"{OUTFOLDER}mae_vs_percentile_bias_r30.pdf")
    print("\nAll done.")


if __name__ == "__main__":
    main()
