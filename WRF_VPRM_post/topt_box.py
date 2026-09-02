"""
topt_box.py
==============================================
Shared helper for drawing the fitted-Topt boxplot panel.

The Topt distribution cannot be taken from the tuning xlsx: there `Topt` is the
*prescribed* per-vegetation-class value (constant within a class). The spread
shown here is the per-site-year Topt fitted from the GPP-vs-T curves, i.e. the
same sample the published p10/p25/p50/p75/p90 ensemble is derived from.

Ported from the standalone panel (b) in Fig3a_Topt_perSiteYear_fit.py so that
Fig3b_AppxG10_G11_Topt_tuneParam.py can draw the same panel as the top row of
its per-vegetation-class and per-site parameter figures.

Note on x alignment: seaborn places categorical ticks at 0, 1, 2, ... whereas
matplotlib's ``ax.bxp`` defaults to 1, 2, 3, ... Callers that stack this panel
on top of seaborn boxplots must pass ``positions=range(len(order))`` and give
the seaborn panels a matching explicit ``order=``.
==============================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Canonical vegetation-class palette for the whole paper -- Fig1, Fig2,
# Fig3a and Fig3b all import this rather than keeping their own copy,
# so a colour change here reaches every figure at once.
PFT_COLORS = {
    "ENF": "#003300",  # Evergreen needleleaf forest (darkest; was #004d00/#006400)
    "DBF": "#228B22",  # Deciduous broadleaf forest
    "MF": "#8FBC8F",  # Mixed forest
    "SHB": "#A0522D",  # Shrubland
    "SAV": "#FFD700",  # Savannas
    "CRO": "#FFA07A",  # Cropland
    "GRA": "#00A86B",  # Grassland (jade: a green with more blue, was #66CC00)
}

_FALLBACK_COLOR = "#8FBC8F"

# Outlier-marker style, shared with the sns.boxplot flierprops in
# Fig3b_AppxG10_G11_Topt_tuneParam.py's parameter rows so every row of the
# stacked figure marks outliers the same way.
FLIER_PROPS = dict(
    marker="o", markersize=4, markerfacecolor="0.35", markeredgecolor="0.35",
    alpha=0.7, linestyle="none",
)


def build_topt_points(min_nee_temp_dict2) -> pd.DataFrame:
    """Per-site-year fitted Topt as a long frame: PFT, site, year, Topt.

    Keeps only non-fallback fits (colour != "red", i.e. the mirrored/upward
    Gaussian converged) with a finite Topt -- the same filter the standalone
    boxplot uses, so both figures describe the same sample.
    """
    rows = []
    for site, yd in min_nee_temp_dict2.items():
        pft = site.split("_")[0]
        for yr, (t, c) in yd.items():
            if c != "red" and isinstance(t, (int, float)) and np.isfinite(t):
                rows.append({"PFT": pft, "site": site, "year": yr, "Topt": float(t)})
    return pd.DataFrame(rows, columns=["PFT", "site", "year", "Topt"])


def _sample_for(sub, group_col):
    """The sample a box's percentiles describe: per-site means when there are
    >=2 sites in a per-class box (matching how the published table -- percentiles
    of per-site means -- was derived), otherwise the raw values (a single site,
    or a per-site panel). Also used to find outliers, so a value is only flagged
    as one relative to the same population its box summarises."""
    if group_col == "PFT" and sub["site"].nunique() >= 2:
        return sub.groupby("site").Topt.mean().values
    return sub["Topt"].values


def _stats_for(key, sub, group_col, perc):
    """(whislo, q1, med, q3, whishi) for one category, or None if no data."""
    if group_col == "PFT" and perc is not None and key in perc.index:
        # Published ensemble: percentiles of the per-site means.
        r = perc.loc[key]
        return r["p10"], r["p25"], r["p50"], r["p75"], r["p90"]
    if sub.empty:
        return None
    sample = _sample_for(sub, group_col)
    if len(sample) == 0:
        return None
    return tuple(np.percentile(sample, [10, 25, 50, 75, 90]))


def draw_topt_box(
    ax,
    df_pts,
    order,
    group_col="PFT",
    perc=None,
    colors=None,
    positions=None,
    show_counts=True,
    label_fontsize=17,
    tick_fontsize=14,
    show_legend=True,
    xlabel=None,
):
    """Draw one Topt boxplot on `ax`, one box per entry of `order`.

    group_col="PFT"  -> box comes from `perc` (topt_percentiles.csv) when the
                        class is present there, otherwise from percentiles of
                        the per-site means (>=2 sites) or of the raw year values.
    group_col="site" -> box always comes from that site's raw year values.

    Categories with no fitted Topt (e.g. the CRO sites, which are excluded
    because cutting events distort the GPP-vs-T fit) are skipped but keep their
    x slot, so every row of a stacked figure stays column-aligned.
    """
    order = list(order)
    if positions is None:
        positions = range(1, len(order) + 1)
    positions = list(positions)
    if colors is None:
        colors = [PFT_COLORS.get(k, _FALLBACK_COLOR) for k in order]

    ax.grid(True, axis="y", alpha=0.6)
    ax.set_axisbelow(True)

    bxp_stats, bxp_pos, bxp_colors = [], [], []
    for key, x, col in zip(order, positions, colors):
        sub = df_pts[df_pts[group_col] == key]
        st = _stats_for(key, sub, group_col, perc)
        if st is None:
            continue  # keep the x slot, draw nothing
        p10, p25, p50, p75, p90 = st
        # Outliers: points from the same sample the box summarises (via
        # _sample_for -- site means for a >=2-site class, else raw values) that
        # fall outside the whiskers. Using the finer site-year values here
        # instead would flag most of them, since within-site year-to-year spread
        # is much wider than the between-site-mean spread the box describes.
        sample = _sample_for(sub, group_col)
        fliers = sample[(sample < p10) | (sample > p90)]
        bxp_stats.append(
            dict(label=str(key), whislo=p10, q1=p25, med=p50, q3=p75, whishi=p90,
                 fliers=fliers)
        )
        bxp_pos.append(x)
        bxp_colors.append(col)

    if bxp_stats:
        bp = ax.bxp(
            bxp_stats,
            positions=bxp_pos,
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            boxprops=dict(linewidth=1.4),
            whiskerprops=dict(linewidth=1.6, color="#333333"),
            capprops=dict(linewidth=1.6, color="#333333"),
            medianprops=dict(linewidth=2.6, color="black"),
            # Same style as the sns.boxplot flierprops in Fig3b_AppxG10_G11's parameter
            # rows (FLIER_PROPS there) -- outliers used to be coloured per box
            # here but left at the seaborn default (dark, uncoloured) below,
            # which read as two different things rather than one "outliers"
            # convention applied to every row.
            flierprops=FLIER_PROPS,
        )
        for patch, col in zip(bp["boxes"], bxp_colors):
            patch.set_facecolor(col)
            # higher opacity so dark-green ENF and medium-green DBF stay distinct
            patch.set_alpha(0.85)
            patch.set_edgecolor("#333333")

    # Match seaborn's categorical x limits so a stacked figure lines up column
    # for column (seaborn uses -0.5 .. n-0.5; bxp would autoscale to the boxes).
    if positions:
        ax.set_xlim(min(positions) - 0.5, max(positions) + 0.5)
    ax.set_xticks(positions)
    if show_counts and group_col == "PFT":
        n_sites = df_pts.groupby("PFT").site.nunique()
        ax.set_xticklabels(
            [
                f"{k}\n({int(n_sites.get(k, 0))} site"
                f"{'' if int(n_sites.get(k, 0)) == 1 else 's'})"
                for k in order
            ]
        )
    else:
        ax.set_xticklabels([str(k) for k in order])

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(r"$T_{\mathrm{opt}}$ [°C]", fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    # Keep all 4 spines (sns.boxplot below doesn't hide top/right either) so
    # every row in the stacked figure has the same frame.

    if show_legend:
        ax.legend(handles=topt_legend_elements(), frameon=True, framealpha=0.9,
                  fontsize=12, loc="upper left")
    return ax


def topt_legend_elements(
    central_label=r"central $T_{opt}$ (V24)",
    iqr_label=r"ALPS p25-p75 (ens.)",
    whisker_label=r"ALPS p10-p90 (ens.)",
    outlier_label="outliers",
):
    """Legend entries explaining what the box, whiskers and dots mean.

    The labels are overridable so one legend can serve both the per-class and the
    per-site figure: the per-site boxes are percentiles of that site's own years,
    not the ALPS T_opt ensemble, so there the wording has to stay generic.
    """
    return [
        Patch(facecolor="#bbbbbb", edgecolor="#333333", alpha=0.55,
              label=iqr_label),
        Line2D([0], [0], color="#333333", lw=1.6, label=whisker_label),
        Line2D([0], [0], color="black", lw=2.6, label=central_label),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666",
               markeredgecolor="#666666", markersize=6, alpha=0.8,
               label=outlier_label),
    ]
