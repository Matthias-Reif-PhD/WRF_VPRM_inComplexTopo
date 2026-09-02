"""
Fig3a_Topt_perSiteYear_fit.py
==============================================
Derives the optimum temperature (Topt) per FLUXNET2015 site-year in the Alps and
produces Fig. 3a (fig:ToptAlps, panel a). Panel b is produced by
Fig3b_AppxG10_G11_Topt_tuneParam.py.

Historical note: this script used to draw both panels of Fig.~\\ref{fig:ToptAlps}:

  (a) optimum_temp_<PFT>_<site>_<year>.pdf  -- the per-site-year GPP-vs-T fit that
      locates Topt (Fig. ToptAlps a). By default only GRA_IT-MBo 2012 is saved.
  (b) boxplot_Topt_Alps.pdf                 -- the per-PFT Topt distribution used as
      the T_opt percentile sensitivity ensemble (Fig. ToptAlps b): box = p25-p75,
      whiskers = p10-p90, central line = V24 (per-site-mean) Topt.

GMD revision (RC2): Topt is now derived from GPP (GPP_NT_VUT_USTAR50) instead of NEE.
An upward Gaussian is fitted to the binned diurnal-mean GPP as a function of
temperature (peak = Topt); a cubic-maximum fallback is used where the Gaussian fit is
poorly constrained. Negative GPP (night-time partitioning artefacts) is clipped to 0.

The p10/p25/p50/p75/p90 shown in panel (b) are read from
VPRM_tools/topt_percentiles.csv (produced by VPRM_tools/derive_topt_members.py on a
per-site-mean basis) so that the figure is identical to the sigma_Topt ensemble used
in the seasonal decomposition. Only ENF/DBF/GRA have enough sites for an ensemble;
MF (1 site), SHB/SAV (no Alpine sites) and CRO (harvest artefacts) use defaults.

needs the file:
"$SCRATCH_PATH"/DATA/Fluxnet2015/Alps/site_info_all_FLUXNET2015.csv
and the FLUXNET2015 FLX_* site folders in "$SCRATCH_PATH"/DATA/Fluxnet2015/Alps
==============================================
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
from scipy.optimize import curve_fit, minimize
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ==================== Configuration ====================
SCRATCH_PATH = os.getenv("SCRATCH_PATH")
GITHUB_PATH = os.getenv("GITHUB_PATH")
OUTFOLDER = os.getenv("OUTFOLDER")

base_path = os.path.join(SCRATCH_PATH, "DATA/Fluxnet2015/Alps/")
plot_path = OUTFOLDER
site_info = pd.read_csv(
    os.path.join(SCRATCH_PATH, "DATA/Fluxnet2015/Alps/site_info_all_FLUXNET2015.csv")
)
# Per-PFT percentile ensemble (per-site-mean basis) -- panel (b) is drawn from this so
# it matches the sigma_Topt members used elsewhere in the revision.
PERC_CSV = ROOT / "VPRM_tools" / "topt_percentiles.csv"

plot_data = True  # save the single per-site-year GPP fit plots (panel a)
save_data = True  # save filtered_Topt_values.csv
plot_boxplot = True  # save the per-PFT boxplot (panel b)
# Only save these (site, year) single plots (keeps output churn small); set to None
# to save every site-year.
SINGLE_PLOTS_TO_SAVE = None  # None → save all site-years

# VPRM vegetation-class colours -- canonical palette, shared with Fig1/Fig2/E1ac.
from topt_box import PFT_COLORS as pft_colors

default_Topt = {
    "ENF": 20.0,
    "DBF": 20.0,
    "MF": 20.0,
    "SHB": 20.0,
    "SAV": 20.0,
    "CRO": 22.0,
    "GRA": 18.0,
}

# Nicer default look.
# Font/figsize convention for the paper: a figure is drawn at ~2x the size it is
# printed at, so the fonts below land at ~8 pt on the page (body text is ~9.5 pt).
# Fig. 3(a) is included at 0.45\linewidth = 225 pt = 3.13 in, hence ~6.25 in wide
# here (scale 0.52: 16 pt -> 8.3 pt printed).
# NOTE: the manuscript must use width=0.45\linewidth for these panels -- see
# FIGURE_TEX_CHANGES.md in the repo root.
FIGSIZE = (6.25, 4.6)  # panel (a): wide, leaves room for the legend underneath
FIGSIZE_BOX = (6.25, 5.5)  # panel (b): the standalone per-class Topt boxplot
plt.rcParams.update(
    {
        "font.size": 14,  # ticks / default text -> ~7.2 pt printed
        "axes.labelsize": 16,  # -> ~8.3 pt printed
        "axes.axisbelow": True,
        "axes.edgecolor": "#444444",
        "grid.color": "#cccccc",
        "grid.linewidth": 0.8,
        "svg.fonttype": "none",
    }
)


# ==================== Fit functions ====================
def gaussian(x, a, b, c):
    """Upward Gaussian -> models the GPP maximum; b is Topt."""
    return a * np.exp(-((x - b) ** 2) / (2 * c**2))


def cubic_polynomial(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def calculate_rmse(observed, predicted):
    return np.sqrt(np.mean((observed - predicted) ** 2))


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


# Read the per-PFT percentile ensemble (for the band in panel a and the boxes in b)
perc = None
if PERC_CSV.exists():
    perc = pd.read_csv(PERC_CSV).set_index("PFT")

# ==================== Per-site-year Topt from GPP ====================
min_nee_temp_dict = {}
min_nee_temp_dict2 = {}

for folder in sorted(os.listdir(base_path)):
    if not folder.startswith("FLX_"):
        continue

    file_base = "_".join(folder.split("_")[0:4])
    yrs = "_".join(folder.split("_")[4:6])
    file_path = os.path.join(base_path, folder, f"{file_base}_HH_{yrs}.csv")

    # Extract site name and PFT
    site_id = folder.split("_")[1]
    target_pft = None
    for i, site_i in enumerate(site_info["site"]):
        if site_id == site_i:
            target_pft = site_info["pft"][i]
            if target_pft == "EBF":
                target_pft = "DBF"
            if target_pft == "OSH":  # TODO check OSH
                target_pft = "SHB"
            if site_id == "AT-Mie":
                target_pft = "ENF"
    if target_pft is None:
        print(f"No PFT found for {site_id}, skipping.")
        continue
    site_name = target_pft + "_" + site_id

    # Load the data
    columns_to_copy = ["TIMESTAMP_START", "TA_F", "GPP_NT_VUT_USTAR50", "NIGHT"]
    converters = {k: (lambda x: float(x)) for k in columns_to_copy}
    df_site = pd.read_csv(file_path, usecols=columns_to_copy, converters=converters)
    df_site["TIMESTAMP_START"] = pd.to_datetime(
        df_site["TIMESTAMP_START"], format="%Y%m%d%H%M"
    )

    # Clean the data
    df_site["TA_F"] = df_site["TA_F"].replace(-9999.0, np.nan)
    df_site = df_site.dropna(subset=["TA_F"])
    df_site["GPP_NT_VUT_USTAR50"] = df_site["GPP_NT_VUT_USTAR50"].replace(-9999.0, np.nan)
    nan_sum = df_site["GPP_NT_VUT_USTAR50"].isna().sum()
    full_year = len(df_site["GPP_NT_VUT_USTAR50"])
    df_site = df_site.dropna(subset=["GPP_NT_VUT_USTAR50"])
    percent_nan = nan_sum / full_year * 100 if full_year else 100
    if percent_nan > 20:
        print(f"WARNING: {site_name} skipped, {percent_nan:2.1f}% GPP missing")
        continue

    # Daytime only, then clip negative GPP (night-time partitioning artefact)
    df_site.loc[df_site["NIGHT"] == 1, "GPP_NT_VUT_USTAR50"] = np.nan
    df_site["GPP_NT_VUT_USTAR50"] = df_site["GPP_NT_VUT_USTAR50"].clip(lower=0)

    # Resample to daily
    df_site.set_index("TIMESTAMP_START", inplace=True)
    df_daily = (
        df_site.resample("D")
        .agg({"TA_F": "mean", "GPP_NT_VUT_USTAR50": "mean"})
        .dropna()
    )
    df_daily["YEAR"] = df_daily.index.year

    min_nee_temp_dict.setdefault(site_name, {})
    min_nee_temp_dict2.setdefault(site_name, {})

    for year in df_daily["YEAR"].unique():
        df_year = df_daily[df_daily["YEAR"] == year].copy()
        df_year = df_year[df_year["TA_F"] >= 3]
        if df_year.empty:
            continue

        df_year.loc[:, "TA_F_rounded"] = df_year["TA_F"].round()
        mean_values = df_year.groupby("TA_F_rounded").mean()

        Topt = np.nan
        fit_func, fit_popt = None, None
        rmse_threshold = 2.0

        # 1) upward Gaussian
        try:
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
            Topt = popt[1]
            fit_func, fit_popt = gaussian, popt
            rmse = calculate_rmse(
                mean_values["GPP_NT_VUT_USTAR50"], gaussian(mean_values.index, *popt)
            )
            if site_name == "ENF_DE-Lbk" and year == 2012:
                rmse_threshold = 1
            if rmse > rmse_threshold:
                raise RuntimeError("Gaussian fit RMSE too high")

            if Topt < df_year["TA_F_rounded"].max():
                real_Topt_col = "green"
            elif Topt > df_year["TA_F_rounded"].max():
                real_Topt_col = "yellow"  # extrapolated
            else:
                Topt = default_Topt.get(target_pft)
                real_Topt_col = "red"

        # 2) cubic-maximum fallback
        except RuntimeError:
            try:
                popt_poly, _ = curve_fit(
                    cubic_polynomial,
                    mean_values.index,
                    mean_values["GPP_NT_VUT_USTAR50"],
                    maxfev=10000,
                )
                Topt = find_maximum_of_cubic_poly(popt_poly, mean_values.index)
                fit_func, fit_popt = cubic_polynomial, popt_poly
                real_Topt_col = (
                    "yellow" if Topt > df_year["TA_F_rounded"].max() else "green"
                )
            except RuntimeError:
                Topt = default_Topt.get(target_pft)
                real_Topt_col = "red"

        # CRO: harvest events disturb the data -> default
        if target_pft == "CRO":
            Topt = default_Topt.get(target_pft)
            real_Topt_col = "red"
        # implausible -> default
        if Topt is None or not np.isfinite(Topt) or Topt < 5 or Topt > 30:
            Topt = default_Topt.get(target_pft)
            real_Topt_col = "red"

        min_nee_temp_dict[site_name][year] = Topt
        min_nee_temp_dict2[site_name][year] = (Topt, real_Topt_col)

        # ---------- panel (a): single per-site-year GPP fit ----------
        save_this = plot_data and (
            SINGLE_PLOTS_TO_SAVE is None or (site_name, year) in SINGLE_PLOTS_TO_SAVE
        )
        if save_this:
            col = pft_colors.get(target_pft, "#1f77b4")
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.grid(True, alpha=0.6)

            # binned data (markers 1.5x the original s=55)
            ax.scatter(
                mean_values.index,
                mean_values["GPP_NT_VUT_USTAR50"],
                s=82.5, color=col, edgecolor="white", linewidth=0.6, zorder=3,
                label=r"binned GPP per $T_{\mathrm{2m}}$",
            )
            # smooth fitted curve
            if fit_func is not None:
                xs = np.linspace(
                    float(mean_values.index.min()),
                    float(mean_values.index.max()),
                    250,
                )
                ax.plot(
                    xs, fit_func(xs, *fit_popt),
                    color="#b22222", lw=2.4, zorder=4, label="fitted curve",
                )
            # site-year Topt
            if np.isfinite(Topt):
                ax.axvline(
                    Topt, color="black", ls="--", lw=2, zorder=5,
                    label=rf"SITE $T_{{\mathrm{{opt}}}}$ = {Topt:.1f} °C",
                )

            # PFT p10-p90 ensemble band (panel a link to the sensitivity measure).
            # Grey-shaded band + black central line; wording matches the ALPS
            # Topt-percentile legend used in Fig4/Fig5_7 (same underlying quantity).
            # Drawn last so these entries sort to the end of the legend; the grey
            # patches' default zorder (~1) still keeps them behind the scatter/fit/
            # site-Topt line (zorder 3-5) regardless of this draw order.
            if perc is not None and target_pft in perc.index:
                r = perc.loc[target_pft]
                ax.axvspan(
                    r["p10"], r["p90"], color="#bbbbbb", alpha=0.35,
                    label=r"ALPS p10-p90 (ens.)",
                )
                ax.axvspan(
                    r["p25"], r["p75"], color="#555555", alpha=0.55,
                    label=r"ALPS p25-p75 (ens.)",
                )
                ax.axvline(
                    r["p50"], color="black", ls="-", lw=2,
                    label=rf"ALPS $T_\text{{opt}}$ = {r['p50']:.1f} °C",
                )

            ax.set_xlabel(r"$T_{\mathrm{2m}}$ [°C]")
            ax.set_ylabel(r"GPP [$\mu$mol m$^{-2}$ s$^{-1}$]")
            ax.margins(x=0.02)
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
            # Legend below the axes: at the printed panel width (0.30\linewidth)
            # these six entries no longer fit inside without covering the fitted
            # curve, so they go underneath in two columns.
            ax.legend(
                frameon=True, framealpha=0.9, fontsize=13,
                loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2,
                handlelength=1.8, columnspacing=1.2, borderaxespad=0.0,
            )
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            fig.tight_layout()
            fig.savefig(
                os.path.join(plot_path, f"optimum_temp_{site_name}_{year}.pdf"),
                dpi=300, bbox_inches="tight",
            )
            plt.close(fig)


# ==================== Save filtered per-site-year Topt ====================
filtered_dict = {
    site: {yr: t for yr, (t, c) in yd.items() if c != "red"}
    for site, yd in min_nee_temp_dict2.items()
}
filtered_df = pd.DataFrame(filtered_dict).T.sort_index()
if save_data:
    filtered_df.to_csv(os.path.join(plot_path, "filtered_Topt_values.csv"))

counts = {
    s: sum(isinstance(v[0], float) and np.isfinite(v[0]) for v in yd.values())
    for s, yd in min_nee_temp_dict2.items()
}
print("valid Topt entries per site:", counts)
print("Total valid Topt entries:", sum(counts.values()))


# ==================== panel (b): per-PFT Topt boxplot ====================
if plot_boxplot:
    # Per-site-year "real" (non-fallback) Topt, grouped by PFT
    rows = []
    for site, yd in min_nee_temp_dict2.items():
        pft = site.split("_")[0]
        for yr, (t, c) in yd.items():
            if c != "red" and isinstance(t, (int, float)) and np.isfinite(t):
                rows.append({"PFT": pft, "site": site, "year": yr, "Topt": float(t)})
    df_pts = pd.DataFrame(rows)

    # Which PFTs get a box: ensemble PFTs first, then any remaining with data
    perc_pfts = list(perc.index) if perc is not None else []
    extra_pfts = [p for p in df_pts["PFT"].unique() if p not in perc_pfts]
    pft_order = perc_pfts + sorted(extra_pfts)

    fig, ax = plt.subplots(figsize=FIGSIZE_BOX)
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_axisbelow(True)

    bxp_stats = []
    for pft in pft_order:
        if perc is not None and pft in perc.index:
            r = perc.loc[pft]
            p10, p25, p50, p75, p90 = (
                r["p10"], r["p25"], r["p50"], r["p75"], r["p90"]
            )
        else:  # fallback: per-site means if ≥2 sites, else individual year values
            sub = df_pts[df_pts.PFT == pft]
            sm = (sub.groupby("site").Topt.mean().values
                  if sub.site.nunique() >= 2 else sub["Topt"].values)
            p10, p25, p50, p75, p90 = np.percentile(sm, [10, 25, 50, 75, 90])
        bxp_stats.append(
            dict(
                label=pft, whislo=p10, q1=p25, med=p50, q3=p75, whishi=p90, fliers=[]
            )
        )

    positions = range(1, len(pft_order) + 1)
    bp = ax.bxp(
        bxp_stats,
        positions=list(positions),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.4),
        whiskerprops=dict(linewidth=1.6, color="#333333"),
        capprops=dict(linewidth=1.6, color="#333333"),
        medianprops=dict(linewidth=2.6, color="black"),
    )
    for patch, pft in zip(bp["boxes"], pft_order):
        patch.set_facecolor(pft_colors.get(pft, "#8FBC8F"))
        patch.set_alpha(0.85)  # higher opacity so dark-green ENF and medium-green DBF stay distinct
        patch.set_edgecolor("#333333")

    # Overlay: faint site-year cloud + prominent per-site means (= the sample the
    # p10/p25/p50/p75/p90 are computed on, so all means sit inside the whiskers).
    rng = np.random.default_rng(0)
    for x, pft in zip(positions, pft_order):
        sub = df_pts[df_pts.PFT == pft]
        yr_vals = sub["Topt"].values
        ax.scatter(
            np.full_like(yr_vals, x, dtype=float)
            + rng.uniform(-0.16, 0.16, size=len(yr_vals)),
            yr_vals,
            s=16, color=pft_colors.get(pft, "#333333"),
            alpha=0.30, zorder=3, linewidth=0,
        )
        site_means = sub.groupby("site").Topt.mean().values
        ax.scatter(
            np.full_like(site_means, x, dtype=float)
            + rng.uniform(-0.10, 0.10, size=len(site_means)),
            site_means,
            s=70, color=pft_colors.get(pft, "#333333"),
            edgecolor="black", linewidth=1.0, alpha=0.95, zorder=5,
        )

    # x tick labels with site count
    n_sites = df_pts.groupby("PFT").site.nunique()
    ax.set_xticks(list(positions))
    ax.set_xticklabels(
        [f"{p}\n({int(n_sites.get(p, 0))} site{'' if int(n_sites.get(p, 0)) == 1 else 's'})"
         for p in pft_order]
    )
    ax.set_xlabel("vegetation class", fontsize=16)
    ax.set_ylabel(r"$T_{\mathrm{opt}}$ [°C]", fontsize=16)
    ax.tick_params(labelsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    legend_elems = [
        Patch(facecolor="#bbbbbb", edgecolor="#333333", alpha=0.55,
              label=r"ALPS p25-p75 (ens.)"),
        Line2D([0], [0], color="#333333", lw=1.6, label=r"ALPS p10-p90 (ens.)"),
        Line2D([0], [0], color="black", lw=2.6, label="central $T_{opt}$ (V24)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666",
               markeredgecolor="black", markersize=9, label="site means"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666",
               markersize=5, alpha=0.4, label="site-years"),
    ]
    ax.legend(handles=legend_elems, frameon=True, framealpha=0.9,
              fontsize=13, loc="upper left")

    fig.tight_layout()
    fig.savefig(
        os.path.join(plot_path, "boxplot_Topt_Alps.pdf"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)
    print("Wrote", os.path.join(plot_path, "boxplot_Topt_Alps.pdf"))
