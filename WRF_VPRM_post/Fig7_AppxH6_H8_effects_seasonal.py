"""
Fig7_AppxH6_H8_effects_seasonal.py
==============================================
Fig. 7 (fig:effects_all), Fig. H6 (fig:effects_clear_fig) and Fig. H8
(fig:effects_cloudy_fig): the seasonal decomposition
of the effect sizes, one panel per flux (GPP, R_eco, NEE), for the pooled
24-day sample and the two 12-day weather regimes (clear sky / clouds and
rain).

Bars per panel, in the same order as the shipped tables' columns:
  Delta_par^p25, Delta_par^p75, Delta_par            (parameter effect, ALPS
                                                       subtracted -- Eq. 2)
  Delta_res^9 ALPS,  Delta_res^9 DEFAULT
  Delta_res^54 ALPS, Delta_res^54 DEFAULT            (resolution effect, Eq. 1)
R_eco has no Topt dependence, so its two percentile slots stay empty.

Delta_par, Delta_par^p25/p75 and Delta_res^9/54 (ALPS) are exactly what
write_seasonal_effects_table[_combined] in
Fig5_6_AppxG5_G6_Table2_TableH_WRFout_hourly_means_and_timeseries.py computes and ships in
Table_seasonal_effects_2012{,_cloudy,_all}.tex -- this script calls the same
_smean/_member_tagmeans helpers on the same per-tag/DEFAULT dataframes (via
the shared load_regime loader) rather than re-deriving the numbers, so the
figure cannot silently diverge from what those tables (and
Table_1_domain_averaged_2012.tex, which shares Table_seasonal_effects'
Delta_par sign convention via _t1_flux_row) print. Delta_res(DEFAULT) has no
ALPS/Topt dependence at all and is not carried by Table_seasonal_effects, so
it is computed directly from the DEFAULT dataframe the same way.

Before anything is drawn, the pooled sample's Delta_res^9/54(DEFAULT) "24 d"
values are cross-checked against the shipped
plots_final/Table_1_domain_averaged_2012.tex (whose 9/54 km DEFAULT columns
carry exactly this quantity) -- this caught two errors during development of
the original (local) version of this script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from Fig5_6_AppxG5_G6_Table2_TableH_WRFout_hourly_means_and_timeseries import (
    ALPS_TAGS,
    MEDIAN_TAG,
    SEASONS,
    _cell_delta,
    _member_tagmeans,
    _read_blocks,
    _signed,
    _smean,
    load_regime,
)

SCRATCH_PATH = os.getenv("SCRATCH_PATH")
GITHUB_PATH = os.getenv("GITHUB_PATH")
OUTFOLDER = os.getenv("OUTFOLDER")

FLUXES = [("GPP", "GPP"), ("RECO", r"$R_\text{eco}$"), ("NEE", "NEE")]


def _row(p50, ref, member_dfs, months, var):
    """One (season, flux) record: all seven quantities the bars need, keyed
    by their table-column slot name. RECO has no p25/p75 keys at all (no
    Topt dependence), which is what makes its two percentile bar slots
    empty further down -- same convention Table_seasonal_effects_* uses."""
    row = {
        "par": _smean(ref, f"{var}_1km", months) - _smean(p50, f"{var}_1km", months),
        "r9": _smean(p50, f"{var}_9km", months) - _smean(p50, f"{var}_1km", months),
        "r54": _smean(p50, f"{var}_54km", months) - _smean(p50, f"{var}_1km", months),
        "r9d": _smean(ref, f"{var}_9km", months) - _smean(ref, f"{var}_1km", months),
        "r54d": _smean(ref, f"{var}_54km", months) - _smean(ref, f"{var}_1km", months),
    }
    if var != "RECO":
        tagmeans = _member_tagmeans(member_dfs, var, months)
        row["p25"] = tagmeans[1] - tagmeans[2]
        row["p75"] = tagmeans[3] - tagmeans[2]
    return row


def _build(p50, ref, member_dfs):
    """-> {row: {flux: {slot: value}}}; rows are the four seasons plus
    'Last' (the 12/24-day bar), which is the mean of the four seasonal rows
    -- not a fresh recompute over all months -- matching the Year-row
    convention write_seasonal_effects_table already uses and verifies."""
    d = {name: {var: _row(p50, ref, member_dfs, months, var) for var, _ in FLUXES}
         for name, months in SEASONS}
    d["Last"] = {
        var: {k: float(np.mean([d[name][var][k] for name, _ in SEASONS]))
              for k in d[SEASONS[0][0]][var]}
        for var, _ in FLUXES
    }
    return d


def _load_samples():
    csv_folder = os.path.join(GITHUB_PATH, "WRF_VPRM_inComplexTopo/WRF_VPRM_post/csv/")
    start_date, end_date = "2012-01-01 00:00:00", "2012-12-31 00:00:00"
    STD_TOPO = 200
    columns = ["GPP", "RECO", "NEE", "T2", "SWDOWN"]
    resolutions = ["1km", "9km", "54km", "CAMS"]

    regimes = {
        sim: load_regime(sim, csv_folder, STD_TOPO, start_date, end_date, resolutions,
                          columns, ref_sim=True)
        for sim in ("", "_cloudy")
    }
    p50 = {sim: regimes[sim]["alps"][ALPS_TAGS.index(MEDIAN_TAG)] for sim in regimes}
    ref = {sim: regimes[sim]["ref"] for sim in regimes}
    members = {sim: regimes[sim]["alps"] for sim in regimes}

    p50_pooled = pd.concat([p50[""], p50["_cloudy"]])
    ref_pooled = pd.concat([ref[""], ref["_cloudy"]])
    members_pooled = [pd.concat([c, d]) for c, d in zip(members[""], members["_cloudy"])]

    # name -> (p50, ref, member_dfs, x-tick label for the 'Last' bar, filename suffix)
    return {
        "all": (p50_pooled, ref_pooled, members_pooled, "24 d", ""),
        "clear": (p50[""], ref[""], members[""], "12 d", "_clear"),
        "cloudy": (p50["_cloudy"], ref["_cloudy"], members["_cloudy"], "12 d", "_cloudy"),
    }


SAMPLES = _load_samples()
D = {name: _build(*args[:3]) for name, args in SAMPLES.items()}

# ---- cross-check the pooled DEFAULT resolution effect against Table 1 --------
_table1 = f"{OUTFOLDER}plots_final/Table_1_domain_averaged_2012.tex"
_blocks = _read_blocks(_table1)
_bad = 0
for var, _lbl in FLUXES:
    cells = _blocks["24-day average"][var]
    exp9, exp54 = _cell_delta(cells[3]), _cell_delta(cells[5])  # 9/54 km DEFAULT parens
    got9, got54 = _signed(D["all"]["Last"][var]["r9d"]), _signed(D["all"]["Last"][var]["r54d"])
    for got, exp, lbl in ((got9, exp9, "9 km"), (got54, exp54, "54 km")):
        ok = got == exp
        print(f"  [{'ok' if ok else 'FAIL'}] {var:5s} Delta_res({lbl}, DEFAULT): "
              f"reconstructed {got}  vs  Table 1 {exp}")
        if not ok:
            _bad += 1
print(f"assert: DEFAULT Delta_res reconstruction vs Table 1 -> {'OK' if not _bad else str(_bad) + ' BAD'}")
assert not _bad

# ---- figures ------------------------------------------------------------------
C = {"p25": "#A5D6A7", "p75": "#A5D6A7", "par": "#2E7D32",
     "r9": "blue", "r9d": "lightskyblue", "r54": "red", "r54d": "lightcoral"}
# MWR (28 Aug 2026): the p25/p75 calibrations are variants of the ALPS parameter
# set, so they must read as one green family with the ALPS bar rather than as
# warm colours the eye groups against the blue/red resolution bars. Hatching,
# leaning opposite ways, distinguishes the two variants without leaving green.
H = {"p25": "///", "p75": "\\\\\\", "par": "", "r9": "", "r9d": "",
     "r54": "", "r54d": ""}
E = {"p25": "#2E7D32", "p75": "#2E7D32"}
# every bar is referenced to ALPS, so the ones that are not the DEFAULT counterpart say so
LBL = {"p25": r"$\Delta_{\mathrm{par}}^{p25}$, ALPS",
       "p75": r"$\Delta_{\mathrm{par}}^{p75}$, ALPS",
       "par": r"$\Delta_{\mathrm{par}}$, ALPS",
       "r9": r"$\Delta_{\mathrm{res}}$ 9 km, ALPS",
       "r9d": r"$\Delta_{\mathrm{res}}$ 9 km, DEFAULT",
       "r54": r"$\Delta_{\mathrm{res}}$ 54 km, ALPS",
       "r54d": r"$\Delta_{\mathrm{res}}$ 54 km, DEFAULT"}
SLOTS = ["p25", "p75", "par", "r9", "r9d", "r54", "r54d"]
OFF = {k: i - 3 for i, k in enumerate(SLOTS)}
BAR_W = 0.1265
# one empty bar slot in front of each group, so consecutive groups are set apart
PITCH = 1.0 + BAR_W
GAP = 0.35                       # wider break before the 12 d / 24 d group
FIG_H, AX_H = 2.6, 2.10
# hold the rendered bar width fixed while the x-range grows by the extra slot
AX_W = 1.43 * ((4 * PITCH + GAP + 1.0) / 5.35)
BOTTOM, RIGHT_PAD = 0.42, 0.06
LEFT_LBL, LEFT_NOLBL = 0.56, 0.10
FIGW_LBL, FIGW_NOLBL = LEFT_LBL + AX_W + RIGHT_PAD, LEFT_NOLBL + AX_W + RIGHT_PAD
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})

vals = [D[n][r][var][k] for n in D for var, _ in FLUXES
        for r in [s for s, _ in SEASONS] + ["Last"]
        for k in SLOTS if k in D[n][r][var]]
lo, hi = min(vals), max(vals)
pad = 0.07 * (hi - lo)
YLIM = (lo - pad, hi + pad)

X = np.array([0, PITCH, 2 * PITCH, 3 * PITCH, 4 * PITCH + GAP], float)
geom = set()
for name, (_p50, _ref, _members, lastlbl, suf) in SAMPLES.items():
    d = D[name]
    xt = [s for s, _ in SEASONS] + [lastlbl]
    for var, _lbl in FLUXES:
        lbl = var == "GPP"
        figw = FIGW_LBL if lbl else FIGW_NOLBL
        fig = plt.figure(figsize=(figw, FIG_H))
        ax = fig.add_axes([(LEFT_LBL if lbl else LEFT_NOLBL) / figw, BOTTOM / FIG_H,
                           AX_W / figw, AX_H / FIG_H])
        for k in SLOTS:
            if k not in d["DJF"][var]:
                continue
            ax.bar(X + OFF[k] * BAR_W,
                   [d[r][var][k] for r, _ in SEASONS] + [d["Last"][var][k]],
                   BAR_W, color=C[k], label=LBL[k], hatch=H[k],
                   edgecolor=E.get(k, "none"), linewidth=0.3)
        ax.axhline(0, color="k", lw=0.6)
        for i in range(3):                       # thin dotted rule between seasons
            ax.axvline((X[i] + X[i + 1]) / 2, color="0.75", lw=0.4, ls=":", zorder=0)
        ax.axvline((X[3] + X[4]) / 2, color="0.45", lw=1.1, ls="--", zorder=0)
        ax.set_xlim(-0.55, X[4] + 0.45)
        ax.set_ylim(*YLIM)
        ax.set_xticks(X)
        ax.set_xticklabels(xt, rotation=90)
        if lbl:
            ax.set_ylabel(r"$\mu$mol m$^{-2}$ s$^{-1}$", labelpad=1)
        else:
            ax.set_yticklabels([])
        if var == "NEE":
            lg = ax.legend(fontsize=4.9, loc="lower right", handlelength=1.1,
                           borderpad=0.3, labelspacing=0.25, frameon=True,
                           facecolor="white", edgecolor="0.75", framealpha=1.0)
            lg.get_frame().set_linewidth(0.4)
            lg.set_zorder(5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(f"{OUTFOLDER}effects_seasonal_{var}{suf}.pdf")
        plt.close(fig)
        b = ax.get_position().bounds
        geom.add((round(b[2] * figw, 6), round(b[3] * FIG_H, 6), ax.get_ylim()))
    print("wrote 3 panels for", name)

assert len(geom) == 1, f"panels differ in axes box or y-limits: {geom}"
w, h, yl = geom.pop()
print(f"assert: all 9 panels share axes box {w:.3f} x {h:.3f} in, ylim "
      f"{yl[0]:+.2f}..{yl[1]:+.2f}, nothing clipped")
print(f"LaTeX widths: GPP 0.341\\linewidth   RECO/NEE "
      f"{0.341 * FIGW_NOLBL / FIGW_LBL:.3f}\\linewidth")
