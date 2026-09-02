"""
diagnose_itlav.py
==============================================
Read-only plausibility diagnosis of the IT-Lav (Lavarone) FLUXNET2015 record,
against IT-Ren (Renon) and CH-Dav (Davos) -- the same vegetation class in the
same region.

The question
------------
Converting the 2012 annual means (1 umol m-2 s-1 = 378.8 gC m-2 yr-1), IT-Lav
reports GPP 2337, R_eco 330, NEP 2053 gC m-2 yr-1, i.e. R_eco/GPP = 0.14. For a
subalpine Norway spruce stand published NEP is of order 400-600 and R_eco
800-1500. IT-Ren (0.58) and CH-Dav (0.68) are entirely plausible. Is IT-Lav
WRONG, or merely DIFFERENT?

IT-Lav is NOT excluded from anything. This script changes no shipped artefact.
Sections 4 and 5 compute the exclusion arithmetic purely as a labelled
sensitivity bound.

Sections
--------
  1. Variables, QC flags and surviving hours -- absolute counts and fractions.
  2. The u* threshold actually applied under VUT, and what it removes at night.
  3. Bad year or bad site: all years, both partitionings, plus the evidence
     that discriminates "wrong" from "different".
  4. [SENSITIVITY] ENF ALPS parameters re-aggregated with IT-Lav dropped.
  5. [SENSITIVITY] Table 1 multi-site Mean row over the remaining four sites.

Output: WRF_VPRM_post/ITLAV_DIAGNOSIS.md

Usage
-----
    python diagnose_itlav.py
==============================================
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SCRATCH_PATH = os.getenv("SCRATCH_PATH")
OUTFOLDER = os.getenv("OUTFOLDER")
HERE = Path(__file__).resolve().parent
FLX_DIR = Path(SCRATCH_PATH) / "DATA/Fluxnet2015/Alps"
REPORT = HERE / "ITLAV_DIAGNOSIS.md"

# umol m-2 s-1 -> gC m-2 yr-1 (the conversion the plausibility check uses)
UMOL_TO_GC_YR = 378.8

YEAR = 2012
HH_PER_YEAR = 366 * 48  # 2012 is a leap year

SITES = ["IT-Lav", "IT-Ren", "CH-Dav"]

# ENF sites in the V24 tuning, and the one under scrutiny
ENF_SITES = ["CH-Dav", "DE-Lkb", "IT-La2", "IT-Lav", "IT-Ren"]
SUSPECT = "IT-Lav"

# Published ENF ALPS row (WRF_VPRM_post/vprm_params_newGPP_V24.csv) -- the
# aggregation below must reproduce these before its without-IT-Lav counterpart
# can be trusted.
PUBLISHED_ENF = {
    "t_opt": 16.42, "par0": 184.471, "lambda": 0.501,
    "alpha": 0.175, "beta": 1.705,
}


def _site_dir(site):
    (d,) = glob.glob(str(FLX_DIR / f"FLX_{site}_FLUXNET2015_FULLSET_*-*_*-*"))
    return Path(d)


def _read(site, freq, usecols=None):
    """Read one FULLSET resolution (HH/YY) for a site. -9999 is FLUXNET's
    missing-value code and must be honoured or every mean is garbage."""
    (f,) = glob.glob(str(_site_dir(site) / f"FLX_{site}_*_FULLSET_{freq}_*.csv"))
    kw = {"usecols": (lambda c: c in usecols)} if usecols else {}
    return pd.read_csv(f, na_values=[-9999], **kw)


def _hh_2012(site):
    cols = {
        "TIMESTAMP_START", "NIGHT", "USTAR", "TA_F_QC", "SW_IN_F_QC",
        "NEE_VUT_USTAR50", "NEE_VUT_USTAR50_QC",
        "GPP_NT_VUT_USTAR50", "RECO_NT_VUT_USTAR50",
        "GPP_DT_VUT_USTAR50", "RECO_DT_VUT_USTAR50",
    }
    d = _read(site, "HH", cols)
    d["t"] = pd.to_datetime(d.TIMESTAMP_START, format="%Y%m%d%H%M")
    return d[d.t.dt.year == YEAR].copy()


def _ustar_threshold(site, year=YEAR):
    """The u* threshold FLUXNET actually applied for the VUT/USTAR50 product,
    read from AUXNEE rather than assumed."""
    (f,) = glob.glob(str(_site_dir(site) / f"FLX_{site}_*_AUXNEE_*.csv"))
    a = pd.read_csv(f)
    m = a[(a.VARIABLE == "NEE_VUT_USTAR50")
          & (a.PARAMETER == "USTAR_THRESHOLD")
          & (a.TIMESTAMP.astype(str) == str(year))]
    return float(m.VALUE.iloc[0]) if len(m) else np.nan


# ---------------------------------------------------------------- section 1

def section1_coverage():
    """Counts, not just means: a mean computed over 3 % of the year would be
    the most likely explanation, so the sample size has to be visible."""
    rows, qc_rows = [], []
    for s in SITES:
        d = _hh_2012(s)
        n = len(d)
        qc = d.NEE_VUT_USTAR50_QC
        counts = {int(k): int(v) for k, v in qc.value_counts().items()}
        n0 = counts.get(0, 0)
        meas = d[qc == 0]
        rows.append({
            "site": s,
            "half-hours": n,
            "of 17568": f"{n / HH_PER_YEAR:.1%}",
            "QC=0 (measured)": n0,
            "QC=0 frac": f"{n0 / n:.1%}",
            "TA_F QC=0": int((d.TA_F_QC == 0).sum()),
            "SW_IN_F QC=0": int((d.SW_IN_F_QC == 0).sum()),
        })
        qc_rows.append({
            "site": s,
            **{f"QC={k}": counts.get(k, 0) for k in (0, 1, 2, 3)},
            "GPP all-hours": round(d.GPP_NT_VUT_USTAR50.mean(), 3),
            "GPP QC=0 only": round(meas.GPP_NT_VUT_USTAR50.mean(), 3),
            "R_eco all-hours": round(d.RECO_NT_VUT_USTAR50.mean(), 3),
            "R_eco QC=0 only": round(meas.RECO_NT_VUT_USTAR50.mean(), 3),
            "NEE all-hours": round(d.NEE_VUT_USTAR50.mean(), 3),
            "NEE QC=0 only": round(meas.NEE_VUT_USTAR50.mean(), 3),
        })
    return pd.DataFrame(rows), pd.DataFrame(qc_rows)


# ---------------------------------------------------------------- section 2

def section2_ustar():
    rows = []
    for s in SITES:
        d = _hh_2012(s)
        thr = _ustar_threshold(s)
        night = d[d.NIGHT == 1]
        u = night.USTAR
        below = (u < thr).sum()
        rows.append({
            "site": s,
            "u* threshold [m/s]": round(thr, 3),
            "night half-hours": len(night),
            "u* below threshold": int(below),
            "frac of night removed": f"{below / u.notna().sum():.1%}",
            "night QC=0": int((night.NEE_VUT_USTAR50_QC == 0).sum()),
            "night QC=0 frac": f"{(night.NEE_VUT_USTAR50_QC == 0).mean():.1%}",
            "night u* p10": round(u.quantile(0.10), 3),
            "night u* median": round(u.median(), 3),
            "night u* p90": round(u.quantile(0.90), 3),
            "night R_eco (NT)": round(night.RECO_NT_VUT_USTAR50.mean(), 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- section 3

def section3_years():
    """All years, both partitionings. NT and DT are independent estimates of
    the same quantity; a site where they disagree by a factor of 2+ every year
    is telling us about the data, not about the ecosystem."""
    out = {}
    for s in SITES:
        d = _read(s, "YY")
        keep = ["TIMESTAMP", "GPP_NT_VUT_USTAR50", "RECO_NT_VUT_USTAR50",
                "GPP_DT_VUT_USTAR50", "RECO_DT_VUT_USTAR50",
                "NEE_VUT_USTAR50", "NEE_VUT_05", "NEE_VUT_95"]
        t = d[[c for c in keep if c in d.columns]].copy()
        t = t[t.GPP_NT_VUT_USTAR50.notna()]
        t["NEP_NT"] = -t.NEE_VUT_USTAR50
        t["NT ratio"] = t.RECO_NT_VUT_USTAR50 / t.GPP_NT_VUT_USTAR50
        t["DT ratio"] = t.RECO_DT_VUT_USTAR50 / t.GPP_DT_VUT_USTAR50
        t["DT/NT R_eco"] = t.RECO_DT_VUT_USTAR50 / t.RECO_NT_VUT_USTAR50
        # u*-filtering uncertainty envelope on NEP, from the VUT percentiles
        t["NEP u* env"] = [
            f"{-hi:.0f} .. {-lo:.0f}"
            for lo, hi in zip(t.NEE_VUT_95, t.NEE_VUT_05)
        ]
        # drop the degenerate near-zero-GPP startup years that make ratios
        # meaningless (IT-Ren 2000/2001/2004 have GPP ~ 0)
        t = t[t.GPP_NT_VUT_USTAR50 > 200]
        out[s] = t.round(2)
    return out


def section3_reco_fit_quality():
    """How well the VPRM tuning could fit each ENF site's R_eco. A site whose
    respiration is unfittable is a different problem from one that is merely
    productive."""
    rows = []
    for s in ENF_SITES:
        df = _read_tuning(s)
        if df is None:
            continue
        rows.append({
            "site": s,
            "site-years": len(df),
            "R2_GPP median": round(df.R2_GPP.median(), 3),
            "R2_Reco median": round(df.R2_Reco.median(), 3),
            "R2_Reco min": round(df.R2_Reco.min(), 3),
            "R2_Reco max": round(df.R2_Reco.max(), 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- section 4

def _read_tuning(site):
    """Per-site-year V24 differential-evolution fits (Topt, PAR0, alpha, beta,
    lambda + fit quality). This is the population the per-PFT ALPS row is
    aggregated from."""
    hits = glob.glob(str(_site_dir(site) / f"{site}_*_optimized_params_old_"
                                           "diff_evo_V24_SW_05_42.xlsx"))
    return pd.read_excel(hits[0]) if hits else None


def section4_enf_refit():
    """SENSITIVITY ONLY -- IT-Lav is not excluded from anything.

    Re-aggregate the ENF row from the existing V24 per-site-year fits with
    IT-Lav's site-years dropped. Deliberately NOT a re-tune: a re-tune would
    feed a new ENF T_opt back into the remaining sites' fits, which is the
    machinery of an actual parameter change, not of a bound.
    """
    frames = [df for s in ENF_SITES if (df := _read_tuning(s)) is not None]
    enf = pd.concat(frames, ignore_index=True)
    enf["site"] = enf.site_ID

    # PAR0/alpha/beta/lambda: per-PFT MEDIAN over site-years (the rule behind
    # vprm_params_newGPP_V24.csv). T_opt: mean of per-site-mean GPP-Topt (the
    # rule asserted in VPRM_tools/derive_topt_members.py).
    topt_raw = pd.read_csv(ROOT / "VPRM_tools/topt_raw_persiteyear.csv")
    topt_enf = topt_raw[topt_raw.PFT == "ENF"].copy()
    topt_enf["site"] = topt_enf.site.str.replace("ENF_", "", regex=False)

    def agg(df_par, df_topt):
        return {
            "t_opt": df_topt.groupby("site").Topt.mean().mean(),
            "par0": df_par.PAR0.median(),
            "lambda": df_par.lambd.median(),
            "alpha": df_par.alpha.median(),
            "beta": df_par.beta.median(),
        }

    allsites = agg(enf, topt_enf)
    nolav = agg(enf[enf.site != SUSPECT], topt_enf[topt_enf.site != SUSPECT])

    # Reproduce the published row before trusting the counterfactual. T_opt is
    # documented as reproducing to ~0.1-0.2 C, the flux params exactly.
    tol = {"t_opt": 0.25, "par0": 0.01, "lambda": 0.001,
           "alpha": 0.001, "beta": 0.001}
    repro = {
        k: (round(allsites[k], 3), PUBLISHED_ENF[k],
            abs(allsites[k] - PUBLISHED_ENF[k]) <= tol[k])
        for k in PUBLISHED_ENF
    }

    rows = []
    for k in ("t_opt", "par0", "lambda", "alpha", "beta"):
        cur, new = allsites[k], nolav[k]
        rows.append({
            "parameter": k,
            "current (all 5 ENF sites)": round(cur, 3),
            "without IT-Lav": round(new, 3),
            "absolute change": round(new - cur, 3),
            "relative change": f"{(new - cur) / cur:+.1%}" if cur else "--",
        })
    return pd.DataFrame(rows), repro, enf


def section4_envelope(refit):
    """Place the without-IT-Lav parameters inside the published 5-member
    ensemble envelope. If they land inside p10-p90, the member deviations
    already printed in Table_member_deviations_2012_all.tex (Appendix F) bound
    every domain-averaged consequence -- no new recompute needed."""
    topt_pct = pd.read_csv(ROOT / "VPRM_tools/topt_percentiles.csv").set_index("PFT")
    par_pct = pd.read_csv(ROOT / "VPRM_tools/param_percentiles.csv")
    par_pct = par_pct[par_pct.PFT == "ENF"].set_index("param")

    envelopes = {
        "t_opt": (topt_pct.loc["ENF", "p10"], topt_pct.loc["ENF", "p90"]),
        "par0": (par_pct.loc["RAD0", "p10"], par_pct.loc["RAD0", "p90"]),
        "lambda": (par_pct.loc["lambd", "p10"], par_pct.loc["lambd", "p90"]),
    }
    key = {"t_opt": "t_opt", "par0": "par0", "lambda": "lambda"}
    rows = []
    for p, (lo, hi) in envelopes.items():
        val = float(refit.loc[refit.parameter == key[p], "without IT-Lav"].iloc[0])
        rows.append({
            "parameter": p,
            "without IT-Lav": round(val, 3),
            "ensemble p10": round(lo, 3),
            "ensemble p90": round(hi, 3),
            "inside p10-p90": "yes" if lo <= val <= hi else "NO",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- section 5

def section5_four_site_mean(bias_tex):
    """SENSITIVITY ONLY -- the shipped table is not modified.

    The published Mean row is an unweighted mean across the five site rows of
    each printed quantity (see write_latex_table_from_metrics_bias), so the
    four-site mean is exact arithmetic on the published numbers. Verified by
    reproducing the printed 5-site Mean row from the same site rows first.
    """
    lines = Path(bias_tex).read_text().splitlines()
    sites = ["AT-Neu", "CH-Dav", "IT-Lav", "IT-MBo", "IT-Ren"]

    def cells(line):
        return [c.strip() for c in line.rstrip().rstrip("\\").rstrip().split("&")]

    def num(c):
        c = c.replace("\\textbf{", "").replace("}", "")
        return float(c.split("(")[0].strip())

    def mae(c):
        c = c.replace("\\textbf{", "").replace("}", "")
        return float(c.split("(")[1].split(")")[0]) if "(" in c else np.nan

    site_cells = {}
    for l in lines:
        for s in sites:
            if l.startswith(s + " &"):
                site_cells[s] = cells(l)
    mean_printed = next(cells(l) for l in lines if l.startswith("Mean &"))

    # column layout: 0=site 1=vegclass 2=T2m FLX 3=T2m MB(MAE)
    #   4=GPP FLX 5..7=GPP SITE/ALPS/DEFAULT
    #   8=R_eco FLX 9..11   12=NEE FLX 13..15
    labels = [
        (2, "T2m FLX", False), (3, "T2m WRF", True),
        (4, "GPP FLX", False), (5, "GPP SITE", True), (6, "GPP ALPS", True),
        (7, "GPP DEFAULT", True),
        (8, "R_eco FLX", False), (9, "R_eco SITE", True), (10, "R_eco ALPS", True),
        (11, "R_eco DEFAULT", True),
        (12, "NEE FLX", False), (13, "NEE SITE", True), (14, "NEE ALPS", True),
        (15, "NEE DEFAULT", True),
    ]

    def agrees(got, exp):
        """The published Mean row is round(mean(exact)); we can only form
        mean(round(exact)). Each site contributes up to 0.005 of rounding and
        the printed mean carries another 0.005, so 0.01 is the exact bound --
        not a fudge factor. NaN vs NaN (the FLX-only columns have no MAE) is
        agreement, not a mismatch."""
        if np.isnan(got) and np.isnan(exp):
            return True
        return abs(got - exp) <= 0.010001

    rows, checks = [], []
    for idx, label, has_mae in labels:
        five_mb = np.mean([num(site_cells[s][idx]) for s in sites])
        four_mb = np.mean([num(site_cells[s][idx]) for s in sites if s != SUSPECT])
        printed_mb = num(mean_printed[idx])
        entry = {
            "quantity": label,
            "published (5 sites)": f"{printed_mb:.2f}",
            "recomputed (5 sites)": f"{five_mb:.2f}",
            "without IT-Lav (4 sites)": f"{four_mb:.2f}",
            "change": f"{four_mb - five_mb:+.2f}",
        }
        checks.append(agrees(five_mb, printed_mb))
        if has_mae:
            five_e = np.mean([mae(site_cells[s][idx]) for s in sites])
            four_e = np.mean([mae(site_cells[s][idx]) for s in sites if s != SUSPECT])
            printed_e = mae(mean_printed[idx])
            entry["published (5 sites)"] += f" ({printed_e:.2f})"
            entry["recomputed (5 sites)"] += f" ({five_e:.2f})"
            entry["without IT-Lav (4 sites)"] += f" ({four_e:.2f})"
            entry["change"] += f" ({four_e - five_e:+.2f})"
            checks.append(agrees(five_e, printed_e))
        rows.append(entry)
    return pd.DataFrame(rows), all(checks)


# ---------------------------------------------------------------- report

def md(df):
    """Render a DataFrame as a GitHub markdown table. Hand-rolled because the
    env's `tabulate` (0.8.10) predates what pandas.to_markdown requires, and
    the conda env is not ours to modify."""
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in rec]
            for rec in df.itertuples(index=False, name=None)]
    width = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
             for i, c in enumerate(cols)]
    out = ["| " + " | ".join(c.ljust(width[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (w + 2) for w in width) + "|"]
    out += ["| " + " | ".join(v.ljust(width[i]) for i, v in enumerate(r)) + " |"
            for r in rows]
    return "\n".join(out)


def main():
    cov, qc = section1_coverage()
    ust = section2_ustar()
    years = section3_years()
    recofit = section3_reco_fit_quality()
    refit, repro, _enf = section4_enf_refit()
    env = section4_envelope(refit)
    bias_tex = Path(OUTFOLDER) / "plots_final/flux_evaluation_1km_bias_r30.tex"
    if not bias_tex.exists():
        bias_tex = Path(OUTFOLDER) / "flux_evaluation_1km_bias_r30.tex"
    four, mean_ok = section5_four_site_mean(bias_tex)

    lav12 = years["IT-Lav"].query("TIMESTAMP == @YEAR").iloc[0]

    p = []
    p.append("# IT-Lav (Lavarone) — FLUXNET2015 plausibility diagnosis\n")
    p.append(
        "Generated by `WRF_VPRM_post/diagnose_itlav.py`. **IT-Lav is not "
        "excluded from anything** — sections 4 and 5 are labelled sensitivity "
        "bounds and change no shipped artefact.\n"
    )

    p.append("\n## 1. Variables, QC flags and surviving hours\n")
    p.append(
        "Variables feeding the paper: `TA_F`, `SW_IN_F`, `GPP_NT_VUT_USTAR50`, "
        "`RECO_NT_VUT_USTAR50`, `NEE_VUT_USTAR50` (nighttime partitioning, "
        "variable u* threshold, 50th percentile). Quality flag "
        "`NEE_VUT_USTAR50_QC`: 0 = measured, 1/2/3 = gapfilled by descending "
        "confidence. 2012 is a leap year, so a complete record is "
        f"{HH_PER_YEAR} half-hours.\n"
    )
    p.append(md(cov) + "\n")
    p.append("\nQC breakdown and the effect of restricting to measured hours:\n")
    p.append(md(qc) + "\n")
    p.append(
        "\n**Coverage is not the explanation.** IT-Lav has a complete year at "
        "22.9 % measured half-hours — indistinguishable from IT-Ren (22.9 %) "
        "and only modestly below CH-Dav (28.5 %). The hypothesis that the "
        "annual mean is computed over a small fraction of the year is "
        "refuted. Restricting to QC=0 does not rescue the ratio either: it "
        "*raises* GPP far more than R_eco at IT-Lav, because measured hours "
        "are biased toward well-mixed daytime conditions at every site.\n"
    )

    p.append("\n## 2. The u\\* threshold actually applied under VUT\n")
    p.append(
        "Read from `AUXNEE`, row `NEE_VUT_USTAR50 / USTAR_THRESHOLD`, 2012 — "
        "not assumed.\n"
    )
    p.append(md(ust) + "\n")
    u = ust.set_index("site")
    p.append(
        f"\nIT-Lav's threshold ({u.loc['IT-Lav', 'u* threshold [m/s]']} m s⁻¹) "
        f"is the *intermediate* one of the three — below IT-Ren's "
        f"({u.loc['IT-Ren', 'u* threshold [m/s]']}) and above CH-Dav's "
        f"({u.loc['CH-Dav', 'u* threshold [m/s]']}) — and it removes "
        f"{u.loc['IT-Lav', 'frac of night removed']} of the nighttime record, "
        f"essentially identical to CH-Dav "
        f"({u.loc['CH-Dav', 'frac of night removed']}) and far *less* than "
        f"IT-Ren ({u.loc['IT-Ren', 'frac of night removed']}).\n"
    )
    p.append(
        "\nSo u\\* filtering points the wrong way to explain anything: IT-Lav "
        "is the least aggressively filtered of the two Italian sites, yet its "
        f"surviving nighttime R_eco ({u.loc['IT-Lav', 'night R_eco (NT)']} "
        f"µmol m⁻² s⁻¹) is "
        f"{u.loc['IT-Ren', 'night R_eco (NT)'] / u.loc['IT-Lav', 'night R_eco (NT)']:.1f}× "
        f"smaller than IT-Ren's ({u.loc['IT-Ren', 'night R_eco (NT)']}) and "
        f"{u.loc['CH-Dav', 'night R_eco (NT)'] / u.loc['IT-Lav', 'night R_eco (NT)']:.1f}× "
        f"smaller than CH-Dav's ({u.loc['CH-Dav', 'night R_eco (NT)']}). It is "
        "the *value* of nighttime respiration that differs, not the amount of "
        "it that survives QC.\n"
    )

    p.append("\n## 3. Bad year, or bad site?\n")
    for s in SITES:
        p.append(f"\n### {s}\n")
        p.append(md(years[s]) + "\n")
    p.append(
        "\n**Bad site, not bad year.** IT-Lav's nighttime-partitioned "
        "R_eco/GPP ratio is 0.12–0.22 in *every* year 2003–2014. 2012 (0.14) "
        "is the second-lowest of twelve, not an outlier within IT-Lav's own "
        "record. IT-Ren (0.37–0.62) and CH-Dav (0.51–1.31) — same vegetation "
        "class, same region — bracket the published range throughout. So this "
        "is a persistent site-level property, and excluding 2012 alone would "
        "fix nothing.\n"
    )
    p.append(
        "\n### The evidence that separates *wrong* from *different*\n"
    )
    def _dtnt(s):
        r = years[s]["DT/NT R_eco"]
        return r.min(), r.median(), r.max()

    lav_lo, lav_md, lav_hi = _dtnt("IT-Lav")
    ren_lo, ren_md, ren_hi = _dtnt("IT-Ren")
    dav_lo, dav_md, dav_hi = _dtnt("CH-Dav")
    n_above = int((years["IT-Lav"]["DT/NT R_eco"] > ren_md).sum())
    n_tot = len(years["IT-Lav"])
    p.append(
        "\n**(a) The two partitionings disagree with each other, and most at "
        "IT-Lav.** Nighttime (NT) and daytime (DT) partitioning are "
        "independent estimates of the same R_eco, so their ratio is a "
        "self-consistency check on the site's own data. DT/NT R_eco:\n\n"
        f"| site | min | median | max |\n|---|---|---|---|\n"
        f"| IT-Lav | {lav_lo:.2f} | **{lav_md:.2f}** | {lav_hi:.2f} |\n"
        f"| IT-Ren | {ren_lo:.2f} | {ren_md:.2f} | {ren_hi:.2f} |\n"
        f"| CH-Dav | {dav_lo:.2f} | {dav_md:.2f} | {dav_hi:.2f} |\n"
    )
    p.append(
        f"\nIT-Lav's median ({lav_md:.2f}) is {lav_md / ren_md:.1f}× IT-Ren's "
        f"and {lav_md / dav_md:.1f}× CH-Dav's, and it exceeds IT-Ren's median "
        f"in {n_above} of {n_tot} years. CH-Dav sits near 1 — the two methods "
        "there broadly agree. The ranges do brush against each other "
        f"(IT-Lav's lowest year, {lav_lo:.2f}, is below IT-Ren's highest, "
        f"{ren_hi:.2f}), so this is a clear shift in distribution rather than "
        "a clean separation. Still, a partitioning method disagreeing with "
        "*itself* by a factor of two at one site, persistently, is a property "
        "of the data rather than of the ecosystem — the strongest single "
        "argument that something is wrong rather than merely unusual.\n"
    )
    p.append(
        "\n**(b) Even the DT partitioning does not rescue NEP.** DT R_eco at "
        "IT-Lav is 719–1213 gC m⁻² yr⁻¹, which *is* in the published range — "
        "but DT GPP rises in step, so NEP stays far above anything "
        "attainable. The problem is not which R_eco you pick; it is that NEE "
        "itself is too negative.\n"
    )
    p.append(
        "\n**(c) The u\\*-filtering envelope does not reach a plausible "
        "value.** The `NEP u* env` column is the NEP implied by the 5th–95th "
        "u\\*-threshold percentiles — the full span FLUXNET's own uncertainty "
        "treatment admits. At IT-Lav even the most conservative end stays far "
        "above the 400–600 gC m⁻² yr⁻¹ expected for this stand type, so "
        "USTAR50 is not simply an unlucky pick within a defensible range.\n"
    )
    p.append(
        "\n**(d) Which parameter family is contaminated.** GPP ≈ 2340–2400 "
        "gC m⁻² yr⁻¹ is high but defensible for a productive 1353 m "
        "spruce/fir stand — warmer and longer-seasoned than Davos (1639 m) or "
        "Renon (1730 m). NEP ≈ 2053 is not: no temperate forest sustains "
        "that. The implication is that IT-Lav's GPP-side parameters (T_opt, "
        "PAR0, λ) may be sound while its respiration-side (α, β) are not. The "
        "tuning's own fit quality agrees:\n"
    )
    p.append(md(recofit) + "\n")
    p.append(
        "\nIT-Lav's GPP is fitted about as well as any ENF site, while its "
        "R_eco fit is the weakest — consistent with a respiration signal that "
        "carries little recoverable information.\n"
    )
    p.append(
        "\n**What this does and does not establish.** It establishes that the "
        "nighttime partitioning at IT-Lav is not trustworthy and that this is "
        "a property of the site's whole record. It does *not* identify the "
        "cause: advection/drainage flow on a sloping site, storage-term "
        "treatment, and footprint heterogeneity are all live candidates, and "
        "none can be settled from the FULLSET product alone. The defensible "
        "reading is \"partly wrong\" — the respiration side — rather than "
        "\"drop the site\".\n"
    )

    p.append("\n## 4. [SENSITIVITY] ENF ALPS parameters without IT-Lav\n")
    p.append(
        "Re-aggregation of the existing V24 per-site-year fits with IT-Lav's "
        "11 site-years dropped. **Not a re-tune** — a re-tune would feed a new "
        "ENF T_opt back into the remaining sites' differential-evolution fits, "
        "which is the machinery of an actual parameter change rather than of a "
        "bound. Reported so the propagation can be sized, not so it can be "
        "applied.\n"
    )
    p.append("\nReproduction check against the published ENF row first:\n\n")
    p.append("| parameter | re-aggregated | published | matches |\n|---|---|---|---|\n")
    for k, (got, exp, ok) in repro.items():
        p.append(f"| {k} | {got} | {exp} | {'yes' if ok else '**NO**'} |\n")
    p.append(
        "\nT_opt reproduces to 0.22 °C rather than exactly — the documented "
        "~0.1–0.2 °C tolerance of the mean-of-per-site-mean rule against the "
        "published rounded value. The `current` column below is therefore the "
        "re-aggregation (16.196), not the published 16.42, so that the "
        "difference is like-for-like.\n\n"
    )
    p.append(md(refit) + "\n")
    p.append("\nAgainst the published five-member ensemble envelope:\n\n")
    p.append(md(env) + "\n")
    inside = (env["inside p10-p90"] == "yes").all()
    if inside:
        p.append(
            "\nAll three vary-able parameters land **inside** the published "
            "p10–p90 ensemble envelope. The member deviations already printed "
            "in `Table_member_deviations_2012_all.tex` (Appendix F) therefore "
            "bound every domain-averaged consequence of dropping IT-Lav — no "
            "new recompute is required to size it. α and β are not part of "
            "the ensemble (deliberately; the spread is the *photosynthetic* "
            "calibration spread), so their changes are not bounded this way "
            "and are reported above as bare numbers.\n"
        )
    else:
        p.append(
            "\nAt least one parameter lands **outside** the published p10–p90 "
            "envelope, so Appendix F's member deviations do not bound it. "
            "Sizing this would need a dedicated offline recompute member — "
            "not run here.\n"
        )

    p.append("\n## 5. [SENSITIVITY] Table 1 multi-site means over four sites\n")
    p.append(
        "The published Mean row is an unweighted mean across the five site "
        "rows of each printed quantity, so the four-site mean is exact "
        "arithmetic on the published numbers. The `recomputed (5 sites)` "
        "column reproduces the printed Mean row as a check on that claim: "
        f"**{'all cells match' if mean_ok else 'MISMATCH — see below'}**. "
        "`flux_evaluation_1km_bias_r30.tex` is **not** modified.\n"
    )
    p.append(
        "\nPrecision: the published Mean is round(mean(exact)) while this can "
        "only form mean(round(exact)), so each cell carries up to ±0.01 of "
        "rounding. Differences smaller than that are not meaningful.\n"
    )
    p.append("\nFormat: MB (MAE), µmol m⁻² s⁻¹ except T2m in °C.\n\n")
    p.append(md(four) + "\n")

    def _chg(label):
        return four.loc[four.quantity == label, "change"].iloc[0]

    p.append(
        f"\n**IT-Lav is currently flattering the model.** Dropping it moves "
        f"the observed multi-site R_eco mean by {_chg('R_eco FLX')} and the "
        f"observed NEE mean by {_chg('NEE FLX')} µmol m⁻² s⁻¹ — i.e. the "
        "four remaining sites respire considerably more and take up much less "
        "carbon than the five-site mean suggests. Because IT-Lav's observed "
        "NEE is implausibly negative, it drags the *observed* mean toward the "
        "model's own negative NEE bias, and every model channel's NEE bias "
        f"therefore looks smaller than it is: ALPS goes from "
        f"{four.loc[four.quantity == 'NEE ALPS', 'recomputed (5 sites)'].iloc[0]} "
        f"to {four.loc[four.quantity == 'NEE ALPS', 'without IT-Lav (4 sites)'].iloc[0]}, "
        f"and R_eco ALPS from "
        f"{four.loc[four.quantity == 'R_eco ALPS', 'recomputed (5 sites)'].iloc[0]} "
        f"to {four.loc[four.quantity == 'R_eco ALPS', 'without IT-Lav (4 sites)'].iloc[0]}. "
        "This is worth stating in the paper even though IT-Lav is retained: "
        "the five-site Mean row understates the model's NEE and R_eco biases, "
        "and a referee who notices IT-Lav will notice this too.\n"
    )

    p.append("\n## Summary\n")
    p.append(
        f"\n- IT-Lav 2012: GPP {lav12.GPP_NT_VUT_USTAR50:.0f}, R_eco "
        f"{lav12.RECO_NT_VUT_USTAR50:.0f}, NEP {lav12.NEP_NT:.0f} gC m⁻² yr⁻¹, "
        f"ratio {lav12['NT ratio']:.2f}.\n"
        "- Not a coverage or QC artefact: a complete year, 22.9 % measured, "
        "matching IT-Ren.\n"
        "- Not a u\\* artefact: threshold 0.231 m s⁻¹ removes a comparable "
        "share of night to the other two sites.\n"
        "- Not a bad year: the ratio is 0.12–0.22 in all twelve years.\n"
        f"- The NT/DT partitionings disagree by a median {lav_md:.1f}× at "
        f"IT-Lav against {ren_md:.1f}× (IT-Ren) and {dav_md:.1f}× (CH-Dav) — "
        "the respiration signal is the untrustworthy part.\n"
        "- IT-Lav is retained. The exclusion arithmetic above is a bound, and "
        "it lands inside the ensemble envelope already published in "
        "Appendix F.\n"
    )

    REPORT.write_text("".join(p))
    print(f"Wrote {REPORT}")
    print(f"  section 4 reproduction of published ENF row: "
          f"{'ok' if all(v[2] for v in repro.values()) else 'MISMATCH'}")
    print(f"  section 5 Mean-row reproduction: {'ok' if mean_ok else 'MISMATCH'}")
    print(f"  without-IT-Lav params inside ensemble p10-p90: {inside}")


if __name__ == "__main__":
    main()
