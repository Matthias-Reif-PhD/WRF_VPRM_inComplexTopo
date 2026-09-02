"""
build_site_param_csv.py
======================================================================
Assemble the GPP-based per-site SITE parameter file consumed by
extract_SITE_timeseries.py (Step 3) and print LaTeX rows for
tab:filtered_sites_alps (Step 4).

For each of the five d03 SITE FLUXNET locations:
  - reads the per-site xlsx written by main_tune_VPRM.py with
    opt_method="diff_evo_V24_SITE" (maxiter 42);
  - takes Topt, PAR0, lambda, alpha, beta from that xlsx (the Topt
    column must match the fixed per-site value used during tuning);
  - takes Tmin/Tmax from vprm_params_newGPP_V24.csv per PFT;
  - writes WRF_VPRM_post/vprm_params_site.csv.

Output columns:
  site, pft, vprm_veg_id, t_opt, t_min, t_max, par0, lambda, alpha, beta

Usage: python build_site_param_csv.py
======================================================================
"""
import os
import glob
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
load_dotenv(ROOT / ".env")

SCRATCH = os.getenv("SCRATCH_PATH", "/scratch/c7071034")
ALPS = Path(SCRATCH) / "DATA/Fluxnet2015/Alps"
V24_CSV = HERE / "vprm_params_newGPP_V24.csv"
OUT_CSV = HERE / "vprm_params_site.csv"

# Five d03 SITE locations: site -> expected PFT and vprm_veg_id
SITES = {
    "AT-Neu": ("GRA", 7),
    "CH-Dav": ("ENF", 1),
    "IT-Lav": ("ENF", 1),
    "IT-MBo": ("GRA", 7),
    "IT-Ren": ("ENF", 1),
}

# Lat/lon/AMSL from the paper table (stable geographic metadata)
SITE_GEO = {
    "AT-Neu": (47.1167, 11.3175,  970),
    "CH-Dav": (46.8153,  9.8559, 1639),
    "IT-Lav": (45.9562, 11.2813, 1353),
    "IT-MBo": (46.0147, 11.0458, 1550),
    "IT-Ren": (46.5869, 11.4337, 1730),
}

MAXITER = 42


def find_xlsx(site):
    pattern = str(ALPS / f"FLX_{site}_*" / f"{site}_*_optimized_params_old_diff_evo_V24_SITE_{MAXITER}.xlsx")
    hits = glob.glob(pattern)
    if not hits:
        raise FileNotFoundError(
            f"No V24_SITE xlsx for {site} (maxiter={MAXITER}). "
            f"Pattern: {pattern}"
        )
    return hits[0]


def main():
    v24 = pd.read_csv(V24_CSV, comment="#").set_index("pft")
    rows = []

    for site, (pft, vid) in SITES.items():
        xlsx = find_xlsx(site)
        df = pd.read_excel(xlsx)
        if len(df) != 1:
            raise ValueError(f"{xlsx}: expected 1 row, got {len(df)}")
        r = df.iloc[0]
        t_opt = float(r["Topt"])
        par0  = float(r["PAR0"])
        lam   = float(r["lambd"])
        alpha = float(r["alpha"])
        beta  = float(r["beta"])
        t_min = float(v24.loc[pft, "t_min"])
        t_max = float(v24.loc[pft, "t_max"])
        rows.append(dict(
            site=site, pft=pft, vprm_veg_id=vid,
            t_opt=t_opt, t_min=t_min, t_max=t_max,
            par0=par0, lambda_=lam, alpha=alpha, beta=beta,
        ))
        print(f"  {site:7s} {pft}  Topt={t_opt:.2f}  PAR0={par0:.2f}  "
              f"lam={lam:.4f}  alpha={alpha:.4f}  beta={beta:.4f}")

    out = pd.DataFrame(rows).rename(columns={"lambda_": "lambda"})
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    # Column order matches tab:filtered_sites_alps header:
    # Site & PFT & Latitude & Longitude & AMSL & T_opt & PAR0 & alpha & beta & lambda
    print("\n--- LaTeX rows for tab:filtered_sites_alps ---")
    for r in rows:
        lat, lon, amsl = SITE_GEO[r["site"]]
        print(
            f"\t\t\t\t{r['site']:7s} & {r['pft']:3s} & "
            f"{lat:.4f} & {lon:.4f} & {amsl} & "
            f"{r['t_opt']:.2f} & {r['par0']:.2f} & "
            f"{r['alpha']:.3f} & {r['beta']:.3f} & {r['lambda_']:.3f} \\\\"
        )


if __name__ == "__main__":
    main()
