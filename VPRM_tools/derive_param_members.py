"""
derive_param_members.py
======================================================================
Build the 5-member per-PFT percentile table for the two GPP parameters
lambda and PAR0 (RAD0), for ENF/DBF/GRA, from the per-site-year tuning dump
`params_raw_persiteyear.csv` (produced by
WRF_VPRM_post/Fig3b_AppxG10_G11_Topt_tuneParam.py).

Basis is identical to derive_topt_members.py (the Topt ensemble): the per-site
MEAN distribution, so each site is weighted equally, and the CENTRAL member
(p50) is pinned to the published V24 value (vprm_params_newGPP_V24.csv) so the
central ensemble member reproduces the current figures exactly. p10/p25/p75/p90
are the per-site-mean percentiles, clipped monotonic around the central.

Output: VPRM_tools/param_percentiles.csv
    columns: PFT, param, p10, p25, p50, p75, p90
    param in {RAD0, lambd};  p50 == V24 central (par0 / lambda).
Consumed by WRF_VPRM_post/Fig11_linPertComp_hourly_mean.py to shade the
p10-p90 parameter-spread band.
======================================================================
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "params_raw_persiteyear.csv")
V24 = os.path.join(HERE, "..", "WRF_VPRM_post", "vprm_params_newGPP_V24.csv")
OUT = os.path.join(HERE, "param_percentiles.csv")

PFTS = ["ENF", "DBF", "MF", "GRA"]
# (output param name, raw column, V24 central column)
PARAMS = [("RAD0", "RAD0", "par0"), ("lambd", "lambd", "lambda")]

df = pd.read_csv(RAW)
v24 = pd.read_csv(V24, comment="#").set_index("pft")

rows = []
for pft in PFTS:
    sub = df[df.PFT == pft]
    for out_name, raw_col, v24_col in PARAMS:
        site_means = sub.groupby("site")[raw_col].mean().values
        if len(site_means) < 2:
            # Single-site class (MF/CH-Lae): the between-site distribution is
            # degenerate, so use its site-YEAR values. Same fallback as
            # derive_topt_members.py, so Topt and PAR0/lambda members for that
            # class describe the same sample.
            site_means = sub[raw_col].values
        p10, p25, _, p75, p90 = np.percentile(site_means, [10, 25, 50, 75, 90])
        central = float(v24.loc[pft, v24_col])
        members = [round(float(v), 4) for v in (p10, p25, central, p75, p90)]
        # enforce monotonic non-decreasing around the pinned central (p50)
        members[1] = min(members[1], central)
        members[0] = min(members[0], members[1])
        members[3] = max(members[3], central)
        members[4] = max(members[4], members[3])
        rows.append([pft, out_name] + members)

out = pd.DataFrame(rows, columns=["PFT", "param", "p10", "p25", "p50", "p75", "p90"])
out.to_csv(OUT, index=False)
print("Wrote", OUT)
print(out.to_string(index=False))
