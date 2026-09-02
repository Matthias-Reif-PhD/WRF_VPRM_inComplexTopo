"""
derive_topt_members.py
======================================================================
Build the 5-member Topt sensitivity table per PFT (ENF/DBF/GRA) from the
per-site-year GPP-based Topt dump produced by
WRF_VPRM_post/Fig3b_AppxG10_G11_Topt_tuneParam.py (topt_raw_persiteyear.csv).

Why per-site MEAN: the published V24 central Topt is the *mean of per-site mean*
Topt (each site weighted equally), NOT the median -- verified to reproduce the
main_tune_VPRM.py T_opt dict (ENF 16.42, DBF 23.94, GRA 18.56) to ~0.1-0.2 C.
So the range is built consistently on the per-site-mean distribution:
    members = { p10, p25, CENTRAL(=V24 mean), p75, p90 }
The mean (V24) falls between p25 and p75 for all three PFTs -> monotonic, and the
central member reuses the existing V24 tuning (no re-tune needed for it).

Output: VPRM_tools/topt_percentiles.csv
    columns: PFT, n_sites, p10, p25, p50, p75, p90
    where p50 is the V24 central (mean) value; p10/p25/p75/p90 are per-site-mean
    percentiles. Consumed by submit_jobs_tune_VPRM.sh / main_tune_VPRM.py.
======================================================================
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "topt_raw_persiteyear.csv")
OUT = os.path.join(HERE, "topt_percentiles.csv")

# V24 central Topt (= main_tune_VPRM.py T_opt dict, the per-site-mean estimate)
V24_CENTRAL = {"ENF": 16.42, "DBF": 23.94, "MF": 19.60, "GRA": 18.56}
PFTS = ["ENF", "DBF", "MF", "GRA"]

df = pd.read_csv(RAW)
rows = []
for pft in PFTS:
    site_means = df[df.PFT == pft].groupby("site").Topt.mean().values
    n_sites = len(site_means)  # record the true site count, not the fallback sample
    if len(site_means) < 2:
        # MF has a single site (CH-Lae), so the between-site distribution is
        # degenerate; fall back to its site-YEAR values -- the same rule
        # topt_box._stats_for uses when drawing that class's box. Note this is an
        # interannual spread, not a between-site one (say so in the manuscript).
        site_means = df[df.PFT == pft].Topt.values
    p10, p25, _, p75, p90 = np.percentile(site_means, [10, 25, 50, 75, 90])
    central = V24_CENTRAL[pft]
    members = [round(float(v), 2) for v in (p10, p25, central, p75, p90)]
    # safety: enforce monotonic non-decreasing (clip tails to the central if needed)
    members[1] = min(members[1], central)
    members[0] = min(members[0], members[1])
    members[3] = max(members[3], central)
    members[4] = max(members[4], members[3])
    rows.append([pft, n_sites] + members)

out = pd.DataFrame(rows, columns=["PFT", "n_sites", "p10", "p25", "p50", "p75", "p90"])
out.to_csv(OUT, index=False)
print("Wrote", OUT)
print(out.to_string(index=False))
