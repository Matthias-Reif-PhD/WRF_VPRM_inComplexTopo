"""
build_member_param_csvs.py
======================================================================
Build the 5 Topt-sensitivity member parameter CSVs consumed by
recompute_vprm_fluxes.py / the sensitivity domain-mean script.

For each percentile member pX in {p10, p25, p75, p90} (p50 == V24 central):
  - ENF/DBF/MF/GRA rows: Topt from topt_percentiles.csv and par0/lambda from
    param_percentiles.csv, both for <pX>. These are the same two tables
    Fig11_linPertComp_hourly_mean.py builds its spread band from, so every figure
    in the paper uses one member definition. A member therefore carries a
    self-consistent (Topt, PAR0, lambda) triple -- varying Topt alone cannot
    represent acclimation, since the parameters are coupled by equifinality.
  - alpha/beta are NOT varied: param_percentiles.csv holds none, and they are
    Reco parameters with no Topt dependence.
  - SHB/SAV(WET)/CRO rows: copied unchanged from vprm_params_newGPP_V24.csv.
  - Tmin/Tmax: from the V24 csv (WRF online values).
p50 member = a copy of vprm_params_newGPP_V24.csv (the published central).

Output: WRF_VPRM_post/vprm_params_topt_p{10,25,50,75,90}.csv
======================================================================
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ALPS = os.path.join(os.getenv("SCRATCH_PATH", "/scratch/c7071034"),
                    "DATA/Fluxnet2015/Alps")
V24_CSV = os.path.join(HERE, "vprm_params_newGPP_V24.csv")
TOPT_PCT = os.path.join(HERE, "..", "VPRM_tools", "topt_percentiles.csv")
PARAM_PCT = os.path.join(HERE, "..", "VPRM_tools", "param_percentiles.csv")

VARY = {"ENF": 1, "DBF": 2, "MF": 3, "GRA": 7}  # pft -> vprm_veg_id (varied PFTs)
MEMBERS = ["p10", "p25", "p50", "p75", "p90"]


def member_params(pct):
    """Per-PFT PAR0/lambda for one percentile member, from param_percentiles.csv."""
    pp = pd.read_csv(PARAM_PCT)                                 # PFT, param, p10..p90
    return pp.pivot(index="PFT", columns="param", values=pct)   # cols: RAD0, lambd


def main():
    v24 = pd.read_csv(V24_CSV, comment="#")
    topt = pd.read_csv(TOPT_PCT).set_index("PFT")

    for pct in MEMBERS:
        out = v24.copy()
        if pct != "p50":
            prm = member_params(pct)
            for pft, vid in VARY.items():
                row = out["vprm_veg_id"] == vid
                out.loc[row, "t_opt"] = float(topt.loc[pft, pct])
                out.loc[row, "par0"] = float(prm.loc[pft, "RAD0"])
                out.loc[row, "lambda"] = float(prm.loc[pft, "lambd"])
        outpath = os.path.join(HERE, f"vprm_params_topt_{pct}.csv")
        out.to_csv(outpath, index=False)
        print(f"wrote {os.path.basename(outpath)}")
        print(out[["pft", "t_opt", "par0", "lambda"]].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
