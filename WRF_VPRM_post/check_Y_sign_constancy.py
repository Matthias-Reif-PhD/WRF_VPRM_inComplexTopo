"""
check_Y_sign_constancy.py
==============================================
9d: verify the manuscript's claim that satellite-derived VPRM driver
contributions (Y_Wscale, Y_Pscale, Y_EVI) keep a constant sign throughout
the year.

Reads the per-timestep records Fig11_linPertComp_hourly_mean.py dumps to
plots/signcheck_Y_components{,_cloudy}.csv (one row per config x resolution x
timestep, config in {ALPS, DEFAULT}), pools both regimes (24 representative
days total), and reports sign counts per (config, resolution, component) --
counts only, no verdict. For any component with a nonzero minority-sign
count, also reports which timesteps and how large the minority values are
relative to the majority.
==============================================
"""

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
OUTFOLDER = os.getenv("OUTFOLDER", "./plots/")

COMPONENTS = ["Y_Wscale", "Y_Pscale", "Y_EVI"]


def main():
    paths = [f"{OUTFOLDER}signcheck_Y_components.csv", f"{OUTFOLDER}signcheck_Y_components_cloudy.csv"]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("Missing sign-check CSV(s), cannot aggregate yet:")
        for p in missing:
            print(f"  {p}")
        return

    df = pd.concat([pd.read_csv(p, parse_dates=["time"]) for p in paths], ignore_index=True)

    n_days = df["time"].dt.date.nunique()
    print(f"pooled sign-check data: {len(df)} rows, {n_days} distinct days "
          f"(expect 24 representative days)")

    print(f"\n{'config':8s} {'res':6s} {'component':10s} {'pos':>5s} {'neg':>5s} "
          f"{'zero':>5s} {'total':>6s}")
    counts = {}
    for cfg in ("ALPS", "DEFAULT"):
        for res in ("_54km", "_9km"):
            sub = df[(df.config == cfg) & (df.res == res)]
            for comp in COMPONENTS:
                pos = int((sub[comp] > 0).sum())
                neg = int((sub[comp] < 0).sum())
                zero = int((sub[comp] == 0).sum())
                total = pos + neg + zero
                counts[(cfg, res, comp)] = (pos, neg, zero, total)
                expect = n_days * 24
                assert total == expect, (
                    f"{cfg} {res} {comp}: got {total} rows, expected {expect} "
                    f"({n_days} days x 24 hours) -- timesteps dropped somewhere"
                )
                print(f"{cfg:8s} {res:6s} {comp:10s} {pos:5d} {neg:5d} {zero:5d} {total:6d}")

    print("\nminority-sign detail (component has both signs present):")
    any_minority = False
    for (cfg, res, comp), (pos, neg, zero, total) in counts.items():
        if pos > 0 and neg > 0:
            any_minority = True
            sub = df[(df.config == cfg) & (df.res == res)]
            majority_sign = 1 if pos >= neg else -1
            majority_vals = sub[comp][np.sign(sub[comp]) == majority_sign]
            minority_vals = sub[comp][np.sign(sub[comp]) == -majority_sign]
            maj_mean_abs = float(majority_vals.abs().mean())
            min_mean_abs = float(minority_vals.abs().mean())
            rel = min_mean_abs / maj_mean_abs if maj_mean_abs > 0 else float("nan")
            print(f"\n  {cfg} {res} {comp}: majority sign = "
                  f"{'positive' if majority_sign > 0 else 'negative'} "
                  f"({max(pos, neg)}/{total}), minority = {min(pos, neg)}/{total}")
            print(f"    mean |majority| = {maj_mean_abs:.4f}, mean |minority| = "
                  f"{min_mean_abs:.4f}, ratio = {rel:.3f}")
            minority_rows = sub[np.sign(sub[comp]) == -majority_sign][["time", comp]]
            for _, row in minority_rows.iterrows():
                print(f"    minority: {row['time']}  {comp}={row[comp]:+.4f}")

    if not any_minority:
        print("  none -- every (config, resolution, component) is single-signed "
              "(ignoring exact zeros) across all 24 days x 24 hours")

    print("\n  [ok] sign counts sum to 24 days x 24 hours for every (config, resolution, component)")


if __name__ == "__main__":
    main()
