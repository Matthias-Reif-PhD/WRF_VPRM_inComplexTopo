"""
report_coarse_eval.py
==============================================
Follow-up 15e: assemble the coarse-domain (9 km / 54 km) evaluation report.

Table 1 evaluates only the 1 km run against the towers, so the coarse runs are
shown to *differ* from the fine one but never to be *worse* than it. This
collects the three MB/MAE tables side by side and -- crucially -- records which
grid point each resolution was evaluated at and by which rule, because at 54 km
that choice is most of what the comparison means.

Reads:
  csv/distances_{res}{sim}_{timespan}_r30.csv   (written by extract_SITE_timeseries.py)
  plots/flux_evaluation_{res}_bias_r30.tex      (written by Fig4_...py, EVAL_RES=<res>)

Writes: WRF_VPRM_post/COARSE_EVAL_15e.md

Usage
-----
    python report_coarse_eval.py
==============================================
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CSVFOLDER = Path(os.getenv("CSVFOLDER"))
OUTFOLDER = Path(os.getenv("OUTFOLDER"))
HERE = Path(__file__).resolve().parent
REPORT = HERE / "COARSE_EVAL_15e.md"

TIMESPAN = "2012-01-01 00:00:00_2012-12-31 00:00:00"
RADIUS = 30
RESOLUTIONS = ["1km", "9km", "54km"]
SITES = ["AT-Neu", "CH-Dav", "IT-Lav", "IT-MBo", "IT-Ren"]

# dx in km -> cell area, for the representativeness caveat
DX = {"1km": 1, "9km": 9, "54km": 54}


def md(df):
    """GitHub markdown table (the env's tabulate is too old for to_markdown)."""
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in rec]
            for rec in df.itertuples(index=False, name=None)]
    w = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
         for i, c in enumerate(cols)]
    out = ["| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    out += ["| " + " | ".join(v.ljust(w[i]) for i, v in enumerate(r)) + " |"
            for r in rows]
    return "\n".join(out)


def grid_points():
    """Which cell each tower was evaluated at, per resolution, and how much of
    a choice the rule actually had."""
    rows = []
    for res in RESOLUTIONS:
        f = CSVFOLDER / f"distances_{res}_{TIMESPAN}_r{RADIUS}.csv"
        if not f.exists():
            print(f"  missing {f.name} -- skipping {res}")
            continue
        d = pd.read_csv(f)
        d["site"] = d.name.str.split("_").str[0]
        d = d[d.site.isin(SITES)]
        for _, r in d.iterrows():
            # The older 1 km file predates the rule/candidate columns.
            rule = r.get("rule", "cost")
            ncand = r.get("n_candidates", np.nan)
            rows.append({
                "res": res,
                "site": r.site,
                "rule": rule if isinstance(rule, str) else "cost",
                "candidates": "" if pd.isna(ncand) else int(ncand),
                "cell (i,j)": (f"({int(r.grid_i)},{int(r.grid_j)})"
                               if "grid_i" in r and not pd.isna(r.grid_i) else ""),
                "dist [km]": round(r.dist, 1),
                "cell lat": round(r.lat_wrf, 3),
                "cell lon": round(r.lon_wrf, 3),
                "site hgt [m]": ("" if "hgt_site" not in r or pd.isna(r.hgt_site)
                                 else int(r.hgt_site)),
                "cell hgt [m]": int(round(r.hgt_wrf)),
                "veg frac": round(r.veg_frac_idx, 2),
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["site", "res"], key=lambda c: c.map(
        {**{s: i for i, s in enumerate(SITES)},
         **{r: i for i, r in enumerate(RESOLUTIONS)}}).fillna(0))


def _parse_bias(res):
    """Pull the Mean row's ALPS/DEFAULT MB (MAE) out of one bias table."""
    f = OUTFOLDER / f"flux_evaluation_{res}_bias_r{RADIUS}.tex"
    if not f.exists():
        return None
    lines = f.read_text().splitlines()
    hdr = next(l for l in lines if l.startswith("Site &"))
    names = [c.strip() for c in hdr.rstrip().rstrip("\\").rstrip().split("&")]
    mean = next(l for l in lines if l.startswith("Mean &"))
    cells = [c.strip() for c in mean.rstrip().rstrip("\\").rstrip().split("&")]

    def clean(c):
        return c.replace("\\textbf{", "").replace("}", "").strip()

    # Group the flux blocks by walking the header: each flux block starts at an
    # $\overline{\text{FLX}}$ column. Layout differs between 1 km (SITE/ALPS/
    # DEFAULT) and the coarse tables (ALPS/DEFAULT), so this is read, not assumed.
    fluxes = ["GPP", "R_eco", "NEE"]
    starts = [i for i, n in enumerate(names) if "overline" in n]
    out = {"T2m WRF": clean(cells[3])}
    for flux, s in zip(fluxes, starts[1:]):
        end = starts[starts.index(s) + 1] if starts.index(s) + 1 < len(starts) else len(names)
        for j in range(s + 1, end):
            out[f"{flux} {names[j]}"] = clean(cells[j])
    return out


def bias_comparison():
    per_res = {res: _parse_bias(res) for res in RESOLUTIONS}
    per_res = {k: v for k, v in per_res.items() if v}
    if not per_res:
        return None
    keys = []
    for v in per_res.values():
        for k in v:
            if k not in keys:
                keys.append(k)
    rows = [{"quantity": k, **{res: per_res[res].get(k, "--") for res in per_res}}
            for k in keys]
    return pd.DataFrame(rows)


def main():
    gp = grid_points()
    bc = bias_comparison()

    p = []
    p.append("# Follow-up 15e — 9 km and 54 km evaluated at the five towers\n")
    p.append(
        "\nTable 1 evaluates only the 1 km run against the FLUXNET towers, so "
        "the coarse runs are shown to *differ* from the fine one but never to "
        "be *worse* than it. These tables close that gap using the existing "
        "archived output — **no new simulations**.\n"
    )
    p.append(
        "\nEmitted: `flux_evaluation_9km_bias_r30.tex` and "
        "`flux_evaluation_54km_bias_r30.tex`, same MB (MAE) layout as Table 1 "
        "(`flux_evaluation_1km_bias_r30.tex`), same five d03 towers, same "
        "24 representative days (12 clear + 12 cloudy pooled).\n"
    )

    p.append("\n## Channels\n")
    p.append(
        "\n- **ALPS** = offline recompute with the `topt_p50` parameter set — "
        "the same channel and the same tag Table 1's ALPS column uses.\n"
        "- **DEFAULT** = the online `EBIO_*_REF` tracers from the archived "
        "wrfout, reconstructed exactly as at 1 km.\n"
        "- **No SITE column.** SITE parameters are 1 km per-site fits and the "
        "coarse domains carry no per-site parameterisation, so a SITE column "
        "there would be fabricated rather than measured.\n"
        "- **p50 only, no ensemble band.** Only the p50 member was recomputed "
        "at 9/54 km, and Table 1's ALPS column is the p50 representative, so "
        "this keeps the comparison one-to-one. Widening to all five members "
        "would be a 5× extraction.\n"
    )

    p.append("\n## Which grid point, and by which rule\n")
    p.append(
        "\nThis is the part that decides what the comparison means.\n"
        "\n- **1 km and 9 km: the terrain/vegetation cost function was rerun "
        "on that resolution's own grid**, radius 30 km. At 9 km it chooses "
        "between 32–37 candidate cells — a real selection.\n"
        "- **54 km: the nearest cell, cost function NOT rerun.** It cannot be: "
        "within 30 km of the towers the 54 km grid holds 0–2 cell centres, and "
        "**AT-Neu has none** (its nearest centre is 32.9 km away), so the "
        "cost function would raise there and be a no-op elsewhere. Widening "
        "the radius until it nominally \"runs\" would select cells up to 80 km "
        "from the tower and dress an arbitrary choice as a terrain match.\n"
        "- **The 1 km choice is not reused** at either resolution: grid "
        "indices do not map across domains and a 1 km terrain match carries no "
        "meaning on a coarse grid.\n"
    )
    p.append("\n" + md(gp) + "\n")

    # The two facts a reader must not miss.
    c54 = gp[gp.res == "54km"]
    dupes = c54[c54["cell (i,j)"].duplicated(keep=False)]
    if len(dupes):
        pairs = dupes.groupby("cell (i,j)").site.apply(lambda x: " and ".join(sorted(x)))
        for cell, sites in pairs.items():
            p.append(
                f"\n**{sites} share the same 54 km cell {cell}.** At this "
                "resolution they are not independent evaluations — two rows of "
                "the 54 km table read the identical model grid point and "
                "differ only in the observations they are compared against.\n"
            )
    if len(c54):
        worst = c54.loc[(c54["cell hgt [m]"] - c54["site hgt [m]"]).abs().idxmax()]
        p.append(
            f"\n**{worst.site}'s 54 km cell misses the tower altitude by "
            f"{abs(int(worst['cell hgt [m]']) - int(worst['site hgt [m]']))} m** "
            f"({worst['site hgt [m]']} m site vs {worst['cell hgt [m]']} m cell, "
            f"{worst['dist [km]']} km away). Its flux bias is largely an "
            "altitude mismatch, not a statement about the 54 km physics.\n"
        )

    if bc is not None:
        p.append("\n## Mean over the five towers, by resolution\n")
        p.append("\nMB (MAE); T2m in °C, fluxes in µmol m⁻² s⁻¹.\n\n")
        p.append(md(bc) + "\n")

    p.append("\n## Which resolution actually wins\n")
    if bc is not None:
        wins = []
        for _, r in bc.iterrows():
            vals = {}
            for res in RESOLUTIONS:
                c = r.get(res, "--")
                if c in ("--", "") or pd.isna(c):
                    continue
                mb = float(c.split("(")[0].strip())
                mae = float(c.split("(")[1].split(")")[0]) if "(" in c else np.nan
                vals[res] = (abs(mb), mae)
            if len(vals) < 2:
                continue
            wins.append({
                "quantity": r.quantity,
                "best abs MB": min(vals, key=lambda k: vals[k][0]),
                "best MAE": min(vals, key=lambda k: vals[k][1]),
            })
        wdf = pd.DataFrame(wins)
        p.append("\n" + md(wdf) + "\n")
        n1 = int((wdf["best MAE"] == "1km").sum())
        p.append(
            f"\n**The 1 km run is not uniformly the best at the towers.** It "
            f"wins MAE in {n1} of {len(wdf)} comparable quantities. It is "
            "clearly better for $T_{2m}$ and for $R_{eco}$, but for **GPP and "
            "NEE on the ALPS channel the coarse runs score better**, and "
            "monotonically so: ALPS GPP MAE falls 3.32 → 2.67 → 2.42 and ALPS "
            "NEE MAE falls 4.41 → 3.61 → 3.40 going from 1 km to 54 km, with "
            "ALPS NEE bias shrinking from -1.86 to +0.12.\n"
        )
    p.append("\n## How to read this\n")
    p.append(
        "\nThis result needs stating plainly, because it does **not** say what "
        "a limitation paragraph would like it to say: on these five towers the "
        "fine run is not shown to be the better one overall.\n"
    )
    p.append(
        "\nThat said, the coarse advantage on GPP/NEE is most plausibly "
        "**error compensation rather than skill**. The ALPS parameter set "
        "over-predicts GPP at 1 km (MB +1.43), and everything about coarsening "
        "pushes GPP down: the domain-mean GPP itself falls (Table 1: 4.33 → "
        "4.07 µmol m⁻² s⁻¹ clear-sky), and the coarse cells are far less "
        "vegetated at these towers (VEGFRA 0.05–0.43 at 54 km against 0.83–1.00 "
        "at 1 km), so the modelled flux is diluted by non-vegetated area. A "
        "downward push applied to an over-prediction improves the score without "
        "improving the physics. The same coarsening makes $T_{2m}$ and "
        "$R_{eco}$ worse, which is what you would expect if the mechanism were "
        "dilution and altitude mismatch rather than better representation.\n"
    )
    p.append(
        "\nSo the defensible claim is the narrow one: **a tower comparison at "
        f"this scale cannot rank the resolutions.** The 54 km cell is a "
        f"{DX['54km']**2:,} km² average judged against a point measurement "
        "12–33 km away — and at 54 km two of the five towers (IT-Lav, IT-MBo) "
        "are the same cell, so the five rows are really four. The honest "
        "framing for the manuscript is that the coarse runs are *not* "
        "validated as worse at the towers, and that this evaluation cannot "
        "separate a resolution effect from a representativeness effect. "
        "Claiming the 1 km run is better on this evidence would be "
        "unsupportable, and a referee would be right to say so.\n"
    )

    REPORT.write_text("".join(p))
    print(f"Wrote {REPORT}")
    print(f"  grid points: {len(gp)} rows across {gp.res.nunique()} resolutions")
    if bc is not None:
        print(f"  bias comparison: {len(bc)} quantities, "
              f"{[c for c in bc.columns if c != 'quantity']}")


if __name__ == "__main__":
    main()
