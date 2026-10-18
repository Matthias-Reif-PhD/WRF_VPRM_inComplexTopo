"""
ship_tables.py
==============================================
Copy the generated LaTeX tables from `plots/` (where the Fig*.py scripts write
them) into `plots/plots_final/` (the staging directory the manuscript
`\\input`s from), then re-run the table assertions AGAINST THE SHIPPED COPIES.

Why this exists
---------------
Follow-up 12 fixed the T2 percentage brackets (12d) and rebuilt the grouped
FLUXNET site-means table (12a/12b), but the corrected files never reached
`plots_final/`, so the manuscript kept rendering the pre-12d numbers. Twice.
Asserting on the file the Fig scripts just wrote is not enough -- the assertion
has to run on the copy the paper actually includes, which is what this script
does.

Nothing here regenerates a table. The sources are taken as-is and copied
byte-for-byte, so shipping cannot move a number that the manuscript text has
already been written against. Regenerate with the Fig*.py scripts first if you
want fresh numbers, then ship.

Usage
-----
    python ship_tables.py            # copy + verify
    python ship_tables.py --check    # verify only, copy nothing
==============================================
"""

from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from Fig4_AppxG1_G4_Table1_TableF5_F6_FLUXNET_eval import verify_fluxnet_site_means_table
from Fig5_6_AppxG5_G6_Table2_TableH_WRFout_hourly_means_and_timeseries import (
    verify_no_ratio_cells,
    verify_no_t2_percent,
)

SRC = Path(os.getenv("OUTFOLDER", HERE / "plots"))
DST = SRC / "plots_final"

# The three domain-averaged tables 12d fixed, from which Follow-up 16h then
# split CAMS out into a fourth. verify_no_t2_percent runs over all four in the
# shipped location; verify_no_ratio_cells over the three de-CAMSed ones, which
# are the only ones with a 1 km ALPS/DEFAULT pair.
DOMAIN_TABLES = [
    "Table_1_domain_averaged_2012.tex",
    "Table_seasonal_domain_averaged_2012.tex",
    "Table_seasonal_domain_averaged_2012_cloudy.tex",
]
CAMS_TABLE = "Table_CAMS_domain_averaged_2012.tex"

# Fig. 7 (fig:effects_all) and its two appendix twins read these three directly
# (Fig7_AppxH6_H8_effects_seasonal.py) rather than the raw csv, so the figure and the
# printed numbers cannot drift apart -- they were previously copied to
# plots_final/ by hand; formalizing that here gets them the same
# byte-identity check as the rest of the shipped tables.
EFFECTS_TABLES = [
    "Table_seasonal_effects_2012.tex",
    "Table_seasonal_effects_2012_cloudy.tex",
    "Table_seasonal_effects_2012_all.tex",
]

# tab:par_members (Appendix H9) -- the manuscript's \input for this table had
# no producer at all (Table_par_members_2012_all.tex did not exist anywhere);
# write_par_members_table in the Fig5_6_... script is that producer.
PAR_MEMBERS_TABLE = "Table_par_members_2012_all.tex"

# The 15-row grouped/footnoted site-means table (12a/12b) and the evaluation
# tables. flux_evaluation_1km_bias_r30.tex is shipped alongside because the
# site-means assertion is a *cross-table* check against it -- verifying the
# shipped means table against an unshipped bias table would leave the pairing
# untested in the place that matters. Both evaluation tables are `\input` by
# the manuscript already and were likewise missing from plots_final/.
SITE_TABLES = [
    "fluxnet_site_means_r30.tex",
    "flux_evaluation_1km_bias_r30.tex",
    "flux_evaluation_1km_r30.tex",
]

SHIP = DOMAIN_TABLES + [CAMS_TABLE] + SITE_TABLES + EFFECTS_TABLES + [PAR_MEMBERS_TABLE]

# Present in plots/ but deliberately NOT shipped: not `\input` anywhere in the
# current manuscript (confirmed by grep) -- internal/diagnostic tables that
# fed the design of Table_par_members above, not a table of their own.
# Reported so the gap is visible rather than silently carried.
REPORT_ONLY = [
    "Table_member_deviations_2012_all.tex",
    "Table_member_deviations_dres_2012_all.tex",
]

# Figures reached plots_final/ by hand until round 2 -- there was no figure-copy
# step at all, and two Fig. 10 files had silently gone stale there (plots_final/
# held an Aug 12 build against an Aug 14 source).
#
# Rather than a hand-maintained list, which drifted the moment the manuscript was
# restructured (the round-2 draft gained a supplement, dropped five seasonal
# tables and never picked up the new terrain panel), the figure set is read
# straight out of the manuscript: whatever it \includegraphics from plots_final/
# is what gets shipped. That makes "shipped" and "used" the same set by
# construction, and turns both kinds of drift into a printed report.
#
# Commented-out lines are skipped, so a figure the author has parked behind a %
# is treated as unused -- which is why removing it is only ever *suggested* here,
# never done.
TEX_GLOBS = ("Draft_*.tex", "Supplement_*.tex")

# Fallback when no manuscript is next to the repo (the .tex files are gitignored,
# so a fresh checkout has none): the round-2 figure set, as last shipped.
SHIP_FIGURES_FALLBACK = [
    "regression_Topt_vs_Tmean_old_Alps_VPRM_optimized_params_diff_evo_V24_SW_05_42.pdf",
    "boxplot_PFTs_old_Alps_VPRM_optimized_params_diff_evo_V24_SW_05_42.pdf",
    "boxplot_siteIDs_old_Alps_VPRM_optimized_params_diff_evo_V24_SW_05_42.pdf",
    "boxplot_NNSE_old_Alps_VPRM_optimized_params_diff_evo_V24_SW_05_42.pdf",
    "legend_topt_boxplots.pdf",
    "areafluxes_dT_model_2012-07-27_12h.pdf",
    "areafluxes_dGPPdT_1km_radt54_swint1_2012-07-27_12h.pdf",
    "areafluxes_dGPP_model_02_radt54_swint1_2012-07-27_12h.pdf",
    "hourly_avg_dGPP_combined_std_topo_200all_"
    "2012-01-01 00:00:00_2012-12-31 00:00:00.pdf",
    "hourly_avg_dRECO_combined_std_topo_200all_"
    "2012-01-01 00:00:00_2012-12-31 00:00:00.pdf",
    "legend_combined_all.pdf",
]


def tex_sources():
    """The manuscript files sitting next to the repo, newest naming first."""
    found = []
    for pat in TEX_GLOBS:
        found += sorted(ROOT.glob(pat))
    return found


def referenced_assets():
    """Everything the manuscript pulls out of plots_final/, as a set of names.

    Lines whose first non-space character is `%` are LaTeX comments and are
    skipped -- the round-2 draft has `%\\input{plots_final/Table_seasonal_effects_
    2012_cloudy.tex}`, which must not count as a live reference.
    """
    names = set()
    for tex in tex_sources():
        for line in tex.read_text(errors="replace").splitlines():
            if line.lstrip().startswith("%"):
                continue
            names.update(re.findall(r'plots_final/([^}"]+\.(?:pdf|tex))', line))
    return names


def figures_to_ship(referenced):
    """Referenced figures that actually exist in plots/, sorted.

    Missing ones are reported by main() rather than raised on: a manuscript
    routinely references figures whose script has not been re-run yet, and that
    should not block shipping the ones that are ready.
    """
    if not referenced:
        return [n for n in SHIP_FIGURES_FALLBACK if (SRC / n).exists()]
    return sorted(n for n in referenced if n.endswith(".pdf") and (SRC / n).exists())


def copy_tables(names=SHIP):
    """Copy each file into plots_final/, reporting whether it is new, changed
    or already identical. Returns the list of shipped destination paths."""
    DST.mkdir(parents=True, exist_ok=True)
    shipped = []
    for name in names:
        src, dst = SRC / name, DST / name
        if not src.exists():
            raise FileNotFoundError(
                f"{src} does not exist -- run the Fig*.py that writes it first"
            )
        if not dst.exists():
            state = "NEW"
        elif filecmp.cmp(src, dst, shallow=False):
            state = "unchanged"
        else:
            state = "UPDATED"
        shutil.copy2(src, dst)
        print(f"  [{state:>9}] {name}")
        shipped.append(dst)
    return shipped


def verify_byte_identical(names=SHIP, what="shipped copies"):
    """'Shipped' has to mean 'identical', not 'similar'. filecmp with
    shallow=False compares contents, not just size/mtime."""
    for name in names:
        src, dst = SRC / name, DST / name
        assert dst.exists(), f"{dst} missing after copy"
        assert filecmp.cmp(src, dst, shallow=False), (
            f"{name}: shipped copy differs from {src}"
        )
    print(f"  [ok] all {len(names)} {what} byte-identical to their sources")


def report_drift(referenced, shipped_figures):
    """Print the two gaps between what the manuscript wants and what exists.

    Neither is an error. A referenced-but-unbuilt figure means its script has not
    been re-run yet; a shipped-but-unreferenced one means the manuscript dropped
    it (or has it behind a `%`). Both are printed so they are visible rather than
    silently carried -- which is exactly how plots_final/ drifted before.
    """
    if not referenced:
        return
    want_pdf = {n for n in referenced if n.endswith(".pdf")}
    unbuilt = sorted(n for n in want_pdf if not (SRC / n).exists())
    in_final = {p.name for p in DST.glob("*") if p.is_file()}
    unused = sorted(in_final - referenced)

    print(f"\nManuscript coverage: {len(want_pdf) - len(unbuilt)}/{len(want_pdf)} "
          "referenced figures are built and shipped.")
    if unbuilt:
        print(f"\nReferenced but NOT BUILT ({len(unbuilt)}) -- re-run the "
              "producing script, then ship again:")
        for n in unbuilt:
            print(f"  {n}")
    if unused:
        print(f"\nIn plots_final/ but not referenced ({len(unused)}) -- stale, or "
              "parked behind a % in the manuscript. Left in place; delete by hand "
              "if genuinely dropped:")
        for n in unused:
            print(f"  {n}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the already-shipped copies without copying anything",
    )
    args = parser.parse_args()

    print(f"source: {SRC}")
    print(f"target: {DST}\n")

    referenced = referenced_assets()
    sources = tex_sources()
    if sources:
        print("manuscript: " + ", ".join(t.name for t in sources))
        print(f"  {len(referenced)} plots_final/ references (commented-out lines ignored)\n")
    else:
        print("manuscript: none found next to the repo -- using the fallback "
              "figure list\n")
    ship_figures = figures_to_ship(referenced)

    if args.check:
        missing = [n for n in SHIP + ship_figures if not (DST / n).exists()]
        if missing:
            raise SystemExit(f"not shipped yet: {', '.join(missing)}")
    else:
        print("Shipping tables:")
        copy_tables()
        print()
        verify_byte_identical()
        print("\nShipping figures (every figure the manuscript references):")
        copy_tables(ship_figures)
        print()
        verify_byte_identical(ship_figures, what="shipped figures")

    report_drift(referenced, ship_figures)

    print("\nAsserting against the SHIPPED copies:")

    # Follow-up 12d: no T2 row in any domain-averaged table carries a
    # percentage bracket (a ratio of Celsius values is not a physical quantity).
    verify_no_t2_percent([str(DST / n) for n in DOMAIN_TABLES + [CAMS_TABLE]])

    # Follow-up 16a/16b: no 1 km cell still reports |ALPS|/|DEFAULT| as a ratio
    # where every other cell of the same table reports a difference.
    verify_no_ratio_cells([str(DST / n) for n in DOMAIN_TABLES])

    # Follow-up 12a/12b/12c: 15 grouped data rows, and every 24-day cell for
    # the five d03 sites equals its counterpart in the bias table -- except
    # IT-MBo T2m, allowed 0.01 by the documented _align() pairing rule.
    verify_fluxnet_site_means_table(
        str(DST / "fluxnet_site_means_r30.tex"),
        str(DST / "flux_evaluation_1km_bias_r30.tex"),
    )

    absent = [n for n in REPORT_ONLY if not (DST / n).exists()]
    if absent:
        print(
            "\nNot shipped (not \\input anywhere in the current manuscript -- "
            "internal/diagnostic only; ship by hand if that changes):\n  "
            + "\n  ".join(absent)
        )

    print("\nAll shipped-copy assertions passed.")


if __name__ == "__main__":
    main()
