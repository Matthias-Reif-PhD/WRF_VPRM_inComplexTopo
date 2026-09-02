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


def copy_tables():
    """Copy each table into plots_final/, reporting whether it is new, changed
    or already identical. Returns the list of shipped destination paths."""
    DST.mkdir(parents=True, exist_ok=True)
    shipped = []
    for name in SHIP:
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


def verify_byte_identical():
    """'Shipped' has to mean 'identical', not 'similar'. filecmp with
    shallow=False compares contents, not just size/mtime."""
    for name in SHIP:
        src, dst = SRC / name, DST / name
        assert dst.exists(), f"{dst} missing after copy"
        assert filecmp.cmp(src, dst, shallow=False), (
            f"{name}: shipped copy differs from {src}"
        )
    print(f"  [ok] all {len(SHIP)} shipped copies byte-identical to their sources")


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

    if args.check:
        missing = [n for n in SHIP if not (DST / n).exists()]
        if missing:
            raise SystemExit(f"not shipped yet: {', '.join(missing)}")
    else:
        print("Shipping:")
        copy_tables()
        print()
        verify_byte_identical()

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
