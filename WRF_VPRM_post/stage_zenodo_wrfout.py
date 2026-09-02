"""
stage_zenodo_wrfout.py
======================================================================
Stage the Topt-corrected WRF output for a new version of Zenodo record
20395680 ("WRF-VPRM v0.2 -- sample WRF input and output data").

Scope
-----
The Topt fix changes ONLY the model output. The record's other three
files (Fluxnet2015.zip, CAMS.zip, VPRM_input.zip) are unaffected and
carry over to the new version untouched -- so this script stages a
replacement for `WRFOUT.zip` alone.

Layout
------
The corrected runs live in `WRFOUT_ALPS_<res>_toptfix/`. In the package
the `_toptfix` suffix is STRIPPED, so the directory names match what
v0.2 published and downstream code keeps working unchanged:

    <staging>/WRFOUT/WRFOUT_ALPS_54km/wrfout_d01_2012-07-27_*
                     WRFOUT_ALPS_9km/...
                     ...
                     MANIFEST.sha256
                     CHANGELOG_v0.3.md

Sidecar files that record provenance (namelist.input, namelist.wps,
job_WRF.slurm_*, geo_em*) are copied alongside the wrfout, matching the
v0.2 convention.

This script only STAGES and checksums. It does not upload -- publishing
to Zenodo is done by hand with the account's own credentials.

Usage
-----
    python stage_zenodo_wrfout.py --date 2012-07-27 \
        --res 54km 27km 9km 3km 1km \
        --staging /scratch/c7071034/DATA/zenodo_v0.3

    python stage_zenodo_wrfout.py --dry-run          # report sizes only
======================================================================
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

SRC_ROOT = Path(os.getenv("SCRATCH_PATH", "/scratch/c7071034")) / "DATA" / "WRFOUT"
SIDECARS = ["namelist.input", "namelist.wps", "geo_em.d01.nc", "geo_em.d02.nc"]

# 1km output is the d02 nest of the 3km run and is archived in its own dir.
DOMAIN_OF = {"1km": "d02"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def source_dir(res):
    return SRC_ROOT / f"WRFOUT_ALPS_{res}_toptfix"


def collect(res, date):
    """wrfout files for `date` plus the provenance sidecars."""
    src = source_dir(res)
    if not src.is_dir():
        return None, []
    dom = DOMAIN_OF.get(res, "d01")
    wrfout = sorted(src.glob(f"wrfout_{dom}_{date}_*"))
    extras = [src / n for n in SIDECARS + [f"job_WRF.slurm_{res}"]
              if (src / n).exists()]
    return src, wrfout + extras


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2012-07-27")
    p.add_argument("--res", nargs="+", default=["54km", "27km", "9km", "3km", "1km"])
    p.add_argument("--staging",
                   default=str(Path(os.getenv("SCRATCH_PATH", "/scratch/c7071034"))
                               / "DATA" / "zenodo_v0.3"))
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be staged, copy nothing")
    args = p.parse_args()

    stage = Path(args.staging) / "WRFOUT"
    total = 0
    staged = {}
    missing = []

    print(f"date    : {args.date}")
    print(f"source  : {SRC_ROOT}")
    print(f"staging : {stage}\n")

    for res in args.res:
        src, files = collect(res, args.date)
        if src is None:
            missing.append(res)
            print(f"  {res:<6} SOURCE MISSING ({source_dir(res).name}) -- run not done yet")
            continue
        if not files:
            missing.append(res)
            print(f"  {res:<6} no files for {args.date} in {src.name}")
            continue
        size = sum(f.stat().st_size for f in files)
        total += size
        staged[res] = files
        nwrf = sum(1 for f in files if f.name.startswith("wrfout_"))
        print(f"  {res:<6} {nwrf:3d} wrfout + {len(files)-nwrf} sidecars   {human(size):>10}")

    print(f"\n  TOTAL {human(total):>34}")
    if missing:
        print(f"  NOT STAGED: {', '.join(missing)}")
    if args.dry_run:
        print("\n(dry run -- nothing copied)")
        return 0
    if not staged:
        print("\nnothing to stage.", file=sys.stderr)
        return 1

    print()
    lines = []
    for res, files in staged.items():
        # strip the _toptfix suffix so the layout matches what v0.2 published
        dest = stage / f"WRFOUT_ALPS_{res}"
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            target = dest / f.name
            if not target.exists() or target.stat().st_size != f.stat().st_size:
                if target.exists():
                    target.unlink()
                # Source and staging are on the same filesystem, so hardlink
                # instead of copying: instant, and costs no extra space for what
                # is ~49 GB of wrfout. Falls back to a real copy across devices.
                try:
                    os.link(f, target)
                except OSError:
                    shutil.copy2(f, target)
            lines.append(f"{sha256(target)}  WRFOUT_ALPS_{res}/{f.name}")
        print(f"  staged {res} -> {dest}")

    manifest = stage / "MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + "\n")
    print(f"\n  wrote {manifest}  ({len(lines)} files)")
    print("  verify later with:  sha256sum -c MANIFEST.sha256")
    print("\nNOT uploaded -- publish the new version on Zenodo by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
