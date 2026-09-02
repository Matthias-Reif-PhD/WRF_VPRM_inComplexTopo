"""
verify_toptfix_channels.py
======================================================================
Regression check for the online VPRM Topt fix + the p10..p90 percentile
channels (WRF branch vprm-topt-fix-percentile-ensemble, tag v0.3).

Why this exists
---------------
In `chem/module_ghg_fluxes.F` the per-channel Topt was set with Fortran
`DATA` statements placed inside a runtime `SELECT CASE`. `DATA` is a
compile-time initializer, so only the first one in source order ever took
effect and EVERY parallel VPRM channel silently ran with the
`VPRM_table_US` values (Topt = 20,20,20,20,20,22,18,0), regardless of
`p_ebio_gee_t`. The fix replaces them with executable array assignments
and adds 5 channels carrying the Topt-percentile ensemble.

This script compares a corrected run against the archived buggy run for
the same date and asserts the specific pattern the fix must produce.

What it asserts
---------------
  1. METEOROLOGY IDENTICAL   T2 / SWDOWN bit-identical old vs new.
     The 54km job is serial (ntasks=1) and aer_ra_feedback=0 with CO2
     passive, so the fix must not perturb the meteorology at all.
  2. REF CHANNELS IDENTICAL  EBIO_GEE_REF / EBIO_RES_REF (channel 6) and
     EBIO_GEE_DPDT_REF (channel 8) bit-identical. Their intended Topt
     equals the value the bug forced, so they are the control: any
     difference means something unintended was perturbed.
  3. FIXED CHANNELS DIFFER   EBIO_GEE, _2.._5 and EBIO_GEE_DPDT must
     differ from the old run. If they don't, the fix is not in the
     binary that ran.
  4. NEW CHANNELS SANE       EBIO_GEE_P10..P90 / EBIO_RES_P10..P90 exist,
     are finite, and are NON-ZERO. All-zero GEE is the specific
     signature of a lambda sign error: the CSVs store a positive
     light-use efficiency but WRF needs it negative, and VPRM's
     min(0.0,...) clamp would zero every flux.
  5. ENSEMBLE SPREAD         The five members differ from one another.
     Identical members would mean the Topt dispatch is still broken.

Usage
-----
    python verify_toptfix_channels.py \
        --new /scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km_toptfix \
        --old /scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km \
        --date 2012-07-27

Exit status is 0 only if every check passes.
======================================================================
"""

import argparse
import glob
import os
import sys

import numpy as np
import netCDF4 as nc

# Channel -> NetCDF field name, as registered in Registry/registry.chem.
REF_CHANNELS = ["EBIO_GEE_REF", "EBIO_RES_REF", "EBIO_GEE_DPDT_REF", "EBIO_RES_DPDT_REF"]
FIXED_CHANNELS = ["EBIO_GEE", "EBIO_GEE_2", "EBIO_GEE_3", "EBIO_GEE_4", "EBIO_GEE_5",
                  "EBIO_GEE_DPDT"]
PCT_GEE = ["EBIO_GEE_P10", "EBIO_GEE_P25", "EBIO_GEE_P50", "EBIO_GEE_P75", "EBIO_GEE_P90"]
PCT_RES = ["EBIO_RES_P10", "EBIO_RES_P25", "EBIO_RES_P50", "EBIO_RES_P75", "EBIO_RES_P90"]
MET = ["T2", "SWDOWN"]


class Result:
    """Collects pass/fail lines so the whole report prints even on failure."""

    def __init__(self):
        self.failures = 0

    def check(self, ok, label, detail=""):
        mark = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        print(f"  [{mark}] {label}" + (f"  --  {detail}" if detail else ""))
        return ok

    def note(self, label, detail=""):
        print(f"  [ .. ] {label}" + (f"  --  {detail}" if detail else ""))


def read(path, name):
    """Read one variable's first time slice, or None if absent."""
    with nc.Dataset(path) as ds:
        if name not in ds.variables:
            return None
        return np.asarray(ds.variables[name][0], dtype=np.float64)


def field_names(path):
    with nc.Dataset(path) as ds:
        return set(ds.variables)


def timesteps(directory, date):
    """wrfout files for the given day, sorted by time."""
    return sorted(glob.glob(os.path.join(directory, f"wrfout_d01_{date}_*")))


def relative_diff(a, b):
    """max|a-b| normalised by the field's own scale, plus the raw max|diff|.

    Compared only over cells finite in BOTH fields. The 9km VPRM input has a
    couple of cells where LSWI_MAX makes VPRM's Wscale divide by zero, so
    EBIO_GEE carries a handful of NaNs -- in the archived runs as well as the
    new ones. Without masking, those few cells would turn every comparison into
    NaN and hide the real result.
    """
    good = np.isfinite(a) & np.isfinite(b)
    if not np.any(good):
        return float("nan"), float("nan"), 0
    absdiff = np.max(np.abs(a[good] - b[good]))
    scale = max(np.max(np.abs(b[good])), 1e-30)
    return absdiff / scale, absdiff, int(np.count_nonzero(~good))


def compare_pair(res, new_f, old_f, tol, label_prefix=""):
    """Checks 1-3: old-vs-new comparison on one timestep.

    `tol` is a RELATIVE tolerance for the "unchanged" assertions:
      tol == 0   -> require bit-identical. Correct for a serial run (54km,
                    ntasks=1), where the fix provably cannot perturb anything.
      tol  > 0   -> allow round-off. Necessary at 9km/3km, where the job runs
                    on 99/256 MPI tasks and reduction ordering can move the
                    last bits even for an unchanged calculation.
    A channel counts as genuinely CHANGED only if it moved by >100x tol, so a
    real Topt change is never confused with decomposition noise.
    """
    new_vars = field_names(new_f)
    changed_floor = max(100.0 * tol, 1e-12)

    def nan_note(name, a, b, masked):
        """Flag NaNs, and whether the new run introduced any beyond the old."""
        n_new = int(np.count_nonzero(~np.isfinite(a)))
        n_old = int(np.count_nonzero(~np.isfinite(b)))
        if masked:
            res.check(n_new <= n_old,
                      f"{label_prefix}{name} introduces no new NaNs",
                      f"NaN cells new={n_new} old={n_old} (pre-existing, masked out)")

    for group, kind in ((MET, "met"), (REF_CHANNELS, "control")):
        for name in group:
            a, b = read(new_f, name), read(old_f, name)
            if a is None or b is None:
                res.note(f"{label_prefix}{name}: absent in one file, skipped")
                continue
            rel, absd, nmask = relative_diff(a, b)
            finite_equal = np.array_equal(a[np.isfinite(a) & np.isfinite(b)],
                                          b[np.isfinite(a) & np.isfinite(b)])
            ok = finite_equal if tol == 0 else rel <= tol
            word = "identical" if tol == 0 else f"unchanged (rel<={tol:g})"
            res.check(ok, f"{label_prefix}{name} {word}"
                          + (" (control)" if kind == "control" else ""),
                      f"rel={rel:.2e} max|diff|={absd:.3e}"
                      + (f" [{nmask} NaN cells masked]" if nmask else ""))
            nan_note(name, a, b, nmask)

    for name in FIXED_CHANNELS:
        a, b = read(new_f, name), read(old_f, name)
        if a is None or b is None:
            res.note(f"{label_prefix}{name}: absent in one file, skipped")
            continue
        rel, absd, nmask = relative_diff(a, b)
        res.check(rel > changed_floor, f"{label_prefix}{name} CHANGED by the fix",
                  f"rel={rel:.2e} max|diff|={absd:.3e}"
                  + (f" [{nmask} NaN cells masked]" if nmask else ""))
        nan_note(name, a, b, nmask)

    return new_vars


def check_new_channels(res, new_f, label_prefix=""):
    """Checks 4-5: the percentile ensemble on one timestep."""
    present = field_names(new_f)

    for group, kind in ((PCT_GEE, "GEE"), (PCT_RES, "RES")):
        missing = [n for n in group if n not in present]
        if not res.check(not missing, f"{label_prefix}all 5 {kind} percentile channels present",
                         f"missing: {missing}" if missing else "P10 P25 P50 P75 P90"):
            continue

        arrays = {n: read(new_f, n) for n in group}

        for n, a in arrays.items():
            nbad = int(np.count_nonzero(~np.isfinite(a)))
            frac = nbad / a.size
            # A handful of NaN cells is a pre-existing property of the VPRM
            # input (Wscale divides by zero where LSWI_MAX is a fill value) and
            # is present in the archived runs too -- only widespread NaN is a
            # real failure.
            res.check(frac < 0.01, f"{label_prefix}{n} finite (>99% of cells)",
                      f"{nbad}/{a.size} NaN" if nbad else "no NaN")
            # All-zero is the lambda-sign-error signature.
            nonzero = np.any(np.abs(a[np.isfinite(a)]) > 0)
            res.check(nonzero, f"{label_prefix}{n} non-zero",
                      f"mean={np.nanmean(a):.4g} min={np.nanmin(a):.4g} "
                      f"max={np.nanmax(a):.4g}")

        # Members must not be identical to one another (compare finite cells).
        names = list(arrays)

        def same(x, y):
            g = np.isfinite(x) & np.isfinite(y)
            return np.array_equal(x[g], y[g])

        identical = [(names[i], names[j])
                     for i in range(len(names)) for j in range(i + 1, len(names))
                     if same(arrays[names[i]], arrays[names[j]])]
        if kind == "GEE":
            res.check(not identical, f"{label_prefix}{kind} members mutually distinct",
                      f"identical pairs: {identical}" if identical else "all 5 differ")
        else:
            # Reco carries no Topt dependence and alpha/beta are shared across
            # members, so the RES members are EXPECTED to be identical.
            res.note(f"{label_prefix}{kind} members identical as expected "
                     f"(Reco has no Topt dependence): {len(identical)}/10 pairs")


def summarise_ensemble(new_f):
    """Domain-mean GEE per member -- the physical ordering sanity check."""
    print("\n  Domain-mean GEE by Topt percentile member "
          "(mol km^-2 hr^-1, negative = uptake):")
    for n in PCT_GEE:
        a = read(new_f, n)
        if a is not None:
            print(f"    {n:<16} mean={np.nanmean(a):12.4f}   min={np.nanmin(a):12.4f}")
    for n in ("EBIO_GEE", "EBIO_GEE_REF"):
        a = read(new_f, n)
        if a is not None:
            print(f"    {n:<16} mean={np.nanmean(a):12.4f}   min={np.nanmin(a):12.4f}"
                  "   (site-combo / reference)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", required=True, help="dir with the corrected run's wrfout")
    p.add_argument("--old", required=True, help="dir with the archived buggy-Topt wrfout")
    p.add_argument("--date", default="2012-07-27", help="YYYY-MM-DD")
    p.add_argument("--hours", default="09,12,15",
                   help="comma-separated hours to test (default: 09,12,15 -- daytime, "
                        "when GPP is non-zero and the Topt difference actually shows)")
    p.add_argument("--tol", type=float, default=None,
                   help="relative tolerance for the 'unchanged' checks. Default: 0 "
                        "(bit-identical) for 54km/27km, which run serial or near-serial; "
                        "1e-6 for 9km/3km, whose 99/256-task MPI decomposition can move the "
                        "last bits. Set explicitly to override the auto-choice.")
    args = p.parse_args()

    if args.tol is None:
        # 9km runs on 99 tasks and 3km on 256; both can differ at round-off even
        # where the physics is unchanged. 54km is serial (ntasks=1) and 27km uses 9.
        parallel = any(k in args.new for k in ("9km", "3km", "1km"))
        args.tol = 1e-6 if parallel else 0.0
    print(f"unchanged-check tolerance: rel <= {args.tol:g}"
          f"{'  (bit-identical)' if args.tol == 0 else '  (round-off allowed, parallel run)'}")

    res = Result()
    new_files = timesteps(args.new, args.date)
    if not new_files:
        print(f"ERROR: no wrfout for {args.date} in {args.new}", file=sys.stderr)
        return 2

    print(f"new: {args.new}\nold: {args.old}\ndate: {args.date}")
    print(f"found {len(new_files)} new timesteps for that day\n")

    for hh in args.hours.split(","):
        stamp = f"{args.date}_{hh.strip()}:00:00"
        new_f = os.path.join(args.new, f"wrfout_d01_{stamp}")
        old_f = os.path.join(args.old, f"wrfout_d01_{stamp}")
        if not os.path.exists(new_f):
            res.note(f"{stamp}: no new file, skipped")
            continue
        print(f"--- {stamp} ---")
        if os.path.exists(old_f):
            compare_pair(res, new_f, old_f, args.tol, label_prefix="")
        else:
            res.note("no matching old file -- old/new comparison skipped for this step")
        check_new_channels(res, new_f)
        print()

    midday = os.path.join(args.new, f"wrfout_d01_{args.date}_12:00:00")
    if os.path.exists(midday):
        summarise_ensemble(midday)

    print(f"\n{'ALL CHECKS PASSED' if res.failures == 0 else f'{res.failures} CHECK(S) FAILED'}")
    return 0 if res.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
