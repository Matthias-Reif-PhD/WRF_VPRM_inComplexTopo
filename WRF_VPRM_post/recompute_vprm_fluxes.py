"""
recompute_vprm_fluxes.py
==============================================
Offline recomputation of VPRM-old fluxes (GPP, Reco, NEE) on the WRF grid,
WITHOUT re-running WRF.

Why this exists
---------------
The WRF output (~3 TB, fixed) stores the *online* VPRM fluxes EBIO_GEE / EBIO_RES
that were computed at run time with the *old* parameter set. To evaluate a new
(GPP-based-Topt) parameter set we recompute the fluxes offline from the WRF
meteorology that was archived in `wrfout` (SWDOWN, T2) together with the static
VPRM input fields (EVI, LSWI, VEGFRA_VPRM) -- exactly the same inputs the WRF
VPRM module uses. The GPP math is a direct, validated port of the per-PFT
`GPP_validate` recompute already in `Fig11_linPertComp_hourly_mean.py`; here it
is generalised to:
  * load parameters from a CSV table (`vprm_params_recalc.csv`) instead of
    hard-coded dicts, so a new parameter set can be dropped in;
  * additionally compute Reco = max(0, alpha*T2 + beta) and NEE = Reco - GPP;
  * write one gridded NetCDF per timestep, mirroring the `gpp_pmodel` /
    `reco_migliavacca` per-timestep files that the `extract_*` scripts already
    know how to ingest (see their `run_Pmodel` branch).

Output (per wrfout timestep), into  <out_root>/vprm_recalc_<tag>_<res>/ :
    vprm_recalc_<tag>_<res>_YYYY-MM-DD_HH:MM:SS.nc
with variables GPP, RECO, NEE  [umol m-2 s-1]  on the native WRF grid of that
resolution (interior 10:-10 trim for 1km, full grid for the coarse domains, to
match the conventions used in the extract chain).

Sign / unit conventions (consistent with the extract chain and VPRM.py)
    GPP  >= 0 ,  uptake
    RECO >= 0 ,  release
    NEE  = RECO - GPP   (= EBIO_RES + EBIO_GEE ;  positive = net source)
    lambda is the POSITIVE light-use efficiency (Fig10 convention); the tuned
    xlsx stores it with the opposite (GEE) sign -- flip it when you fill the CSV.

Usage
-----
    python recompute_vprm_fluxes.py -s "2012-07-01 00:00:00" \
                                    -e "2012-07-07 00:00:00" \
                                    -r 1km -t "" --tag ALPSgpp

    # cluster: see job_recompute_vprm_fluxes.slurm
==============================================
"""

import os
import sys
import glob
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from dotenv import load_dotenv

# Load environment variables from .env file (repo root)
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ==================== Configuration ====================
SCRATCH_PATH = os.getenv("SCRATCH_PATH")
GITHUB_PATH = os.getenv("GITHUB_PATH")
CSVFOLDER = os.getenv("CSVFOLDER")

EPS = 1e-7  # numerical safeguard (matches Fig10)
WRF_GEE_TO_UMOL = 1.0 / 3600.0  # EBIO_* [mmol m-2 h-1 sign] -> umol m-2 s-1 (validation only)

# Default location of the parameter table (next to this script)
DEFAULT_PARAM_CSV = Path(__file__).resolve().parent / "vprm_params_recalc.csv"

# Default destination for the recomputed gridded NetCDF files
DEFAULT_OUT_ROOT = os.path.join(SCRATCH_PATH or "", "DATA", "VPRM_recalc")


def resolve_input_path(path, default=None):
    """Resolve a file path relative to the current cwd, the script dir, or the repo root."""
    if path is None:
        return Path(default).expanduser() if default is not None else None

    p = Path(path).expanduser()
    if p.is_absolute():
        return p

    for base in [Path.cwd(), Path(__file__).resolve().parent, ROOT]:
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate

    return (Path.cwd() / p).resolve()


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
def load_params(param_csv):
    """Read the per-class VPRM parameter table into ordered numpy arrays.

    Returns a dict of length-7 arrays indexed by m = vprm_veg_id - 1
    (m = 0..6 -> classes 1..7), matching the PFT axis of VEGFRA_VPRM and the
    m-loop in Fig10.
    """
    param_csv = resolve_input_path(param_csv, DEFAULT_PARAM_CSV)
    df = pd.read_csv(param_csv, comment="#")
    df = df.sort_values("vprm_veg_id").reset_index(drop=True)
    ids = df["vprm_veg_id"].to_numpy()
    if not np.array_equal(ids, np.arange(1, 8)):
        raise ValueError(
            f"param table must contain vprm_veg_id 1..7 exactly, got {ids.tolist()}"
        )
    return {
        "pft": df["pft"].tolist(),
        "t_opt": df["t_opt"].to_numpy(float),
        "t_min": df["t_min"].to_numpy(float),
        "t_max": df["t_max"].to_numpy(float),
        "par0": df["par0"].to_numpy(float),
        "lambda": df["lambda"].to_numpy(float),
        "alpha": df["alpha"].to_numpy(float),
        "beta": df["beta"].to_numpy(float),
    }


# --------------------------------------------------------------------------- #
# File handling (same selection logic as the extract_* scripts)
# --------------------------------------------------------------------------- #
def extract_datetime_from_filename(filename):
    """Datetime from 'wrfout_d0x_YYYY-MM-DD_HH:MM:SS'."""
    base = os.path.basename(filename)
    date_str = base.split("_")[-2] + "_" + base.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")


def select_full_day_files(wrf_dir, d0x, start_date, end_date):
    """Return basenames of wrfout files that belong to complete 24 h days
    (00:00..23:00) inside [start_date, end_date]. Mirrors the extract chain."""
    start_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").date()
    end_obj = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").date()

    all_files = sorted(glob.glob(os.path.join(wrf_dir, f"{d0x}_*")))
    by_day = defaultdict(list)
    for f in all_files:
        dt = extract_datetime_from_filename(f)
        if start_obj <= dt.date() <= end_obj:
            by_day[dt.date()].append((dt, os.path.basename(f)))

    file_list = []
    for day in sorted(by_day.keys()):
        files = sorted(by_day[day])
        if len(files) == 24 and all(dt.hour == i for i, (dt, _) in enumerate(files)):
            file_list.extend(f for _, f in files)
    return file_list


# --------------------------------------------------------------------------- #
# Core VPRM-old recompute (port of Fig10 GPP_validate + Reco/NEE)
# --------------------------------------------------------------------------- #
def _load_vprm_input(path, trim):
    """Load the per-PFT static VPRM fields for one day, optionally interior-trimmed.
    `trim` is True for the 1km domain (10:-10), False for the coarse domains."""
    ds = xr.open_dataset(path)

    def get(name, fill):
        da = ds[name]
        if trim:
            da = da.isel(south_north=slice(10, -10), west_east=slice(10, -10))
        return np.nan_to_num(da.values, nan=fill)

    fields = {
        "vegfra": get("VEGFRA_VPRM", 0.0),
        "evi": get("EVI", 0.0),
        "evi_min": get("EVI_MIN", 0.0),
        "evi_max": get("EVI_MAX", 0.0),
        "lswi": get("LSWI", -1.0),
        "lswi_min": get("LSWI_MIN", -1.0),
        "lswi_max": get("LSWI_MAX", -1.0),
    }
    ds.close()
    return fields


def compute_fluxes(swdown, t2_celsius, vin, p):
    """Recompute GPP, Reco, NEE on the grid for one timestep.

    Parameters
    ----------
    swdown      : 2D array, SWDOWN [W m-2]
    t2_celsius  : 2D array, 2 m temperature [degC]
    vin         : dict of per-PFT static fields from `_load_vprm_input`
                  (each shaped [1, 7, ny, nx] on the PFT axis)
    p           : parameter dict from `load_params`

    Returns GPP, RECO, NEE arrays [umol m-2 s-1].
    """
    GPP = np.zeros_like(swdown, dtype=np.float64)
    RECO = np.zeros_like(swdown, dtype=np.float64)

    for m in range(7):
        vegfrac = vin["vegfra"][0, m, :, :]
        if np.all(vegfrac < EPS):
            continue
        vegfrac = np.where(np.isnan(vegfrac) | (vegfrac < EPS), 0.0, vegfrac)

        evi = vin["evi"][0, m, :, :]
        evi_min = vin["evi_min"][0, m, :, :]
        evi_max = vin["evi_max"][0, m, :, :]
        lswi = vin["lswi"][0, m, :, :]
        lswi_min = vin["lswi_min"][0, m, :, :]
        lswi_max = vin["lswi_max"][0, m, :, :]

        # --- PAR / RAD saturation term:  SWDOWN / (1 + SWDOWN/PAR0) ---
        par0 = p["par0"][m]
        if par0 > 0:
            rad = (1.0 / (1.0 + swdown / par0)) * swdown
            rad = np.nan_to_num(rad, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            rad = np.zeros_like(swdown)

        # --- Tscale (Raich/VPRM form) ---
        t_opt, t_min, t_max = p["t_opt"][m], p["t_min"][m], p["t_max"][m]
        a1 = t2_celsius - t_min
        a2 = t2_celsius - t_max
        a3 = t2_celsius - t_opt
        tscale = np.where((a1 < 0) | (a2 > 0), 0.0, a1 * a2 / (a1 * a2 - a3 ** 2))
        tscale = np.nan_to_num(tscale, nan=0.0, posinf=0.0, neginf=0.0)
        tscale = np.where(tscale < 0, 0.0, tscale)

        # --- Wscale ---  (xeric systems: SHB m==3, GRA m==6 use min-max form)
        if m == 3 or m == 6:
            num = lswi - lswi_min
            den = lswi_max - lswi_min
            wscale = np.divide(num, den, out=np.zeros_like(num), where=den >= EPS)
        else:
            wscale = (1.0 + lswi) / (1.0 + lswi_max)
        wscale[np.isnan(wscale)] = 0.0

        # --- Pscale ---  (ENF m==0 -> 1; SAV/GRA m==4,6 -> (1+LSWI)/2; else EVI thresh)
        if m == 0:
            pscale = np.ones_like(swdown)
        elif m == 4 or m == 6:
            pscale = (1.0 + lswi) / 2.0
        else:
            evithresh = evi_min + 0.55 * (evi_max - evi_min)
            pscale = np.where(evi >= evithresh, 1.0, (1.0 + lswi) / 2.0)
        pscale = np.nan_to_num(pscale, nan=0.0)

        # --- GPP for this PFT, weighted by vegetation fraction ---
        gpp_m = p["lambda"][m] * tscale * wscale * pscale * rad * evi * vegfrac
        gpp_m[gpp_m < 0] = 0.0
        GPP += gpp_m

        # --- Reco for this PFT:  max(0, alpha*T2 + beta), vegfrac-weighted ---
        reco_m = p["alpha"][m] * t2_celsius + p["beta"][m]
        reco_m[reco_m < 0] = 0.0
        RECO += reco_m * vegfrac

    NEE = RECO - GPP
    return GPP, RECO, NEE


# --------------------------------------------------------------------------- #
# NetCDF writer
# --------------------------------------------------------------------------- #
def write_flux_nc(out_path, time, lats, lons, gpp, reco, nee, attrs):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ny, nx = gpp.shape
    with nc.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)

        def add(name, data, units, long_name):
            v = ds.createVariable(name, "f4", ("south_north", "west_east"),
                                  zlib=True, complevel=4)
            v[:, :] = data.astype(np.float32)
            v.units = units
            v.long_name = long_name

        add("XLAT", lats, "degrees_north", "latitude")
        add("XLONG", lons, "degrees_east", "longitude")
        add("GPP", gpp, "umol m-2 s-1", "offline-recomputed VPRM GPP")
        add("RECO", reco, "umol m-2 s-1", "offline-recomputed VPRM ecosystem respiration")
        add("NEE", nee, "umol m-2 s-1", "offline-recomputed VPRM NEE (RECO - GPP)")

        ds.valid_time = time.strftime("%Y-%m-%d_%H:%M:%S")
        ds.title = "Offline-recomputed VPRM-old fluxes (no WRF rerun)"
        ds.source = "recompute_vprm_fluxes.py"
        for k, val in attrs.items():
            setattr(ds, k, str(val))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def recompute(start_date, end_date, res, sim_type, tag, param_csv, out_root,
              overwrite=False):
    param_csv = resolve_input_path(param_csv, DEFAULT_PARAM_CSV)
    out_root = str(resolve_input_path(out_root, DEFAULT_OUT_ROOT))
    params = load_params(param_csv)

    # resolution -> wrfout prefix + vprm_input prefix/dir
    if res == "1km":
        d0x = "wrfout_d02"
        trim = True
        vprm_dir = os.path.join(SCRATCH_PATH, "DATA/VPRM_input/vprm_corine_1km")
        vprm_prefix = "vprm_input_d02"
    else:
        d0x = "wrfout_d01"
        trim = False
        vprm_dir = os.path.join(SCRATCH_PATH, f"DATA/VPRM_input/vprm_corine_{res}")
        vprm_prefix = "vprm_input_d01"

    wrf_dir = os.path.join(SCRATCH_PATH, f"DATA/WRFOUT/WRFOUT_ALPS_{res}{sim_type}")
    out_dir = os.path.join(out_root, f"vprm_recalc_{tag}_{res}{sim_type}")

    file_list = select_full_day_files(wrf_dir, d0x, start_date, end_date)
    print(f"[{res}{sim_type}] {len(file_list)} hourly files selected "
          f"({start_date} .. {end_date}) -> {out_dir}")
    if not file_list:
        print("  nothing to do.")
        return

    for wrf_file in file_list:
        time = extract_datetime_from_filename(wrf_file)
        time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
        out_path = os.path.join(out_dir, f"vprm_recalc_{tag}_{res}{sim_type}_{time_str}.nc")
        if os.path.exists(out_path) and not overwrite:
            print(f"  skip (exists): {os.path.basename(out_path)}")
            continue

        # WRF meteorology
        with nc.Dataset(os.path.join(wrf_dir, wrf_file), "r") as fid:
            if trim:
                sl = (0, slice(10, -10), slice(10, -10))
            else:
                sl = (0, slice(None), slice(None))
            swdown = np.asarray(fid.variables["SWDOWN"][sl], dtype=np.float64)
            t2c = np.asarray(fid.variables["T2"][sl], dtype=np.float64) - 273.15
            lats = np.asarray(fid.variables["XLAT"][sl])
            lons = np.asarray(fid.variables["XLONG"][sl])

        # static VPRM input for that day
        day = time.strftime("%Y-%m-%d")
        vprm_path = os.path.join(vprm_dir, f"{vprm_prefix}_{day}_00:00:00.nc")
        if not os.path.exists(vprm_path):
            print(f"  WARNING missing VPRM input, skipping: {vprm_path}")
            continue
        vin = _load_vprm_input(vprm_path, trim)

        gpp, reco, nee = compute_fluxes(swdown, t2c, vin, params)

        write_flux_nc(
            out_path, time, lats, lons, gpp, reco, nee,
            attrs={
                "param_csv": os.path.abspath(param_csv),
                "resolution": res,
                "sim_type": sim_type or "(base)",
                "tag": tag,
                "pft_order": ",".join(params["pft"]),
                "t_opt": ",".join(f"{v:g}" for v in params["t_opt"]),
            },
        )
        print(f"  wrote {os.path.basename(out_path)} | "
              f"GPP~{np.nanmean(gpp):.3f} RECO~{np.nanmean(reco):.3f} "
              f"NEE~{np.nanmean(nee):.3f} umol/m2/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--start", default="2012-01-01 00:00:00",
                        help='start, "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument("-e", "--end", default="2012-12-31 00:00:00",
                        help='end,   "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument("-r", "--res", default="1km",
                        help="WRF resolution dir suffix: 1km, 9km, 54km, 3km, 27km")
    parser.add_argument("-t", "--type", dest="sim_type", default="",
                        help="simulation type suffix: '', '_cloudy', '_parm_err'")
    parser.add_argument("--tag", default="recalc",
                        help="label for the parameter set (used in output dir/file names)")
    parser.add_argument("--params", default=str(DEFAULT_PARAM_CSV),
                        help="path to the parameter CSV")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                        help="root dir for the recomputed NetCDF (kept on /scratch)")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute even if the output file already exists")
    args = parser.parse_args()

    recompute(args.start, args.end, args.res, args.sim_type, args.tag,
              args.params, args.out_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
