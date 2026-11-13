"""
vprm_class_area_fractions.py
==============================================
VPRM vegetation-class composition of the area of analysis, at 1, 9 and 54 km.

Written for referee round 2 (L208, GC2, GC4), which asked how much of the domain
each vegetation class actually covers, and how much of that composition survives
coarsening. REVISION_NUMBERS.md section 3 had only four classes (ENF 35.6, GRA
27.4, MF 13.0, DBF 10.0 %) with the remaining 14 % lumped as "SHB/WET/CRO".

Three different numbers answer "what fraction of the area is class X", and they
are NOT interchangeable -- all three are printed so the manuscript can quote the
one it means:

  area_frac   mean of VEGFRA_VPRM over the mask. The honest "area fraction":
              every cell contributes its actual sub-grid mixture, and the eight
              classes sum to 100 %.
  dom_1_7     share of cells whose dominant class is X, argmax over classes 1-7
              with OTH dropped. This is what section 3 reports, and dropping OTH
              inflates every vegetated class by ~1/(1-f_OTH) ~ 1.32.
  dom_1_8     share of cells whose dominant class is X, argmax over all eight.

Area of analysis
----------------
LANDMASK == 1 AND VAR >= 200 m, on the rim-trimmed 1 km grid -- the mask
acclim_experiment/FigAppxF_acclim.py:load_static() builds and the one
REVISION_NUMBERS.md is written against (n = 63 594). The script asserts both that
count and section 3's four percentages before printing anything new, so a silent
change of mask definition cannot pass unnoticed.

Coarse grids are compared by mapping each coarse field onto the 1 km mask with
NEAREST-neighbour interpolation (the class index is categorical; the "linear"
default of three of the repo's five proj_on_finer_WRF_grid copies would be
meaningless for it). Every resolution is therefore scored over the same 63 594
1 km cells, which is what makes the mismatch share below well defined.

Class 5 is WETLANDS
-------------------
WRF_VPRM_pre/pyVPRM/copernicus_land_cover_corine_mapping.yaml:32-39 maps CORINE
grid codes 35/36/37 (CLC 411 inland marshes, 412 peat bogs, 421 salt marshes) to
vprm_class 5 under "Wetlands", and main_tune_VPRM.py carries T_opt["WET"]=20.23,
which is the value sitting in the `5,SAV` row of vprm_params_newGPP_V24.csv. The
palette, Fig. 2's legend and the parameter CSVs all label slot 5 "SAV"/savanna.
This script prints WET, with the code name alongside, and changes nothing
elsewhere -- the mislabel is reported, not fixed (see REVISION_NUMBERS.md).

Usage
-----
    python vprm_class_area_fractions.py
    python vprm_class_area_fractions.py --date 2012-07-27   # static field; any day
==============================================
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import netCDF4 as nc
import numpy as np
from dotenv import load_dotenv
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SCRATCH_PATH = os.getenv("SCRATCH_PATH")

STD_TOPO = 200.0  # m, subgrid orography std (WRF `VAR`) -- the area-of-analysis cut
RESOLUTIONS = ["1km", "9km", "54km"]

# vprm_class 1..8, in VEGFRA_VPRM's own index order. Names per the pyVPRM CORINE
# mapping YAML; the repo's palette calls index 5 "SAV", which is wrong (see the
# module docstring), so both names are carried here.
CLASS_NAMES = ["ENF", "DBF", "MF", "SHB", "WET", "CRO", "GRA", "OTH"]
CLASS_ALIAS = {"WET": "SAV in code/palette"}

# Guards: REVISION_NUMBERS.md section 3, reproduced before anything new is printed.
EXPECTED_MASK_N = 63594
EXPECTED_DOM_1_7 = {"ENF": 35.6, "DBF": 10.0, "MF": 13.0, "GRA": 27.4}


def load_mask(date):
    """Area of analysis on the 1 km grid, plus its lat/lon.

    Same construction as acclim_experiment/FigAppxF_acclim.py:load_static() --
    LANDMASK and VAR off a 1 km wrfout, rim-trimmed by 10 cells. Deliberately NOT
    the coastal-buffer mask of Fig9/extract_dPdT (a different definition), and
    without the `totveg >= 0.05` cut of FigAppxF_Topt_map (which would give
    n = 57 891 and shift ENF by 2.6 points).
    """
    wd = os.path.join(SCRATCH_PATH, "DATA", "WRFOUT", "WRFOUT_ALPS_1km")
    files = sorted(glob.glob(os.path.join(wd, f"wrfout_d02_{date}_*")))
    if not files:
        files = sorted(glob.glob(os.path.join(wd, "wrfout_d02_*")))
    if not files:
        raise FileNotFoundError(f"no wrfout_d02_* under {wd}")
    sl = (0, slice(10, -10), slice(10, -10))
    with nc.Dataset(files[0]) as d:
        land = np.asarray(d.variables["LANDMASK"][sl]) == 1
        var = np.asarray(d.variables["VAR"][sl], dtype=np.float64)
        lat = np.asarray(d.variables["XLAT"][sl], dtype=np.float64)
        lon = np.asarray(d.variables["XLONG"][sl], dtype=np.float64)
    mask = land & (var >= STD_TOPO)
    return mask, lat, lon, os.path.basename(files[0])


def load_vegfra(res, date):
    """VEGFRA_VPRM (8, ny, nx) and its lat/lon for one resolution.

    The 1 km domain is the d02 nest of the 3 km parent and is rim-trimmed to match
    the mask; 9 and 54 km are standalone d01 runs and are not trimmed.
    """
    trim = res == "1km"
    d0x = "d02" if trim else "d01"
    vprm_dir = os.path.join(SCRATCH_PATH, "DATA", "VPRM_input", f"vprm_corine_{res}")
    path = os.path.join(vprm_dir, f"vprm_input_{d0x}_{date}_00:00:00.nc")
    if not os.path.exists(path):
        cand = sorted(glob.glob(os.path.join(vprm_dir, f"vprm_input_{d0x}_*.nc")))
        if not cand:
            raise FileNotFoundError(f"no vprm_input_{d0x}_*.nc under {vprm_dir}")
        path = cand[0]
    with nc.Dataset(path) as fid:
        # VEGFRA_VPRM is (Time, vprm_vgcls, south_north, west_east), but XLAT/XLONG
        # in vprm_input carry NO Time dimension (unlike wrfout) -- they are plain
        # (south_north, west_east), so they must not be indexed with [0].
        veg = np.asarray(fid.variables["VEGFRA_VPRM"][0], dtype=np.float64)
        lat = np.asarray(fid.variables["XLAT"][:], dtype=np.float64)
        lon = np.asarray(fid.variables["XLONG"][:], dtype=np.float64)
    if trim:
        veg = veg[:, 10:-10, 10:-10]
        lat = lat[10:-10, 10:-10]
        lon = lon[10:-10, 10:-10]
    return veg, lat, lon, os.path.basename(path)


def to_1km(field, lat_c, lon_c, lat_f, lon_f):
    """Nearest-neighbour projection of a coarse field onto the 1 km grid.

    Same call as proj_on_finer_WRF_grid (Fig9_AppxG7_areafluxes_per_timestep.py:104),
    pinned to "nearest": these fields are either a categorical class index or the
    fraction belonging to one, and a coarse cell's value applies to every fine cell
    inside it rather than being blended across neighbours.
    """
    return griddata(
        (lat_c.ravel(), lon_c.ravel()),
        field.ravel(),
        (lat_f, lon_f),
        method="nearest",
    )


def composition(veg, mask):
    """area_frac / dom_1_7 / dom_1_8 percentages for one (8, ny, nx) field."""
    flat = veg[:, mask]  # (8, n_mask)
    area = flat.mean(axis=1) * 100.0
    dom8 = np.argmax(flat, axis=0)
    dom7 = np.argmax(flat[:7], axis=0)
    n = flat.shape[1]
    pct8 = np.array([(dom8 == i).sum() for i in range(8)]) / n * 100.0
    pct7 = np.array([(dom7 == i).sum() for i in range(8)]) / n * 100.0
    return area, pct7, pct8, dom7


def fmt_table(rows, header):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    out = ["  ".join(str(h).ljust(w[i]) for i, h in enumerate(header))]
    out.append("  ".join("-" * w[i] for i in range(len(header))))
    for r in rows:
        out.append("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default="2012-07-27",
                    help="vprm_input day to read (the field is static; default 2012-07-27)")
    args = ap.parse_args()

    mask, lat_f, lon_f, wrf_name = load_mask(args.date)
    n_mask = int(mask.sum())
    print(f"area of analysis: LANDMASK==1 & VAR>={STD_TOPO:.0f} m, 10-cell rim removed")
    print(f"  from {wrf_name}: {mask.shape[0]}x{mask.shape[1]} trimmed cells, "
          f"n_mask = {n_mask}")
    assert n_mask == EXPECTED_MASK_N, (
        f"mask is {n_mask} cells, expected {EXPECTED_MASK_N} -- the mask definition "
        "has drifted from the one REVISION_NUMBERS.md section 3 is written against"
    )

    results = {}
    dom_maps = {}
    for res in RESOLUTIONS:
        veg, lat_c, lon_c, src = load_vegfra(res, args.date)
        if res == "1km":
            veg_on_1km = veg
        else:
            veg_on_1km = np.stack(
                [to_1km(veg[i], lat_c, lon_c, lat_f, lon_f) for i in range(veg.shape[0])]
            )
        area, pct7, pct8, dom7 = composition(veg_on_1km, mask)
        results[res] = (area, pct7, pct8)
        dom_maps[res] = dom7
        print(f"  {res:5s} <- {src}  (grid {veg.shape[1]}x{veg.shape[2]})")

    # --- guard: section 3's four published percentages, on the 1 km dominant-1-7 column
    a1, p7_1, p8_1 = results["1km"]
    for name, want in EXPECTED_DOM_1_7.items():
        got = p7_1[CLASS_NAMES.index(name)]
        assert abs(got - want) < 0.1, (
            f"{name} dominant(1-7) = {got:.2f} %, REVISION_NUMBERS.md section 3 says "
            f"{want} % -- refusing to report new numbers off a changed basis"
        )
    print(f"\nreproduced REVISION_NUMBERS.md section 3 "
          f"(n and the four published dominant-class shares)\n")

    # --- composition at each resolution
    for res in RESOLUTIONS:
        area, pct7, pct8 = results[res]
        rows = []
        for i, cls in enumerate(CLASS_NAMES):
            label = cls + (f" ({CLASS_ALIAS[cls]})" if cls in CLASS_ALIAS else "")
            rows.append([label, f"{area[i]:6.2f}",
                         f"{pct7[i]:6.2f}" if i < 7 else "     -",
                         f"{pct8[i]:6.2f}"])
        rows.append(["TOTAL", f"{area.sum():6.2f}", f"{pct7.sum():6.2f}",
                     f"{pct8.sum():6.2f}"])
        print(f"--- {res} (mapped onto the 1 km mask)" if res != "1km" else "--- 1km")
        print(fmt_table(rows, ["class", "area_%", "dom1-7", "dom1-8"]))
        print()

    # --- what section 3 lumped as "the remaining 14 %"
    rem = [(c, results["1km"][0][CLASS_NAMES.index(c)],
            results["1km"][1][CLASS_NAMES.index(c)]) for c in ("SHB", "WET", "CRO")]
    print("section 3's lumped 'remaining 14 % (SHB/WET/CRO)', split out:")
    for c, a, d in rem:
        print(f"  {c:4s} area {a:5.2f} %   dominant(1-7) {d:5.2f} %")
    print(f"  sum  area {sum(a for _, a, _ in rem):5.2f} %   "
          f"dominant(1-7) {sum(d for _, _, d in rem):5.2f} %\n")

    # --- dominant-class mismatch between 1 km and each coarse grid
    # composition() already returns dom7 over the masked cells only, so these are
    # 1-D arrays of length n_mask, aligned cell-for-cell across resolutions.
    print("1 km cells whose dominant class differs from their coarse cell's:")
    base = dom_maps["1km"]
    for res in ("9km", "54km"):
        diff = (base != dom_maps[res]).sum()
        print(f"  {res:5s}  {int(diff):6d} / {n_mask} = {diff / n_mask * 100:5.1f} %")
    print("\nNote: the area fractions are near resolution-invariant while the")
    print("dominant class is not -- coarsening inflates the majority class and")
    print("erases the minor ones. Quote whichever of the two the sentence means.")


if __name__ == "__main__":
    main()
