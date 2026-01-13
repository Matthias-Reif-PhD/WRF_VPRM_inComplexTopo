"""
Plot plant functional types (PFTs) for nested WRF domains (d01 and d03).
Visualizes dominant vegetation types and overlays fractional PFT pie charts.
"""

import matplotlib

matplotlib.use("Agg")

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
import netCDF4 as nc
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# --- Input Parameters ---
dx = "_54km"
domain = "_d01"
base_mz = f"/scratch/c7071034/DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/out{domain}_2012{dx}"
t_file_fra = "VPRM_input_VEG_FRA_d01_2012.nc"
outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"

# --- Load vegetation fraction map ---
"""Load vegetation fraction data from xarray dataset."""
ds = xr.open_dataset(os.path.join(base_mz, t_file_fra))
veg_frac_map = ds["vegetation_fraction_map"]
lat = ds["lat"].values
lon = ds["lon"].values

# --- Load d02 domain and restrict data to that extent ---
"""Load d02 WRF domain extent and mask vegetation data accordingly."""
d2 = xr.open_dataset("/scratch/c7071034/WPS/geo_em.d02.nc")
lat2 = (
    d2["XLAT_M"]
    .isel(Time=0, south_north=slice(10, -10), west_east=slice(10, -10))
    .values
)
lon2 = (
    d2["XLONG_M"]
    .isel(Time=0, south_north=slice(10, -10), west_east=slice(10, -10))
    .values
)

lat2_min, lat2_max = lat2.min(), lat2.max()
lon2_min, lon2_max = lon2.min(), lon2.max()

mask = (lat >= lat2_min) & (lat <= lat2_max) & (lon >= lon2_min) & (lon <= lon2_max)

# Get indices where mask is True
y_idx, x_idx = np.where(mask)
if len(y_idx) == 0 or len(x_idx) == 0:
    raise ValueError("No overlapping data between domain and vegetation fraction map")

# Slice bounds
ymin, ymax = y_idx.min(), y_idx.max() + 1
xmin, xmax = x_idx.min() - 1, x_idx.max() + 1

# Crop datasets
veg_frac_map = veg_frac_map.isel(
    south_north=slice(ymin, ymax), west_east=slice(xmin, xmax)
)
lat = lat[ymin:ymax, xmin:xmax]
lon = lon[ymin:ymax, xmin:xmax]

# --- PFT Settings (Colors and Labels) ---
"""
Define colors and labels for plant functional types (VPRM classification).
8 classes: ENF, DBF, MF, SHB, SAV, CRO, GRA, OTH (Others)
"""
colors = [
    "#006400",  # Evergreen forest
    "#228B22",  # Deciduous forest
    "#8FBC8F",  # Mixed forest
    "#A0522D",  # Shrubland
    "#FFD700",  # Savannas
    "#FFA07A",  # Cropland
    "#7CFC00",  # Grassland
    "#808080",  # Others
]
pft_labels = [
    "Evergreen forest",
    "Deciduous forest",
    "Mixed forest",
    "Shrubland",
    "Savannas",
    "Cropland",
    "Grassland",
    "Others",
]
cmap = mcolors.ListedColormap(colors)
bounds = np.arange(1, 10)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# --- Compute Dominant PFT ---
"""Calculate dominant PFT (1-8) for each grid cell as argmax of vegetation fractions."""
dominant_type = veg_frac_map.argmax(dim="vprm_classes") + 1

# --- Create Base Map with Cartopy ---
"""Plot domain map with dominant vegetation type background and topographic features."""
fig = plt.figure(figsize=(12, 15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(
    [float(lon2.min()), float(lon2.max()), float(lat2.min()), float(lat2.max())],
    crs=ccrs.PlateCarree(),
)

# --- Background: Dominant Vegetation Type ---
"""Plot dominant PFT as pcolormesh background using colormap."""
im = ax.pcolormesh(
    lon,
    lat,
    dominant_type,
    cmap=cmap,
    norm=norm,
    shading="auto",
    transform=ccrs.PlateCarree(),
)

# --- Add Map Features ---
"""Add coastlines, borders, lakes, and rivers."""
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, edgecolor="black")
ax.add_feature(cfeature.LAKES, edgecolor="black")
ax.add_feature(cfeature.RIVERS)

# --- Add Gridlines ---
"""Add geographic gridlines with labels."""
gl = ax.gridlines(
    draw_labels=True, linewidth=1.5, color="black", alpha=0.3, linestyle="--"
)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 18}
gl.ylabel_style = {"size": 18}

# --- Overlay: Fractional PFT Pies via Inset Axes ---
"""Plot mini pie charts at regular grid intervals showing fractional composition of each PFT."""
step = 1
pie_size = 1.2

for i in range(1, lat.shape[0] - 1, step):
    for j in range(1, lat.shape[1] - 1, step):
        f = veg_frac_map[:, i, j].values
        if np.isnan(f).all() or np.sum(f) == 0:
            continue
        f = f / np.sum(f)
        x, y = lon[i, j], lat[i, j]

        ax_inset = inset_axes(
            ax,
            width=pie_size,
            height=pie_size,
            loc="center",
            bbox_to_anchor=(x, y),
            bbox_transform=ax.transData,
            borderpad=0,
        )
        ax_inset.pie(f, colors=colors, wedgeprops=dict(edgecolor="k", linewidth=0.2))
        ax_inset.set_aspect("equal")
        ax_inset.axis("off")

plt.tight_layout()
# plt.show()
plt.savefig(f"{outfolder}/domain_d02_PFTs_pie_per_cell.pdf", bbox_inches="tight")
plt.close()

# ============================================================================
# === Part 2: FLUXNET Site Matching and Vegetation Type Comparison ===
# ============================================================================

warnings.filterwarnings("ignore")


def compute_slope_aspect(hgt, lats, lons):
    """
    Compute slope and aspect from digital elevation model.

    Args:
        hgt (ndarray): Elevation grid (m).
        lats (ndarray): 2D latitude array.
        lons (ndarray): 2D longitude array.

    Returns:
        tuple: (slope in degrees, aspect in degrees 0-360).
    """
    lat_rad = np.radians(lats)
    dy = 111000  # meters per degree latitude
    dx = 111000 * np.cos(lat_rad)  # meters per degree longitude

    dz_dy = np.gradient(hgt, axis=0) / dy
    dz_dx = np.gradient(hgt, axis=1) / dx

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope = np.degrees(slope_rad)

    aspect = (np.degrees(np.arctan2(dz_dy, -dz_dx)) + 360) % 360
    return slope, aspect


def haversine_dist(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between points on Earth.

    Args:
        lat1, lon1, lat2, lon2 (float or ndarray): Latitude/longitude in degrees.

    Returns:
        Distance in km.
    """
    R = 6371.0  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def find_best_fluxnet_match(
    lat_target,
    lon_target,
    lats,
    lons,
    location_pft,
    veg_frac_map,
    hgt,
    hgt_site,
    radius,
    slope=None,
    aspect=None,
):
    """
    Find best matching grid cell for a FLUXNET site using multi-criteria cost function.

    Balances height difference, slope, aspect, PFT vegetation fraction, and distance.

    Args:
        lat_target, lon_target: Site coordinates (degrees).
        lats, lons: 2D coordinate grids.
        location_pft: Target PFT class (1-8).
        veg_frac_map: 4D vegetation fraction array (time, pft, y, x).
        hgt: Elevation grid (m).
        hgt_site: Site elevation (m).
        radius: Search radius (km).
        slope, aspect: Pre-computed slope/aspect grids (optional).

    Returns:
        tuple: (distance_km, grid_idx, slope_target, slope_diff, aspect_target,
                aspect_diff, veg_frac, height_diff, cost)
    """
    if slope is None or aspect is None:
        slope, aspect = compute_slope_aspect(hgt, lats, lons)

    # Site reference
    flat_idx = np.abs(lats - lat_target) + np.abs(lons - lon_target)
    lat_idx, lon_idx = np.unravel_index(np.argmin(flat_idx), lats.shape)
    target_slope = slope[lat_idx, lon_idx]
    target_aspect = aspect[lat_idx, lon_idx]

    # Distance constraint
    dist_km = haversine_dist(lat_target, lon_target, lats, lons)
    dist_mask = dist_km <= abs(radius)

    # Terrain differences
    height_diff = np.abs(hgt - hgt_site)
    slope_diff = np.abs(slope - target_slope)
    aspect_diff = np.abs((aspect - target_aspect + 180) % 360 - 180)

    # Vegetation fraction (axis 1 = PFT)
    veg_frac = veg_frac_map[0, location_pft - 1, :, :]  # shape (ny, nx)

    # Define weighted cost function
    cost = (
        0.5 * (height_diff / np.nanmax(height_diff))
        + 0.1 * (slope_diff / 90.0)
        + 0.1 * (aspect_diff / 180.0)
        + 0.6 * (1.0 - veg_frac) ** 2
    )

    # Apply radius mask
    cost = np.where(dist_mask, cost, np.inf)

    if not np.any(np.isfinite(cost)):
        raise ValueError("No valid grid cell within radius.")

    min_idx = np.unravel_index(np.argmin(cost), cost.shape)
    min_dist = dist_km[min_idx]

    return (
        min_dist,
        min_idx,
        target_slope,
        slope_diff[min_idx],
        target_aspect,
        aspect_diff[min_idx],
        veg_frac[min_idx],
        height_diff[min_idx],
        cost[min_idx],
    )


def find_nearest_grid_hgt_sa(
    lat_target, lon_target, lats, lons, location_pft, IVGTYP_vprm, hgt, hgt_site, radius
):
    """
    Find nearest grid cell matching PFT with similar slope and aspect.

    Args:
        lat_target, lon_target: Site coordinates (degrees).
        lats, lons: 2D coordinate grids.
        location_pft: Target PFT class (1-8).
        IVGTYP_vprm: 2D PFT type grid.
        hgt: Elevation grid (m).
        hgt_site: Site elevation (m).
        radius: Search radius (km).

    Returns:
        tuple: (distance_km, grid_idx, slope_target, aspect_target)
    """
    slope, aspect = compute_slope_aspect(hgt, lats, lons)

    flat_idx = np.abs(lats - lat_target) + np.abs(lons - lon_target)
    min_flat_idx = np.argmin(flat_idx)
    lat_idx, lon_idx = np.unravel_index(min_flat_idx, lats.shape)
    target_slope = slope[lat_idx, lon_idx]
    target_aspect = aspect[lat_idx, lon_idx]

    pft_mask = IVGTYP_vprm == location_pft
    dist_km = haversine_dist(lat_target, lon_target, lats, lons)
    dist_mask = dist_km <= abs(radius)

    height_diff = np.abs(hgt - hgt_site)
    slope_diff = np.abs(slope - target_slope)
    aspect_diff = np.abs((aspect - target_aspect + 180) % 360 - 180)

    aspect_mask = aspect_diff <= 20
    slope_mask = slope_diff <= 10

    combined_mask = pft_mask & dist_mask & aspect_mask & slope_mask

    if not np.any(combined_mask):
        relaxed_mask = pft_mask & dist_mask
        if np.any(relaxed_mask):
            masked_height_diff = np.where(relaxed_mask, height_diff, np.inf)
            min_idx = np.unravel_index(np.argmin(masked_height_diff), hgt.shape)
            min_dist = dist_km[min_idx]
            return min_dist, min_idx, target_slope, target_aspect
        else:
            fallback_flat_idx = np.argmin(dist_km)
            fallback_idx = np.unravel_index(fallback_flat_idx, lats.shape)
            fallback_dist = dist_km[fallback_idx]
            return fallback_dist, fallback_idx, target_slope, target_aspect

    masked_height_diff = np.where(combined_mask, height_diff, np.inf)
    min_idx = np.unravel_index(np.argmin(masked_height_diff), hgt.shape)
    min_dist = dist_km[min_idx]

    return min_dist, min_idx, target_slope, target_aspect


def find_nearest_grid_hgt(
    lat_target, lon_target, lats, lons, location_pft, IVGTYP_vprm, hgt, hgt_site, radius
):
    """
    Find nearest grid cell matching PFT with minimum height difference within radius.

    Fallback method: only considers PFT type and elevation match.

    Args:
        lat_target, lon_target: Site coordinates (degrees).
        lats, lons: 2D coordinate grids.
        location_pft: Target PFT class (1-8).
        IVGTYP_vprm: 2D PFT type grid.
        hgt: Elevation grid (m).
        hgt_site: Site elevation (m).
        radius: Search radius (km).

    Returns:
        tuple: (distance_km, grid_idx)
    """

    # Get valid lat/lon values
    valid_mask = IVGTYP_vprm == location_pft
    valid_lats = np.where(valid_mask, lats, np.nan)
    valid_lons = np.where(valid_mask, lons, np.nan)

    # Convert latitude and longitude differences to km
    lat_diff = (
        valid_lats - lat_target
    )  # * 111  # approximate conversion factor for degrees to km
    lon_diff = (
        valid_lons - lon_target  # * 111 * np.cos(np.radians(lat_target))
    )  # adjust for latitude
    dist = np.sqrt(lat_diff**2 + lon_diff**2)
    dist_km = dist * 111  # approximate conversion factor for degrees to km

    # Mask the distance to only consider points within the radius
    within_radius_mask = dist_km <= abs(radius)
    dist_within_radius = np.where(within_radius_mask, dist_km, np.nan)

    # Check if there are valid points within the radius
    if np.all(np.isnan(dist_within_radius)):
        min_index = np.unravel_index(np.nanargmin(dist), lats.shape)
        return (
            np.nanmin(dist_km),
            min_index,
        )  # No valid points found within the radius, closest index is given back without checking height

    # Calculate the height difference for points within the radius
    height_diff_within_radius = np.where(
        within_radius_mask, np.abs(hgt - hgt_site), np.nan
    )

    # Get the index of the minimum height difference within the radius
    min_height_diff_idx = np.unravel_index(
        np.nanargmin(height_diff_within_radius), lats.shape
    )

    # Get the value of the distance with the minimum height
    dist_idx = dist_within_radius[min_height_diff_idx[0], min_height_diff_idx[1]]

    # Return the minimum distance and the index of the minimum height difference
    return dist_idx, min_height_diff_idx


def get_int_var(lat_target, lon_target, lats, lons, WRF_var):
    """
    Interpolate WRF variable to target location using RegularGridInterpolator.

    Args:
        lat_target, lon_target: Target coordinates (degrees).
        lats, lons: 2D coordinate grids.
        WRF_var: 2D WRF variable grid.

    Returns:
        Interpolated value at target location.
    """
    interpolator = RegularGridInterpolator((lats[:, 0], lons[0, :]), WRF_var)
    interpolated_value = interpolator((lat_target, lon_target))
    return interpolated_value


############# INPUT PARAMETERS ############
"""Configuration for domain, resolution, and file paths."""
domain = "_d02"
dx = "_1km"
dx_int = dx[1:-2]
radius = 30  # should be set to dx*20 to fit to extract_timeseries.py
std_threshold = 200
base_mz = (
    "/scratch/c7071034/DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/out"
    + domain
    + "_2012"
    + dx
)

t_file_fra = "VPRM_input_VEG_FRA_d01_2012.nc"
if dx == "_1km":
    wrfinput_path = "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km/wrfinput_d02"
    t_file_fra = "VPRM_input_VEG_FRA_d02_2012.nc"
elif dx == "_3km":
    wrfinput_path = "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km/wrfinput_d01"
elif dx == "_9km":
    wrfinput_path = "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_9km/wrfinput_d01"
elif dx == "_27km":
    wrfinput_path = "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_27km/wrfinput_d01"
elif dx == "_54km":
    wrfinput_path = "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km/wrfinput_d01"

# --- FLUXNET Site Elevation Reference ---
"""Elevation of known FLUXNET sites (m above sea level)."""
locations_hgt = {
    "AT-Neu": 970,
    "CH-Cha": 393,
    "CH-Dav": 1639,
    "CH-Fru": 982,
    "CH-Lae": 689,
    "CH-Oe1": 450,
    "CH-Oe2": 452,
    "DE-Lkb": 1308,
    "IT-Isp": 210,
    "IT-La2": 1350,
    "IT-Lav": 1353,
    "IT-MBo": 1550,
    "IT-PT1": 60,
    "IT-Ren": 1730,
    "IT-Tor": 2160,
}

if dx == "_1km":
    """1 km resolution: high-altitude mountain sites."""
    sites = [
        {
            "site": "CH-Dav",
            "country": "Switzerland",
            "pft": "ENF",
            "pft_id": "1",
            "lat": 46.8153,
            "lon": 9.8559,
            "hgt_site": 1639,
        },
        {
            "site": "IT-Lav",
            "country": "Italy",
            "pft": "ENF",
            "pft_id": "1",
            "lat": 45.9562,
            "lon": 11.2813,
            "hgt_site": 1353,
        },
        {
            "site": "IT-Ren",
            "country": "Italy",
            "pft": "ENF",
            "pft_id": "1",
            "lat": 46.5869,
            "lon": 11.4337,
            "hgt_site": 1730,
        },
        {
            "site": "AT-Neu",
            "country": "Austria",
            "pft": "GRA",
            "pft_id": "7",
            "lat": 47.1167,
            "lon": 11.3175,
            "hgt_site": 970,
        },
        {
            "site": "IT-MBo",
            "country": "Italy",
            "pft": "GRA",
            "pft_id": "7",
            "lat": 46.0147,
            "lon": 11.0458,
            "hgt_site": 1550,
        },
    ]
else:
    """Coarser resolutions: comprehensive site coverage."""
    sites = [
        {
            "site": "IT-Isp",
            "country": "Italy",
            "pft": "DBF",
            "pft_id": "2",
            "lat": 45.8126,
            "lon": 8.6336,
            "hgt_site": 210,
        },
        {
            "site": "CH-Oe2",
            "country": "Switzerland",
            "pft": "CRO",
            "pft_id": "6",
            "lat": 47.2863,
            "lon": 7.7343,
            "hgt_site": 452,
        },
        {
            "site": "IT-PT1",
            "country": "Italy",
            "pft": "DBF",
            "pft_id": "2",
            "lat": 45.2009,
            "lon": 9.061,
            "hgt_site": 60,
        },
        {
            "site": "CH-Dav",
            "country": "Switzerland",
            "pft": "ENF",
            "pft_id": "1",
            "lat": 46.8153,
            "lon": 9.8559,
            "hgt_site": 1639,
        },
        {
            "site": "DE-Lkb",
            "country": "Germany",
            "pft": "ENF",
            "pft_id": "1",
            "lat": 49.0996,
            "lon": 13.3047,
            "hgt_site": 1308,
        },
        {
            "site": "IT-La2",
            "country": "Italy",
            "pft": "ENF",
            "pft_id": "1",
            "lat": 45.9542,
            "lon": 11.2853,
            "hgt_site": 1350,
        },
        {
            "site": "IT-Lav",
            "country": "Italy",
            "pft": "ENF",
            "pft_id": "1",
            "lat": 45.9562,
            "lon": 11.2813,
            "hgt_site": 1353,
        },
        {
            "site": "IT-Ren",
            "country": "Italy",
            "pft": "ENF",
            "pft_id": "1",
            "lat": 46.5869,
            "lon": 11.4337,
            "hgt_site": 1730,
        },
        {
            "site": "AT-Neu",
            "country": "Austria",
            "pft": "GRA",
            "pft_id": "7",
            "lat": 47.1167,
            "lon": 11.3175,
            "hgt_site": 970,
        },
        {
            "site": "CH-Cha",
            "country": "Switzerland",
            "pft": "GRA",
            "pft_id": "7",
            "lat": 47.2102,
            "lon": 8.4104,
            "hgt_site": 393,
        },
        {
            "site": "CH-Fru",
            "country": "Switzerland",
            "pft": "GRA",
            "pft_id": "7",
            "lat": 47.1158,
            "lon": 8.5378,
            "hgt_site": 982,
        },
        {
            "site": "CH-Oe1",
            "country": "Switzerland",
            "pft": "GRA",
            "pft_id": "7",
            "lat": 47.2858,
            "lon": 7.7319,
            "hgt_site": 450,
        },
        {
            "site": "IT-MBo",
            "country": "Italy",
            "pft": "GRA",
            "pft_id": "7",
            "lat": 46.0147,
            "lon": 11.0458,
            "hgt_site": 1550,
        },
        {
            "site": "IT-Tor",
            "country": "Italy",
            "pft": "GRA",
            "pft_id": "7",
            "lat": 45.8444,
            "lon": 7.5781,
            "hgt_site": 2160,
        },
        {
            "site": "CH-Lae",
            "country": "Switzerland",
            "pft": "MF",
            "pft_id": "3",
            "lat": 47.4781,
            "lon": 8.3644,
            "hgt_site": 689,
        },
    ]

# CORINE vegetation type labels
"""Mapping of 44 CORINE vegetation type classes to names."""
vegetation_labels = [
    "Continuous Urban Fabric",
    "Discontinuous Urban Fabric",
    "Industrial or Commercial Units",
    "Road and Rail Networks",
    "Port Areas",
    "Airports",
    "Mineral Extraction Sites",
    "Dump Sites",
    "Construction Sites",
    "Green Urban Areas",
    "Sport and Leisure Facilities",
    "Non-irrigated Arable Land",
    "Permanently Irrigated Land",
    "Rice Fields",
    "Vineyards",
    "Fruit Trees and Berry Plantations",
    "Olive Groves",
    "Pastures",
    "Annual Crops Associated with Permanent Crops",
    "Complex Cultivation Patterns",
    "Land Principally Occupied by Agriculture",
    "Agro-forestry Areas",
    "Broad-leaved Forest",
    "Coniferous Forest",
    "Mixed Forest",
    "Natural Grasslands",
    "Moors and Heathland",
    "Sclerophyllous Vegetation",
    "Transitional Woodland-Shrub",
    "Beaches, Dunes, Sands",
    "Bare Rocks",
    "Sparsely Vegetated Areas",
    "Burnt Areas",
    "Glaciers and Perpetual Snow",
    "Inland Marshes",
    "Peat Bogs",
    "Salt Marshes",
    "Salines",
    "Intertidal Flats",
    "Water Courses",
    "Water Bodies",
    "Coastal Lagoons",
    "Estuaries",
]

# Define a color map for CORINE vegetation types
"""Colors for 44 CORINE classes (red urban to blue water)."""
vegetation_colors = [
    "#FF0000",
    "#FF4500",
    "#B22222",
    "#808080",
    "#4682B4",
    "#87CEFA",
    "#8B4513",
    "#A0522D",
    "#D2691E",
    "#32CD32",
    "#7FFF00",
    "#FFFF00",
    "#FFD700",
    "#F0E68C",
    "#8B0000",
    "#B8860B",
    "#CD853F",
    "#ADFF2F",
    "#DDA0DD",
    "#F5DEB3",
    "#FFE4C4",
    "#DEB887",
    "#228B22",
    "#006400",
    "#8FBC8F",
    "#7CFC00",
    "#556B2F",
    "#6B8E23",
    "#8B7500",
    "#F4A460",
    "#A9A9A9",
    "#D3D3D3",
    "#FFFFFF",
    "#2E8B57",
    "#3CB371",
    "#20B2AA",
    "#5F9EA0",
    "#48D1CC",
    "#00CED1",
    "#1E90FF",
    "#0000CD",
    "#4682B4",
    "#708090",
]

corine_to_vprm = {
    24: 1,  # Coniferous Forest (Evergreen)
    23: 2,  # Broad-leaved Forest (Deciduous)
    25: 3,
    29: 3,  # Mixed Forest and Transitional Woodland-Shrub
    27: 4,
    28: 4,  # Moors and Heathland, Sclerophyllous Vegetation (Shrubland)
    35: 5,
    36: 5,
    37: 5,  # Wetlands: Inland Marshes, Peat Bogs, Salt Marshes
    12: 6,
    13: 6,
    14: 6,
    15: 6,
    16: 6,
    17: 6,
    19: 6,
    20: 6,
    21: 6,
    22: 6,  # Cropland
    18: 7,
    26: 7,  # Grassland: Pastures, Natural Grasslands
    # Others mapped to 8 (gray)
    1: 8,
    2: 8,
    3: 8,
    4: 8,
    5: 8,
    6: 8,
    7: 8,
    8: 8,
    9: 8,
    10: 8,
    11: 8,
    30: 8,
    31: 8,
    32: 8,
    33: 8,
    34: 8,
    38: 8,
    39: 8,
    40: 8,
    41: 8,
    42: 8,
    43: 8,
    44: 8,
}

# --- Define Labels for Simplified Vegetation Types ---
"""VPRM 8-class PFT abbreviations corresponding to CORINE reclassification."""
labels_vprm_short = [
    "ENF",
    "DBF",
    "MF",
    "SHB",
    "SAV",
    "CRO",
    "GRA",
    "OTH",  # Include "Others"
]
# Load the NetCDF file
"""Load WRF input file with vegetation type (IVGTYP) and topography."""
nc_fid = nc.Dataset(wrfinput_path, "r")
IVGTYP = nc_fid.variables["IVGTYP"][0, 10:-10, 10:-10]
IVGTYP.shape
# --- Retrieve WRF Coordinates and Variables ---
"""Extract CORINE vegetation types (IVGTYP), latitude, longitude, and auxiliary variables."""
XLAT = nc_fid.variables["XLAT"][0, 10:-10, 10:-10]  # Assuming the first time slice
XLONG = nc_fid.variables["XLONG"][0, 10:-10, 10:-10]
HGT_M = nc_fid.variables["HGT"][0, 10:-10, 10:-10]
STDVAR = nc_fid.variables["VAR"][0, 10:-10, 10:-10]
IVGTYP_vprm = np.vectorize(corine_to_vprm.get)(IVGTYP[:, :])
IVGTYP_vprm.shape

# --- Load Vegetation Fraction Map ---
"""Load pre-processed vegetation fraction data and reclassify water bodies."""
in_veg_frac = xr.open_dataset(os.path.join(base_mz, t_file_fra))
veg_frac_map = in_veg_frac["vegetation_fraction_map"][:, 10:-10, 10:-10]

veg_frac_class_8 = veg_frac_map.sel(vprm_classes=8)
mask_ivgtyp_trimmed = IVGTYP == 44
veg_frac_class_8 = veg_frac_class_8.where(~mask_ivgtyp_trimmed, 1)

veg_frac_map.loc[{"vprm_classes": 8}] = veg_frac_class_8

# --- Compute Dominant Vegetation Type ---
"""Get dominant VPRM class for each grid cell and convert to DataFrame."""
VPRM_in_dom_vgtyp = veg_frac_map.argmax(dim="vprm_classes") + 1
df_VPRM_in_dom_vgtyp = VPRM_in_dom_vgtyp.to_dataframe(
    name="vegetation_type"
).reset_index()
if dx == "_1km":
    """Load 1km VPRM input data for high-resolution domain."""
    d0X = "wrfout_d02"
    vprm_input_path_1km = f"/scratch/c7071034/DATA/VPRM_input/vprm_corine_1km/vprm_input_d02_2012-06-23_00:00:00.nc"
    ds = xr.open_dataset(vprm_input_path_1km)
    # ['Times', 'XLONG', 'XLAT', 'EVI_MIN', 'EVI_MAX', 'EVI', 'LSWI_MIN', 'LSWI_MAX', 'LSWI', 'VEGFRA_VPRM']
    veg_frac_map = (
        ds["VEGFRA_VPRM"]
        .isel(south_north=slice(10, -10), west_east=slice(10, -10))
        .values
    )
    veg_frac_map = np.nan_to_num(veg_frac_map, nan=0.0)
else:
    """Load coarser resolution VPRM input data."""
    vprm_input_path = f"/scratch/c7071034/DATA/VPRM_input/vprm_corine_{res}/vprm_input_d01_2012-06-23_00:00:00.nc"
    ds = xr.open_dataset(vprm_input_path)
    # ['Times', 'XLONG', 'XLAT', 'EVI_MIN', 'EVI_MAX', 'EVI', 'LSWI_MIN', 'LSWI_MAX', 'LSWI', 'VEGFRA_VPRM']
    veg_frac_map = ds["VEGFRA_VPRM"].values
    veg_frac_map = np.nan_to_num(veg_frac_map, nan=0.0)

# --- Find PFT Match in Search Radius with Height Criteria ---
"""
Match each FLUXNET site to nearest WRF grid cell using multi-criteria optimization.
Considers: distance, elevation, slope, aspect, and vegetation fraction.
"""
for site in sites:
    lat, lon = site["lat"], site["lon"]
    # dist_km, grid_idx, target_slope,target_aspect = find_nearest_grid_hgt_sa(lat, lon, XLAT, XLONG,int(site["pft_id"]),VPRM_in_dom_vgtyp,HGT_M,site["hgt_site"],radius)

    (
        dist_km,
        grid_idx,
        target_slope,
        target_slope_diff,
        target_aspect,
        target_aspect_diff,
        veg_frac_idx,
        height_diff_idx,
        cost,
    ) = find_best_fluxnet_match(
        lat,
        lon,
        XLAT,
        XLONG,
        int(site["pft_id"]),
        veg_frac_map,
        HGT_M,
        site["hgt_site"],
        radius,
    )
    print(
        f"Cost: \n  [{site['site']}] "
        f"Dist={dist_km:.2f} km | "
        f"Height Diff={height_diff_idx:.2f} m"
        f"Idx={grid_idx} | "
        f"Slope={target_slope:.1f}° | "
        f"Slope diff={target_slope_diff:.1f}° | "
        f"Aspect={target_aspect:.1f}° | "
        f"Aspect diff={target_aspect_diff:.1f}° | "
        f"VegFrac={veg_frac_idx:.2f} | "
        f"Cost={cost:.3f}"
    )

    # --- Append matched model cell properties to site dict ---
    """Store location and characteristics of best-matching grid cell."""
    site["model_pft"] = IVGTYP[grid_idx[0], grid_idx[1]]
    site["model_hgt_NN"] = HGT_M[grid_idx[0], grid_idx[1]]
    site["model_lat"] = XLAT[grid_idx[0], grid_idx[1]]
    site["model_lon"] = XLONG[grid_idx[0], grid_idx[1]]
    site["dist_km"] = dist_km
    site["target_slope"] = target_slope
    site["target_aspect"] = target_aspect
    site["model_pft_CLC"] = vegetation_labels[site["model_pft"] - 1]
    site["model_pft_CLC_to_VPRM"] = labels_vprm_short[
        corine_to_vprm[site["model_pft"]] - 1
    ]
    # site["model_pft_VPRM_dom_id"] = int(VPRM_in_dom_vgtyp[grid_idx[0],grid_idx[1]])
    if site["pft"] == site["model_pft_CLC_to_VPRM"]:
        site["pft_match"] = True
    else:
        site["pft_match"] = False

# --- Convert to DataFrame and Save ---
"""Tabular output of site-to-grid matching results."""
df_sites_match = pd.DataFrame(sites)

# Print the table
print(df_sites_match)
df_sites_match.to_csv("pft_site_match_at" + dx + ".csv")


# --- Define Color Map for Simplified Vegetation Types ---
"""Create colormap and normalization for 8 VPRM classes."""
custom_colors = [
    "#006400",
    "#228B22",
    "#8FBC8F",
    "#A0522D",
    "#FFD700",
    "#FFA07A",
    "#7CFC00",
    "#808080",
]  # Added gray for "Others"

# Create a colormap using the defined colors
cmap_simplified = mcolors.ListedColormap(custom_colors)

# --- Define Vegetation Type Colormap and Normalization ---
"""ListedColormap for 8 VPRM classes with BoundaryNorm for discrete intervals."""
custom_colors = [
    "#006400",
    "#228B22",
    "#8FBC8F",
    "#A0522D",
    "#FFD700",
    "#FFA07A",
    "#7CFC00",
    "#808080",
]  # Added gray for "Others"

# Create a colormap using the defined colors
cmap = mcolors.ListedColormap(custom_colors)

# Boundaries for each vegetation type (1 to 8, with an extra to close the last bin)
bounds = list(range(1, 10))
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# --- Prepare 2D Vegetation Data for Plotting ---
"""Pivot DataFrame to 2D array and create topography-based mask."""
veg_2d = df_VPRM_in_dom_vgtyp.pivot(
    index="lat", columns="lon", values="vegetation_type"
)
# veg_2d.fillna(0, inplace=True)

# mask low std areas (kept separate from veg values)
mask_low_std = STDVAR < std_threshold

# --- Create Vegetation Plot ---
"""Main plot: dominant vegetation on Cartopy map with hatched overlay for low topography variability."""
fig, ax = plt.subplots(figsize=(12, 15), subplot_kw={"projection": ccrs.PlateCarree()})

# --- Plot Dominant Vegetation Types ---
"""Draw dominant PFT as pcolormesh background (zorder=0)."""
veg_plot = ax.pcolormesh(
    veg_2d.columns.values,
    veg_2d.index.values,
    veg_2d.values,
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    zorder=0,
)

# --- HATCHED overlay on top using PolyCollection (correct coverage + border) ---
"""Apply hatching to cells where topographic standard deviation is below threshold."""
from matplotlib.collections import PolyCollection

low_polys = []

lat_vals = veg_2d.index.values
lon_vals = veg_2d.columns.values

for i in range(len(lat_vals) - 1):
    for j in range(len(lon_vals) - 1):
        if mask_low_std[i, j]:
            # build the cell polygon
            poly = [
                (lon_vals[j], lat_vals[i]),
                (lon_vals[j + 1], lat_vals[i]),
                (lon_vals[j + 1], lat_vals[i + 1]),
                (lon_vals[j], lat_vals[i + 1]),
            ]
            low_polys.append(poly)

# build hatched collection
coll = PolyCollection(
    low_polys,
    facecolor="none",
    edgecolor="black",
    hatch="/",
    linewidth=0.01,
    zorder=20,
    transform=ccrs.PlateCarree(),
)

ax.add_collection(coll)

# --- Create Legend with Hatch Pattern ---
"""Patch legend for PFT colors + hatched pattern indicator."""
# Define vegetation type labels
veg_labels = [
    "Evergreen forest",
    "Deciduous forest",
    "Mixed forest",
    "Shrubland",
    "Savannas",
    "Cropland",
    "Grassland",
    "Others",
]
patches = [Patch(color=custom_colors[k], label=lab) for k, lab in enumerate(veg_labels)]

# add hatched legend entry
hatched_patch = Patch(
    facecolor="none", edgecolor="black", hatch="///", label=r"STD$_\text{TOPO}$ < 200 m"
)
patches.append(hatched_patch)

leg = ax.legend(
    handles=patches,
    loc="lower left",
    fontsize=20,
    frameon=True,
    framealpha=0.7,
)
leg.set_zorder(1000)  # force legend to front
leg.get_frame().set_facecolor("white")  # optional: ensure readability
leg.get_frame().set_edgecolor("black")  # optional border

# --- Add Gridlines ---
"""Geographic gridlines with coordinate labels."""
gl = ax.gridlines(
    draw_labels=True, linewidth=1.5, color="black", alpha=0.3, linestyle="--"
)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 20}
gl.ylabel_style = {"size": 20}

# --- Plot FLUXNET Site Matching ---
"""
Mark observation sites (black circles) and matched model cells (blue circles).
Connect with blue lines. Color-code labels based on PFT match success.
"""
for index, site in df_sites_match.iterrows():
    label = f"{site['site']}-{site['pft']}"  # Create label as 'site-pft'

    # Adjust specific label positions if necessary
    if site["site"] == "IT-Lav":
        text_offset_x, text_offset_y = 0.05, -0.1
    else:
        text_offset_x, text_offset_y = 0.05, 0.1

    if site["pft_match"]:
        colorx = "blue"
    else:
        colorx = "purple"

    # --- Plot Site Markers ---
    """Red circle: observation site. Blue circle: matched model cell."""
    ax.plot(
        [site["lon"], site["model_lon"]],
        [site["lat"], site["model_lat"]],
        color="black",
        linestyle="-",
        linewidth=2,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        site["lon"],
        site["lat"],
        marker="o",
        color="red",
        markersize=11,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        site["model_lon"],
        site["model_lat"],
        marker="o",
        color="blue",
        markersize=9,
        transform=ccrs.PlateCarree(),
    )

    # --- Add Site Label ---
    """Text label with site name and PFT, color-coded by match success."""
    ax.text(
        site["lon"] + text_offset_x,
        site["lat"] + text_offset_y,
        label,
        color="black",
        transform=ccrs.PlateCarree(),
        fontsize=24,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.3, boxstyle="round, pad=0.3"),
        ha="left",
        va="center",
    )

# --- Finalize and Save Figure ---
"""Set tick sizes and save to PDF."""
plt.xticks(fontsize=20)  # for tick labels
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(
    f"{outfolder}/VGT_VPRM_{domain+dx}_STD{std_threshold}.pdf", bbox_inches="tight"
)
