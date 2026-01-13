"""
Plot WRF domains with topography and FLUXNET site locations.

This module generates publication-quality maps showing:
- WRF model domains (single and nested configurations)
- Topographic contours
- FLUXNET site locations color-coded by plant functional type (PFT)
"""

import matplotlib

matplotlib.use("Agg")

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.lines as mlines

# ==================== Configuration ====================

OUTFOLDER = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"

# Map rendering configuration
FIGSIZE_SINGLE = (12, 15)
FIGSIZE_NESTED = (12, 15)
FONTSIZE_AXIS_LABELS = 22
FONTSIZE_LEGEND = 20
FONTSIZE_TITLE = 24

# Topography contour configuration
TOPO_CONTOUR_INTERVAL = 100  # meters
TOPO_MAX_LEVEL = 4000  # meters
TOPO_COLORMAP = "terrain"

# Site marker configuration
SITE_MARKERSIZE = 12
SITE_EDGEWIDTH = 1.5
LEGEND_LOCATION = "lower right"
LEGEND_FRAMEALPHA = 0.5

# Site metadata
SITES = {
    "names": [
        "AT-Neu",
        "CH-Cha",
        "CH-Dav",
        "CH-Fru",
        "CH-Lae",
        "CH-Oe1",
        "CH-Oe2",
        "DE-Lkb",
        "IT-Isp",
        "IT-La2",
        "IT-Lav",
        "IT-MBo",
        "IT-PT1",
        "IT-Ren",
        "IT-Tor",
    ],
    "lats": [
        47.1167,
        47.2102,
        46.8153,
        47.1158,
        47.4781,
        47.2858,
        47.2863,
        49.0996,
        45.8126,
        45.9542,
        45.9562,
        46.0147,
        45.2009,
        46.5869,
        45.8444,
    ],
    "lons": [
        11.3175,
        8.4104,
        9.8559,
        8.5378,
        8.3650,
        7.7319,
        7.7343,
        13.3047,
        8.6336,
        11.2853,
        11.2813,
        11.0458,
        9.0610,
        11.4337,
        7.5781,
    ],
    "types": [
        "GRA",
        "GRA",
        "ENF",
        "GRA",
        "MF",
        "GRA",
        "CRO",
        "ENF",
        "DBF",
        "ENF",
        "ENF",
        "GRA",
        "DBF",
        "ENF",
        "GRA",
    ],
}

# PFT configuration
PFT_CODES = ["ENF", "DBF", "MF", "SHB", "SAV", "CRO", "GRA"]
PFT_LABELS = [
    "Evergreen forest",
    "Deciduous forest",
    "Mixed forest",
    "Shrubland",
    "Savannas",
    "Cropland",
    "Grassland",
]
PFT_COLORS = [
    "#006400",
    "#228B22",
    "#8FBC8F",
    "#A0522D",
    "#FFD700",
    "#FFA07A",
    "#7CFC00",
]

PFT_COLOR_MAP = dict(zip(PFT_CODES, PFT_COLORS))

# ==================== Functions ====================


def load_wrf_data(filepath):
    """
    Load WRF dataset and extract coordinates and topography.

    Args:
        filepath (str): Path to WRF input file (wrfinput_d*).

    Returns:
        tuple: (hgt, lat, lon) as xarray DataArray objects.
            - hgt: Terrain height [m]
            - lat: Latitude coordinates
            - lon: Longitude coordinates

    Raises:
        FileNotFoundError: If filepath does not exist.
        KeyError: If required variables not found in dataset.
    """
    ds = xr.open_dataset(filepath)
    hgt = ds["HGT"].isel(Time=0)
    lat = ds["XLAT"].isel(Time=0)
    lon = ds["XLONG"].isel(Time=0)
    return hgt, lat, lon


def setup_map(ax, lon_min, lon_max, lat_min, lat_max, map_type=None):
    """
    Configure map axes with geographic features and gridlines.

    Args:
        ax (matplotlib.axes.Axes): Cartopy axes object to configure.
        lon_min (float): Minimum longitude extent.
        lon_max (float): Maximum longitude extent.
        lat_min (float): Minimum latitude extent.
        lat_max (float): Maximum latitude extent.
        map_type (str, optional): Type of map ('single' or 'nested').
            Affects font sizes for gridlines. Defaults to None.

    Returns:
        None. Modifies ax in-place.
    """
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.LAKES, edgecolor="black")
    ax.add_feature(cfeature.RIVERS)

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=1.5, color="black", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    fontsize = FONTSIZE_AXIS_LABELS if map_type == "single" else 20
    gl.xlabel_style = {"size": fontsize}
    gl.ylabel_style = {"size": fontsize}


def plot_topography(ax, lon, lat, hgt, add_colorbar=False):
    """
    Plot topographic contours on map.

    Args:
        ax (matplotlib.axes.Axes): Cartopy axes object.
        lon (xarray.DataArray): 2D longitude coordinate array.
        lat (xarray.DataArray): 2D latitude coordinate array.
        hgt (xarray.DataArray): 2D topography array [m].
        add_colorbar (bool, optional): Whether to add colorbar. Defaults to False.

    Returns:
        matplotlib.contour.ContourSet: The contourf object.
    """
    levels = list(range(0, 3000, TOPO_CONTOUR_INTERVAL)) + [TOPO_MAX_LEVEL]
    cf = ax.contourf(lon, lat, hgt, levels=levels, cmap=TOPO_COLORMAP, extend="both")

    if add_colorbar:
        cb = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.01, shrink=0.4)
        cb.set_label("[m] AMSL", fontsize=FONTSIZE_LEGEND)
        cb.ax.tick_params(labelsize=FONTSIZE_LEGEND)

    return cf


def plot_sites(ax, site_lons, site_lats, site_types, add_legend=True):
    """
    Plot FLUXNET site locations with plant functional type (PFT) colors.

    Args:
        ax (matplotlib.axes.Axes): Cartopy axes object.
        site_lons (list): Longitude coordinates of sites.
        site_lats (list): Latitude coordinates of sites.
        site_types (list): Plant functional type for each site (e.g., 'ENF', 'GRA').
        add_legend (bool, optional): Whether to add PFT legend. Defaults to True.

    Returns:
        None. Modifies ax in-place.
    """
    for lon, lat, pft_type in zip(site_lons, site_lats, site_types):
        ax.plot(
            lon,
            lat,
            "o",
            color=PFT_COLOR_MAP.get(pft_type, "#808080"),
            markersize=SITE_MARKERSIZE,
            transform=ccrs.PlateCarree(),
            markeredgecolor="black",
            markeredgewidth=SITE_EDGEWIDTH,
        )

    if add_legend:
        handles = [
            mlines.Line2D(
                [],
                [],
                color=col,
                marker="o",
                linestyle="None",
                markersize=10,
                label=lbl,
            )
            for col, lbl in zip(PFT_COLORS, PFT_LABELS)
        ]
        # Place legend with semi-transparent background
        ax.legend(
            handles=handles,
            loc=LEGEND_LOCATION,
            fontsize=FONTSIZE_LEGEND,
            frameon=True,
            framealpha=LEGEND_FRAMEALPHA,
        )


def plot_nested_domain_boundary(ax, lon2, lat2):
    """
    Plot the boundary of a nested domain on parent domain map.

    Args:
        ax (matplotlib.axes.Axes): Cartopy axes object.
        lon2 (xarray.DataArray): 2D longitude array of nested domain.
        lat2 (xarray.DataArray): 2D latitude array of nested domain.

    Returns:
        None. Modifies ax in-place.
    """
    ny2, nx2 = lon2.shape
    xbox = [
        float(lon2[0, 0]),
        float(lon2[0, nx2 - 1]),
        float(lon2[ny2 - 1, nx2 - 1]),
        float(lon2[ny2 - 1, 0]),
        float(lon2[0, 0]),
    ]
    ybox = [
        float(lat2[0, 0]),
        float(lat2[0, nx2 - 1]),
        float(lat2[ny2 - 1, nx2 - 1]),
        float(lat2[ny2 - 1, 0]),
        float(lat2[0, 0]),
    ]
    ax.plot(xbox, ybox, color="black", linewidth=1.5, transform=ccrs.PlateCarree())


def add_domain_labels(ax, lon, lat, domain_labels):
    """
    Add text labels identifying domains on map.

    Args:
        ax (matplotlib.axes.Axes): Cartopy axes object.
        lon (xarray.DataArray): 2D longitude coordinate array (for context).
        lat (xarray.DataArray): 2D latitude coordinate array (for context).
        domain_labels (list): List of tuples (label_str, lon_pos, lat_pos).

    Returns:
        None. Modifies ax in-place.
    """
    for label, lo, la in domain_labels:
        ax.text(
            float(lo) + 0.4,
            float(la) + 0.2,
            label,
            fontsize=FONTSIZE_TITLE,
            weight="bold",
            color="black",
            transform=ccrs.PlateCarree(),
        )


def plot_single_domain(hgt1, lat1, lon1, domain_label, filename):
    """
    Generate map of single domain with topography and FLUXNET sites.

    Args:
        hgt1 (xarray.DataArray): 2D topography array [m].
        lat1 (xarray.DataArray): 2D latitude coordinate array.
        lon1 (xarray.DataArray): 2D longitude coordinate array.
        domain_label (str): Label for domain (e.g., 'd01').
        filename (str): Output filename for PDF plot.

    Returns:
        None. Saves figure to OUTFOLDER/filename.
    """
    fig = plt.figure(figsize=FIGSIZE_SINGLE)
    ax = plt.axes(projection=ccrs.PlateCarree())

    setup_map(
        ax,
        float(lon1.min()),
        float(lon1.max()),
        float(lat1.min()),
        float(lat1.max()),
        "single",
    )
    plot_topography(ax, lon1, lat1, hgt1, add_colorbar=False)
    plot_sites(ax, SITES["lons"], SITES["lats"], SITES["types"], add_legend=True)
    add_domain_labels(ax, lon1, lat1, [(domain_label, lon1[0, 0], lat1[0, 0])])

    plt.tight_layout()
    plt.savefig(OUTFOLDER + filename, bbox_inches="tight")
    plt.close()


def plot_nested_domains(hgt1, lat1, lon1, hgt2, lat2, lon2, domain_labels, filename):
    """
    Generate map of nested domains with topography and FLUXNET sites.

    Args:
        hgt1 (xarray.DataArray): 2D topography array of parent domain [m].
        lat1 (xarray.DataArray): 2D latitude coordinates of parent domain.
        lon1 (xarray.DataArray): 2D longitude coordinates of parent domain.
        hgt2 (xarray.DataArray): 2D topography array of nested domain [m].
        lat2 (xarray.DataArray): 2D latitude coordinates of nested domain.
        lon2 (xarray.DataArray): 2D longitude coordinates of nested domain.
        domain_labels (list): List of tuples (label_str, lon_pos, lat_pos) for domains.
        filename (str): Output filename for PDF plot.

    Returns:
        None. Saves figure to OUTFOLDER/filename.
    """
    fig = plt.figure(figsize=FIGSIZE_NESTED)
    ax = plt.axes(projection=ccrs.PlateCarree())

    setup_map(
        ax, float(lon1.min()), float(lon1.max()), float(lat1.min()), float(lat1.max())
    )
    plot_topography(ax, lon1, lat1, hgt1, add_colorbar=True)
    plot_topography(ax, lon2, lat2, hgt2, add_colorbar=False)
    plot_nested_domain_boundary(ax, lon2, lat2)
    plot_sites(ax, SITES["lons"], SITES["lats"], SITES["types"], add_legend=False)
    add_domain_labels(ax, lon1, lat1, domain_labels)

    plt.tight_layout()
    plt.savefig(OUTFOLDER + filename, bbox_inches="tight")
    plt.close()


# ==================== Main ====================


def main():
    """
    Main entry point: Generate both single and nested domain plots.

    Generates two publication-quality PDF maps:
    1. Single 54km domain covering Alps
    2. Nested domains (9km parent + 3km nested) covering Alps

    Returns:
        None. Outputs saved to OUTFOLDER.
    """
    # ---- Plot 1: Single domain (54km) ----
    print("Generating single domain plot (d01)...")
    hgt_d01, lat_d01, lon_d01 = load_wrf_data(
        "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km/wrfinput_d01"
    )
    plot_single_domain(hgt_d01, lat_d01, lon_d01, "d01", "domains_topo_sites_d01.pdf")
    print("✓ Saved: domains_topo_sites_d01.pdf")

    # ---- Plot 2: Nested domains (9km parent + 3km nested) ----
    print("Generating nested domain plot (d02 + d03)...")
    hgt_d02, lat_d02, lon_d02 = load_wrf_data(
        "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_9km/wrfinput_d01"
    )
    hgt_d03, lat_d03, lon_d03 = load_wrf_data(
        "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km/wrfinput_d02"
    )

    domain_labels = [
        ("d02", lon_d02[0, 0], lat_d02[0, 0]),
        ("d03", lon_d03[0, 0], lat_d03[0, 0]),
    ]
    plot_nested_domains(
        hgt_d02,
        lat_d02,
        lon_d02,
        hgt_d03,
        lat_d03,
        lon_d03,
        domain_labels,
        "domains_topo_sites.pdf",
    )
    print("✓ Saved: domains_topo_sites.pdf")
    print(f"\nAll plots saved to: {OUTFOLDER}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        raise
