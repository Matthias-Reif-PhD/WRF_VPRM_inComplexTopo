import subprocess
import numpy as np
import rasterio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== Configuration ====================
SCRATCH_PATH = os.getenv("SCRATCH_PATH")
GITHUB_PATH = os.getenv("GITHUB_PATH")

# Paths
input_tif = os.path.join(
    SCRATCH_PATH,
    "DATA/CORINE_LC/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif",
)
reprojected_tif = os.path.join(
    SCRATCH_PATH, "DATA/CORINE_LC/reprojected_for_VPRM_U2018_CLC2018_V2020_20u1.tif"
)
converted_tif = os.path.join(
    SCRATCH_PATH, "DATA/CORINE_LC/converted_for_VPRM_U2018_CLC2018_V2020_20u1.tif"
)
modified_tif = os.path.join(
    SCRATCH_PATH,
    "DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/data/copernicus/modified_for_VPRM_U2018_CLC2018_V2020_20u1.tif",
)

# Step 1: Use gdalwarp and gdal_translate for reprojection and conversion
subprocess.run(
    [
        "gdalwarp",
        "-co",
        "COMPRESS=DEFLATE",
        "-srcnodata",
        "EPSG:3035",
        "-t_srs",
        "EPSG:4326",
        input_tif,
        reprojected_tif,
    ],
    check=True,
)

subprocess.run(
    [
        "gdal_translate",
        "-of",
        "GTiff",
        "-b",
        "1",
        "-co",
        "COMPRESS=DEFLATE",
        reprojected_tif,
        converted_tif,
    ],
    check=True,
)

# Step 2: Read the converted TIFF with rasterio
with rasterio.open(converted_tif) as src:
    band_1_data = src.read(1)  # Read first band

    # Replace -128 and 127 with 255
    band_1_data = np.where(np.isin(band_1_data, [-128, 127]), 255, band_1_data)

    # Define the metadata for the new file (copy from the original)
    meta = src.meta
    meta.update(
        dtype=rasterio.uint8, count=1
    )  # Update data type if necessary (e.g., uint8 for 0-255 range)

# Step 3: Write the modified data to a new file
with rasterio.open(modified_tif, "w", **meta) as dst:
    dst.write(band_1_data, 1)

print(f"Modified raster saved to {modified_tif}")
