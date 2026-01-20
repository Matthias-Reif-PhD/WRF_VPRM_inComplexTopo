import numpy as np
import rasterio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== Configuration ====================
SCRATCH_PATH = os.getenv("SCRATCH_PATH", "/mnt/ssd2/WRF-VPRM_zenodo")

file = "E000N60_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
# file = "modified_for_VPRM_U2018_CLC2018_V2020_20u1.tif"

with rasterio.open(
    os.path.join(
        SCRATCH_PATH, "DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/data/copernicus/"
    )
    + file
) as src:
    band_1_data = src.read(1)  # Read first band
    print(np.unique(band_1_data))  # Check unique values (will help identify NaNs)
