from osgeo import gdal
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== Configuration ====================
SCRATCH_PATH = os.getenv("SCRATCH_PATH", "/mnt/ssd2/WRF-VPRM_zenodo")

# Open the dataset
dataset = gdal.Open("cropped_3km_U2018_CLC2018_V2020_20u1_with_255.tif")
band = dataset.GetRasterBand(1)

# Read the data as an array
array = band.ReadAsArray()

# Get unique values and statistics
unique_values = np.unique(array)
mean_value = np.mean(array)
min_value = np.min(array)
max_value = np.max(array)

print("Unique values:", unique_values)
print("Mean value:", mean_value)
print("Min value:", min_value)
print("Max value:", max_value)
