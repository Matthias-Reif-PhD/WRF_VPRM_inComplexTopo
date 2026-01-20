import warnings
import os
import glob
import xarray as xr
import numpy as np
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== Configuration ====================
SCRATCH_PATH = os.getenv("SCRATCH_PATH", "/mnt/ssd2/WRF-VPRM_zenodo")

warnings.filterwarnings("ignore")

# Base directory path
base = os.path.join(
    SCRATCH_PATH, "DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/out_d01_2012_9km/"
)

# Iterate over unique base file prefixes
for fbase in np.unique(
    [i.split("_part")[0] for i in glob.glob(os.path.join(base, "VPRM_input*part_*"))]
):
    print(f"Processing base file: {fbase}")

    # Skip if the combined file already exists
    # if os.path.exists(fbase + ".nc"):
    #    continue

    lon_stripes = []

    for n in np.arange(1, 10, 1):
        file_list = sorted(glob.glob(fbase + "*_{}.nc".format(n)))

        # Debug: Print file list
        print(f"Files found for n={n}: {file_list}")

        if not file_list:
            print(f"No files found for pattern: {fbase}*_.nc".format(n))
            continue  # Skip this iteration if no files found

        # Extract and sort parts numerically
        spslits = [fname.split("_part_")[1] for fname in file_list]
        base_name = [fname.split("_part_")[0] for fname in file_list][
            0
        ]  # Safe if file_list is non-empty
        sorted_parts = sorted(spslits, key=lambda x: int(x.split("_")[0]))
        sorted_files = [base_name + "_part_" + part for part in sorted_parts]

        try:
            lon_stripes.append(
                xr.concat(
                    [
                        xr.open_dataset(f).drop_dims(["x_b", "y_b"])
                        for f in sorted_files
                    ],
                    dim="west_east",
                    compat="no_conflicts",
                )
            )
        except Exception as e:
            print(f"Error concatenating parts: {e}")
            lon_stripes.append(
                xr.concat(
                    [xr.open_dataset(f) for f in sorted_files],
                    dim="west_east",
                    compat="no_conflicts",
                )
            )

        # Move processed files to 'splits' subdirectory
        split_dir = os.path.join(base, "splits")
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        for file in sorted_files:
            shutil.move(file, os.path.join(split_dir, os.path.basename(file)))

    if lon_stripes:
        # Concatenate all longitude stripes across the 'south_north' dimension
        full_dataset = xr.concat(lon_stripes, dim="south_north")
        print(f"Saving {fbase}.nc")

        # Save the final merged NetCDF file
        full_dataset.to_netcdf(fbase + ".nc")
    else:
        print(f"No longitude stripes found for {fbase}. Skipping output.")

# mv regridder_* splits/.
# mv veg_map_on_modis_grid_* splits/.
# shutil.move(base+"regridder*", "splits/.")
# shutil.move(base+"veg_map*", "splits/.")
