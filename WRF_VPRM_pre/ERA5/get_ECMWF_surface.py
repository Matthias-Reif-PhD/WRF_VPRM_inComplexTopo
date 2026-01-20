import cdsapi
import argparse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== Configuration ====================
SCRATCH_PATH = os.getenv("SCRATCH_PATH", "/mnt/ssd2/WRF-VPRM_zenodo")

parser = argparse.ArgumentParser(
    description="Download ERA5 pressure level data for a given date range."
)
parser.add_argument("--start", type=int, required=True, help="Start day")
parser.add_argument("--end", type=int, required=True, help="End day")
parser.add_argument("--month", type=int, required=True, help="Month")
parser.add_argument("--year", type=int, required=True, help="Year")
args = parser.parse_args()

start = args.start
end = args.end
month = args.month
year = args.year

for day in range(start, end + 1):

    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "land_sea_mask",
            "mean_sea_level_pressure",
            "sea_ice_cover",
            "sea_surface_temperature",
            "skin_temperature",
            "snow_density",
            "snow_depth",
            "soil_temperature_level_1",
            "soil_temperature_level_2",
            "soil_temperature_level_3",
            "soil_temperature_level_4",
            "surface_pressure",
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3",
            "volumetric_soil_water_layer_4",
        ],
        "year": [f"{year}"],
        "month": [f"{month}"],
        "day": [
            f"{day:02d}",
        ],
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [53, 3, 41, 19],
    }

    client = cdsapi.Client()
    # Construct a dynamic file name based on the day
    file_name = os.path.join(
        SCRATCH_PATH,
        f"DATA/ECMWF/surface/era5_surface_{year}_{month:02d}_{day:02d}.grib",
    )

    # Download and save the file with the dynamic name
    client.retrieve(dataset, request).download(file_name)
