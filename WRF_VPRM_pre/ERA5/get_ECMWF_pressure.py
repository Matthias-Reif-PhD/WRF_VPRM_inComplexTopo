import cdsapi
import argparse

"""
Example usage:
    python get_ECMWF_pressure.py --start 1 --end 3 --month 6 --year 2012

This will download ERA5 pressure level data
"""

parser = argparse.ArgumentParser(description="Download ERA5 pressure level data for a given date range.")
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

    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "divergence",
            "fraction_of_cloud_cover",
            "geopotential",
            "ozone_mass_mixing_ratio",
            "potential_vorticity",
            "relative_humidity",
            "specific_cloud_ice_water_content",
            "specific_cloud_liquid_water_content",
            "specific_humidity",
            "specific_rain_water_content",
            "specific_snow_water_content",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "vorticity",
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
        "pressure_level": [
            "10",
            "20",
            "30",
            "50",
            "70",
            "100",
            "125",
            "150",
            "175",
            "200",
            "225",
            "250",
            "300",
            "350",
            "400",
            "450",
            "500",
            "550",
            "600",
            "650",
            "700",
            "750",
            "775",
            "800",
            "825",
            "850",
            "875",
            "900",
            "925",
            "950",
            "975",
            "1000",
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [53, 3, 41, 19],
    }

    client = cdsapi.Client()
    # Construct a dynamic file name based on the day
    file_name = f"/scratch/c7071034/DATA/ECMWF/pressure/era5_data_{year}_{month:02d}_{day:02d}.grib"

    # Download and save the file with the dynamic name
    client.retrieve(dataset, request).download(file_name)
