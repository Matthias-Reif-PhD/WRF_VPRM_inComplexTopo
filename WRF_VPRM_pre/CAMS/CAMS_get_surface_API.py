import cdsapi
import argparse
import os
from dotenv import load_dotenv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SCRATCH_PATH = os.getenv("SCRATCH_PATH")


def get_surface_data(start_date, end_date, output_folder):
    dataset = "cams-global-ghg-reanalysis-egg4"
    request = {
        "variable": [
            "2m_dewpoint_temperature",
            "2m_temperature",
            "accumulated_carbon_dioxide_ecosystem_respiration",
            "accumulated_carbon_dioxide_gross_primary_production",
            "accumulated_carbon_dioxide_net_ecosystem_exchange",
            "anthropogenic_emissions_of_carbon_dioxide",
            "co2_column_mean_molar_fraction",
            "flux_of_carbon_dioxide_ecosystem_respiration",
            "flux_of_carbon_dioxide_gross_primary_production",
            "flux_of_carbon_dioxide_net_ecosystem_exchange",
            "gpp_coefficient_from_biogenic_flux_adjustment_system",
            "rec_coefficient_from_biogenic_flux_adjustment_system",
            "land_sea_mask",
            "surface_solar_radiation_downwards",
            "photosynthetically_active_radiation_at_the_surface",
        ],
        "date": [f"{start_date}/{end_date}"],
        "step": ["0", "3", "6", "9", "12", "15", "18", "21"],
        "data_format": "netcdf",
        "area": [53, 3, 41, 19],
    }

    target = f"{output_folder}/ghg-reanalysis_surface_{start_date}_{end_date}.nc"

    client = cdsapi.Client()

    response = client.retrieve(dataset, request, target)
    print(response)


def main():
    start_date = "2012-01-01"
    end_date = "2012-12-31"
    output_folder = os.path.join(SCRATCH_PATH, "DATA/CAMS")
    # for CAMS you have to ads url in ~/.cdsapirc
    # file_path="$HOME/.cdsapirc"
    # sed -i '/^url:/c\url: https://ads.atmosphere.copernicus.eu/api' "$file_path"

    parser = argparse.ArgumentParser(description="Retrieve CAMS atmospheric CO2 data.")
    parser.add_argument(
        "-s",
        "--start_date",
        type=str,
        default=start_date,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "-e",
        "--end_date",
        type=str,
        default=end_date,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default=output_folder,
        help="Folder to save the output data",
    )
    args = parser.parse_args()

    get_surface_data(args.start_date, args.end_date, args.output_folder)


if __name__ == "__main__":
    main()
