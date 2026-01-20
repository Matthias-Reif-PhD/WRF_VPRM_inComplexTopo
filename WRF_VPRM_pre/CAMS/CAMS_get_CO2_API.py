import cdsapi
import argparse
import os
from dotenv import load_dotenv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SCRATCH_PATH = os.getenv("SCRATCH_PATH")


def get_CO2_data(start_date, end_date, output_folder):
    dataset = "cams-global-ghg-reanalysis-egg4"
    request = {
        "variable": ["carbon_dioxide"],
        "model_level": [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33",
            "34",
            "35",
            "36",
            "37",
            "38",
            "39",
            "40",
            "41",
            "42",
            "43",
            "44",
            "45",
            "46",
            "47",
            "48",
            "49",
            "50",
            "51",
            "52",
            "53",
            "54",
            "55",
            "56",
            "57",
            "58",
            "59",
            "60",
        ],
        "date": [f"{start_date}/{end_date}"],
        "step": ["0", "3", "6", "9", "12", "15", "18", "21"],
        "data_format": "netcdf_zip",
        "area": [53, 3, 41, 19],
    }

    target = f"{output_folder}/ghg-reanalysis_CO2_{start_date}_{end_date}.zip"

    client = cdsapi.Client()
    response = client.retrieve(dataset, request, target)
    print(response)


def main():
    start_date = "2012-07-01"
    end_date = "2012-08-01"
    output_folder = os.path.join(SCRATCH_PATH, "DATA/CAMS")
    # for CAMS you have to use ads url in ~/.cdsapirc
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

    get_CO2_data(args.start_date, args.end_date, args.output_folder)


if __name__ == "__main__":
    main()
