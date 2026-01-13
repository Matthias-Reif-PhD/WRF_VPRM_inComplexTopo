import cdsapi
import argparse

# use the correct url in ~/.cdsapirc
# url: https://ads.atmosphere.copernicus.eu/api


def get_lnsp_data(start_date, end_date, output_folder):
    dataset = "cams-global-ghg-reanalysis-egg4"
    request = {
        "variable": ["logarithm_of_surface_pressure"],
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

    target = f"{output_folder}/ghg-reanalysis_lnsp_{start_date}_{end_date}.zip"

    client = cdsapi.Client()
    response = client.retrieve(dataset, request, target)
    print(response)


def main():
    start_date = "2012-02-01"
    end_date = "2012-03-01"
    output_folder = "/scratch/c7071034/DATA/CAMS"
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

    get_lnsp_data(args.start_date, args.end_date, args.output_folder)


if __name__ == "__main__":
    main()
