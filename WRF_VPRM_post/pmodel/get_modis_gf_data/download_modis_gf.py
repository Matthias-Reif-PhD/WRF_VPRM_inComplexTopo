import os
import requests
import xarray as xr
from datetime import datetime

# Define the base URL for the dataset
base_url = "https://globalland.vito.be/download/netcdf/fraction_absorbed_par/fapar_1km_v2_10daily/2012/"

# Define the output directories
download_dir = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/gap_filled/fapar_2012"
extracted_dir = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/gap_filled/fapar_2012_europe"
os.makedirs(download_dir, exist_ok=True)
os.makedirs(extracted_dir, exist_ok=True)

# List of available dekad directories for 2012
dekad_dirs = [
    "20120110", "20120120", "20120131", "20120210", "20120220", "20120229",
    "20120310", "20120320", "20120331", "20120410", "20120420", "20120430",
    "20120510", "20120520", "20120531", "20120610", "20120620", "20120630",
    "20120710", "20120720", "20120731", "20120810", "20120820", "20120831",
    "20120910", "20120920", "20120930", "20121010", "20121020", "20121031",
    "20121110", "20121120", "20121130", "20121210", "20121220", "20121231"
]

# Europe's bounding box (latitude and longitude)
lat_min, lat_max = 41, 51
lon_min, lon_max = 3, 21

# Function to download a dekad folder
def download_dekad_folder(dekad_dir):
    dekad_url = base_url + dekad_dir + "/"
    response = requests.get(dekad_url)
    
    if response.status_code == 200:
        # Get the list of NetCDF files in the directory
        # If the directory contents are listed in HTML, you can extract the filenames.
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", href=True)
        
        for link in links:
            if link['href'].endswith('.nc'):
                # Download each NetCDF file
                file_url = dekad_url + link['href']
                file_path = os.path.join(download_dir, link['href'])
                download_file(file_url, file_path)
    else:
        print(f"Failed to retrieve the list of files from {dekad_url}")

# Function to download a specific NetCDF file
def download_file(file_url, file_path):
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return
    
    print(f"Downloading: {file_url}")
    response = requests.get(file_url)
    
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file_path}")
    else:
        print(f"Failed to download: {file_url} (Status code: {response.status_code})")

# Function to extract Europe from a NetCDF file
def extract_europe_from_file(input_path, output_path):
    # Open the dataset
    ds = xr.open_dataset(input_path)
    
    # Select the FAPAR variable (adjust the variable name if different)
    if 'FAPAR' in ds.variables:
        fapar = ds['FAPAR']
    else:
        print(f"FAPAR variable not found in {input_path}")
        return
    
    # Subset the data for Europe
    fapar_europe = fapar.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    
    # Save the subset to a new NetCDF file
    fapar_europe.to_netcdf(output_path)
    print(f"Saved European subset: {output_path}")

# Download all dekad folders for 2012
for dekad_dir in dekad_dirs:
    download_dekad_folder(dekad_dir)

# After downloading, extract the European region for each downloaded NetCDF file
for filename in os.listdir(download_dir):
    if filename.endswith(".nc"):
        input_path = os.path.join(download_dir, filename)
        output_path = os.path.join(extracted_dir, filename.replace(".nc", "_europe.nc"))
        extract_europe_from_file(input_path, output_path)
