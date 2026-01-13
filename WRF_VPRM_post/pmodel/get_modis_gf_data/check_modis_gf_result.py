import xarray as xr
import matplotlib.pyplot as plt

# Example file to visualize
file_path = "fapar_2012_europe/FAPAR_1km_20120101_global_v2.0.1_europe.nc"

# Open the dataset
ds = xr.open_dataset(file_path)

# Plot the FAPAR data
ds['FAPAR'].plot(cmap='YlGn')
plt.title("FAPAR - Europe - 2012-01-01")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
