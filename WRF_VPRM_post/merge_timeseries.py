import pandas as pd
import glob

# Path where your CSV files are stored
csv_path = "/scratch/c7071034/DATA/WRFOUT/csv/"

# Get all CSV filenames matching the pattern
csv_files = glob.glob(f"{csv_path}wrf_FLUXNET_sites_3km_*.csv")

# Read and merge all CSV files
df_list = []
for file in csv_files:
    try:
        # Read CSV with the first column named as 'TIMESTAMP' explicitly
        df = pd.read_csv(file, header=None)
        df.columns = ["TIMESTAMP"] + list(
            df.columns[1:]
        )  # Set first column as 'TIMESTAMP'
        df_list.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue  # Skip this file and move to the next one

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(df_list, ignore_index=True)

# Convert the 'TIMESTAMP' column to datetime format for proper sorting
merged_df["TIMESTAMP"] = pd.to_datetime(merged_df["TIMESTAMP"], errors="coerce")

# Drop rows where 'TIMESTAMP' conversion failed (NaT values)
merged_df = merged_df.dropna(subset=["TIMESTAMP"])

# Sort the DataFrame by 'TIMESTAMP'
merged_df = merged_df.sort_values(by="TIMESTAMP")

# Drop duplicate rows based on 'TIMESTAMP', keeping the first occurrence
merged_df = merged_df.drop_duplicates(subset=["TIMESTAMP"], keep="first")

# Reset index after sorting and removing duplicates
merged_df.reset_index(drop=True, inplace=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(csv_path + "merged_wrf_FLUXNET_sites_3km.csv", index=False)

# Optionally, print the first few rows to check
print(merged_df.head())
