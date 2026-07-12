# expect OG dataset to be downloaded already into DATA_PATH
from typing import Dict, List
import os
import numpy as np
import pandas as pd
import xarray as xr

BASE_PATH = os.path.join("..", "..", "data")
# where to find the raw dataset
DATA_PATH = os.path.join(BASE_PATH, "raw", "Global_TAVG_Gridded_5deg.nc")
# where to save the transformed dataset to
OUT_PATH = os.path.join(BASE_PATH, "actual", "temperature", "temperature_final.csv")
# which year(s) we want to visualize
YEAR_START = 2020
YEAR_END = 2025

print(f"0) Loading data from {DATA_PATH}")
# 1) transform to DataFrame
print("1) Transforming to DataFrame")
ds = xr.open_dataset(DATA_PATH, engine="netcdf4")
years = np.floor(ds.time.values).astype(int)
annual = (
    ds["temperature"]
    .assign_coords(year=("time", years))
    .groupby("year")
    .mean()
)
# select a specific year
ds_year = annual.sel(
    year=slice(str(YEAR_START), str(YEAR_END))
).mean("year", skipna=True)
df = ds_year.to_dataframe().reset_index()
df_filtered = df[df['temperature'].notnull()].reset_index(drop=True)

# 2) normalize values
print("2) Normalizing Values")
def normalize_temperature_dataset(dataframe: pd.DataFrame, category: str = "Temperature") -> pd.DataFrame:
    max_temp, min_temp = dataframe['temperature'].max(), dataframe['temperature'].min()
    temp_range = max_temp - min_temp
    temp_offset = min_temp
    data: List[Dict] = []
    for _, row in dataframe.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        temp = row['temperature']
        value = (temp - temp_offset) / temp_range
        
        data.append({
            'aggrId': None,
            'value': np.round(value, 5),
            'category': category,
            'lat': lat,
            'lng': lon
        })

    return pd.DataFrame(data)
df_temp = normalize_temperature_dataset(df_filtered)
print(f"3) Saving data to {OUT_PATH}")
df_temp.to_csv(OUT_PATH, index=True, index_label="id")
print("-done-")
