# expect OG dataset to be downloaded already into DATA_PATH
from typing import Dict, List
import argparse
import os
import numpy as np
import pandas as pd
import xarray as xr

MIN_YEAR = 1900
MAX_YEAR = 2025
MIN_TEMPERATURE = 1.0   # decided to be our minimal temperature (i.e., temperatures <= MIN_TEMPERATURE will have a pain value of 0)
MAX_TEMPERATURE = 5.0   # decided to be our maximum temperature (i.e., temperatures >= MAX_TEMPERATURE will have a pain value of 1)


def _normalize_temperature_dataset(dataframe: pd.DataFrame, category: str = "Temperature") -> pd.DataFrame:
  # max_temp, min_temp = dataframe['temperature'].max(), dataframe['temperature'].min()
  temp_range = MAX_TEMPERATURE - MIN_TEMPERATURE
  temp_offset = MIN_TEMPERATURE
  # clip 0: every temperature below MIN_TEMPERATURE will get a value of 0 (the result without clip would be < 0)
  # clip 1: every temperature above MAX_TEMPERATURE will get a value of 1 (the result without clip would be > 1)
  value = (
    (dataframe["temperature"] - temp_offset) / temp_range
  ).clip(0, 1).round(5)
  return pd.DataFrame({
      "aggrId": None,
      "value": value,
      "category": category,
      "lat": dataframe["latitude"],
      "lng": dataframe["longitude"],
  })

def generate_temperature_dataset(input_path: str, output_path: str, start_year: int, end_year: int, verbose: bool = True):
  assert MIN_YEAR <= start_year <= MAX_YEAR
  assert MIN_YEAR <= end_year <= MAX_YEAR
  assert start_year <= end_year

  if verbose: print(f"0) Loading data from {input_path}")

  # 1) transform to DataFrame
  if verbose: print("1) Transforming to DataFrame")
  ds = xr.open_dataset(input_path, engine="netcdf4")
  years = np.floor(ds.time.values).astype(int)
  annual = (
      ds["temperature"]
      .assign_coords(year=("time", years))
      .groupby("year")
      .mean()
  )
  # select a specific year
  ds_year = annual.sel(
      year=slice(str(start_year), str(end_year))
  ).mean("year", skipna=True)
  df = ds_year.to_dataframe().reset_index()
  df_filtered = df[df['temperature'].notnull()].reset_index(drop=True)

  # 2) normalize values
  if verbose: print("2) Normalizing Values")
  df_temp = _normalize_temperature_dataset(df_filtered)
  if verbose: print(f"3) Saving data to {output_path}")
  df_temp.to_csv(output_path, index=True, index_label="id")
  if verbose: print("-done-")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog='Script for generating the Temperature Pain Dataset',
    description='Transforms the downloaded, raw temperature dataset into a Temperature Pain Dataset usable by the PPP-Map. ' \
    'The script was built for the "Global_TAVG_Gridded_0p25deg_2020s.nc" dataset from "https://berkeleyearth.org/high-resolution-data-access-page/" ' \
    'but should work for other datasets on the same site as well.',
    epilog=''
  )
  parser.add_argument("-bp", "--base-path", type=str, help="Common base path shared among input and output file")
  parser.add_argument("-i", "-in", "--input", type=str, help="Path to the input file")
  parser.add_argument("-o", "-out", "--output", type=str, help="Path to the output file")
  parser.add_argument("-sy", "--start-year", type=int, help="Start year to consider for computing the mean temperature")
  parser.add_argument("-ey", "--end-year", type=int, help="End year to consider for computing the mean temperature")
  parser.add_argument("-q", "--quiet", action='store_true', help="Whether to be quiet or print messages informing about completed steps")

  args = parser.parse_args()
  start_year = args.start_year if args.start_year else \
    2020
  end_year = args.end_year if args.end_year else \
    MAX_YEAR
  if start_year < MIN_YEAR or MAX_YEAR < start_year:
    print(f"Invalid start_year! {MIN_YEAR} <= {start_year} <= {MAX_YEAR} must be True")
    exit(1)
  if end_year < MIN_YEAR or MAX_YEAR < end_year:
    print(f"Invalid end_year! {MIN_YEAR} <= {end_year} <= {MAX_YEAR} must be True")
    exit(1)
  if end_year < start_year:
    print(f"Invalid year range! {start_year}(start_year) <= {end_year}(end_year) must be True")
    exit(1)

  if args.base_path:
    if args.input:
      input_path = os.path.join(args.base_path, args.input)
    else:
      print("Specifying a base_path requires an input argument but none was provided!")
      exit(1)
    if args.output:
      output_path = os.path.join(args.base_path, args.output)
    else:
      print("Specifying a base_path requires an output argument but none was provided!")
      exit(1)
  else:
    input_path = args.input if args.input else \
      os.path.join("..", "..", "data", "raw", "Global_TAVG_Gridded_0p25deg_2020s.nc") # "Global_TAVG_Gridded_5deg.nc")
    output_path = args.output if args.output else \
      os.path.join("..", "..", "data", "actual", "temperature", "temperature_final.csv")

  generate_temperature_dataset(input_path, output_path, start_year, end_year, verbose=not args.quiet)
