import argparse
import os
import numpy as np
import pandas as pd

def _normalize_dataset(df: pd.DataFrame, category: str) -> pd.DataFrame:
    avg_co2 = df['co2e_100yr_tonnes'].mean()
    max_co2 = df['co2e_100yr_tonnes'].max()
    min_co2 = df['co2e_100yr_tonnes'].min()
    value = (
        #(df["co2e_100yr_tonnes"] >= avg_co2).astype(float)   # 0 for below average, 1 for above average
        (df["co2e_100yr_tonnes"] - min_co2) / (max_co2 - min_co2) # linear distribution between min and max co2
        #0.5 - 0.5 * np.exp(-df["co2e_100yr_tonnes"] / max_co2) # experimental exponential distribution
    ).clip(0, 1).round(5)
    return pd.DataFrame({
        "aggrId": None,
        "value": value,
        "category": category,
        "lat": df["lat"],
        "lng": df["lon"],
    })

def perform(input_path: str, output_path: str, verbose: bool = True):
  if verbose: print(f"0) Loading data from {input_path}")
  dataset = pd.read_csv(input_path)
  
  if verbose: print("1) Transforming to DataFrame")
  df_pain = _normalize_dataset(dataset, category="CO2")

  if verbose: print(f"3) Saving data to {output_path}")
  df_pain.to_csv(output_path, index=True, index_label="id")
  
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
  parser.add_argument("-q", "--quiet", action='store_true', help="Whether to be quiet or print messages informing about completed steps")

  args = parser.parse_args()
  print(args)

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
      os.path.join("data", "raw", "env", "co2", "climate_trace_compiled.csv")
    output_path = args.output if args.output else \
      os.path.join("data", "actual", "env", "co2", "co2_final.csv")

  perform(input_path, output_path, verbose=not args.quiet)
