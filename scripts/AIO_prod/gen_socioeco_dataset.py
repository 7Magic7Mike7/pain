from typing import Dict
import os
import numpy as np
import pandas as pd

def _compute_gdp_pain(df_gdp: pd.DataFrame) -> pd.Series:
  # (logarithmic relative to the max value)
  log_max_gdp = np.log(df_gdp["gdp"].max())
  return (log_max_gdp - np.log(df_gdp["gdp"].astype(float))) / log_max_gdp # pain = 1 - log(gdp) / log(maxGdp)

def _generate_gdp_dataset(input_path: str) -> pd.DataFrame:
  # 1) read dataset
  df_gdp = pd.read_csv(input_path, skiprows=4)     # first four rows are just metadata

  # 2) convert comma-separated gdp strings to integers
  df_gdp["gdp"] = df_gdp["US dollars"] \
      .str.replace(",", "", regex=False) \
      .apply(pd.to_numeric, errors="coerce") \
      .astype("Int64")

  # 3) drop non-country rows (i.e., unranked entries)
  df_gdp = df_gdp.dropna(subset=["rank", "gdp"])

  # 4) introduce our custom columns
  df_gdp["id"] = np.arange(1, len(df_gdp) + 1)
  df_gdp["aggrid"] = pd.NA
  df_gdp["category"] = "GDP"
  df_gdp["country"] = df_gdp["SOVA3"]     # rename column
  
  # 4) compute pain value
  df_gdp["value"] = _compute_gdp_pain(df_gdp)

  # 5) filter for our required columns
  return df_gdp[
      ["id", "aggrid", "value", "category", "country"]
  ]

def perform(input_paths: Dict[str, str], output_path: str):
  assert "GDP" in input_paths, "GDP dataset is missing!"

  print("Starting...")
  # 1) generate gdp dataset
  df_gdp = _generate_gdp_dataset(input_paths["GDP"])

  # -) generate other socioeco pain dataset (not implemented)

  # -) combine different socioeco pain to one hollistic socioeco pain dataset
  df_socioeco = df_gdp

  # 6) save transformed dataset
  df_socioeco.to_csv(output_path, index=False)
  print("-done-")

if __name__ == "__main__":
  print("TODO: implement args?")

  BASE_PATH = os.path.join("data")#'..', '..', 'data')
  DATASET_PATH = os.path.join(BASE_PATH, 'raw', 'socioeco', 'GDP.csv')
  OUTPUT_PATH = os.path.join(BASE_PATH, 'dummy', 'data_types', 'socioeco.csv')
  perform({ "GDP": DATASET_PATH }, OUTPUT_PATH)
