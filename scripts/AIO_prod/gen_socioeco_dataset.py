import argparse
from typing import Dict
import os
import numpy as np
import pandas as pd

# Constants
DEFAULT_BASE_PATH = os.path.join("data")
DEFAULT_INPUT_PATH = os.path.join(DEFAULT_BASE_PATH, 'raw', 'socioeco', 'GDP.csv')
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_BASE_PATH, 'dummy', 'data_types', 'socioeco.csv')

def _compute_gdp_pain(df_gdp: pd.DataFrame) -> pd.Series:
  # (logarithmic relative to the max value)
  log_max_gdp = np.log(df_gdp["gdp"].max())
  return 1 - np.log(df_gdp["gdp"].astype(float)) / log_max_gdp # pain = 1 - log(gdp) / log(maxGdp)

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
  """
  GDP Info:
  - Dataset Source = https://datacatalog.worldbank.org/search/dataset/0038130/gdp-ranking
    - Downloadlink = https://datacatalogfiles.worldbank.org/ddh-published/0038130/DR0046440/GDP.csv
      - values are given in Million $
  """
  parser = argparse.ArgumentParser(
    prog='Socio-economic Pain Dataset Generation Script',
    description='Generates the socio-economic pain dataset from GDP data.'
  )
  parser.add_argument("-bp", "--base-path", type=str, help="common base path shared among input and output file")
  parser.add_argument("-i", "-in", "--input", type=str, help="path to the input file")
  parser.add_argument("-o", "-out", "--output", type=str, help="path to the output file")

  args = parser.parse_args()
  # parse paths
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
    if args.word_data:
      words_path = os.path.join(args.base_path, args.word_data)
    else:
      print("Specifying a base_path requires a word-data argument but none was provided!")
      exit(1)
  else:
    input_path = args.input if args.input else DEFAULT_INPUT_PATH
    output_path = args.output if args.output else DEFAULT_OUTPUT_PATH

  perform({ "GDP": input_path }, output_path)
