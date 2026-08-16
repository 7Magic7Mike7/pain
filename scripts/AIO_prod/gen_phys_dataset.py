# Imports
from typing import List, Optional
from pathlib import Path
import argparse
import numpy as np
import os
import pandas as pd
import subprocess
import sys
import time

# Constants
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent # points to the parent of pain/ (parent of scripts/)
SCRIPT_BASE_PATH = Path(__file__).resolve().parent  # points to AIO_prod/
PREP_SCRIPT_PATH = os.path.join(SCRIPT_BASE_PATH, "prep_physical_dataset.py")
AGGR_SCRIPT_PATH = os.path.join(SCRIPT_BASE_PATH, "aggregate_streaming.py")
CATEGORIES = ['Headache disorders', 'Low back pain', 'Migraine', 'Neck pain', 'Osteoarthritis', 'Rheumatoid arthritis']  # extracted beforehand
## Defaults
DEFAULT_BASE_PATH = os.path.join("data", "raw", "physical")
DEFAULT_INPUT_PATH = os.path.join(DEFAULT_BASE_PATH, "prevalence_by_pixel.csv.gz")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_BASE_PATH, "phys_aio.csv.gz")
# Globals
TMP_FILES: List[str] = []

def _compute_pain(category_values: pd.Series, use_log: Optional[bool] = None) -> pd.Series:
    if use_log is None:
       use_log = False

    max_pain = category_values.max()
    min_pain = category_values.min()
    assert min_pain >= 0

    if use_log:
      log_max = np.log(max_pain)
      log_min = np.log(min_pain)
      pain_range = log_max - log_min
      pain_offset = log_min
      #assert pain_range > 0, f"Invalid pain_range! log_max = {log_max}, log_min = {log_min}, pain_range = {pain_range}"
    else:
       pain_range = max_pain - min_pain
       pain_offset = min_pain

    return (
        (np.log(category_values) if use_log else category_values - pain_offset) / pain_range
    ).clip(0, 1).round(5)

def _normalize_dataset(df: pd.DataFrame, categories: Optional[List[str]] = None, use_log: Optional[bool] = None) -> pd.DataFrame:
    if categories is None:
        categories = sorted(df["category"].dropna().astype(str).unique())

    required_columns = {"value", "category", "lat", "lng"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    work_df = df.loc[:, ["value", "category", "lat", "lng"]].copy()
    work_df = work_df.dropna(subset=["value", "category", "lat", "lng"])
    work_df["category"] = work_df["category"].astype(str)

    normalized_frames = []
    for category in categories:
        category_df = work_df.loc[work_df["category"] == str(category)].copy()
        if category_df.empty:
            continue

        max_pain = category_df["value"].max()
        min_pain = min(category_df["value"].min(), 0)
        pain_range = max_pain - min_pain
        if pain_range == 0:
            normalized_values = pd.Series(0.0, index=category_df.index)
        else:
            normalized_values = _compute_pain(category_df["value"], use_log)

        normalized_frames.append(pd.DataFrame({
            "aggrId": pd.NA,
            "value": normalized_values,
            "category": category_df["category"],
            "lat": category_df["lat"],
            "lng": category_df["lng"],
        }))

    if not normalized_frames:
        empty_df = pd.DataFrame({
            "aggrId": pd.NA, # pd.Series(dtype="Float64"),
            "value": pd.Series(dtype="float64"),
            "category": pd.Series(dtype="object"),
            "lat": pd.Series(dtype="float64"),
            "lng": pd.Series(dtype="float64"),
        })
        empty_df.index = pd.RangeIndex(start=1, stop=1)
        return empty_df

    result = pd.concat(normalized_frames, ignore_index=True)
    result["id"] = np.arange(1, len(result) + 1)
    return result

def _create_intermediate_file(prefix: str) -> str:
   return os.path.join(temp_dir, f"{prefix}_{time.time_ns()}.csv")

def perform(input_path: str, output_path: str, temp_dir: str, use_log: Optional[bool] = None,
            skip_prep: Optional[bool] = None, skip_downscale: Optional[bool] = None, skip_pain_computing: Optional[bool] = None):
  CHUNK_SIZE = 200_000
  TMP_FILES = []

  ####################################################
  # PREPARING (metric filter)
  ####################################################
  if skip_prep:
    print("# Skipping preparation step!")
    intermediate_file = input_path
  else:
    if not os.path.exists(temp_dir):
      os.makedirs(temp_dir)
    intermediate_file = _create_intermediate_file("prep")

    print("# Executing preparation script...")
    METRIC_NAME = "Percent"
    t_start = time.time()
    subprocess.run(
        [sys.executable, PREP_SCRIPT_PATH, "--input", input_path, "--output", intermediate_file, "--chunk-size", str(CHUNK_SIZE), "--metric-name", METRIC_NAME],
        check=True,
        capture_output=True,
        text=True
    )
    print(f"Elapsed time for preparing = {time.time() - t_start}")
    TMP_FILES.append(intermediate_file)
  # intermediate file is now either a new file or input_path

  ####################################################
  # DOWNSCALING
  ####################################################
  if skip_downscale:
    print("# Skipping downscale step!")
  else:
    if not os.path.exists(temp_dir):
      os.makedirs(temp_dir)
    downscale_file = _create_intermediate_file("downscale") if skip_pain_computing else output_path

    LAT_DEGREES = 0.2
    LNG_DEGREES = 0.4
    COOR_FUNCTION = "wmid"
    AGGR_FUNCTION = "sum"
    COMPRESSION = "gzip"
    errmsg = "The following has to be manually executed due to import problems with custom modules (python config):\n"
    errmsg += f"python -m scripts.AIO_prod.aggregate_streaming -i {intermediate_file} -o {downscale_file} -lat {LAT_DEGREES} -lng {LNG_DEGREES} -fcoor {COOR_FUNCTION} -faggr {AGGR_FUNCTION} -c {COMPRESSION} -cs {CHUNK_SIZE} -nobd -nvv"
    raise Exception(errmsg)
    print("cwd:", os.getcwd())
    print("PROJECT_ROOT_PATH: ", PROJECT_ROOT_PATH)
    print("# Executing downscaling (throw-away aggregation) script...")
    t_start = time.time()
    subprocess.run(
        [
          sys.executable, AGGR_SCRIPT_PATH, "--input", intermediate_file, "--output", downscale_file, 
          "-lat", str(LAT_DEGREES), "-lng", str(LNG_DEGREES), 
          "-fcoor", COOR_FUNCTION, "-faggr", AGGR_FUNCTION,
          "--chunk-size", str(CHUNK_SIZE), "--compression", COMPRESSION,
          "--no-base-data",         # don't store the unaggregated data points since we want to downscale the data
          "--no-value-validation",  # don't validate values since we are not working with pain values
        ],
        cwd=PROJECT_ROOT_PATH,
        check=True
    )
    print(f"Elapsed time for downscaling = {time.time() - t_start}")
    intermediate_file = downscale_file
    TMP_FILES.append(intermediate_file)
  # intermediate file is now either a new file or whatever it was before

  ####################################################
  # COMPUTING PAIN
  ####################################################
  if skip_pain_computing:
    print("# Skipping pain computing step!")
    # move the result of a previous step to the output path if we don't compute the pain but did perform a previous step
    if (not skip_prep or not skip_downscale) and os.path.exists(intermediate_file):
      os.rename(intermediate_file, output_path)
      print(f"renamed \"{intermediate_file}\" to \"{output_path}\"")
  else:
    df_phys = pd.read_csv(intermediate_file, compression="gzip")
    df_phys = _normalize_dataset(df_phys, CATEGORIES, use_log)
    # reorder columns
    df_phys = df_phys[["id", "aggrId", "value", "category", "lat", "lng"]]
    df_phys.to_csv(output_path, index=False, compression="gzip")

  print("# -done-")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog='Physical Pain Dataset Generation Script',
    description='Generates the physical pain dataset by filtering and transforming a geospatial IHME dataset.'
  )
  parser.add_argument("-bp", "--base-path", type=str, help="common base path shared among input and output file")
  parser.add_argument("-i", "-in", "--input", type=str, help="path to the input file")
  parser.add_argument("-o", "-out", "--output", type=str, help="path to the output file")
  parser.add_argument("-ul", "--use-log", action='store_true', help="whether to use logarithmic pain computation (default = false)")
  parser.add_argument("-skp", "--skip-preparing", action='store_true', help="whether to skip executing the preparation step (default = false)")
  parser.add_argument("-skds", "--skip-downscaling", action='store_true', help="whether to skip executing the down scaling step (default = false)")
  parser.add_argument("-skpc", "--skip-pain-computing", action='store_true', help="whether to skip executing the pain value computing step (default = false)")
  parser.add_argument("-tmp", "--temp-dir", type=str, help="optional directory to store temporary files in (default = invocation directory)")
  parser.add_argument("-keep", "--keep-temp", action='store_true', help="optional flag to keep the temporary files (default = false)")

  args = parser.parse_args()
  if not args.input:
    print("No input file path provided!")
    exit(1)
  if not args.output:
    print("No output file path provided!")
    exit(1)

  temp_dir = args.temp_dir if args.temp_dir else "."

  if args.base_path:
    input_path = os.path.join(args.base_path, args.input)
    output_path = os.path.join(args.base_path, args.output)
    temp_dir = os.path.join(args.base_path, temp_dir)
  else:
    input_path = args.input
    output_path = args.output

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

  t_start = time.time()
  try:
    perform(input_path, output_path, temp_dir, use_log=args.use_log,
            skip_prep=args.skip_preparing, skip_downscale=args.skip_downscaling, skip_pain_computing=args.skip_pain_computing)
  finally:
    if not args.keep_temp:
      # delete all temporary files
      for file in TMP_FILES:
        if os.path.exists(file):
          os.remove(file)
  t_end = time.time()

  print(f"Elapsed time = {t_end - t_start}")
