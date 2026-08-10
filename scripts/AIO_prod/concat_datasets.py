from typing import List, Optional
import argparse
import os
import pandas as pd

def concat(input_paths: List[str], output_path: str, base_path: Optional[str] = None):
  if base_path:
    input_paths = [os.path.join(base_path, input_path) for input_path in input_paths]

  # 1) load all datasets from input_paths and concat them into one pd.DataFrame
  #   Note that the datasets have two important columns: id & aggrId (referencing the id of a different row)
  #   Since the concatenation will mess with ids, we have to make sure to update aggrIds accordingly such that they still reference the same row
  frames = []
  next_id = 1
  for input_path in input_paths:
    df = pd.read_csv(input_path)
    if "id" not in df.columns:
      raise ValueError(f"Missing required 'id' column in {input_path}")

    if "aggrId" not in df.columns:
      #df["aggrId"] = pd.NA
      raise ValueError(f"Missing required 'aggrId' column in {input_path}")

    df = df.copy()
    old_ids = df["id"].tolist()
    new_ids = list(range(next_id, next_id + len(df)))
    id_map = {old_id: new_id for old_id, new_id in zip(old_ids, new_ids)}
    df["id"] = new_ids

    def _remap_aggr_id(value):
      if pd.isna(value):
        return pd.NA
      return id_map.get(value, value)
    df["aggrId"] = df["aggrId"].apply(_remap_aggr_id)

    frames.append(df)
    next_id += len(df)

  combined_df = pd.concat(frames, ignore_index=True)

  # 2) store pd.DataFrame as csv in output_path
  output_dir = os.path.dirname(output_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  combined_df.to_csv(output_path, index=False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog='Data Aggregation Script',
    description='Aggregates datapoints from an input file in the specified lat/lng resolution and stores them in an output file.',
    epilog=''
  )
  parser.add_argument("-bp", "--base-path", type=str, help="common base path shared among input and output file")
  parser.add_argument("-i", "-in", "--inputs", type=str, help="paths to the input files")
  parser.add_argument("-o", "-out", "--output", type=str, help="path to the output file")

  args = parser.parse_args()
  if not args.inputs:
    print("No input file paths provided!")
    exit(1)
  if not args.output:
    print("No output file path provided!")
    exit(1)

  input_paths = args.inputs.split(",")
  if len(input_paths) <= 1:
    print("Only one input path specified by at least two are needed to concatenate!")
    exit(1)

  print(input_paths)
  concat(input_paths, args.output, args.base_path)
