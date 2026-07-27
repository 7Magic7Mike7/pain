# Imports
from typing import Dict, List, Optional, Set, Tuple
import argparse
import gzip
import os
import pandas as pd
import time

EXPECTED_CHUNK_COUNT = 21_708 * 100_000  # for chunk_size = 100_000, the first performance was done after 21_708 chunks
# expected time: 4702 s = 78,36 min = 1h18m21

def perform(input_path: str, output_path: str, chunk_size: int, metric: str = "Percent"):
  expected_chunks = int(EXPECTED_CHUNK_COUNT / chunk_size)
  # Create a stripped down dataset with only the needed columns and rows
  output_columns = ["lat", "lon", "cause_name", "pixel_abs_prevalence"]
  reader = pd.read_csv(
      input_path,
      chunksize=chunk_size,
      usecols=output_columns + ["metric_name"],
      dtype={
        "cause_name": "category",
      },
      compression="gzip",
  )
  causes: Set[str] = set()
  last_size = len(causes)
  last_progress = 0
  i = 0
  written_rows = 0
  first_write = True
  print(f"Start streaming with chunk size = {chunk_size}")
  with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
    while True:
      try:
        chunk = next(reader)
        filtered_chunk = chunk.loc[chunk["metric_name"] == metric, output_columns]
        if not filtered_chunk.empty:
          filtered_chunk.to_csv(handle, index=False, header=first_write)
          first_write = False
          written_rows += len(filtered_chunk)
        causes.update(chunk["cause_name"].cat.categories)
        cur_size = len(causes)
        if cur_size > last_size:
          print("causes = ", causes)
          last_size = cur_size
        i += 1
        progress = i / expected_chunks
        if progress > last_progress + 0.01:
          last_progress = progress
          print(f"predicted progress ~ {int(100 * progress)}%")
      except StopIteration:
        break
      except Exception as ex:
        print("#####################################################")
        print(f"Error occurred at chunk #{i}: ", ex)
        print("#####################################################")
  print(f"Wrote {written_rows} rows to {output_path}")
  print(f"DONE after {i} chunks")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog='Data Aggregation Script',
    description='Aggregates datapoints from an input file in the specified lat/lng resolution and stores them in an output file.',
    epilog=''
  )
  parser.add_argument("-bp", "--base-path", type=str, help="common base path shared among input and output file")
  parser.add_argument("-i", "-in", "--input", type=str, help="path to the input file")
  parser.add_argument("-o", "-out", "--output", type=str, help="path to the output file")
  parser.add_argument("-cs", "--chunk-size", type=int, help="size of the chunks to process while streaming the input file")

  args = parser.parse_args()
  if not args.input:
    print("No input file path provided!")
    exit(1)
  if not args.output:
    print("No output file path provided!")
    exit(1)

  if args.base_path:
    input_path = os.path.join(args.base_path, args.input)
    output_path = os.path.join(args.base_path, args.output)
  else:
    input_path = args.input
    output_path = args.output

  chunk_size = args.chunk_size if args.chunk_size else 100_000

  t_start = time.time()
  perform(input_path, output_path, chunk_size)
  t_end = time.time()
  print(f"Elapsed time = {t_end - t_start}")

