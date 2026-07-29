# Imports
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse
import numpy as np
import os
import pandas as pd
import time

# my scripts
from scripts.pain2map import AggregationManager, AggregatedPainData, PainData, Coordinate

# constants
COL_VAL = "value"
COL_CAT = "category"
COL_LAT = "lat"
COL_LNG = "lng"

# helper methods
def _compute_center(ix: int, iy: int, lng_degrees: float, lat_degrees: float) -> Coordinate:
  """
  Computes the center by first computing the bottom left corner of the area according to step sizes
  and indices in the grid and then adding half steps to both axis.
  """
  bot_left = Coordinate(ix * lng_degrees, iy * lat_degrees)
  return bot_left + Coordinate(lng_degrees * 0.5, lat_degrees * 0.5)

@dataclass
class AggregationSummary:
  count: int = 0
  sum_val: float = 0.0
  max_val: float = -1.0
  sum_lat: float = 0.0
  sum_lng: float = 0.0
  weighted_lat: float = 0.0
  weighted_lng: float = 0.0
  max_lat: float = 0.0
  max_lng: float = 0.0

  def add_point(self, lat: float, lng: float, value: float):
    if self.count == 0 or value > self.max_val:
      self.max_val = value
      self.max_lat = lat
      self.max_lng = lng
    self.count += 1
    self.sum_val += value
    self.sum_lat += lat
    self.sum_lng += lng
    self.weighted_lat += lat * value
    self.weighted_lng += lng * value

  def __str__(self):
    return f"AggrSum{{#{self.count}, max=({self.max_lat}|{self.max_lng}), weighted=({self.weighted_lat}|{self.weighted_lng})}}"


def _compute_aggregated_value(summary: AggregationSummary, aggr_func: callable) -> float:
  if aggr_func == AggregatedPainData.avg_aggregation:
    return summary.sum_val / summary.count
  if aggr_func == AggregatedPainData.max_aggregation:
    return summary.max_val
  if aggr_func == AggregatedPainData.sum_aggregation:
    return summary.sum_val
  raise Exception(f"Unsupported aggregation function: {aggr_func}")


def _compute_aggregated_coordinate(summary: AggregationSummary, coor_func: callable, center: Coordinate) -> Coordinate:
  if coor_func == AggregatedPainData.center_coordinate:
    return center
  if coor_func == AggregatedPainData.mid_point_coordinate:
    return Coordinate(summary.sum_lng / summary.count, summary.sum_lat / summary.count)
  if coor_func == AggregatedPainData.weighted_mid_point_coordinate:
    if summary.sum_val == 0:
      return Coordinate(summary.sum_lng / summary.count, summary.sum_lat / summary.count)
    return Coordinate(summary.weighted_lng / summary.sum_val, summary.weighted_lat / summary.sum_val)
  if coor_func == AggregatedPainData.max_point_coordinate:
    return Coordinate(summary.max_lng, summary.max_lat)
  raise Exception(f"Unsupported coordinate function: {coor_func}")


def _validate_chunk(chunk: pd.DataFrame, verbose: bool, validate_values: bool, column_names: Dict[str, str]) -> None:
  """
  :param validate_values: whether to check if values are between 0 and 1 (both inclusive)
  """
  if "aggrId" not in chunk.columns:
    chunk["aggrId"] = pd.NA

  if not chunk["aggrId"].isna().all():
    raise AssertionError("Can only aggregate datasets that are not yet aggregated!")

  chunk[column_names[COL_VAL]] = chunk[column_names[COL_VAL]].astype(float)
  values = chunk[column_names[COL_VAL]].to_numpy(copy=False)
  near_zero = (values > -1.0) & (values < 0.0)
  if near_zero.any():
    if verbose:
      print("Clamping values slightly below 0.0 to 0.0")
    values[near_zero] = 0.0

  if validate_values:
    invalid_values = (values < 0.0) | (values > 1.0)
    if invalid_values.any():
      invalid_index = int(chunk.index[invalid_values][0])
      invalid_row = chunk.iloc[invalid_index]
      raise AssertionError(f"Invalid value for id={int(invalid_row['id'])}: {invalid_row['value']}")

  chunk[column_names[COL_LAT]] = chunk[column_names[COL_LAT]].astype(float)
  chunk[column_names[COL_LNG]] = chunk[column_names[COL_LNG]].astype(float)
  invalid_lat = (chunk[column_names[COL_LAT]] < PainData.MIN_LAT()) | (chunk[column_names[COL_LAT]] >= PainData.MAX_LAT())
  invalid_lng = (chunk[column_names[COL_LNG]] < PainData.MIN_LNG()) | (chunk[column_names[COL_LNG]] >= PainData.MAX_LNG())
  invalid_coords = invalid_lat | invalid_lng
  if invalid_coords.any():
    invalid_index = int(chunk.index[invalid_coords][0])
    invalid_row = chunk.iloc[invalid_index]
    raise AssertionError(
      f"Invalid coordinate for id={int(invalid_row['id'])}: lat={invalid_row[column_names[COL_LAT]]}, lng={invalid_row[column_names[COL_LNG]]}"
    )


def perform(input_path: str, output_path: str, lat_degrees: float, lng_degrees: float, aggr_func_name: str, coor_func_name: str, chunk_size: int,
            include_base_data: bool = True, compression_in: str = "infer", compression_out: Optional[str] = None, column_names: Optional[Dict[str, str]] = None,
            validate_values: bool = True, verbose: bool = True):
  # prepare parameters
  if verbose:
    print("0) Preparing aggregation...")
  aggr_func = AggregationManager.get_aggr_func_from_name(aggr_func_name)
  coor_func = AggregationManager.get_coor_func_from_name(coor_func_name)

  if not column_names:
    column_names = {}
  if COL_VAL not in column_names:
    column_names[COL_VAL] = COL_VAL
  if COL_CAT not in column_names:
    column_names[COL_CAT] = COL_CAT
  if COL_LAT not in column_names:
    column_names[COL_LAT] = COL_LAT
  if COL_LNG not in column_names:
    column_names[COL_LNG] = COL_LNG
  #print(column_names)
  #return

  # perform aggregation (first pass: build summaries and count raw rows)
  if verbose:
    print("1) Scanning input to build aggregation summaries...")

  groups: Dict[Tuple[int, int, str], AggregationSummary] = {}
  num_raw = 0
  chunk_index = 0
  file_size = os.path.getsize(input_path)

  for chunk in pd.read_csv(input_path, chunksize=chunk_size, compression=compression_in):
      chunk["aggr_x"] = np.floor(chunk[column_names[COL_LNG]] / lng_degrees).astype(int)
      chunk["aggr_y"] = np.floor(chunk[column_names[COL_LAT]] / lat_degrees).astype(int)
      chunk = chunk.reset_index(drop=True)
      chunk["id"] = np.arange(num_raw + 1, num_raw + len(chunk) + 1)

      _validate_chunk(chunk, verbose, validate_values, column_names)

      num_raw += len(chunk)
      chunk_index += 1
      if verbose:
        progress = min(num_raw / file_size * 100.0, 100.0)
        print(f"Scanned chunk {chunk_index}; approx {progress:.1f}% of input")  # Note: for very small chunk & file sizes this will be wrong as the file is read in one go

      for row in chunk.itertuples(index=False):
        key = (
          int(getattr(row, "aggr_x")),
          int(getattr(row, "aggr_y")),
          getattr(row, column_names[COL_CAT]),
        )
        summary = groups.get(key)
        if summary is None:
          summary = AggregationSummary()
          groups[key] = summary
        summary.add_point(
          float(getattr(row, column_names[COL_LAT])),
          float(getattr(row, column_names[COL_LNG])),
          float(getattr(row, column_names[COL_VAL])),
        )

  # assign aggregated IDs
  if include_base_data:
    next_aggr_id = num_raw + 1
  else:
    next_aggr_id = 1

  group_to_aggr_id: Dict[Tuple[int, int, str], int] = {}
  for key in groups.keys():
    group_to_aggr_id[key] = next_aggr_id
    next_aggr_id += 1

  # 2nd pass: write raw rows with assigned aggrId (if requested)
  if include_base_data:
    if verbose:
      print("2) Writing raw rows with assigned aggrId...")
    first_chunk = True
    row_id = 0
    for chunk in pd.read_csv(input_path, chunksize=chunk_size, compression=compression_in):
      chunk["aggr_x"] = np.floor(chunk[column_names[COL_LNG]] / lng_degrees).astype(int)
      chunk["aggr_y"] = np.floor(chunk[column_names[COL_LAT]] / lat_degrees).astype(int)
      chunk = chunk.reset_index(drop=True)
      chunk["id"] = np.arange(row_id + 1, row_id + len(chunk) + 1)
      row_id += len(chunk)

      # map each row's group to its aggregated id
      def _map_aggr_id(r):
        k = (int(r.aggr_x), int(r.aggr_y), r.category)
        return group_to_aggr_id.get(k, pd.NA)

      chunk["aggrId"] = [_map_aggr_id(r) for r in chunk.itertuples(index=False)]

      raw_columns = ["id", "aggrId", column_names[COL_VAL], column_names[COL_CAT], column_names[COL_LAT], column_names[COL_LNG]]
      chunk[raw_columns].to_csv(output_path, index=False, mode="w" if first_chunk else "a", header=first_chunk, compression=compression_out)
      first_chunk = False
  else:
    if verbose:
      print("2) Skipping raw row output because include_base_data is false")

  # build aggregated rows and write them
  if verbose:
    print("3) Storing aggregated data...")

  aggregated_rows: List[dict] = []
  for (ix, iy, category), summary in groups.items():
    center = _compute_center(ix, iy, lng_degrees, lat_degrees)
    try:
      PainData.assert_coordinate(center, -1)
    except AssertionError:
      print(
        f"Center for ix={ix}, iy={iy}, lng-step={lng_degrees}, lat-step={lat_degrees} is invalid: center={center}"
      )

    aggr_id = group_to_aggr_id[(ix, iy, category)]
    value = _compute_aggregated_value(summary, aggr_func)
    coordinate = _compute_aggregated_coordinate(summary, coor_func, center)

    aggregated_rows.append({
      "id": aggr_id,
      "aggrId": pd.NA,
      COL_VAL: value,
      COL_CAT: category,
      COL_LAT: coordinate.y,
      COL_LNG: coordinate.x,
    })

  df_aggr = pd.DataFrame(aggregated_rows, columns=["id", "aggrId", COL_VAL, COL_CAT, COL_LAT, COL_LNG])
  # append aggregated rows to the output (or write alone if base-data was skipped)
  df_aggr.to_csv(output_path, index=False, mode="a" if include_base_data else "w", header=not include_base_data, compression=compression_out)

  if verbose:
    print("-done-")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog='Data Aggregation Script',
    description='Aggregates datapoints from an input file in the specified lat/lng resolution and stores them in an output file.',
    epilog=''
  )
  parser.add_argument("-bp", "--base-path", type=str, help="common base path shared among input and output file")
  parser.add_argument("-i", "-in", "--input", type=str, help="path to the input file")
  parser.add_argument("-o", "-out", "--output", type=str, help="path to the output file")
  parser.add_argument("-lat", "--lat-degrees", type=float, 
                      help="Latitude resolution for the aggregation regions (e.g., '1' means that data at lat=10 & lat=10.5 are in the same region while lat=11 is in another)." \
                      f"Values must be in [0, {PainData.MAX_LAT() - PainData.MIN_LAT()}[ and we advise to use values that divide {PainData.MAX_LAT() - PainData.MIN_LAT()} without remainder.")
  parser.add_argument("-lng", "-lon", "--lng-degrees", type=float, 
                      help=f"Longitude resolution for the aggregation regions (e.g., '1' means that data at lng=10 & lng=10.5 are in the same region while lng=11 is in another)" \
                      f"Values must be in [0, {PainData.MAX_LNG() - PainData.MIN_LNG()}[ and we advise to use values that divide {PainData.MAX_LNG() - PainData.MIN_LNG()} without remainder.")
  parser.add_argument("-faggr", "--aggregation-function", type=str, 
                      help="Which function to use for computing the aggregated data point's pain value (avg, max)")
  parser.add_argument("-fcoor", "--coordinate-function", type=str, 
                      help="Which function to use for computing the aggregated data point's coordinate (center, mid, weightedmid, max)")
  parser.add_argument("-cnval", "-col-val", "--column-name-value", type=str, help="Dataset's name for the value column (default: value)")
  parser.add_argument("-cncat", "-col-cat", "--column-name-category", type=str, help="Dataset's name for the category column (default: category)")
  parser.add_argument("-cnlat", "-col-lat", "--column-name-latitude", type=str, help="Dataset's name for the latitude column (default: lat)")
  parser.add_argument("-cnlng", "-col-lng", "--column-name-longitude", type=str, help="Dataset's name for the longitude column (default: lng)")
  parser.add_argument("-cs", "--chunk-size", help="Chunk size for streaming")
  parser.add_argument("-nobd", "--no-base-data", action='store_true', help="Whether to include the unaggregated base data points in the exported file")
  parser.add_argument("-q", "--quiet", action='store_true', help="Whether to be quiet or print messages informing about completed steps")
  parser.add_argument("-c", "--compression", type=str, help="Compression used to read the input file and write the output file. Overwrites the corresponding arguments.")
  parser.add_argument("-cin", "--compression-in", type=str, help="Compression used to read the input file.")
  parser.add_argument("-cout", "--compression-out", type=str, help="Compression used to write the output file.")
  parser.add_argument("-nvv", "--no-value-validation", action='store_true', help="Whether aggregation should raise an error for values <0 or >1.")

  args = parser.parse_args()
  print(args)
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

  lat_degrees = args.lat_degrees if args.lat_degrees else \
    1
  lng_degrees = args.lng_degrees if args.lng_degrees else \
    1
  if lat_degrees < 0 or (PainData.MAX_LAT() - PainData.MIN_LAT()) <= lat_degrees:
    print(f"Invalid lat_degrees: 0 <= {lat_degrees} < {PainData.MAX_LAT() - PainData.MIN_LAT()} must be true!")
    exit(1)
  if lng_degrees < 0 or (PainData.MAX_LNG() - PainData.MIN_LNG()) <= lng_degrees:
    print(f"Invalid lng_degrees: 0 <= {lng_degrees} < {PainData.MAX_LNG() - PainData.MIN_LNG()} must be true!")
    exit(1)

  aggr_func_name = args.aggregation_function if args.aggregation_function else \
    "avg"
  coor_func_name = args.coordinate_function if args.coordinate_function else \
    "center"

  chunk_size = int(args.chunk_size) if args.chunk_size else \
    200_000

  compression_in = args.compression_in if args.compression_in else \
    "infer"
  compression_out = args.compression_out if args.compression_out else \
    None
  if args.compression:
    compression_in = args.compression
    compression_out = args.compression

  column_names = {}
  if args.column_name_value:
    column_names[COL_VAL] = args.column_name_value
  if args.column_name_category:
    column_names[COL_CAT] = args.column_name_category
  if args.column_name_latitude:
    column_names[COL_LAT] = args.column_name_latitude
  if args.column_name_longitude:
    column_names[COL_LNG] = args.column_name_longitude

  t_start = time.time()
  perform(input_path, output_path, lat_degrees, lng_degrees, aggr_func_name, coor_func_name, chunk_size,
          include_base_data=not args.no_base_data, compression_in=compression_in, compression_out=compression_out,
          column_names=column_names, validate_values=not args.no_value_validation, verbose=not args.quiet)
  t_end = time.time()
  print(f"Elapsed time = {t_end - t_start}")
