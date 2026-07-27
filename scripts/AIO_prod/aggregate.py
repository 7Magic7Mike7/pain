# Imports
from typing import List
import argparse
import os
import pandas as pd
import time

# my scripts
from scripts.pain2map import AggregationManager, AggregatedPainData, PainData, Coordinate

def _aggrdp_to_df(aggr_data: List[AggregatedPainData], include_base_data: bool) -> pd.DataFrame:
  data: List[PainData] = []
  for data_point in aggr_data:
    if include_base_data:
      data += [dp.to_df_object() for dp in data_point.resolve()]
    data.append(data_point.to_df_object())
  return pd.DataFrame(data)

def perform(input_path: str, output_path: str, lat_degrees: float, lng_degrees: float, aggr_func_name: str, coor_func_name: str, include_base_data: bool = True, verbose: bool = True):
  # prepare parameters
  if verbose: print("0) Preparing aggregation...")
  aggr_func = AggregationManager.get_aggr_func_from_name(aggr_func_name)
  coor_func = AggregationManager.get_coor_func_from_name(coor_func_name)
  dataset = pd.read_csv(input_path)

  # perform aggregation
  if verbose: print("1) Starting aggregation...")
  aggr_man = AggregationManager(lat_degrees, lng_degrees)
  aggr_data = aggr_man.aggregate(dataset, aggr_func, coor_func)
  # store aggregated data
  if verbose: print("2) Storing aggregated data...")
  df_aggr = _aggrdp_to_df(aggr_data, include_base_data)
  df_aggr.to_csv(output_path, index=False, index_label='id')
  if verbose: print("-done-")

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
  parser.add_argument("-nobd", "--no-base-data", action='store_true', help="Whether to include the unaggregated base data points in the exported file")
  parser.add_argument("-q", "--quiet", action='store_true', help="Whether to be quiet or print messages informing about completed steps")

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

  t_start = time.time()
  perform(input_path, output_path, lat_degrees, lng_degrees, aggr_func_name, coor_func_name, 
          include_base_data=not args.no_base_data, verbose=not args.quiet)
  t_end = time.time()
  print(f"Elapsed time = {t_end - t_start}")
