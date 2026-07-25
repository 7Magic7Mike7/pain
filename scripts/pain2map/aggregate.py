from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from .navigation import Coordinate

class PainData:
  __VALUE_TOLERANCE = 1.0
  __VERBOSE = False

  @staticmethod
  def set_verbose(verbose: bool):
    PainData.__VERBOSE = verbose

  @staticmethod
  def create(id: int, lat: float, lng: float, val: float, src: str) -> "PainData":
    return PainData(id, Coordinate(lng, lat), val, src)

  def __init__(self, id: int, coor: Coordinate, val: float, src: str):
    assert 0 < id, f"Invalid id: 0 < {id} is false!"
    assert -90 <= coor.y < 90, f"Invalid latitude: -90 <= {coor.y} < 90 is false!"
    assert -180 <= coor.x < 180, f"Invalid longitude: -180 <= {coor.x} < 180 is false!"
    if -self.__VALUE_TOLERANCE < val < 0:
      if PainData.__VERBOSE: print(f"WARN: clamped val = {val}")
      val = 0
    elif 1 < val < self.__VALUE_TOLERANCE:
      if PainData.__VERBOSE: print(f"WARN: clamped val = {val}")
      val = 1
    assert 0 <= val <= 1, f"Invalid value: 0 <= {val} <= 1 is false!"

    self.__id = id
    self.__aggrId: Optional[int] = None
    self.__val = val
    self.__src = src
    self.__coor = coor

  @property
  def id(self) -> int:
    return self.__id

  @property
  def aggrId(self) -> Optional[int]:
    return self.__aggrId

  @property
  def val(self) -> float:
    return self.__val

  @property
  def src(self) -> str:
    return self.__src

  @property
  def lat(self) -> float:
    return self.__coor.y

  @property
  def lng(self) -> float:
    return self.__coor.x

  @property
  def _coor(self) -> Coordinate:
    return self.__coor

  def _set_aggrId(self, aggr_id: int) -> bool:
    if self.__aggrId is None:
      self.__aggrId = aggr_id
      return True
    return False

  def is_in_area(self, bot_left: Coordinate, top_right: Coordinate) -> bool:
    assert bot_left.x < top_right.x, f"top_left={bot_left} must not be right of top_right={top_right}"
    assert bot_left.y < top_right.y, f"top_left={bot_left} must not be above top_right={top_right}"

    return bot_left.x <= self.lng < top_right.x and bot_left.y <= self.lat < top_right.y

  def to_df_object(self) -> Dict[str, Union[float, str]]:
    return {
      "id": self.id,
      "aggrId": self.aggrId,
      "value": self.val,
      "category": self.src,
      "lat": self.lat,
      "lng": self.lng,
    }

  def __str__(self) -> str:
    return f"{self.src} @ ({self.lat:.2f}|{self.lng:.2f}) = {self.val:.4f}"


class AggregatedPainData(PainData):
  @staticmethod
  def avg_aggregation(data: List[PainData]) -> float:
    val_sum = sum([pd.val for pd in data])
    return val_sum / len(data)

  @staticmethod
  def center_coordinate(data: List[PainData], center: Coordinate) -> Coordinate:
    return Coordinate(center.x, center.y)   # todo: compute center so we don't have to pass it and can skip it?

  @staticmethod
  def mid_point_coordinate(data: List[PainData], center: Coordinate) -> Coordinate:
    c_sum = Coordinate(0, 0)
    for pd in data:
      c_sum += pd._coor
    return c_sum / len(data)

  @staticmethod
  def weighted_mid_point_coordinate(data: List[PainData], center: Coordinate) -> Coordinate:
    c_sum = Coordinate(0, 0)
    val_sum = 0
    for pd in data:
      c_sum += pd._coor * pd.val
      val_sum += pd.val
    if val_sum == 0:
      # every point has a 0-weight so we simply choose their center
      val_sum = len(data)
    return c_sum / val_sum

  @staticmethod
  def max_point_coordinate(data: List[PainData], center: Coordinate) -> Coordinate:
    c_max = center  # default to center
    max_val = -1    # val is >= 0 for PainData, so there *will* be a bigger one
    for pd in data:
      if pd.val > max_val:
        c_max = pd._coor
        max_val = pd.val
    return Coordinate(c_max.x, c_max.y)

  @staticmethod
  def from_funcs(id: int, data: List[PainData], center: Coordinate, aggregation_func: Callable[[List[PainData]], float], coordinate_func: Callable[[List[PainData], Coordinate], Coordinate]) -> "AggregatedPainData":
    coor = coordinate_func(data, center)
    return AggregatedPainData(id, coor, data, aggregation_func)

  def __init__(self, id: int, coor: Coordinate, data: List[PainData], aggregation_func: Callable[[List[PainData]], float]):
    assert len(data) > 0, "No empty list allowed!"
    src = data[0].src
    for pd in data:
      assert pd.src == src, f"All data points must be from the same source: {pd.src} != {src}"
      assert pd._set_aggrId(id), f"Failed to set aggrId={id} for datapoint #{pd.id} as it already has aggrId={pd.aggrId}"
    val = aggregation_func(data)

    super().__init__(id, coor, val, src)
    self.__datapoints = data

  @property
  def depth(self) -> int:
    return len(self.__datapoints)

  def resolve(self) -> List[PainData]:
    return list(self.__datapoints)

  def __str__(self) -> str:
    return super().__str__() + f" (#{self.depth})"


class AggregationManager:
  __MIN_LAT = -90   # inclusive
  __MAX_LAT = 90    # exclusive
  __MIN_LNG = -180  # inclusive
  __MAX_LNG = 180   # exclusive

  @staticmethod
  def _to_src(category: str) -> str:
    return f"{category}"

  @staticmethod
  def _compute_center(ix: int, iy: int, lng_step: float, lat_step: float) -> Coordinate:
    """
    Computes the center by first computing the bottom left corner of the area according to step sizes
    and indices in the grid and then adding half steps to both axis.
    """
    bot_left = Coordinate(ix * lng_step, iy * lat_step)
    return bot_left + Coordinate(lng_step * 0.5, lat_step * 0.5)

  def from_grid_numbers(num_cols: int, num_rows: int, src_filter: Optional[str] = None):
    lat_range = abs(AggregationManager.__MAX_LAT - AggregationManager.__MIN_LAT)
    lng_range = abs(AggregationManager.__MAX_LNG - AggregationManager.__MIN_LNG)
    lat_step = lat_range / num_rows
    lng_step = lng_range / num_cols
    return AggregationManager(lat_step, lng_step, src_filter)

  def __init__(self, lat_step: float, lng_step: float, src_filter: Optional[str] = None):
    self.__lat_step = lat_step
    self.__lng_step = lng_step
    self.__src_filter = src_filter
    self.__next_id = 1

  def aggregate(self, dataset: pd.DataFrame, aggr_func: Optional[Callable[[List[PainData]], float]] = None, coor_func: Optional[Callable[[List[PainData], Coordinate], Coordinate]] = None) -> List[AggregatedPainData]:
    """
    @param aggr_func: function that determines how the value of AggregatedPainData is computed
    @param coor_func: function that determines how the coordinate of AggregatedPainData is computed based on the underlying data points and the area's center
    """
    if aggr_func is None:
      aggr_func = AggregatedPainData.avg_aggregation
    if coor_func is None:
      coor_func = AggregatedPainData.center_coordinate

    # asign every point its x & y aggregation index (i.e., points with the same indices will be aggregated)
    dataset["aggr_x"] = np.floor(dataset["lng"] / self.__lng_step).astype(int)
    dataset["aggr_y"] = np.floor(dataset["lat"] / self.__lat_step).astype(int)

    self.__next_id = 1
    def get_next_id() -> int:
      cur_id = self.__next_id
      self.__next_id += 1
      return cur_id

    # perform the aggregation
    cache: Dict[Coordinate, List[PainData]] = {}   # stores info for building AggregatedPainData
    error_occurred = False
    for _, row in dataset.iterrows():
      aggrId, value, category, lat, lng, aggr_x, aggr_y = row   # id is missing since we recreate it
      assert not aggrId or np.isnan(aggrId), "Can only aggregate datasets that are not yet aggregated!"

      if value < 0 or 1 < value:
        debug = True

      src = self._to_src(category)
      if self.__src_filter is None or self.__src_filter in src:
        center = self._compute_center(aggr_x, aggr_y, self.__lng_step, self.__lat_step) # Todo: not sure if center is actually needed

        if center not in cache:
          cache[center] = []

        try:
          pd = PainData.create(get_next_id(), lat, lng, value, src)
          cache[center].append(pd)
        except AssertionError as err:
          print(f"Assertion for id={id}: ", err)
          error_occurred = True
          continue

      # else: filter out this row
    if error_occurred:
      raise Exception("Error occurred. Consider stdout for details.")
    return [AggregatedPainData.from_funcs(get_next_id(), cache[key], key, aggr_func, coor_func) for key in cache.keys()]
