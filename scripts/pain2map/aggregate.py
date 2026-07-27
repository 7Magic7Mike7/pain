from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from .navigation import Coordinate

class PainData:
  __VALUE_TOLERANCE = 1.0
  __VERBOSE = False
  _MIN_LAT = -90
  _MAX_LAT = 90
  _MIN_LNG = -180
  _MAX_LNG = 180

  @staticmethod
  def MIN_LAT():
    return PainData._MIN_LAT

  @staticmethod
  def MAX_LAT():
    return PainData._MAX_LAT

  @staticmethod
  def MIN_LNG():
    return PainData._MIN_LNG

  @staticmethod
  def MAX_LNG():
    return PainData._MAX_LNG

  @staticmethod
  def set_verbose(verbose: bool):
    PainData.__VERBOSE = verbose

  @staticmethod
  def create(id: int, lat: float, lng: float, val: float, src: str) -> "PainData":
    return PainData(id, Coordinate(lng, lat), val, src)

  @staticmethod
  def assert_coordinate(coor: Coordinate, id: int):
    assert PainData._MIN_LAT <= coor.y < PainData._MAX_LAT, f"Invalid latitude for id={id}: {PainData._MIN_LAT} <= {coor.y} < {PainData._MAX_LAT} is false!"
    assert PainData._MIN_LNG <= coor.x < PainData._MAX_LNG, f"Invalid longitude for id={id}: {PainData._MIN_LNG} <= {coor.x} < {PainData._MAX_LNG} is false!"

  def __init__(self, id: int, coor: Coordinate, val: float, src: str):
    assert 0 < id, f"Invalid id: 0 < {id} is false!"
    PainData.assert_coordinate(coor, id)
    if -self.__VALUE_TOLERANCE < val < 0:
      if PainData.__VERBOSE: print(f"WARN for id={id}: clamped val = {val}")
      val = 0
    elif 1 < val < self.__VALUE_TOLERANCE:
      if PainData.__VERBOSE: print(f"WARN for id={id}: clamped val = {val}")
      val = 1
    assert 0 <= val <= 1, f"Invalid value for id={id}: 0 <= {val} <= 1 is false!"

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

  def to_df_object(self) -> Dict[str, Union[int, float, str]]:
    return {
      "id": self.id,
      "aggrId": pd.NA if self.aggrId is None else self.aggrId,
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
  def max_aggregation(data: List[PainData]) -> float:
    return max([pd.val for pd in data])

  @staticmethod
  def center_coordinate(data: List[PainData], center: Coordinate) -> Coordinate:
    return Coordinate(center.x, center.y)

  @staticmethod
  def mid_point_coordinate(data: List[PainData], center: Coordinate) -> Coordinate:
    c_sum = Coordinate(0, 0)
    for pd in data:
      c_sum += pd._coor
    return c_sum / len(data)

  @staticmethod
  def weighted_mid_point_coordinate(data: List[PainData], center: Coordinate) -> Coordinate:
    if sum([pd.val for pd in data]) <= 0:
      # every point has a 0-weight so we simply choose their center
      return AggregatedPainData.mid_point_coordinate(data, center)
    # otherwise we have at least one actual weight
    c_sum = Coordinate(0, 0)
    val_sum = 0
    for pd in data:
      c_sum += pd._coor * pd.val
      val_sum += pd.val
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
    PainData.assert_coordinate(coor, id)
    return AggregatedPainData(id, coor, data, aggregation_func)

  def __init__(self, id: int, coor: Coordinate, data: List[PainData], aggregation_func: Callable[[List[PainData]], float]):
    assert len(data) > 0, "Invalid data for id={id}: No empty list allowed!"
    src = data[0].src
    for pd in data:
      assert pd.src == src, f"Invalid src at id={id} for pd.id={pd.id}: All data points must be from the same source: {pd.src} != {src}"
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
  def get_aggr_func_from_name(name: str) -> Callable[[List[PainData]], float]:
    """
    Normalizes the given name (case-insensitive, ignoring whitespace, - and _) and returns its associated function to compute a pain value
    from a list of PainData.
    Supported values include: "average" and "max"
    param name: name of the function to retrieve
    returns: the pain value computation function associated with the given name
    throws: Exception for unknown names
    """
    norm_name = name.lower()
    if norm_name in ["avg", "average"]:
      return AggregatedPainData.avg_aggregation
    elif norm_name in ["max", "maximum"]:
      return AggregatedPainData.max_aggregation
    raise Exception(f"Unknown aggregation function: \"{name}\"")

  @staticmethod
  def get_coor_func_from_name(name: str) -> Callable[[List[PainData], Coordinate], Coordinate]:
    """
    Normalizes the given name (case-insensitive, ignoring whitespace, - and _) and returns its associated function to compute a Coordiante
    from a list of PainData and the center Coordinate of their aggregation region.
    Supported values include: "center", "mid", "weightedmid" and "max"
    param name: name of the function to retrieve
    returns: the coordinate computation function associated with the given name
    throws: Exception for unknown names
    """
    norm_name = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    if norm_name in ["center"]:
      return AggregatedPainData.center_coordinate
    elif norm_name in ["mid", "midpoint"]:
      return AggregatedPainData.mid_point_coordinate
    elif norm_name in ["wmid", "weightedmid", "weightedmidpoint"]:
      return AggregatedPainData.weighted_mid_point_coordinate
    elif norm_name in ["max", "maxpoint"]:
      return AggregatedPainData.max_point_coordinate
    raise Exception(f"Unknown coordinate function: \"{name}\"")

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

  def aggregate(self, dataset: pd.DataFrame, aggr_func: Optional[Callable[[List[PainData]], float]] = None, coor_func: Optional[Callable[[List[PainData], Coordinate], Coordinate]] = None) -> List[AggregatedPainData]:
    """
    @param aggr_func: function that determines how the value of AggregatedPainData is computed
    @param coor_func: function that determines how the coordinate of AggregatedPainData is computed based on the underlying data points and the area's center
    """
    if aggr_func is None:
      aggr_func = AggregatedPainData.avg_aggregation
    if coor_func is None:
      coor_func = AggregatedPainData.center_coordinate

    # compute aggregation indices vectorized
    df = dataset.copy()
    df["aggr_x"] = np.floor(df["lng"] / self.__lng_step).astype(int)
    df["aggr_y"] = np.floor(df["lat"] / self.__lat_step).astype(int)

    # ensure input is not already aggregated
    if not df["aggrId"].isna().all():
      raise AssertionError("Can only aggregate datasets that are not yet aggregated!")

    # apply source filter (vectorized)
    if self.__src_filter is None:
      filtered = df
    else:
      filtered = df[df["category"].astype(str).str.contains(self.__src_filter)].copy()

    # assign sequential ids for raw PainData
    filtered = filtered.reset_index(drop=True)
    num_raw = len(filtered)
    filtered["id"] = np.arange(1, num_raw + 1)

    # group by grid cell and category so every group's points share the same src
    groups = filtered.groupby(["aggr_x", "aggr_y", "category"], sort=False)

    cache: List[Tuple[Coordinate, List[PainData]]] = []
    error_occurred = False

    for (ix, iy, category), grp in groups:
      center = self._compute_center(ix, iy, self.__lng_step, self.__lat_step)
      try:
        PainData.assert_coordinate(center, -1)
      except AssertionError as err:
        print(f"Center for ix={ix}, iy={iy}, lng-step={self.__lng_step}, lat-step={self.__lat_step} is invalid: center={center}")

      pts: List[PainData] = []
      # iterate only within each group (smaller loops)
      for row in grp.itertuples(index=False):
        try:
          # access by attribute names (column names are known)
          id = int(getattr(row, "id"))
          lat = float(getattr(row, "lat"))
          lng = float(getattr(row, "lng"))
          value = float(getattr(row, "value"))
          pd_obj = PainData.create(id, lat, lng, value, category)
          pts.append(pd_obj)
        except AssertionError as err:
          print(f"Assertion for id={id}: ", err)
          error_occurred = True
          continue
      if pts:
        cache.append((center, pts))

    if error_occurred:
      raise Exception("Error occurred. Consider stdout for details.")

    # create aggregated IDs continuing after raw ids
    next_aggr_id = num_raw + 1
    result: List[AggregatedPainData] = []
    for center, pts in cache:
      result.append(AggregatedPainData.from_funcs(next_aggr_id, pts, center, aggr_func, coor_func))
      next_aggr_id += 1

    return result
