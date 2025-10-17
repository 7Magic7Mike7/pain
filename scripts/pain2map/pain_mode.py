import sys
import warnings

from typing import Optional, List, Tuple, Dict
from enum import Enum

import os
import pandas as pd
import geopandas as gpd
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
from PIL import Image

Image.MAX_IMAGE_PIXELS = 20_000 * 10_000    # we need to overwrite the default value since we have huge output images


# Projections map (easy names -> EPSG/PROJ strings)
PROJECTIONS = {
    "PlateCarree": "EPSG:4326",
    "Mercator": "EPSG:3395",
    "Robinson": "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
    "Mollweide": "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
    "EqualEarth": "+proj=eqearth +lon_0=0 +datum=WGS84 +units=m +no_defs",
    "WinkelTripel": "+proj=wintri +lon_0=0 +datum=WGS84 +units=m +no_defs",
}
USED_PROJECTION = "PlateCarree"
CODE_COLUMN = "SOV_A3"
EDGE_WIDTH = 2.25
DPI_VALUE = 100
HEIGHT_VALUE = 128
WIDTH_VALUE = int(HEIGHT_VALUE * 2 / 1)
args_edge_color, args_missing_color = "#00FF37", "#EEEEEE"

MAP_SIZE_COUNTRIES = (2000, 1000)
MAP_SIZE_SOUNDS = (600, 300)
MAP_SIZE_CRATERS = (1000, 500)


class PainMode(Enum):
    SocioEcological = ("socio-eco", os.path.join("out", "sov_data", "socioeconomic-pain-sov.csv"), "YlOrBr", MAP_SIZE_COUNTRIES, "TODO")
    Emotional = ("emo", os.path.join("out", "sov_data", "emo-sov.csv"), "Blues", MAP_SIZE_COUNTRIES, "TODO")
    Physical = ("physical", os.path.join("out", "sov_data", "physical-sov.csv"), "RdPu", MAP_SIZE_COUNTRIES, "TODO")
    Environmental = ("env", os.path.join("out", "sov_data", "env-sov.csv"), "Greys", MAP_SIZE_CRATERS, "TODO")

    Fire = ("fire", os.path.join("out", "sov_data", "std-env-fire-sov.csv"), "Reds", MAP_SIZE_SOUNDS, "TODO")
    Water = ("water", os.path.join("out", "sov_data", "env-water-sov.csv"), "Blues", MAP_SIZE_SOUNDS, "TODO")
    Earth = ("earth", os.path.join("out", "sov_data", "env-earth-sov.csv"), "Greens", MAP_SIZE_SOUNDS, "TODO")
    Wood = ("wood", os.path.join("out", "sov_data", "env-wood-sov.csv"), "Greens", MAP_SIZE_SOUNDS, "TODO")
    Metal = ("metal", os.path.join("out", "sov_data", "env-metal-sov.csv"), "Reds", MAP_SIZE_SOUNDS, "TODO")

    @staticmethod
    def from_string(name: str) -> Optional["PainMode"]:
        for val in PainMode:
            if name == val.__abbr:
                return val
        return None

    def __init__(self, abbreviation: str, data_path: str, cmap: str, default_size: Tuple[int, int], description: str):
        super().__init__()
        self.__abbr = abbreviation
        self.__path = data_path
        self.__cmap = cmap
        self.__default_size = default_size
        self.__desc = description
    
    @property
    def abbreviation(self) -> str:
        return self.__abbr
    
    @property
    def path(self) -> str:
        return self.__path
    
    @property
    def cmap(self) -> str:
        return self.__cmap
    
    @property
    def default_size(self) -> Tuple[int, int]:
        return self.__default_size
    
    @property
    def description(self) -> str:
        return self.__desc
    
    def resolve_path(self, base_path: str) -> str:
        return os.path.join(base_path, self.__path)

    def _load_world(self, file_path: str, verbose: bool = False):
        # Use Natural Earth low-res that ships with GeoPandas
        world = gpd.read_file(file_path)
        if verbose: 
            print(world)
        # Fix some known ISO code quirks
        # Natural Earth uses "France" overseas territories as single polygon, ISO A3 is in 'iso_a3'
        world.loc[world["SOVEREIGNT"] == "France", "sov_a3"] = "FRA"
        world.loc[world["SOVEREIGNT"] == "Norway", "sov_a3"] = "NOR"
        world.loc[world["SOVEREIGNT"] == "Somaliland", "sov_a3"] = "SOL"  # non-ISO, avoid collision
        return world
    
    def _resize_map(self, target_size: Tuple[int, int], target_path: str, output_path: str) -> bool:
        """
        target_size: (width, height)
        """
        try:
            img = Image.open(target_path)
            img = img.resize(target_size)
            img.save(output_path)
            print(f"Resized image to {target_size}")
            return True
        except Exception as ex:
            print(f"Failed to resize map: {ex}", file=sys.stderr)
            return False

    def generate_map(
            self, 
            base_path: str, 
            world_path: str, 
            output_path: Optional[str] = None, 
            args_value_col: str = "value", 
            args_code_col = "sov_a3", 
            target_size: Optional[Tuple[int, int]] = None, 
            skip_resizing: bool = False
            ) -> bool:
        if output_path is None: output_path = os.path.join(base_path, "out", "world.png")
        if target_size is None: target_size = self.default_size
        dataframe = pd.read_csv(self.resolve_path(base_path))

        # Load data
        if args_code_col not in dataframe.columns or args_value_col not in dataframe.columns:
            print(f"CSV must include columns '{args_code_col}' and '{args_value_col}'", file=sys.stderr)
            return False

        # Clean codes
        dataframe[args_code_col] = dataframe[args_code_col].astype(str).str.upper().str.strip()

        # Load world geometries
        world = self._load_world(world_path)

        # Merge
        merged = world.merge(dataframe[[args_code_col, args_value_col]], left_on=CODE_COLUMN, right_on=args_code_col, how="left")

        # Warn about codes that didn't match
        provided_codes = set(dataframe[args_code_col].unique())
        matched_codes = set(merged.loc[~merged[args_value_col].isna(), CODE_COLUMN].unique())
        missing_codes = sorted(provided_codes - matched_codes)
        if missing_codes:
            warnings.warn(f"{len(missing_codes)} code(s) in your CSV did not match Natural Earth {CODE_COLUMN}: {missing_codes}")

        # Projection
        try:
            merged = merged.to_crs(PROJECTIONS.get(USED_PROJECTION))
        except Exception as e:
            warnings.warn(f"Could not project to {USED_PROJECTION}, using PlateCarree. Error: {e}")
            merged = merged.to_crs(PROJECTIONS["PlateCarree"])
        
        if False:
            # Plot
            #legend_kwds = {"loc": "lower left", "title": "TODO", "fmt": ".2f"}
            #fig = plt.figure(figsize=(WIDTH_VALUE, HEIGHT_VALUE), dpi=DPI_VALUE)
            #ax = fig.gca()
            #fig.patch.set_alpha(0)

            #plot_kwargs = dict(
            #    column=column_to_plot,
            #    #cmap=plt.get_cmap(cmap),
            #    linewidth=EDGE_WIDTH,
            #    edgecolor=args_edge_color,
            #    missing_kwds={"color": args_missing_color, "edgecolor": args_edge_color, "hatch": None, "linewidth": EDGE_WIDTH},
            #)
            #merged.plot(ax=ax, **plot_kwargs)
            #ax.set_axis_off()
            #ax.set_aspect("equal")
            #ax.set_title("TODO Title", fontsize=14, pad=12)
            pass
        
        merged.plot(column=args_value_col, cmap=self.cmap, figsize=target_size)

        # Save
        try:
            plt.savefig(output_path, bbox_inches="tight", dpi=DPI_VALUE)
            if not skip_resizing:
                resized_path = os.path.join(base_path, "out", "maps_resized", f"{self.abbreviation}_{target_size[0]}x{target_size[1]}.png")
                if self._resize_map(target_size, output_path, resized_path):
                    print(f"Saved {resized_path}")
            else:
                print(f"Saved {output_path}")
            return True
        except Exception as e:
            print(f"Failed to save figure: {e}", file=sys.stderr)
        return False
