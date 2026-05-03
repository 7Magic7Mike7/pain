import sys
from typing import Dict, Tuple, Set, Optional, Union
import warnings

from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import os


class Util:
    __BASE_PATH = None
    
    PROJECTIONS = {
        "PlateCarree": "EPSG:4326",
        "Mercator": "EPSG:3395",
        "Robinson": "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
        "Mollweide": "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
        "EqualEarth": "+proj=eqearth +lon_0=0 +datum=WGS84 +units=m +no_defs",
        "WinkelTripel": "+proj=wintri +lon_0=0 +datum=WGS84 +units=m +no_defs",
    }
    WORLD_CODE_COLUMN = "SOV_A3"

    def set_base_path(path: str):
        """
        The base path is the folder where countries_map.zip needs to be located to load world data.
        """
        Util.__BASE_PATH = path

    def load_world(path: Optional[str] = None, verbose: bool = False):
        # Use Natural Earth low-res that ships with GeoPandas
        #world_path = gpd.datasets.get_path("naturalearth_lowres")
        if path is None:
            if Util.__BASE_PATH is None:
                raise ValueError("Base path not set. Please call Util.set_base_path(path) with the folder containing countries_map.zip.")
            path = os.path.join(Util.__BASE_PATH, "countries_map.zip")
        world = gpd.read_file(path)#"https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
        #world = gpd.read_file("https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip") #geodatasets.get_path("naturalearth_lowres"))
        if verbose: 
            print(world)
        # Fix some known ISO code quirks
        # Natural Earth uses "France" overseas territories as single polygon, ISO A3 is in 'iso_a3'
        world.loc[world["SOVEREIGNT"] == "France", "sov_a3"] = "FRA"
        world.loc[world["SOVEREIGNT"] == "Norway", "sov_a3"] = "NOR"
        world.loc[world["SOVEREIGNT"] == "Somaliland", "sov_a3"] = "SOL"  # non-ISO, avoid collision
        return world

    def get_sova3_countries(path: Optional[str] = None) -> Dict[str, str]:
        world = Util.load_world(path)
        return dict(zip(world["SOVEREIGNT"], world["SOV_A3"]))

    def compute_sova3_subset(dataframe: pd.DataFrame, world_path: Optional[str] = None, df_country_label: str = "Country", df_value_label: str = "value") -> Tuple[pd.DataFrame, Set]:
        # Find intersection of country/location names
        new_world = Util.load_world(world_path)
        common_data = set(new_world["SOVEREIGNT"]).intersection(set(dataframe[df_country_label]))
        diff_data = set(new_world["SOVEREIGNT"]).difference(set(dataframe[df_country_label]))

        # Subset both DataFrames to only those in the intersection
        dataframe_sub = dataframe[dataframe["Country"].isin(common_data)].copy()

        result_data = []
        for _, row in dataframe_sub.iterrows():
            country = row[df_country_label]
            value = row[df_value_label]
            nw_row = new_world.loc[new_world["SOVEREIGNT"] == country]  # get the row corresponding to country
            sova3 = nw_row.iloc[0][Util.WORLD_CODE_COLUMN]
            result_data.append({
                "sov_a3": sova3,
                "Country": country,
                "value": value
            })
        # add 0 values for mismatching data
        for country in diff_data:
            nw_row = new_world.loc[new_world["SOVEREIGNT"] == country]  # get the row corresponding to country
            sova3 = nw_row.iloc[0][Util.WORLD_CODE_COLUMN]
            result_data.append({
                "sov_a3": sova3,
                "Country": country,
                "value": 0
            })
        #dataframe_sub.merge(new_world[["SOVEREIGNT", "SOV_A3"]], how="left")

        return pd.DataFrame(result_data), diff_data
    
    def generate_map(data: Union[str, pd.DataFrame], world_path: str, output_path: str, args_value_col: str = "value", args_code_col: str = "sov_a3", width: int = 512, dpi: int = 100, cmap: str = "viridis", projection: str = "PlateCarree"):
        """
        width: width of the output image in pixels - height will be automatically determined to maintain aspect ratio of the world geometries (is roughly 100:48)
        """
        if isinstance(data, str):
            dataframe = pd.read_csv(data)
        else:
            dataframe = data

        # Load data
        if args_code_col not in dataframe.columns or args_value_col not in dataframe.columns:
            print(f"CSV must include columns '{args_code_col}' and '{args_value_col}'", file=sys.stderr)
            return

        # Clean codes
        dataframe[args_code_col] = dataframe[args_code_col].astype(str).str.upper().str.strip()

        # Load world geometries
        world = Util.load_world(world_path)

        # Merge
        merged = world.merge(dataframe[[args_code_col, args_value_col]], left_on=Util.WORLD_CODE_COLUMN, right_on=args_code_col, how="left")

        # Warn about codes that didn't match
        provided_codes = set(dataframe[args_code_col].unique())
        matched_codes = set(merged.loc[~merged[args_value_col].isna(), Util.WORLD_CODE_COLUMN].unique())
        missing_codes = sorted(provided_codes - matched_codes)
        if missing_codes:
            warnings.warn(f"{len(missing_codes)} code(s) in your CSV did not match Natural Earth {Util.WORLD_CODE_COLUMN}: {missing_codes}")

        # Projection
        try:
            merged = merged.to_crs(Util.PROJECTIONS.get(projection))
        except Exception as e:
            warnings.warn(f"Could not project to {projection}, using PlateCarree. Error: {e}")
            merged = merged.to_crs(Util.PROJECTIONS["PlateCarree"])
        
        # Plot
        # Convert pixel dimensions to inches for figsize
        fig_width_inches = width / dpi
        fig_height_inches = fig_width_inches / 2
        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=dpi)
        ax = plt.gca()
        fig.patch.set_alpha(0)
        merged.plot(column=args_value_col, ax=ax, cmap=cmap)

        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.margins(0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #ax.set_title("TODO Title", fontsize=14, pad=12)

        # Save without extra transparent borders
        try:
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=dpi, transparent=True)
            print(f"Saved {output_path}")
            return True
        except Exception as e:
            print(f"Failed to save figure: {e}", file=sys.stderr)
            return False
