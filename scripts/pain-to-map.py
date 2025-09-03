import sys
import warnings

from typing import Optional

import os
import pandas as pd
import geopandas as gpd
import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba

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


def _load_world(file_path: str, verbose: bool = False):
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


def generate_map(dataframe: pd.DataFrame, world_path: str, output_path: Optional[str] = None, cmap: str = "Greys", args_value_col: str = "value", args_code_col = "sov_a3"):
    if output_path is None: output_path = os.path.join("..", "data", "world.png")

    # Load data
    if args_code_col not in dataframe.columns or args_value_col not in dataframe.columns:
        print(f"CSV must include columns '{args_code_col}' and '{args_value_col}'", file=sys.stderr)
        return

    # Clean codes
    dataframe[args_code_col] = dataframe[args_code_col].astype(str).str.upper().str.strip()

    # Load world geometries
    world = _load_world(world_path)

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
    
    column_to_plot = args_value_col
    legend_kwds = {"loc": "lower left", "title": "TODO", "fmt": ".2f"}

    # Plot
    fig = plt.figure(figsize=(WIDTH_VALUE, HEIGHT_VALUE), dpi=DPI_VALUE)
    ax = plt.gca()
    fig.patch.set_alpha(0)

    plot_kwargs = dict(
        column=column_to_plot,
        #cmap=plt.get_cmap(cmap),
        linewidth=EDGE_WIDTH,
        edgecolor=args_edge_color,
        missing_kwds={"color": args_missing_color, "edgecolor": args_edge_color, "hatch": None, "linewidth": EDGE_WIDTH},
    )
    #merged.plot(ax=ax, **plot_kwargs)
    merged.plot(column="value", ax=ax, cmap=cmap)

    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title("TODO Title", fontsize=14, pad=12)

    # Save
    try:
        plt.savefig(output_path, bbox_inches="tight", dpi=DPI_VALUE, facecolor=fig.get_facecolor())
        print(f"Saved {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save figure: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    BASE_PATH = os.path.join("D:\\", "Workspaces", "vscode-workspace", "ai_x_medicine", "data")
    WORLD_PATH = os.path.join(BASE_PATH, "countries_map.zip")
    INPUT_PATHS = {
        "socio-eco": os.path.join(BASE_PATH, "out", "sov_data", "socioeconomic-pain-sov.csv"),
        "emo": os.path.join(BASE_PATH, "out", "sov_data", "emo-sov.csv"),
        "physical": os.path.join(BASE_PATH, "out", "sov_data", "physical-sov.csv"),
        "env": os.path.join(BASE_PATH, "out", "sov_data", "env-sov.csv"),

        "fire": os.path.join(BASE_PATH, "out", "sov_data", "env-fire-sov.csv"),
        "water": os.path.join(BASE_PATH, "out", "sov_data", "env-water-sov.csv"),
        "earth": os.path.join(BASE_PATH, "out", "sov_data", "env-earth-sov.csv"),
        "wood": os.path.join(BASE_PATH, "out", "sov_data", "env-wood-sov.csv"),
        "metal": os.path.join(BASE_PATH, "out", "sov_data", "env-metal-sov.csv"),
    }
    CMAPS = {
        "socio-eco": "YlOrBr",
        "emo": "Blues",
        "physical": "RdPu",
        "env": "Greys",

        "fire": "Reds",
        "water": "Blues",
        "earth": "Greens",
        "wood": "Greens",
        "metal": "Reds",
    }
    # todo (optional): arguments
    parser = argparse.ArgumentParser(description="Color a world map by country and export PNG.")
    parser.add_argument("--mode", required=True, help="'socio-eco', 'emo', 'physical'")
    args = parser.parse_args()
    DATA_MODE = args.mode
    
    df_pain = pd.read_csv(INPUT_PATHS[DATA_MODE]) #args.data)
    generate_map(df_pain, world_path=WORLD_PATH, output_path=os.path.join(BASE_PATH, "out", "maps", f"map-{DATA_MODE}.png"), cmap=CMAPS[DATA_MODE]) #args.world)
