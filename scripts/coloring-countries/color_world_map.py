#!/usr/bin/env python3
"""
Color a world map country-wise and export as PNG.

Usage:
  python color_world_map.py --data mydata.csv --value-col value --code-col iso_a3 --out map.png \
      --title "My Choropleth" --cmap viridis --missing-color "#EEEEEE" --projection Robinson

Inputs:
  - CSV with at least a country code column (ISO-3 like "USA", "FRA") and a numeric value column.
    Example headers: iso_a3,value
Output:
  - A high-resolution PNG choropleth map.

Requires: geopandas, pandas, matplotlib, mapclassify
"""
import argparse
import sys
import warnings

import pandas as pd
import geopandas as gpd
from geodatasets import get_path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
import mapclassify as mc

# Projections map (easy names -> EPSG/PROJ strings)
PROJECTIONS = {
    "PlateCarree": "EPSG:4326",
    "Mercator": "EPSG:3395",
    "Robinson": "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
    "Mollweide": "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
    "EqualEarth": "+proj=eqearth +lon_0=0 +datum=WGS84 +units=m +no_defs",
    "WinkelTripel": "+proj=wintri +lon_0=0 +datum=WGS84 +units=m +no_defs",
}

def load_world():
    # Use Natural Earth low-res that ships with GeoPandas
    #world_path = gpd.datasets.get_path("naturalearth_lowres")
    #world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
    world = gpd.read_file(get_path("naturalearth_lowres"))
    # Fix some known ISO code quirks
    # Natural Earth uses "France" overseas territories as single polygon, ISO A3 is in 'iso_a3'
    world.loc[world["name"] == "France", "iso_a3"] = "FRA"
    world.loc[world["name"] == "Norway", "iso_a3"] = "NOR"
    world.loc[world["name"] == "Somaliland", "iso_a3"] = "SOL"  # non-ISO, avoid collision
    return world

def classify_values(series, scheme, k):
    scheme = (scheme or "").lower()
    if scheme in ("quantiles", "quantile", "q"):
        return mc.Quantiles(series, k=k)
    elif scheme in ("equal_interval", "equalinterval", "equal", "ei"):
        return mc.EqualInterval(series, k=k)
    elif scheme in ("natural_breaks", "jenks", "nb"):
        return mc.NaturalBreaks(series, k=k)
    elif scheme in ("std_mean", "std", "zscore"):
        return mc.StdMean(series)
    else:
        return None  # continuous

def main():
    parser = argparse.ArgumentParser(description="Color a world map by country and export PNG.")
    parser.add_argument("--data", required=True, help="Path to CSV with values per country")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--code-col", default="iso_a3", help="Column with ISO-3 codes (default: iso_a3)")
    parser.add_argument("--value-col", default="value", help="Numeric value column (default: value)")
    parser.add_argument("--name-col", default=None, help="Optional column with custom country labels")
    parser.add_argument("--title", default=None, help="Optional map title")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name")
    parser.add_argument("--missing-color", default="#EEEEEE", help="Color for countries without data")
    parser.add_argument("--edge-color", default="#FFFFFF", help="Country border line color")
    parser.add_argument("--edge-width", type=float, default=0.25, help="Country border line width")
    parser.add_argument("--projection", default="Robinson", choices=list(PROJECTIONS.keys()), help="Map projection")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument("--width", type=float, default=12, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=6.5, help="Figure height in inches")
    parser.add_argument("--scheme", default=None, help="Classification scheme: quantiles, equal_interval, natural_breaks, std_mean")
    parser.add_argument("--k", type=int, default=5, help="Number of classes (if scheme is given)")
    parser.add_argument("--legend", action="store_true", help="Show legend")
    parser.add_argument("--legend-title", default=None, help="Legend title")
    parser.add_argument("--label-topn", type=int, default=0, help="Label top-N countries by value (0=none)")
    parser.add_argument("--format-values", default=".2f", help="Format string for labels/legend (default: .2f)")

    args = parser.parse_args()

    # Load data
    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if args.code_col not in df.columns or args.value_col not in df.columns:
        print(f"CSV must include columns '{args.code_col}' and '{args.value_col}'", file=sys.stderr)
        sys.exit(1)

    # Clean codes
    df[args.code_col] = df[args.code_col].astype(str).str.upper().str.strip()

    # Load world geometries
    world = load_world()

    # Merge
    merged = world.merge(df[[args.code_col, args.value_col] + ([args.name_col] if args.name_col else [])],
                         left_on="iso_a3", right_on=args.code_col, how="left")

    # Warn about codes that didn't match
    provided_codes = set(df[args.code_col].unique())
    matched_codes = set(merged.loc[~merged[args.value_col].isna(), "iso_a3"].unique())
    missing_codes = sorted(provided_codes - matched_codes)
    if missing_codes:
        warnings.warn(f"{len(missing_codes)} code(s) in your CSV did not match Natural Earth ISO-3: {missing_codes}")

    # Projection
    crs = PROJECTIONS.get(args.projection, "EPSG:4326")
    try:
        merged = merged.to_crs(crs)
    except Exception as e:
        warnings.warn(f"Could not project to {args.projection}, using PlateCarree. Error: {e}")
        merged = merged.to_crs(PROJECTIONS["PlateCarree"])

    # Prepare classification or continuous mapping
    values = merged[args.value_col]
    classifier = None
    if args.scheme:
        valid = values.dropna()
        if len(valid) == 0:
            print("No numeric values to classify.", file=sys.stderr)
            sys.exit(1)
        classifier = classify_values(valid, args.scheme, args.k)
        bins = classifier.bins
        # Build a discrete colormap with k bins
        cmap = plt.get_cmap(args.cmap, len(bins))
        merged["_class"] = classifier(y=values)
        column_to_plot = "_class"
        legend_kwds = {"loc": "lower left", "title": args.legend_title or args.value_col, "fmt": args.format_values}
    else:
        cmap = plt.get_cmap(args.cmap)
        column_to_plot = args.value_col
        legend_kwds = {"loc": "lower left", "title": args.legend_title or args.value_col, "fmt": args.format_values}

    # Plot
    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    ax = plt.gca()
    fig.patch.set_alpha(0)

    plot_kwargs = dict(
        column=column_to_plot,
        cmap=cmap,
        linewidth=args.edge_width,
        edgecolor=args.edge_color,
        missing_kwds={"color": args.missing_color, "edgecolor": args.edge_color, "hatch": None, "linewidth": args.edge_width},
    )
    merged.plot(ax=ax, **plot_kwargs)

    ax.set_axis_off()
    ax.set_aspect("equal")

    if args.title:
        ax.set_title(args.title, fontsize=14, pad=12)

    if args.legend:
        # geopandas legend for continuous data is limited; we use built-in
        merged.plot(ax=ax, **plot_kwargs, legend=True, legend_kwds=legend_kwds)

    # Label top-N countries
    if args.label_topn and args.label_topn > 0:
        try:
            # Compute centroids in projected CRS for good placement
            centroids = merged.copy()
            centroids["centroid"] = centroids.geometry.centroid
            top = centroids.nlargest(args.label_topn, args.value_col)[["centroid", args.value_col, "name", args.code_col if args.code_col in centroids.columns else "iso_a3"]]
            for _, row in top.iterrows():
                x, y = row["centroid"].x, row["centroid"].y
                label = f"{row['name']} ({row[args.value_col]:{args.format_values}})"
                ax.text(x, y, label, fontsize=6, ha="center", va="center")
        except Exception as e:
            warnings.warn(f"Could not place labels: {e}")

    # Save
    try:
        plt.savefig(args.out, bbox_inches="tight", dpi=args.dpi, facecolor=fig.get_facecolor())
        print(f"Saved {args.out}")
    except Exception as e:
        print(f"Failed to save figure: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
