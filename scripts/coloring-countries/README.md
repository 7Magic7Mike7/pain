# World Map Choropleth (PNG)

A tiny command-line tool to color a world map by country and export a PNG.

## Quick start

1) Install dependencies (ideally in a fresh virtual env):
```
pip install -r requirements.txt
```

2) Prepare a CSV with ISO-3 country codes and a numeric value. Example: `sample_data.csv`

3) Run:
```
python color_world_map.py --data sample_data.csv --value-col value --code-col iso_a3 --out world.png --title "Sample Map" --legend --scheme quantiles --k 5 --projection Robinson
```

This will create `world.png`.

## CSV format

- `iso_a3`: ISO 3166-1 alpha-3 country code (e.g., USA, FRA, JPN)
- `value`: numeric value used for coloring
- (optional) `label`: custom label for countries (not required)

## Options (highlights)
- `--cmap viridis` : pick any matplotlib colormap (e.g., plasma, inferno, Blues)
- `--scheme quantiles|equal_interval|natural_breaks|std_mean` : use discrete classes
- `--legend` and `--legend-title "My Legend"` : add a legend
- `--projection Robinson` : choose PlateCarree, Mercator, Robinson, Mollweide, EqualEarth, WinkelTripel
- `--missing-color "#EEEEEE"` : color for countries without data
- `--label-topn 10` : label the top 10 countries by value

## Notes

- The base map comes from Natural Earth (low resolution) via GeoPandas.
- Some small territories may not be represented at this resolution.
