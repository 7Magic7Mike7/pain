from typing import Optional

import os
import argparse

from pain2map import PainMode


if __name__ == "__main__":
    BASE_PATH = os.path.join("D:\\", "Workspaces", "vscode-workspace", "ai_x_medicine", "data")
    WORLD_PATH = os.path.join(BASE_PATH, "countries_map.zip")
    # todo (optional): arguments
    parser = argparse.ArgumentParser(description="Color a world map by country and export PNG.")
    parser.add_argument("--mode", required=True, help="'socio-eco', 'emo', 'physical'")
    args = parser.parse_args()
    pain_mode: Optional[PainMode] = PainMode.from_string(args.mode)
    if pain_mode is None:
        print("Invalid mode! Use one of:")
        print("-" + "\n-".join([mode.abbreviation for mode in PainMode.values()]))
    else:
        output_path = os.path.join(BASE_PATH, "out", "maps", f"map-{pain_mode.abbreviation}.png")
        pain_mode.generate_map(BASE_PATH, WORLD_PATH, output_path)
