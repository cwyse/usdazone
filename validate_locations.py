#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from usda_zone_access import USDAZoneDataset

DATASET = Path("data/processed/global_usda_zone_temperature_1991_2020.nc")

LOCATIONS = [
    ("Hartford, CT", 41.77, -72.68),
    ("Bangor, ME", 44.80, -68.78),
    ("Minneapolis, MN", 44.98, -93.27),
    ("Bismarck, ND", 46.81, -100.78),
    ("Denver, CO", 39.74, -104.99),
    ("Flagstaff, AZ", 35.20, -111.65),
    ("Atlanta, GA", 33.75, -84.39),
    ("Houston, TX", 29.76, -95.37),
    ("Seattle, WA", 47.61, -122.33),
    ("Fresno, CA", 36.74, -119.78),
    ("Fairbanks, AK", 64.84, -147.72),
    ("Berlin, DE", 52.52, 13.40),
    ("Helsinki, FI", 60.17, 24.94),
    ("Tokyo, JP", 35.68, 139.76),
    ("Buenos Aires, AR", -34.60, -58.38),
    ("Dori, Burkina Faso", 14.030960, -0.032262),
    ("boundary north↔tropics @ (+19.940,-5.000)", 19.940000, -5.000000),
    ("boundary north↔tropics @ (+20.060,-5.000)", 20.060000, -5.000000),
    ("boundary north↔tropics @ (+19.940,+5.000)", 19.940000, 5.000000),
    ("boundary north↔tropics @ (+20.060,+5.000)", 20.060000, 5.000000),
    ("boundary north↔tropics @ (+19.940,+15.000)", 19.940000, 15.000000),
    ("boundary north↔tropics @ (+20.060,+15.000)", 20.060000, 15.000000),
    ("boundary tropics↔south @ (-19.940,+15.000)", -19.940000, 15.000000),
    ("boundary tropics↔south @ (-20.060,+15.000)", -20.060000, 15.000000),
    ("boundary tropics↔south @ (-19.940,+30.000)", -19.940000, 30.000000),
    ("boundary tropics↔south @ (-20.060,+30.000)", -20.060000, 30.000000),
    ("boundary tropics↔south @ (-19.940,+45.000)", -19.940000, 45.000000),
    ("boundary tropics↔south @ (-20.060,+45.000)", -20.060000, 45.000000),
    ("boundary north lon=0 @ (+25.000,-0.060)", 25.000000, -0.060000),
    ("boundary north lon=0 @ (+25.000,+0.060)", 25.000000, 0.060000),
    ("boundary north lon=0 @ (+35.000,-0.060)", 35.000000, -0.060000),
    ("boundary north lon=0 @ (+35.000,+0.060)", 35.000000, 0.060000),
    ("boundary north lon=0 @ (+45.000,-0.060)", 45.000000, -0.060000),
    ("boundary north lon=0 @ (+45.000,+0.060)", 45.000000, 0.060000),
    ("boundary north lon=90 @ (+25.000,+89.940)", 25.000000, 89.940000),
    ("boundary north lon=90 @ (+25.000,+90.060)", 25.000000, 90.060000),
    ("boundary north lon=90 @ (+35.000,+89.940)", 35.000000, 89.940000),
    ("boundary north lon=90 @ (+35.000,+90.060)", 35.000000, 90.060000),
    ("boundary north lon=90 @ (+45.000,+89.940)", 45.000000, 89.940000),
    ("boundary north lon=90 @ (+45.000,+90.060)", 45.000000, 90.060000),
    ("boundary north lon=270 @ (+35.000,-90.060)", 35.000000, -90.060000),
    ("boundary north lon=270 @ (+35.000,-89.940)", 35.000000, -89.940000),
    ("boundary north lon=270 @ (+45.000,-90.060)", 45.000000, -90.060000),
    ("boundary north lon=270 @ (+45.000,-89.940)", 45.000000, -89.940000),
    ("boundary north lon=270 @ (+55.000,-90.060)", 55.000000, -90.060000),
    ("boundary north lon=270 @ (+55.000,-89.940)", 55.000000, -89.940000),
    ("boundary tropics lon=0 @ (+10.000,-0.060)", 10.000000, -0.060000),
    ("boundary tropics lon=0 @ (+10.000,+0.060)", 10.000000, 0.060000),
    ("boundary tropics lon=0 @ (+15.000,-0.060)", 15.000000, -0.060000),
    ("boundary tropics lon=0 @ (+15.000,+0.060)", 15.000000, 0.060000),

]

def main() -> None:
    # Make location column wide enough for the longest name
    loc_w = max(len("Location"), max(len(name) for name, _, _ in LOCATIONS))
    temp_w = len("Temp (°F)")
    zone_w = len("Zone")

    header = f"{'Location':<{loc_w}}  {'Temp (°F)':>{temp_w}}  {'Zone':>{zone_w}}"
    print(header)
    print("-" * len(header))

    with USDAZoneDataset(DATASET) as zds:
        for name, lat, lon in LOCATIONS:
            p = zds.point(lat, lon)
            if p.temp_f is None or p.zone_label == "—":
                temp_s = "N/A"
                zone_s = "—"
            else:
                temp_s = f"{p.temp_f:.2f}"
                zone_s = p.zone_label

            print(f"{name:<{loc_w}}  {temp_s:>{temp_w}}  {zone_s:>{zone_w}}")


if __name__ == "__main__":
    main()
