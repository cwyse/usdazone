#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import xarray as xr

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
]


def normalize_lon(lon: float) -> float:
    """Convert to 0–360 for ERA5 grid"""
    return lon if lon >= 0 else lon + 360


def zone_label(zone_num: int, subzone: int) -> str:
    if zone_num <= 0:
        return "—"
    return f"{zone_num}{'a' if subzone == 0 else 'b'}"


def main() -> None:
    ds = xr.open_dataset(DATASET)

    # Temperature (prefer stored F, else derive from C)
    if "usda_zone_temp_f" in ds:
        t_f = ds["usda_zone_temp_f"]
    elif "usda_zone_temp_c" in ds:
        t_f = ds["usda_zone_temp_c"] * 9 / 5 + 32
    else:
        raise RuntimeError("No USDA temperature variable found (usda_zone_temp_f or usda_zone_temp_c)")

    # Zone fields (must exist now)
    if "usda_zone_num" not in ds or "usda_zone_subzone" not in ds:
        raise RuntimeError("Dataset missing usda_zone_num/usda_zone_subzone; re-run merge_usda_zone_tiles.py")

    zn = ds["usda_zone_num"]
    zs = ds["usda_zone_subzone"]

    print(f"{'Location':<20} {'Temp (°F)':>10} {'Zone':>6}")
    print("-" * 40)

    for name, lat, lon in LOCATIONS:
        lon = normalize_lon(lon)

        if not (
            float(t_f.latitude.min()) <= lat <= float(t_f.latitude.max())
            and float(t_f.longitude.min()) <= lon <= float(t_f.longitude.max())
        ):
            print(f"{name:<20} {'N/A':>10} {'—':>6}")
            continue

        sel = dict(latitude=lat, longitude=lon, method="nearest")

        temp_val = float(t_f.sel(**sel))
        zone_num = int(zn.sel(**sel))
        subzone = int(zs.sel(**sel))
        zlab = zone_label(zone_num, subzone)

        # If zone_num is 0 (missing), temp may also be NaN; format defensively
        if zone_num <= 0 or temp_val != temp_val:  # NaN check
            print(f"{name:<20} {'N/A':>10} {'—':>6}")
        else:
            print(f"{name:<20} {temp_val:10.2f} {zlab:>6}")


if __name__ == "__main__":
    main()
