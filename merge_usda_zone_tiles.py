#!/usr/bin/env python3
# merge_usda_zone_tiles.py

from __future__ import annotations

"""
Merge per-tile USDA-zone climatology NetCDFs into a single global raster, and
(re)compute USDA zone fields using the shared single-source-of-truth logic.

This script is intentionally "dumb merge + decorate":
- It does NOT implement any zone math itself (delegates to usda_zone_core).
- It does NOT do query-time fixes (wraparound / interpolation / nearest-point
  policy). Those belong in the centralized query API (USDAZoneDataset).

Inputs
------
Tile climatology files produced by your pipeline (Step: per-tile climatology):

  data/processed/usda_zone_temperature_tiles/<tile>_mean_<start>_<end>.nc

Where <tile> is typically:
  - north_lon000 / lon090 / lon180 / lon270   (90° lon bins)
  - south_lon000 / lon090 / lon180 / lon270   (90° lon bins)
  - tropics_lon000 / lon180                   (180° lon bins)

This merge expects each tile file to contain at least:
  - dimensions: latitude, longitude
  - a temperature variable in either:
      * preferred: usda_zone_temp_f
      * alternate: mean_annual_extreme_min_f
      * alternate: usda_zone_temp_c
      * alternate: mean_annual_extreme_min

The script will create/ensure:
  - usda_zone_temp_f (degF) as the canonical temperature field for downstream use.

Outputs
-------
A global merged file:

  data/processed/global_usda_zone_temperature_<start>_<end>.nc

The output includes:
  - usda_zone_temp_f (degF)
  - usda_zone_num (uint8): 1..13, 0 = missing
  - usda_zone_subzone (uint8): 0='a', 1='b', 255 = missing
  - usda_zone_code (uint16): zone_num*2 + subzone01, 0 = missing

Important assumptions / invariants
----------------------------------
- Longitude domain: expected to already be in ERA5 convention [0, 360)
  across all tiles. This script does not normalize -180..180 to 0..360.
- Tile boundaries: expected to be half-open in longitude bins
  (e.g. [0,90), [90,180), ...), avoiding duplicate boundary columns.
  If a tile includes a 360.0 endpoint column, it can create a seam unless it
  was removed upstream.
- Latitude overlap: expected to be handled upstream (tropics vs non-tropics).
  This script simply merges by coordinates.

CLI usage
---------
  python merge_usda_zone_tiles.py --start-year 1991 --end-year 2020

This will merge all files matching:
  data/processed/usda_zone_temperature_tiles/*_mean_1991_2020.nc

Implementation notes
--------------------
- Merge strategy:
    1) Read each tile dataset.
    2) Sort coords (lat/lon), drop duplicates defensively.
    3) Align with join="outer" and combine_first() so that:
         - where multiple tiles overlap, earlier-loaded tiles win
           (but your pipeline should avoid overlap).
- Zone fields are computed AFTER merge on the global usda_zone_temp_f field so
  zone math is applied consistently once, at global scope.
"""

from pathlib import Path
import argparse
import numpy as np
import xarray as xr

# Use the shared, single-source-of-truth USDA zone logic.
# These functions should be pure and deterministic.
from usda_zone_core import zone_num_from_temp_f, subzone_from_temp_f

TILES_DIR = Path("data/processed/usda_zone_temperature_tiles")


def _dedupe_1d(ds: xr.Dataset, dim: str, name: str) -> xr.Dataset:
    """
    Drop duplicate coordinate values along a 1D dimension, keeping the first.

    Why:
    - If any upstream step accidentally creates duplicated lat/lon values
      (e.g., float rounding artifacts or inclusive endpoints), xarray alignment
      and merge operations can behave unexpectedly.

    This is defensive; in the ideal pipeline, duplicates never occur.
    """
    idx = ds[dim].to_index()
    if idx.has_duplicates:
        mask = ~idx.duplicated()
        ds = ds.isel({dim: mask})
        print(f"[WARN] {name}: dropped {int((~mask).sum())} duplicate {dim} values")
    return ds


def _normalize(ds: xr.Dataset, name: str) -> xr.Dataset:
    """
    Normalize coordinate ordering and defensively remove duplicates.

    This does NOT change coordinate values (no lon normalization, no snapping).
    It only enforces:
      - latitude sorted ascending
      - longitude sorted ascending
      - no duplicates in either coordinate axis
    """
    # Ensure predictable coord order
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")

    # Defensive: drop duplicates inside a dataset if any ever appear
    if "latitude" in ds.dims:
        ds = _dedupe_1d(ds, "latitude", name)
    if "longitude" in ds.dims:
        ds = _dedupe_1d(ds, "longitude", name)
    return ds


def _get_temp_f(merged: xr.Dataset) -> xr.DataArray:
    """
    Return a canonical Fahrenheit temperature field from a merged dataset.

    Accepted inputs:
      - Preferred (new pipeline tiles): usda_zone_temp_f
      - Alternate: mean_annual_extreme_min_f
      - Alternate legacy: usda_zone_temp_c
      - Alternate legacy: mean_annual_extreme_min  (assumed degC)

    Returns:
      - An xarray DataArray named "usda_zone_temp_f" with units=degF.

    Note:
    - This function does not write into `merged`; it returns a DataArray.
      The caller can choose to assign it to merged["usda_zone_temp_f"].
    """
    if "usda_zone_temp_f" in merged:
        t = merged["usda_zone_temp_f"]
        t.attrs.setdefault("units", "degF")
        return t

    if "mean_annual_extreme_min_f" in merged:
        t = merged["mean_annual_extreme_min_f"]
        tf = t.copy()
        tf.name = "usda_zone_temp_f"
        tf.attrs.update(
            {
                "long_name": "Mean annual extreme minimum temperature (climatology)",
                "units": "degF",
                "source": "mean_annual_extreme_min_f",
            }
        )
        return tf

    if "usda_zone_temp_c" in merged:
        tc = merged["usda_zone_temp_c"]
        tf = tc * 9 / 5 + 32
        tf.name = "usda_zone_temp_f"
        tf.attrs.update(
            {
                "long_name": "Mean annual extreme minimum temperature (converted from C)",
                "units": "degF",
                "source": "usda_zone_temp_c",
            }
        )
        return tf

    if "mean_annual_extreme_min" in merged:
        tc = merged["mean_annual_extreme_min"]
        tf = tc * 9 / 5 + 32
        tf.name = "usda_zone_temp_f"
        tf.attrs.update(
            {
                "long_name": "Mean annual extreme minimum temperature (converted from C)",
                "units": "degF",
                "source": "mean_annual_extreme_min",
            }
        )
        return tf

    raise RuntimeError(
        "Merged dataset has no recognized temperature variable. "
        "Expected one of: mean_annual_extreme_min_f, mean_annual_extreme_min, "
        "usda_zone_temp_f, usda_zone_temp_c."
    )


def _compute_usda_zone_fields(
    temp_f: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Compute numeric USDA zone rasters from a Fahrenheit temperature field.

    Inputs
    ------
    temp_f:
      - Fahrenheit mean annual extreme minimum temperature.
      - Missing values should be NaN.

    Outputs (numeric rasters)
    -------------------------
    usda_zone_num: uint8
      - Values 1..13
      - 0 indicates missing (no temperature data)

    usda_zone_subzone: uint8
      - 0 = 'a' (colder half), 1 = 'b' (warmer half)
      - 255 indicates missing (IMPORTANT: we cannot use 0 as missing because 0 is 'a')

    usda_zone_code: uint16
      - Encodes zone + subzone in one integer:
          code = zone_num*2 + subzone01
      - 0 indicates missing
      - Note: subzone01 here is 0/1 (not the 255-missing raster)

    Notes
    -----
    USDA zone logic is delegated to usda_zone_core:
      - zone_num_from_temp_f(temp_f) -> float in [1..13]
      - subzone_from_temp_f(temp_f, zone_num_f) -> 0/1 (boundary -> b)

    This keeps zone math centralized and consistent with query-time logic.
    """
    valid = xr.apply_ufunc(np.isfinite, temp_f)

    # Shared logic
    zone_num_f = zone_num_from_temp_f(temp_f)  # float array, 1..13
    subzone01 = subzone_from_temp_f(temp_f, zone_num_f)  # 0=a, 1=b (boundary -> b)

    zone_num = zone_num_f.where(valid, other=0.0).astype(np.uint8)
    zone_num.name = "usda_zone_num"
    zone_num.attrs.update(
        {
            "long_name": "USDA Plant Hardiness Zone number",
            "valid_range": [1, 13],
            "missing_value": 0,
            "note": "0 indicates missing (no temperature data)",
        }
    )
    zone_num.encoding["_FillValue"] = np.uint8(0)

    # IMPORTANT: do NOT use 0 as missing because 0 is the real value for 'a'
    sub_fill = np.uint8(255)
    subzone = subzone01.where(valid, other=sub_fill).astype(np.uint8)
    subzone.name = "usda_zone_subzone"
    subzone.attrs.update(
        {
            "long_name": "USDA Plant Hardiness Zone subzone",
            "flag_values": [0, 1],
            "flag_meanings": "a b",
            "missing_value": int(sub_fill),
            "note": "0='a', 1='b'; missing points stored as 255",
        }
    )
    subzone.encoding["_FillValue"] = sub_fill

    zone_code = (
        (zone_num.astype(np.uint16) * 2 + subzone01.astype(np.uint16))
        .where(valid, other=0)
        .astype(np.uint16)
    )
    zone_code.name = "usda_zone_code"
    zone_code.attrs.update(
        {
            "long_name": "USDA zone code (zone_num*2 + subzone)",
            "missing_value": 0,
            "note": "Decode with usda_zone_num and usda_zone_subzone; 0 indicates missing",
        }
    )
    zone_code.encoding["_FillValue"] = np.uint16(0)

    return zone_num, subzone, zone_code


def main() -> None:
    """
    Entry point: locate tile files for a given period, merge them, and write a global NetCDF.

    Args:
      --start-year / --end-year:
        Used only to select input tile files matching:
          *_mean_<start>_<end>.nc
        and to name the output file:
          global_usda_zone_temperature_<start>_<end>.nc
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--end-year", type=int, required=True)
    args = ap.parse_args()

    pattern = f"*_mean_{args.start_year}_{args.end_year}.nc"
    tile_files = sorted(TILES_DIR.glob(pattern))
    if not tile_files:
        raise RuntimeError(f"No USDA zone temperature tiles found matching {pattern}")

    out_path = Path(f"data/processed/global_usda_zone_temperature_{args.start_year}_{args.end_year}.nc")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(tile_files)} tile(s)")

    merged: xr.Dataset | None = None

    for f in tile_files:
        print(f"[LOAD] {f.name}")
        ds = xr.open_dataset(f)

        # All tiles must be spatial rasters
        if not {"latitude", "longitude"} <= set(ds.dims):
            raise RuntimeError(f"{f} missing spatial dimensions")

        ds = _normalize(ds, f.name)

        if merged is None:
            merged = ds
            continue

        # Outer-align to create a global grid union, then fill from each tile.
        # combine_first() prefers values already in `merged` where both have data.
        merged, ds = xr.align(merged, ds, join="outer")
        merged = merged.combine_first(ds)

    assert merged is not None
    merged = _normalize(merged, "merged")

    # Add canonical Fahrenheit variable for downstream scripts.
    # (If already present, keep as-is.)
    temp_f = _get_temp_f(merged)
    if "usda_zone_temp_f" not in merged:
        merged["usda_zone_temp_f"] = temp_f

    # Compute and store USDA zone fields (via shared logic)
    zone_num, subzone, zone_code = _compute_usda_zone_fields(merged["usda_zone_temp_f"])
    merged[zone_num.name] = zone_num
    merged[subzone.name] = subzone
    merged[zone_code.name] = zone_code


    # Global dataset metadata
    merged.attrs.update(
        {
            "title": f"USDA Plant Hardiness Zone Temperature ({args.start_year}–{args.end_year})",
            "summary": (
                "Mean annual extreme minimum temperature over the specified period "
                "derived from ERA5-Land hourly data, with USDA hardiness zone fields."
            ),
            "source": "ERA5-Land (Copernicus Climate Data Store)",
            "institution": "Computed locally",
            "conventions": "CF-1.8",
            "period": f"{args.start_year}-{args.end_year}",
            "usda_zone_definition": (
                "zone_num=floor((F+60)/10)+1 clipped to 1..13; "
                "subzone a if F<lower+5 else b (boundary -> b)"
            ),
            "usda_zone_logic_source": "usda_zone_core.py",
        }
    )

    merged.to_netcdf(out_path)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
