#!/usr/bin/env python3
from __future__ import annotations

"""
retile_region.py

Retile ERA5-Land NetCDF files into standardized longitude bins for a single region
(north / south / tropics), optionally removing overlap between tropics and non-tropics.

This is a *data-shaping* script. It does not compute climatologies; it:
- groups input files by year-month,
- opens all inputs for each year-month,
- normalizes longitude to 0..360,
- filters by a region-specific latitude band,
- slices a target longitude bin,
- concatenates the pieces to produce one output per (year, month, lon_bin).

Typical use case
----------------
You have one or more source directories containing monthly ERA5-Land NetCDF files
(with filenames ending in YYYY_MM.nc or YYYY_MM) and you want to reorganize them
into canonical lon bins under an output root:

  <out-root>/
    lon000/<prefix>_lon000_YYYY_MM.nc
    lon090/<prefix>_lon090_YYYY_MM.nc
    lon180/<prefix>_lon180_YYYY_MM.nc
    lon270/<prefix>_lon270_YYYY_MM.nc

For tropics, only lon000 and lon180 bins are produced.

Why this exists (design goals)
------------------------------
- Deterministic binning: downstream steps can assume consistent lon-bin partitioning.
- Robust to messy sources: different coordinate names (lat vs latitude) and small float
  differences in longitude are normalized (rounding + duplicates dropped).
- Avoid overlap at tropics boundary: north/south exclude the tropics band and tropics keeps
  only -20..20 to prevent duplicated latitude rows between datasets.

Filename assumptions
--------------------
- Input files must end in "...YYYY_MM.nc" or "...YYYY_MM" (case-insensitive). The year-month
  is used only for grouping, not for verifying internal time coordinates.
- This script does not rewrite time coordinates; it preserves what is in each input file.

Coordinate assumptions
----------------------
- Inputs contain 1D latitude and longitude coordinates (typical for gridded ERA5 NetCDF).
- Longitude might be -180..180 or 0..360; this script normalizes to 0..360 and sorts.
- Small float noise in longitude is rounded to 6 decimals, then duplicate lon values are dropped.

Region latitude filtering
-------------------------
- north:   keep lat >  20.0
- south:   keep lat < -20.0
- tropics: keep -20.0 <= lat <= 20.0

Notes:
- This “20 degree” split is aligned to your pipeline’s tropics/non-tropics split.
- This script uses strict > / < for non-tropics and inclusive range for tropics to avoid overlap.

Longitude bin definitions
-------------------------
Non-tropics (4 x 90° bins):
- [0, 90)    -> lon000
- [90, 180)  -> lon090
- [180, 270) -> lon180
- [270, 360) -> lon270

Tropics (2 x 180° bins):
- [0, 180)   -> lon000
- [180, 360) -> lon180

Bins are half-open intervals [lo, hi) to avoid including boundary columns twice.

Dry-run by default
------------------
By default this script prints what it would write (PLAN: ...) and writes nothing.
Use --apply to actually write outputs.

No behavior changes
-------------------
This file is documented only. Logic and defaults are unchanged.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import xarray as xr

# Match filenames ending with "...YYYY_MM" or "...YYYY_MM.nc" (case-insensitive).
# This is used to group files that represent the same month across multiple source dirs.
YEAR_MONTH_RE = re.compile(r"(?P<year>\d{4})_(?P<month>\d{2})(?:\.nc)?$", re.IGNORECASE)

# Non-tropics: 4 bins of 90° (0–89.9, 90–179.9, 180–269.9, 270–359.9)
BINS_NON_TROPICS: List[Tuple[int, int, str]] = [
    (0, 89.9, "lon000"),
    (90, 179.9, "lon090"),
    (180, 269.9, "lon180"),
    (270, 359.9, "lon270"),
]

# Tropics:     2 bins of 180° (0–179.9, 180–359.9)
BINS_TROPICS: List[Tuple[int, int, str]] = [
    (0, 179.0, "lon000"),
    (180, 259.9, "lon180"),
]


def detect_year_month(p: Path) -> Tuple[str, str]:
    """
    Extract (year, month) from the input filename.

    The filename must end with YYYY_MM or YYYY_MM.nc, e.g.:
      north_lon000_1991_12.nc
      era5_land_2010_06.nc
      1991_12.nc

    Returns:
      (year_str, month_str)

    Raises:
      ValueError if no YYYY_MM suffix is found.
    """
    m = YEAR_MONTH_RE.search(p.name)
    if not m:
        raise ValueError("missing YYYY_MM at end of filename")
    return m.group("year"), m.group("month")


def find_coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    """
    Find which coordinate/variable name is present in the dataset.

    ERA5-Land files typically use 'latitude' and 'longitude', but this allows
    for 'lat'/'lon' or other capitalization variants.

    ds.coords and ds.variables are both checked because some files store latitude/longitude
    as data variables rather than coordinates.

    Returns:
      The first candidate name present.

    Raises:
      KeyError if none of the candidates are present.
    """
    names = set(ds.coords) | set(ds.variables)
    for c in candidates:
        if c in names:
            return c
    raise KeyError(f"missing coord among {list(candidates)}; have coords={sorted(list(ds.coords))[:20]}")


def normalize_lon_0_360(lon: xr.DataArray) -> xr.DataArray:
    """
    Normalize longitude values into [0, 360).

    This handles either convention:
    - -180..180 (common GIS)
    - 0..360    (ERA5 convention)

    It also preserves attributes on the longitude array.
    """
    lon2 = (lon % 360 + 360) % 360
    lon2 = lon2.assign_attrs(lon.attrs)
    return lon2


def round_lon(ds: xr.Dataset, lon_name: str, decimals: int = 6) -> xr.Dataset:
    """
    Round longitude coordinate values to reduce float jitter.

    Why:
    - Concatenating datasets can produce near-equal longitude values that differ by tiny
      float noise (e.g., 89.9999999998 vs 90.0). Rounding stabilizes equality checks
      and makes duplicate detection effective.

    decimals=6 is usually ample: 1e-6 degrees is sub-meter scale at the equator.
    """
    lon = ds[lon_name]
    lonr = xr.apply_ufunc(lambda x: np.round(x, decimals), lon)
    lonr = lonr.assign_attrs(lon.attrs)
    return ds.assign_coords({lon_name: lonr})


def drop_duplicate_1d_coord(ds: xr.Dataset, coord_name: str) -> xr.Dataset:
    """
    Drop duplicate values from a 1D coordinate axis by keeping the first occurrence.

    This is important after:
    - lon normalization (e.g., -0.0 becomes 0.0)
    - rounding (multiple near-equal values become identical)
    - concatenation (two sources might both include the same boundary column)

    Only applies when the coordinate is 1D (coord.ndim == 1). For 2D lon grids, this is a no-op.
    """
    coord = ds[coord_name].values
    if coord.ndim != 1:
        return ds
    _, idx = np.unique(coord, return_index=True)  # keep first occurrence
    if len(idx) == len(coord):
        return ds
    idx_sorted = np.sort(idx)
    return ds.isel({coord_name: idx_sorted})


def subset_lat(ds: xr.Dataset, lat_name: str, region: str) -> xr.Dataset:
    """
    Apply the region-specific latitude band filter.

    Region rules (intentionally chosen to avoid overlap):
    - north:   lat >= 20.1       (strict)
    - south:   lat <= -20.1      (strict)
    - tropics: -20.0 <= lat <= 20.0 (inclusive)

    Returns:
      Dataset filtered along the latitude coordinate.

    Notes:
    - This uses boolean selection; it does not assume latitude is sorted ascending/descending.
    - If the filter produces 0 latitude points, caller usually discards this piece.
    """
    lat = ds[lat_name]
    if region == "north":
        return ds.sel({lat_name: lat >= 20.1})
    if region == "south":
        return ds.sel({lat_name: lat <= -20.1})
    if region == "tropics":
        return ds.sel({lat_name: (lat >= -20.0) & (lat <= 20.0)})
    raise ValueError(f"unknown region: {region}")


def subset_lon_range(ds: xr.Dataset, lon_name: str, lo: int, hi: int) -> xr.Dataset:
    """
    Slice the dataset to a half-open longitude interval [lo, hi).

    Bins use [lo, hi) by design to prevent a boundary longitude column from
    being included in two adjacent bins.
    """
    lon = ds[lon_name]
    return ds.sel({lon_name: (lon >= lo) & (lon < hi)})


def open_one(path: Path) -> xr.Dataset:
    """
    Open one NetCDF file as an xarray.Dataset.

    decode_times=True:
      Parse time coordinates into datetime-like values if possible.

    mask_and_scale=True:
      Apply CF scale_factor/add_offset and mask missing values.

    chunks=None:
      Open eagerly (no dask). This keeps behavior simple and avoids introducing
      chunking-dependent behavior across environments.
    """
    return xr.open_dataset(path, decode_times=True, mask_and_scale=True, chunks=None)


def build_out_name(prefix: str, bin_tag: str, year: str, month: str) -> str:
    """
    Construct the output filename for a given prefix, bin tag, year, and month.

    Output format:
      <prefix>_<bin_tag>_<YYYY>_<MM>.nc
    """
    return f"{prefix}_{bin_tag}_{year}_{month}.nc"


def choose_bins(region: str) -> List[Tuple[int, int, str]]:
    """
    Choose longitude bins based on region.

    Tropics uses 2 bins (0–180, 180–360). Non-tropics uses 4 bins (90-degree increments).
    """
    return BINS_TROPICS if region == "tropics" else BINS_NON_TROPICS


def main() -> int:
    """
    CLI workflow.

    1) Validate inputs (root, out-root, src dirs).
    2) Scan all *.nc files under each source directory and group them by (year, month)
       extracted from filenames.
    3) For each (year, month) group:
         For each target longitude bin:
           - open each source file, normalize lon, filter lat, slice lon bin
           - collect non-empty pieces
           - concatenate pieces along longitude coordinate
           - sort + drop duplicate longitudes
           - in dry-run: print plan and close
           - in apply mode: write NetCDF

    Dry-run:
      Default behavior prints PLAN lines and writes nothing.

    Apply:
      Use --apply to write outputs.
    """
    ap = argparse.ArgumentParser(
        description="Retile ERA5-Land into lon bins and (optionally) remove overlap with tropics."
    )
    ap.add_argument("--root", required=True, help="Parent directory containing the source subdirectories.")
    ap.add_argument("--out-root", required=True, help="Output root directory.")
    ap.add_argument("--src-dirs", nargs="+", required=True, help="Source subdirectory names under --root.")
    ap.add_argument(
        "--region",
        choices=["north", "south", "tropics"],
        required=True,
        help="north keeps lat>20; south keeps lat<-20; tropics keeps -20..20.",
    )
    ap.add_argument("--apply", action="store_true", help="Actually write outputs (default is dry-run).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    ap.add_argument(
        "--out-prefix",
        default="",
        help="Output filename prefix. Default: value of --region.",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_prefix = args.out_prefix or args.region
    bins = choose_bins(args.region)


    # Validate that all requested source directories exist.
    src_paths = [root / d for d in args.src_dirs]
    missing = [p for p in src_paths if not p.exists()]
    if missing:
        print("ERROR: missing source directories:")
        for p in missing:
            print(f"  - {p}")
        return 2

    # Group all *.nc files by (year, month) parsed from filename suffix.
    by_ym: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    bad = 0
    total = 0
    for d in src_paths:
        for p in sorted(d.glob("*.nc")):
            total += 1
            try:
                ym = detect_year_month(p)
            except Exception:
                bad += 1
                continue
            by_ym[ym].append(p)

    print(f"Found {total} .nc file(s) across {len(src_paths)} directories")
    if bad:
        print(f"WARNING: {bad} file(s) skipped: missing YYYY_MM at end of filename")

    # Ensure out_root/<bin_tag>/ exists for each bin.
    for _, _, tag in bins:
        (out_root / tag).mkdir(parents=True, exist_ok=True)

    planned = 0
    written = 0
    errors = 0

    # Main loop over months and bins.
    for (year, month), paths in sorted(by_ym.items()):
        for lo, hi, tag in bins:
            out_path = (out_root / tag) / build_out_name(out_prefix, tag, year, month)
            planned += 1

            if out_path.exists() and not args.overwrite:
                print(f"SKIP (exists): {out_path}")
                continue

            pieces: List[xr.Dataset] = []
            lat_name = None
            lon_name = None

            # For this (year, month, lon-bin), open every source file that matches year-month,
            # normalize/clean coords, filter lat by region, slice lon bin, and collect the result.
            for p in paths:
                ds = None
                try:
                    ds = open_one(p)

                    # Support multiple possible coordinate names.
                    lat_name = find_coord_name(ds, ("latitude", "lat", "Latitude", "Lat"))
                    lon_name = find_coord_name(ds, ("longitude", "lon", "Longitude", "Lon"))

                    # Normalize longitude to 0..360, round to eliminate float jitter,
                    # sort, then drop duplicates.
                    ds = ds.assign_coords({lon_name: normalize_lon_0_360(ds[lon_name])})
                    ds = round_lon(ds, lon_name, decimals=6)
                    ds = ds.sortby(lon_name)
                    ds = drop_duplicate_1d_coord(ds, lon_name)

                    # Apply region latitude filter (north/south exclude tropics, tropics keeps -20..20).
                    ds = subset_lat(ds, lat_name, args.region)
                    if ds.sizes.get(lat_name, 0) == 0:
                        continue


                    # Slice longitude bin [lo, hi).
                    ds = subset_lon_range(ds, lon_name, lo, hi)
                    if ds.sizes.get(lon_name, 0) == 0:
                        continue

                    pieces.append(ds)
                    ds = None
                except Exception as e:
                    errors += 1
                    print(f"ERROR: {p} ({year}_{month}) bin {tag}: {e}")
                finally:
                    # If an exception happened after opening, ensure we close the dataset.
                    if ds is not None:
                        try:
                            ds.close()
                        except Exception:
                            pass

            # If no source files contributed data for this bin, skip output.
            if not pieces:
                continue

            # Combine all contributing pieces along longitude.
            # concat(dim=lon_name): treat each piece as occupying a longitude slice.
            # join="outer": union of coordinates
            # coords="minimal": avoid bloating coordinate variables
            # compat="override": allow variable attrs/encoding to differ across pieces
            try:
                assert lon_name is not None and lat_name is not None
                combined = xr.concat(pieces, dim=lon_name, join="outer", coords="minimal", compat="override")
                combined = combined.sortby(lon_name)
                combined = drop_duplicate_1d_coord(combined, lon_name)
            finally:
                # Close all opened piece datasets now that combined is constructed.
                for ds in pieces:
                    try:
                        ds.close()
                    except Exception:
                        pass

            if not args.apply:
                print(
                    f"PLAN: {out_path}  region={args.region} lon[{lo},{hi})  "
                    f"lon_points={combined.sizes.get(lon_name, 0)} lat_points={combined.sizes.get(lat_name, 0)}"
                )
                combined.close()
                continue

            # Write output NetCDF for this (year, month, bin).
            combined.to_netcdf(out_path)
            written += 1
            print(f"WROTE: {out_path}")
            combined.close()

    print("\nSummary")
    print(f"  Year-month groups: {len(by_ym)}")
    print(f"  Planned outputs (YM x {len(bins)} bins): {planned}")
    print(f"  Written: {written}" if args.apply else "  Dry-run only (nothing written).")
    print(f"  Errors: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
