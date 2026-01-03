#!/usr/bin/env python3

"""
compute_annual_extreme_min_tiles.py

Compute annual extreme minimum 2m air temperature (per grid cell) for each tile/year
from hourly ERA5-Land NetCDF inputs.

This script is the first “processing” step after raw downloads. It reduces hourly
temperature time series to a single annual raster per tile/year:

  annual_extreme_min(lat, lon) = min_over_time( t2m(lat, lon, hourly_times) )

Units:
- Input ERA5-Land t2m is Kelvin (K).
- Output is Celsius (degC).

New raw layout only
-------------------
This script assumes the newer raw layout that partitions by region and longitude bin:

  data/raw/era5_land_hourly_temp/
    north/
      lon000/*.nc   (e.g. north_lon000_1991_12.nc)
      lon090/*.nc
      lon180/*.nc
      lon270/*.nc
    south/
      lon000/*.nc   (e.g. south_lon000_1991_06.nc)
      ...
    tropics/
      lon000/*.nc   (e.g. tropics_lon000_1991_01.nc)
      lon180/*.nc

The filename is parsed to discover:
- region: north | south | tropics
- lon_tag: lon000 | lon090 | lon180 | lon270 (non-tropics) or lon000 | lon180 (tropics)
- year: YYYY
- month: MM

Outputs
-------
One NetCDF per (tile, year):

  data/processed/annual_extreme_min_tiles/{tile_name}_{year}.nc

where tile_name is:
  "{region}_{lon_tag}"
examples:
  north_lon090_1991.nc
  south_lon270_2007.nc
  tropics_lon180_2015.nc

Each output file contains a single 2D DataArray:
  annual_extreme_min(latitude, longitude)  [degC]

Cold-season month filtering (by region)
---------------------------------------
To approximate a “hardiness-style” annual extreme minimum based on the cold season,
we only consider certain months per region:

- north   -> DJF (Dec, Jan, Feb) = {12, 1, 2}
- south   -> JJA (Jun, Jul, Aug) = {6, 7, 8}
- tropics -> all months          = {1..12}

Important detail about “year”
-----------------------------
This script groups by the filename’s YYYY, and then filters months inside those files.
For north (DJF), the “cold season” spans Dec of the prior calendar year plus Jan/Feb.
This script does *not* automatically pull December from year-1 into the year group;
it only uses months that exist within the year group being processed.

In practice:
- If your raw layout stores north DJF months using a “season year” convention (e.g., Dec 1991
  is stored with year=1992), then grouping will match DJF naturally.
- If your raw layout uses plain calendar months (Dec 1991 file is labeled 1991_12), then
  the “1991” year group will include Dec 1991, but not Jan/Feb 1992. That still produces
  a valid annual extreme minimum, but it is not a perfect “DJF season-year” definition.

This script’s correctness does not depend on that choice; it just affects the exact
interpretation of “year” for north tiles.

Coordinate canonicalization (why it exists)
-------------------------------------------
This version canonicalizes BOTH longitude and latitude grids per tile to avoid subtle
floating-point drift from CDS / NetCDF coordinate generation that can break
`xr.align(..., join="exact")`.

Observed issue:
- Different months for the same tile can have latitude/longitude values that *should* match
  but differ in the last few bits (e.g. 20.200000000000855 vs 20.200000000000003).
- If you attempt `xr.align(..., join="exact")`, xarray will raise because coordinates differ.

Canonicalization approach:
- Normalize lon to 0..360
- Round coords to 6 decimals
- Sort coordinates
- Select exact tile bounds (lon half-open, lat inclusive endpoints)
- Snap/reindex onto a canonical 0.1° grid (step=0.1 for lat; lon step inferred if needed)
- Drop duplicates after rounding

This preserves physical coverage but makes coordinates deterministic.

Dependencies
------------
- xarray
- netcdf4 engine support (used via xr.open_dataset(..., engine="netcdf4"))
- numpy

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
import argparse
from typing import Iterable, Optional

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Tropics definition and band edge epsilon (0.1° grid).
TROPICS_MIN = -20.0
TROPICS_MAX = 20.0
BAND_EPS = 0.1  # 0.1° grid, to avoid overlap between tropics and non-tropics

# Default input/output roots. CLI can override.
HOURLY_DIR = Path("data/raw/era5_land_hourly_temp")
OUT_DIR = Path("data/processed/annual_extreme_min_tiles")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# New layout filename pattern:
#   north_lon000_1991_12.nc
#   south_lon090_2003_07.nc
#   tropics_lon180_2010_01.nc
NEW_FILENAME_RE = re.compile(
    r"^(?P<region>north|south|tropics)_(?P<lon_tag>lon\d{3})_(?P<year>\d{4})_(?P<month>\d{2})\.nc$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HourlyFileInfo:
    """
    Parsed metadata for one hourly input file.

    tile_name:
      "{region}_{lon_tag}", e.g. "south_lon090"

    region:
      "north" | "south" | "tropics"

    year/month:
      Taken from the filename; used for grouping and filtering.

    path:
      Full Path to the input NetCDF file.
    """
    tile_name: str  # e.g. "south_lon090"
    region: str     # "north"|"south"|"tropics"
    year: int
    month: int
    path: Path


def parse_hourly_filename(path: Path) -> Optional[HourlyFileInfo]:
    """
    Parse a file in the new raw layout.

    Returns:
      HourlyFileInfo if the filename matches NEW_FILENAME_RE, else None.

    This allows the discover step to skip unrelated files quietly.
    """
    m = NEW_FILENAME_RE.match(path.name)
    if not m:
        return None
    region = m.group("region").lower()
    lon_tag = m.group("lon_tag").lower()
    year = int(m.group("year"))
    month = int(m.group("month"))
    tile_name = f"{region}_{lon_tag}"
    return HourlyFileInfo(tile_name=tile_name, region=region, year=year, month=month, path=path)


def allowed_months_for_region(region: str) -> set[int]:
    """
    Cold-season month filter.

    north:   DJF (12, 1, 2)
    south:   JJA (6, 7, 8)
    tropics: all months (1..12)
    """
    region = region.lower()
    if region == "north":
        return {12, 1, 2}
    if region == "south":
        return {6, 7, 8}
    if region == "tropics":
        return set(range(1, 13))
    raise ValueError(f"unknown region: {region}")


def default_min_valid_months(region: str) -> int:
    """
    Default minimum number of valid months required to produce an annual result.

    - tropics: expect all 12 months (more complete coverage)
    - north/south: expect 3 months (DJF or JJA)
    """
    return 12 if region.lower() == "tropics" else 3


def _time_coord(ds: xr.Dataset) -> Optional[str]:
    """
    Identify the time coordinate used in the dataset.

    ERA5-Land NetCDF exports may use:
      - 'valid_time' (common in some CDS exports)
      - 'time'
    """
    for name in ("valid_time", "time"):
        if name in ds.coords:
            return name
    return None


def _pick_temp_var(ds: xr.Dataset) -> Optional[str]:
    """
    Identify the 2m temperature variable name.

    Common possibilities:
      - 't2m'          (standard ERA5-Land short name)
      - '2m_temperature' (more verbose variant)
    """
    if "t2m" in ds.variables:
        return "t2m"
    if "2m_temperature" in ds.variables:
        return "2m_temperature"
    return None


def is_valid_hourly_file(path: Path) -> bool:
    """
    Quick validity check for an hourly NetCDF file.

    Criteria:
    - exists
    - file size >= 1MB (avoid obvious truncations)
    - can be opened with netcdf4 engine
    - contains a recognized temperature variable (t2m or 2m_temperature)

    This is intentionally conservative and fast; deeper checks happen during processing.
    """
    if not path.exists():
        return False
    if path.stat().st_size < 1_000_000:
        return False
    try:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            return _pick_temp_var(ds) is not None
    except Exception:
        return False


def _normalize_lon_0_360(da: xr.DataArray) -> xr.DataArray:
    """
    Normalize longitudes to [0, 360).

    Keeps attributes.
    """
    lon = (da % 360 + 360) % 360
    lon = lon.assign_attrs(da.attrs)
    return lon


def _lon_bounds_for_tile(tile_name: str) -> tuple[float, float]:
    """
    Return longitude bounds (lo, hi) for the given tile.

    tile_name format: "<region>_<lon_tag>"
      - lon_tag is like lon000, lon090, ...

    Width:
      - tropics tiles are 180° wide (lon000, lon180)
      - north/south tiles are 90° wide  (lon000, lon090, lon180, lon270)

    These bounds are used as HALF-OPEN selection [lo, hi) for longitude.
    """
    region, lon_tag = tile_name.split("_", 1)
    lo = float(int(lon_tag[3:]))
    width = 180.0 if region == "tropics" else 90.0
    return lo, lo + width


def _lat_bounds_for_region(region: str) -> tuple[float, float]:
    """
    Return latitude bounds (lo, hi) for the given region on an ascending latitude axis.

    These are intended to match the downloader’s “no overlap” convention on a 0.1° grid:
      - north:   (20.1 .. 90.0]    -> [20.1, 90.0] in ascending form
      - tropics: [-20.0 .. 20.0]   -> [-20.0, 20.0]
      - south:   [-90.0 .. -20.1)  -> [-90.0, -20.1]

    Implementation detail:
    - BAND_EPS=0.1 means “move off the tropics boundary by one grid step”.
    - Later we select latitude INCLUSIVELY (<= hi) in canonicalization for stability across months.
    """
    region = region.lower()
    if region == "north":
        return (TROPICS_MAX + BAND_EPS, 90.0)  # 20.1 .. 90
    if region == "tropics":
        return (TROPICS_MIN, TROPICS_MAX)      # -20 .. 20
    if region == "south":
        return (-90.0, TROPICS_MIN - BAND_EPS) # -90 .. -20.1
    raise ValueError(f"unknown region: {region}")


def _infer_step(values: np.ndarray) -> Optional[float]:
    """
    Infer the coordinate step size from an array of coordinate values.

    Method:
    - sort finite values
    - compute positive diffs
    - return median diff rounded to 6 decimals

    Used primarily when canonicalizing longitude where spacing might be expected to be 0.1
    but we don’t hardcode it universally.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return None
    v = np.sort(v[np.isfinite(v)])
    if v.size < 2:
        return None
    diffs = np.diff(v)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    step = float(np.median(diffs))
    step = float(np.round(step, 6))
    return step if step > 0 else None


def _canonicalize_axis_grid(
    ds: xr.Dataset,
    name: str,
    lo: float,
    hi: float,
    *,
    step: Optional[float] = None,
    half_open: bool = True,
    round_dp: int = 6,
) -> xr.Dataset:
    """
    Canonicalize a coordinate axis onto a deterministic grid.

    Goals:
    - eliminate tiny float differences between months
    - ensure xr.align(..., join="exact") succeeds
    - prevent extra boundary columns from breaking concatenation/reduction

    Process:
    1) round the coordinate to round_dp decimals
    2) sort by coordinate
    3) select bounds:
         - half_open=True  -> keep [lo, hi)  (typical for longitude)
         - half_open=False -> keep [lo, hi]  (typical for latitude endpoints)
    4) infer or use `step`:
         - if step cannot be inferred, only drop duplicates and return
    5) build a target grid and reindex if needed:
         - if current count != expected count, reindex using nearest-with-tolerance
    6) drop duplicates after rounding/reindex

    Notes:
    - Reindex tolerance is set to 0.25*step. This is intentionally tight to prevent
      snapping a coordinate to the wrong grid line.
    - The "off-by-one" drop handles the common case where an extra coordinate is essentially
      at the boundary due to inclusive exports.
    """
    if name not in ds.coords:
        return ds

    coord = ds[name].astype(np.float64)
    coord = xr.apply_ufunc(lambda x: np.round(x, round_dp), coord)
    coord = coord.assign_attrs(ds[name].attrs)

    ds = ds.assign_coords({name: coord}).sortby(name)

    coordv = ds[name]


    # Expected grid size and target coordinate values.
    if half_open:
        ds = ds.sel({name: (coordv >= lo) & (coordv < hi)})
    else:
        ds = ds.sel({name: (coordv >= lo) & (coordv <= hi)})

    n = int(ds.sizes.get(name, 0))
    if n == 0:
        return ds

    vals = ds[name].values.astype(np.float64)
    vals.sort()

    if step is None:
        step = _infer_step(vals)
    if step is None or step <= 0:
        idx = ds[name].to_index()
        if idx.has_duplicates:
            ds = ds.isel({name: ~idx.duplicated()})
        return ds

    # Expected count
    if half_open:
        expected_n = int(round((hi - lo) / step))
        target = np.round(np.arange(lo, hi, step, dtype=np.float64), round_dp)
    else:
        expected_n = int(round((hi - lo) / step)) + 1
        target = np.round(lo + step * np.arange(expected_n, dtype=np.float64), round_dp)

    # Common off-by-one: if one extra point is essentially at the boundary, drop it
    if n == expected_n + 1:
        boundary = hi if half_open else hi
        if abs(float(vals[-1]) - float(boundary)) <= (step * 0.55):
            ds = ds.isel({name: slice(0, expected_n)})

            # Drop any duplicates created by rounding or reindexing.
            idx = ds[name].to_index()
            if idx.has_duplicates:
                ds = ds.isel({name: ~idx.duplicated()})
            return ds

    # If the axis length differs from expected, snap/reindex to the target grid.
    if n != expected_n:
        tol = step * 0.25
        ds = ds.reindex({name: target}, method="nearest", tolerance=tol)

    # Drop any duplicates
    idx = ds[name].to_index()
    if idx.has_duplicates:
        ds = ds.isel({name: ~idx.duplicated()})

    return ds


def _canonicalize_lon_grid(ds: xr.Dataset, tile_name: str, lon_name: str = "longitude") -> xr.Dataset:
    """
    Canonicalize longitude for a given tile.

    - Normalize to 0..360
    - Round to 6 decimals
    - Select to the tile’s longitude range [lo, hi) (half-open)
    - Canonicalize grid via _canonicalize_axis_grid

    Tile bounds:
      derived from tile_name ("region_lonTAG").
    """
    lo, hi = _lon_bounds_for_tile(tile_name)

    if lon_name not in ds.coords:
        return ds

    lon = _normalize_lon_0_360(ds[lon_name])
    lon = xr.apply_ufunc(lambda x: np.round(x, 6), lon)
    lon = lon.assign_attrs(ds[lon_name].attrs)
    ds = ds.assign_coords({lon_name: lon})

    # Use half-open [lo, hi)
    return _canonicalize_axis_grid(ds, lon_name, lo, hi, half_open=True, round_dp=6)


def _canonicalize_lat_grid(ds: xr.Dataset, region: str, lat_name: str = "latitude") -> xr.Dataset:
    """
    Canonicalize latitude per region.

    Why:
    - Different months can have tiny coordinate drift even when they represent the same grid.
    - Canonicalizing prevents xr.align(join="exact") failures during running_min aggregation.

    Behavior:
    - Use inclusive endpoints for latitude (half_open=False).
    - Force step=0.1 explicitly (ERA5-Land lat grid spacing).
    - Snap values to canonical grid within tolerance, while preserving the intended band.
    """
    if lat_name not in ds.coords:
        return ds

    lo, hi = _lat_bounds_for_region(region)

    # On ERA5-Land, latitude spacing is 0.1. Force step=0.1 explicitly.
    return _canonicalize_axis_grid(
        ds,
        lat_name,
        lo,
        hi,
        step=0.1,
        half_open=False,   # inclusive endpoints for latitude
        round_dp=6,
    )


def discover_tile_years(hourly_dir: Path = HOURLY_DIR) -> dict[tuple[str, int], list[HourlyFileInfo]]:
    """
    Walk the raw directory tree and group hourly files by (tile_name, year).

    Returns:
      dict mapping (tile_name, year) -> [HourlyFileInfo, ...]

    This is driven entirely by filenames; files that do not match NEW_FILENAME_RE
    are ignored (debug logged).
    """
    groups: dict[tuple[str, int], list[HourlyFileInfo]] = {}

    if not hourly_dir.exists():
        raise FileNotFoundError(hourly_dir)

    for region_dir in sorted(p for p in hourly_dir.iterdir() if p.is_dir()):
        region = region_dir.name.lower()
        if region not in ("north", "south", "tropics"):
            continue

        for lon_dir in sorted(p for p in region_dir.iterdir() if p.is_dir()):
            for f in sorted(lon_dir.glob("*.nc")):
                parsed = parse_hourly_filename(f)
                if not parsed:
                    log.debug("Skipping unsupported file: %s", f)
                    continue
                groups.setdefault((parsed.tile_name, parsed.year), []).append(parsed)

    return groups


def compute_annual_extreme_min(
    tile_name: str,
    year: int,
    files: Iterable[HourlyFileInfo],
    min_valid_months: Optional[int] = None,
    out_dir: Path = OUT_DIR,
) -> None:
    """
    Compute annual extreme minimum temperature for one (tile_name, year) group.

    Pipeline steps:
    1) Filter to allowed months for region (DJF/JJA/all months).
    2) Validate each file quickly (existence, size, openable, contains t2m).
    3) Require at least `min_valid_months` distinct months present (default: 3 or 12).
    4) For each usable file:
         - open dataset
         - canonicalize latitude and longitude coordinates
         - find time coord and temp variable
         - compute monthly minimum over time (degC)
         - merge into running minimum via elementwise fmin after exact alignment
    5) Write the resulting 2D array to out_dir/{tile_name}_{year}.nc

    Output variable:
      annual_extreme_min  (degC)

    Notes:
    - Canonicalization happens before reduction so that the reduced arrays align exactly.
    - The code does an extra month mask using ds[time].dt.month, which is usually redundant
      because each file is a single month, but it protects against inputs that contain
      multiple months or unexpected time encoding.
    """
    files = list(files)
    if not files:
        return

    region = files[0].region.lower()
    allowed_months = allowed_months_for_region(region)
    required_months = min_valid_months if min_valid_months is not None else default_min_valid_months(region)

    out_path = out_dir / f"{tile_name}_{year}.nc"
    if out_path.exists():
        log.info("[SKIP] %s already exists", out_path.name)
        return

    good: list[HourlyFileInfo] = []
    bad: list[HourlyFileInfo] = []

    # File-level filtering and validation.
    for info in files:
        if info.month not in allowed_months:
            continue
        if is_valid_hourly_file(info.path):
            good.append(info)
        else:
            bad.append(info)

    if bad:
        log.warning("[WARN] %s %d: %d corrupt files", tile_name, year, len(bad))
        for b in bad:
            log.warning("       %s", b.path.name)

    if not good:
        log.info("[SKIP] %s %d: no usable monthly files after filtering", tile_name, year)
        return

    # Require enough distinct months to consider the year complete enough.
    seen_months: set[int] = {i.month for i in good}
    if len(seen_months) < required_months:
        log.info(
            "[SKIP] %s %d: only %d valid months (%s), need %d",
            tile_name,
            year,
            len(seen_months),
            " ".join(f"{m:02d}" for m in sorted(seen_months)),
            required_months,
        )
        return

    running_min: Optional[xr.DataArray] = None
    shapes: set[tuple[int, int]] = set()

    # Reduce month files into annual running minimum.
    for info in sorted(good, key=lambda i: (i.month, i.path.name)):
        ds = xr.open_dataset(info.path, engine="netcdf4")
        try:
            # Canonicalize coords before reducing across time.
            if "latitude" in ds.coords:
                ds = ds.sortby("latitude")
                ds = _canonicalize_lat_grid(ds, region, lat_name="latitude")
                if ds.sizes.get("latitude", 0) == 0:
                    continue

            if "longitude" in ds.coords:
                ds = ds.sortby("longitude")
                ds = _canonicalize_lon_grid(ds, tile_name, lon_name="longitude")
                if ds.sizes.get("longitude", 0) == 0:
                    continue

            tname = _time_coord(ds)
            vname = _pick_temp_var(ds)
            if not tname or not vname:
                continue

            # Safety month filter based on decoded time coord (if supported).
            try:
                month_mask = ds[tname].dt.month.isin(sorted(allowed_months))
            except Exception:
                month_mask = None

            ds2 = ds.where(month_mask, drop=True) if month_mask is not None else ds
            if ds2.sizes.get(tname, 0) == 0:
                continue

            # Convert Kelvin to Celsius, then take min over time for this file/month
            arr = (ds2[vname] - 273.15).min(dim=tname, skipna=True).load()

            shapes.add((int(arr["latitude"].size), int(arr["longitude"].size)))

            if running_min is None:
                running_min = arr
            else:
                # After canonicalization this should succeed; join="exact" is intentional.
                a, b = xr.align(running_min, arr, join="exact")
                running_min = xr.apply_ufunc(np.fmin, a, b).load()
        finally:
            ds.close()

    if running_min is None:
        log.info("[SKIP] %s %d: no usable data after reduction", tile_name, year)
        return

    # Warn if something still caused different shapes after canonicalization.
    if len(shapes) > 1:
        log.warning("[WARN] %s %d: inconsistent grid shapes %s", tile_name, year, sorted(shapes))

    months_used = sorted(seen_months)
    log.info(
        "[INFO] %s %d: using %d month(s) (%s) across %d file(s)",
        tile_name,
        year,
        len(months_used),
        " ".join(f"{m:02d}" for m in months_used),
        len(good),
    )

    # Annotate output with metadata to preserve provenance/debugging context.
    running_min.name = "annual_extreme_min"
    running_min.attrs.update(
        {
            "long_name": "Annual extreme minimum 2m temperature",
            "units": "degC",
            "year": year,
            "source": "ERA5-Land hourly (raw)",
            "region": region,
            "cold_season_definition": (
                f"NH (>={TROPICS_MAX:+.0f}): DJF | "
                f"SH (<={TROPICS_MIN:+.0f}): JJA | "
                "Tropics: all months"
            ),
            "cold_season_latitude_threshold_deg": TROPICS_MAX,
            "minimum_valid_months": required_months,
            "months_used": " ".join(f"{m:02d}" for m in months_used),
            "files_used": len(good),
            "coord_canonicalization": "latitude+longitude rounded/reindexed to canonical 0.1° tile grids",
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    running_min.to_netcdf(out_path)
    log.info("[OK] Wrote %s", out_path)


def main(argv: Optional[list[str]] = None) -> None:
    """
    CLI entrypoint.

    Discovers tile-year groups under --hourly-dir, then computes annual extreme minimum
    for each group into --out-dir.

    CLI options:
      --hourly-dir        Root of raw hourly layout (default: data/raw/era5_land_hourly_temp)
      --out-dir           Output directory for annual tiles (default: data/processed/annual_extreme_min_tiles)
      --min-valid-months  Override minimum distinct months required per tile-year.
                          If omitted: 3 for north/south, 12 for tropics.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly-dir", type=Path, default=HOURLY_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--min-valid-months", type=int, default=None)
    args = parser.parse_args(argv)

    groups = discover_tile_years(args.hourly_dir)
    log.info("[INFO] Discovered %d tile-year groups", len(groups))

    for (tile_name, year), files in sorted(groups.items()):
        compute_annual_extreme_min(
            tile_name,
            year,
            files,
            min_valid_months=args.min_valid_months,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
