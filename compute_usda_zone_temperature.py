#!/usr/bin/env python3
"""
compute_usda_zone_temperature_tiles.py

Compute a climatology (multi-year mean) USDA-style “zone temperature” per tile from
annual extreme minimum temperature rasters produced in the prior step.

Conceptually:
1) Prior step produced per-tile/per-year rasters:
     annual_extreme_min(lat, lon) = minimum hourly 2m temperature in that year/season
   stored as degC in:
     data/processed/annual_extreme_min_tiles/{tile}_{YYYY}.nc

2) This script aggregates a climatology period:
     mean_c(lat, lon) = mean over years of annual_extreme_min(lat, lon)
   and derives USDA zone number + subzone from the mean in °F.

Inputs
------
Expected input files (Step 2 outputs):

  data/processed/annual_extreme_min_tiles/<tile>_<YYYY>.nc

Where <tile> is like:
  north_lon000 / north_lon090 / north_lon180 / north_lon270  (90° wide)
  south_lon000 / south_lon090 / south_lon180 / south_lon270  (90° wide)
  tropics_lon000 / tropics_lon180                            (180° wide)

Each input file must contain:
  annual_extreme_min(latitude, longitude)  [degC]

Outputs
-------
Writes one NetCDF per tile for the requested climatology window:

  data/processed/usda_zone_temperature_tiles/<tile>_mean_<start>_<end>.nc

Variables written:
- usda_zone_temp_c               (degC) mean annual extreme minimum temperature over years
- usda_zone_temp_f               (degF) derived from usda_zone_temp_c
- usda_zone_num                  (int16) USDA zone number in [1..13], 0=missing
- usda_zone_subzone              (int8)  0=a, 1=b, -1=missing
- n_years                        (int16) count of contributing (non-null) years per cell
- stdev_annual_extreme_min       (degC) standard deviation across years

Processing modes
----------------
- --tile <name>      : compute for one tile
- --all-tiles         : discover tiles by scanning ANNUAL_DIR and compute each tile

Coordinate alignment / canonicalization
---------------------------------------
Annual tiles may have tiny coord differences across years (or be off by one extra
boundary column). This script:
- sorts lat/lon
- normalizes lon to [0, 360)
- rounds coords to 6 decimals
- drops duplicate coord values
- enforces lat/lon bounds per tile
- drops an inclusive endpoint column if present (n == expected + 1)

If exact alignment still fails, it will reindex each year onto the reference year’s
lat/lon coordinates using nearest with a tight tolerance (0.25*grid step), then
require join="exact".

Hardiness zone math
-------------------
Zones are computed from mean extreme-min temperature in °F using 10°F bands starting
at -60°F:

  zone_num = floor((temp_f + 60) / 10) + 1

Then clamped to [1, 13] for this project.
(Cells with no contributing years are zone_num = 0.)

Subzone is the warmer/colder half of the zone:
- boundary is (zone_lower + 5°F)
- exact boundary goes to 'b' (warmer half)

  subzone = 1 (b) if temp_f >= zone_lower + 5
            0 (a) otherwise
Missing cells use -1.

Optional Hartford check
-----------------------
If --check-hartford is used, the script will:
- identify whether Hartford’s point (lat=41.77, lon=287.32 in 0..360) lies in the tile
- print the nearest usda_zone_temp_f value for Hartford

Note on latitude bounds
-----------------------
This script’s tile latitude bounds are currently “approximate” (see _tile_lat_bounds):
- tropics: [-20, 20]
- north:   [20, 90]
- south:   [-90, -20]

If your latest pipeline definition uses non-overlapping bands:
- north:   20.1 .. 90
- tropics: -20 .. 20
- south:   -90 .. -20.1

then _tile_lat_bounds should be updated accordingly (20.1 and -20.1) to exactly match
the downloader/annual tile canonicalization. Otherwise you can get small overlap or
mismatch at the band boundary.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import warnings

import xarray as xr
import numpy as np

# Input directory containing annual tile/year outputs from the prior step.
ANNUAL_DIR = Path("data/processed/annual_extreme_min_tiles")
# Output directory for climatology zone-temperature tiles.

OUT_DIR = Path("data/processed/usda_zone_temperature_tiles")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional sanity-check point (Hartford, CT) expressed in 0..360 longitude domain.
HARTFORD_LAT = 41.77
HARTFORD_LON_360 = 360.0 - 72.68  # 287.32


def discover_tiles_from_annual_dir() -> list[str]:
    """
    Discover tile names by scanning ANNUAL_DIR for files named:
      <tile>_<YYYY>.nc

    Returns:
      Sorted list of unique <tile> strings.

    Example:
      annual_extreme_min_tiles/north_lon090_1991.nc -> tile "north_lon090"
    """
    tiles: set[str] = set()
    for p in ANNUAL_DIR.glob("*.nc"):
        stem = p.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        try:
            int(parts[-1])  # last token must be a year
        except ValueError:
            continue
        tile = "_".join(parts[:-1])
        tiles.add(tile)
    return sorted(tiles)


def _open_annual_file(path: Path) -> xr.DataArray:
    """
    Open a single annual tile/year file and return the DataArray.

    Requirements:
    - variable name must be 'annual_extreme_min'
    - dimensions must include ('latitude', 'longitude')
    """
    ds = xr.open_dataset(path)

    if "annual_extreme_min" not in ds:
        raise RuntimeError(f"{path}: missing variable 'annual_extreme_min'")

    da = ds["annual_extreme_min"]

    for dim in ("latitude", "longitude"):
        if dim not in da.dims:
            raise RuntimeError(f"{path}: expected '{dim}' dimension not found")

    return da


def _normalize_lon_0_360(lon: xr.DataArray) -> xr.DataArray:
    """
    Normalize longitude values into [0, 360) and round to 6 decimals.

    This makes 0..360 tiles consistent even if upstream data used -180..180 or had float noise.
    """
    v = (lon % 360 + 360) % 360
    return xr.apply_ufunc(lambda x: np.round(x, 6), v)


def _tile_lon_bounds(tile_name: str) -> tuple[float, float, float]:
    """
    Return (lo, hi, width) for a tile’s longitude span in the 0..360 domain.

    Tile names:
      - tropics_lon000, tropics_lon180 have width=180
      - north/south lon000/lon090/lon180/lon270 have width=90

    These bounds are applied as half-open [lo, hi) when subsetting longitude.
    """
    region, lon_tag = tile_name.split("_", 1)
    lo = float(int(lon_tag[3:]))
    width = 180.0 if region.lower() == "tropics" else 90.0
    return lo, lo + width, width

TROPICS_MIN = -20.0
TROPICS_MAX = 20.0
BAND_EPS = 0.1  # 0.1° grid

def _tile_lat_bounds(tile_name: str) -> tuple[float, float]:
    """
    Latitude bounds used to subset the tile.

    Non-overlapping bands (match the downloader + annual-min scripts):
      - north:   [20.1, 90.0]
      - tropics: [-20.0, 20.0]
      - south:   [-90.0, -20.1]
    """
    region = tile_name.split("_", 1)[0].lower()
    if region == "tropics":
        return TROPICS_MIN, TROPICS_MAX
    if region == "north":
        return TROPICS_MAX + BAND_EPS, 90.0
    if region == "south":
        return -90.0, TROPICS_MIN - BAND_EPS
    return -90.0, 90.0


def _drop_duplicate_coords(da: xr.DataArray, dim: str) -> xr.DataArray:
    """
    If coord values in the given dim contain duplicates (after rounding/normalization),
    keep the first occurrence and drop the rest.
    """
    idx = da[dim].to_index()
    if idx.has_duplicates:
        mask = ~idx.duplicated()
        da = da.isel({dim: mask})
    return da


def _canonicalize_grid(da: xr.DataArray, tile_name: str) -> xr.DataArray:
    """
    Make lat/lon grids deterministic across years for the same tile.

    Steps:
    - latitude:
        * sort
        * round to 6 decimals
        * drop duplicates
        * subset to tile latitude bounds (inclusive endpoints)
    - longitude:
        * normalize to [0,360)
        * round to 6 decimals (via _normalize_lon_0_360)
        * sort
        * drop duplicates
        * subset to [lo, hi) (half-open)
        * detect and drop an extra inclusive endpoint column if present
          (common: one extra column due to CDS inclusive export)
    """
    # Latitude canonicalization
    da = da.sortby("latitude")
    da = da.assign_coords(latitude=xr.apply_ufunc(lambda x: np.round(x, 6), da["latitude"]))
    da = _drop_duplicate_coords(da, "latitude")

    lat_lo, lat_hi = _tile_lat_bounds(tile_name)
    da = da.sel(latitude=(da["latitude"] >= lat_lo) & (da["latitude"] <= lat_hi))

    # Longitude canonicalization
    da = da.assign_coords(longitude=_normalize_lon_0_360(da["longitude"]))
    da = da.sortby("longitude")
    da = _drop_duplicate_coords(da, "longitude")

    lo, hi, width = _tile_lon_bounds(tile_name)

    lonv = da["longitude"]
    da = da.sel(longitude=(lonv >= lo) & (lonv < hi))
    if da.sizes.get("longitude", 0) == 0:
        return da


    # Extra endpoint-column drop heuristic:
    # If n == expected + 1, drop last column.
    lon_vals = da["longitude"].values.astype(np.float64)
    lon_vals.sort()
    diffs = np.diff(lon_vals)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size:
        step = float(np.round(np.median(diffs), 6))
        if step > 0:
            expected = int(round(width / step))
            n = int(da.sizes["longitude"])
            if n == expected + 1:
                da = da.isel(longitude=slice(0, expected))

    return da


def _hartford_in_tile(tile_name: str) -> bool:
    """
    Check whether the Hartford point lies within this tile’s nominal bounds.
    Longitude uses the [lo, hi) half-open test.
    """
    lat_lo, lat_hi = _tile_lat_bounds(tile_name)
    lo, hi, _ = _tile_lon_bounds(tile_name)
    return (lat_lo <= HARTFORD_LAT <= lat_hi) and (lo <= HARTFORD_LON_360 < hi)


def _infer_step(coord: xr.DataArray) -> float:
    """
    Infer median coordinate spacing from a 1D coordinate array.
    Returns 0.0 if spacing cannot be inferred.
    """
    vals = coord.values.astype(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return 0.0
    vals.sort()
    diffs = np.diff(vals)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _reindex_to_ref_with_tol(da: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """
    Snap da onto ref’s latitude and longitude coordinates using nearest-neighbor
    reindexing with tight tolerances, then require exact coordinate match.

    Tolerances are 0.25 * (median step) for each axis, falling back to 1e-6 if step is 0.

    This is the fallback when xr.align(join="exact") fails due to tiny float differences
    that weren’t fully resolved by _canonicalize_grid.
    """
    lat_step = _infer_step(ref["latitude"])
    lon_step = _infer_step(ref["longitude"])

    lat_tol = float(lat_step * 0.25) if lat_step > 0 else 1e-6
    lon_tol = float(lon_step * 0.25) if lon_step > 0 else 1e-6

    da = da.reindex(latitude=ref["latitude"], method="nearest", tolerance=lat_tol)
    da = da.reindex(longitude=ref["longitude"], method="nearest", tolerance=lon_tol)

    da, _ = xr.align(da, ref, join="exact")
    return da


def _compute_zone_num_from_temp_f(temp_f: xr.DataArray) -> xr.DataArray:
    """
    Compute USDA zone number (float) from mean extreme-min temperature in °F.

    Standard USDA binning:
      zone 1 begins at [-60, -50)
      zone 2 begins at [-50, -40)
      ...
    Formula:
      zone = floor((temp_f + 60)/10) + 1

    Then clamp to [1, 13] for this project.

    Missing is handled separately; this function assumes temp_f is valid.
    """
    z = np.floor((temp_f + 60.0) / 10.0) + 1.0
    z = z.clip(1.0, 13.0)
    return z


def _compute_subzone_from_temp_f_and_zone(temp_f: xr.DataArray, zone_num: xr.DataArray) -> xr.DataArray:
    """
    Compute subzone within each zone:
      'a' = colder half (0)
      'b' = warmer half (1)

    Boundary:
      zone_lower + 5°F
    Exact boundary goes to 'b'.
    """
    zone_lower = -60.0 + 10.0 * (zone_num - 1.0)
    return xr.where(temp_f >= (zone_lower + 5.0), 1, 0)


def compute_usda_zone_temperature(
    tile_name: str,
    start_year: int,
    end_year: int,
    check_hartford: bool = False,
) -> None:
    """
    Compute a climatology for one tile over [start_year, end_year].

    For each year y in the range:
      - load annual_extreme_min from ANNUAL_DIR/{tile}_{y}.nc
      - canonicalize coords to stable per-tile grid
      - align to reference grid (exact if possible; else reindex with tolerance)

    Then:
      - stack into a 3D array tmin(year, lat, lon)
      - compute:
          mean (degC) + derived mean (degF)
          n_years per cell (non-null count)
          stdev across years
          zone number and subzone from mean °F

    Writes:
      OUT_DIR/{tile}_mean_{start_year}_{end_year}.nc

    If check_hartford=True:
      prints a Hartford nearest-point mean °F value when Hartford lies in this tile.
    """
    yearly: list[xr.DataArray] = []
    years_found: list[int] = []

    ref: xr.DataArray | None = None

    for y in range(start_year, end_year + 1):
        p = ANNUAL_DIR / f"{tile_name}_{y}.nc"
        if not p.exists():
            continue

        da = _open_annual_file(p)
        da = _canonicalize_grid(da, tile_name)

        if da.sizes.get("latitude", 0) == 0 or da.sizes.get("longitude", 0) == 0:
            continue

        if ref is None:
            ref = da
        else:
            try:
                da, _ = xr.align(da, ref, join="exact")
            except Exception:
                da = _reindex_to_ref_with_tol(da, ref)

        yearly.append(da)
        years_found.append(y)

    if not yearly:
        raise FileNotFoundError(
            f"No usable annual files found for tile={tile_name} in {ANNUAL_DIR} "
            f"for years {start_year}..{end_year}. Expected files like {tile_name}_1991.nc"
        )

    # Combine yearly rasters along a new "year" dimension.
    tmin = xr.concat(
        yearly,
        dim=xr.IndexVariable("year", np.array(years_found, dtype=np.int32)),
        join="exact",
    )


    # Count contributing years per cell.
    n_years = tmin.notnull().sum(dim="year").astype(np.int16)

    # Mean in °C and °F.
    mean_c = tmin.mean(dim="year", skipna=True)
    mean_f = mean_c * 9 / 5 + 32

    # Standard deviation across years; mask cells with no contributing years.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice", category=RuntimeWarning)
        stdev_c = tmin.std(dim="year", skipna=True)

    stdev_c = stdev_c.where(n_years > 0)

    # Compute zone + subzone everywhere temp is valid (n_years>0).
    valid = n_years > 0

    zone_num_f = _compute_zone_num_from_temp_f(mean_f)
    zone_num = xr.where(valid, zone_num_f, 0).astype(np.int16)
    zone_num.name = "usda_zone_num"

    subzone_f = _compute_subzone_from_temp_f_and_zone(mean_f, zone_num_f)
    subzone = xr.where(valid, subzone_f, -1).astype(np.int8)
    subzone.name = "usda_zone_subzone"

    # Sanity: subzone should be non-missing wherever valid
    missing_sub = int(((valid) & (subzone < 0)).sum().values)
    if missing_sub:
        print(f"[WARN] {tile_name}: {missing_sub} valid cells still have missing subzone (-1)")

    # Assemble output dataset.
    out = mean_c.to_dataset(name="usda_zone_temp_c")
    out["usda_zone_temp_f"] = mean_f
    out["usda_zone_num"] = zone_num
    out["usda_zone_subzone"] = subzone
    out["n_years"] = n_years
    out["stdev_annual_extreme_min"] = stdev_c

    # Attach metadata for provenance.
    out.usda_zone_temp_c.attrs.update(
        long_name="Mean annual extreme minimum temperature (climatology)",
        units="degC",
        standard_name="annual_extreme_minimum_temperature",
        source="ERA5-Land hourly",
        method="Mean over years of annual minimum hourly 2m temperatures",
        period=f"{start_year}-{end_year}",
        years_used=f"{years_found[0]}-{years_found[-1]} (n={len(years_found)})",
    )

    out.usda_zone_temp_f.attrs.update(
        long_name="Mean annual extreme minimum temperature (climatology)",
        units="degF",
        source="Derived from usda_zone_temp_c",
        period=f"{start_year}-{end_year}",
    )

    out.usda_zone_num.attrs.update(
        long_name="USDA hardiness zone number",
        units="1",
        description="Computed from usda_zone_temp_f using 10°F bins from -60°F; clamped to [1,13]; 0=missing",
    )

    out.usda_zone_subzone.attrs.update(
        long_name="USDA hardiness subzone (0=a, 1=b)",
        units="1",
        description="0=a colder half, 1=b warmer half; -1=missing",
        values="0=a, 1=b, -1=missing",
    )

    out.n_years.attrs.update(
        long_name="Count of contributing years per grid cell",
        units="1",
        period=f"{start_year}-{end_year}",
    )

    out.stdev_annual_extreme_min.attrs.update(
        long_name="Standard deviation across years of annual extreme minimum temperature",
        units="degC",
        period=f"{start_year}-{end_year}",
    )

    out_path = OUT_DIR / f"{tile_name}_mean_{start_year}_{end_year}.nc"
    out.to_netcdf(out_path)

    print(f"[OK] Wrote {out_path} (years found: {len(years_found)}/{end_year - start_year + 1})")

    # Optional Hartford check.
    if check_hartford:
        if _hartford_in_tile(tile_name):
            hartford = out.usda_zone_temp_f.sel(
                latitude=HARTFORD_LAT,
                longitude=HARTFORD_LON_360,
                method="nearest",
            )
            print(f"Hartford USDA zone temperature: {float(hartford):.2f} °F")
        else:
            print(f"[INFO] {tile_name}: Hartford not in tile bounds; skipping check")


def _parse_args() -> argparse.Namespace:
    """
    CLI args:
      --tile <tile>     Process one tile
      --all-tiles       Process all tiles discovered in ANNUAL_DIR
      --start-year      Start of climatology window (inclusive)
      --end-year        End of climatology window (inclusive)
      --check-hartford  Print Hartford value when applicable
    """
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--tile", help="Single tile name, e.g. north_lon270")
    g.add_argument("--all-tiles", action="store_true", help="Process all tiles found in Step 2 outputs")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--check-hartford", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.all_tiles:
        # Scan for available tiles based on annual files.
        tiles = discover_tiles_from_annual_dir()
        if not tiles:
            raise SystemExit(f"No annual files found in {ANNUAL_DIR}")

        did_any = False
        for tile in tiles:
            out_path = OUT_DIR / f"{tile}_mean_{args.start_year}_{args.end_year}.nc"
            # If output exists, optionally just do Hartford reporting and skip recompute.
            if out_path.exists():
                if args.check_hartford and _hartford_in_tile(tile):
                    ds = xr.open_dataset(out_path)
                    try:
                        hartford = ds["usda_zone_temp_f"].sel(
                            latitude=HARTFORD_LAT,
                            longitude=HARTFORD_LON_360,
                            method="nearest",
                        )
                        print(f"Hartford USDA zone temperature: {float(hartford):.2f} °F (from {out_path.name})")
                        did_any = True
                    finally:
                        ds.close()
                continue

            compute_usda_zone_temperature(
                tile_name=tile,
                start_year=args.start_year,
                end_year=args.end_year,
                check_hartford=args.check_hartford,
            )
            did_any = True

        if not did_any:
            print("[INFO] All outputs already existed; nothing to compute.")
