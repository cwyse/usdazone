#!/usr/bin/env python3
from __future__ import annotations

"""
make_raw_geojson.py

Create GeoJSON tile polygons for each “leaf” directory under a raw ERA5-Land download tree.

Context
-------
Your raw hourly ERA5-Land layout is organized as:

  <root>/
    <region>/
      <londir>/
        <many_month_files>.nc

Example:
  data/raw/era5_land_hourly_temp/north/lon090/north_lon090_1991_12.nc

Each leaf directory (e.g., north/lon090) contains many NetCDF files with identical spatial
extent (lat/lon grid), differing only in time coverage. This script:
- selects one representative .nc file in each leaf directory,
- reads its latitude/longitude coordinate arrays,
- infers the min/max bounds,
- writes a GeoJSON polygon for that tile.

Outputs
-------
Per-leaf GeoJSON files:
  <out-dir>/<region>_<londir>.geojson

Combined FeatureCollection:
  <out-dir>/all_tiles.geojson

Default values match your repo layout:
  --root    data/raw/era5_land_hourly_temp
  --out-dir data/processed/geojson/era5_land_hourly_temp

Intended use
------------
- Visualization / QA: quickly inspect tile coverage in a GIS viewer (QGIS, geojson.io, etc.).
- Debugging: confirm expected bounds for each tile directory match your lat-band/lon-bin logic.
- Metadata: produce a lightweight index (“tile_id -> polygon”) without opening all NetCDF files.

Assumptions
-----------
- Leaf directory heuristic: any directory under --root that contains at least one *.nc file.
- Within each leaf directory, *any one* .nc file is sufficient to infer bounds.
- Latitude and longitude coordinates are present as 1D arrays named either:
    - latitude / longitude (preferred)
    - lat / lon           (alternate)

Coordinate conventions
----------------------
This script does not attempt to:
- normalize longitude to 0..360 vs -180..180
- handle dateline wrapping polygons
- infer true geodesic bounds

It simply uses:
  swlat = min(latitude)
  nelat = max(latitude)
  swlon = min(longitude)
  nelon = max(longitude)

and produces an axis-aligned bounding-box polygon (rectangle) in lon/lat.

Dependencies
------------
- Prefers `netCDF4` for fast, low-overhead coordinate reads.
- Falls back to `xarray` if netCDF4 is not installed.
- If neither is installed, it fails with an explicit error.

"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# Prefer netCDF4 for speed/low overhead; fall back to xarray if needed.
try:
    import netCDF4  # type: ignore
    HAVE_NETCDF4 = True
except Exception:
    HAVE_NETCDF4 = False

try:
    import xarray as xr  # type: ignore
    HAVE_XARRAY = True
except Exception:
    HAVE_XARRAY = False


@dataclass(frozen=True)
class TileFeature:
    """
    Metadata for one tile (one leaf directory).

    tile_id:
      Derived identifier of the form "<region>_<londir>", e.g. "north_lon090".

    region / londir:
      Parsed from the leaf directory path relative to --root. Expected to be:
        <root>/<region>/<londir>/...

    sample_file:
      Relative path (from --root) to the .nc file used to compute bounds.

    swlat / swlon / nelat / nelon:
      Axis-aligned bounding box corners:
        SW = (min_lat, min_lon)
        NE = (max_lat, max_lon)
    """
    tile_id: str
    region: str
    londir: str
    sample_file: str
    swlat: float
    swlon: float
    nelat: float
    nelon: float


def _find_coord_var(var_names: set[str], preferred: tuple[str, ...]) -> Optional[str]:
    """
    Find a coordinate variable name in a NetCDF variable-name set.

    preferred:
      Names to try first, e.g. ("latitude","lat").

    Behavior:
    - First checks exact matches against `preferred`.
    - Then does a case-insensitive match against all variables.

    Returns:
      The matching variable name (as it appears in the file) or None if not found.
    """
    for k in preferred:
        if k in var_names:
            return k
    # common alternatives (case-insensitive)
    for k in var_names:
        lk = k.lower()
        if lk in [p.lower() for p in preferred]:
            return k
    return None


def _isfinite(x: float) -> bool:
    """True if x is a finite float (not NaN/inf)."""
    return math.isfinite(x)


def bounds_from_netcdf4(path: Path) -> Tuple[float, float, float, float]:
    """
    Compute (swlat, swlon, nelat, nelon) using netCDF4.Dataset.

    This is the preferred path when netCDF4 is installed:
    - minimal overhead
    - reads coord arrays directly

    Raises:
      KeyError if latitude/longitude variables are missing.
      ValueError if any computed bounds are non-finite.
    """
    with netCDF4.Dataset(str(path), "r") as ds:
        names = set(ds.variables.keys())
        lat_name = _find_coord_var(names, ("latitude", "lat"))
        lon_name = _find_coord_var(names, ("longitude", "lon"))
        if not lat_name or not lon_name:
            raise KeyError(f"missing lat/lon vars (vars: {sorted(list(names))[:30]})")

        lat = ds.variables[lat_name][:]
        lon = ds.variables[lon_name][:]

        swlat = float(lat.min())
        nelat = float(lat.max())
        swlon = float(lon.min())
        nelon = float(lon.max())

        if not all(_isfinite(v) for v in (swlat, nelat, swlon, nelon)):
            raise ValueError("non-finite bounds")

        return swlat, swlon, nelat, nelon


def bounds_from_xarray(path: Path) -> Tuple[float, float, float, float]:
    """
    Compute (swlat, swlon, nelat, nelon) using xarray.

    This is a fallback when netCDF4 is not installed.
    decode_times=False is used because we only care about coordinate arrays and
    want to avoid unnecessary time decoding overhead.
    """
    ds = xr.open_dataset(path, decode_times=False)
    try:
        coords = set(ds.coords) | set(ds.variables)
        lat_name = None
        lon_name = None
        for k in ("latitude", "lat"):
            if k in coords:
                lat_name = k
                break
        for k in ("longitude", "lon"):
            if k in coords:
                lon_name = k
                break
        if not lat_name or not lon_name:
            raise KeyError(f"missing lat/lon (coords: {sorted(list(ds.coords))})")

        lat = ds[lat_name]
        lon = ds[lon_name]
        swlat = float(lat.min().values)
        nelat = float(lat.max().values)
        swlon = float(lon.min().values)
        nelon = float(lon.max().values)

        # NOTE: the original code checks (swlat, nelat, swlon, nelat).
        # That is a harmless redundancy in most cases because nelat is checked twice.
        # We are not changing it here to avoid behavior changes.
        if not all(_isfinite(v) for v in (swlat, nelat, swlon, nelat)):
            raise ValueError("non-finite bounds")

        return swlat, swlon, nelat, nelon
    finally:
        ds.close()


def get_bounds(path: Path) -> Tuple[float, float, float, float]:
    """
    Choose the coordinate-reading backend based on installed dependencies.

    Priority:
      1) netCDF4
      2) xarray

    Raises:
      RuntimeError if neither dependency is installed.
    """
    if HAVE_NETCDF4:
        return bounds_from_netcdf4(path)
    if HAVE_XARRAY:
        return bounds_from_xarray(path)
    raise RuntimeError("Need netCDF4 or xarray installed.")


def polygon_from_bounds(swlat: float, swlon: float, nelat: float, nelon: float) -> dict:
    """
    Build a GeoJSON Polygon from min/max bounds.

    GeoJSON polygons use coordinates in [lon, lat] order and the ring must be closed.
    This produces an axis-aligned rectangle:
      SW -> SE -> NE -> NW -> SW
    """
    ring = [
        [swlon, swlat],
        [nelon, swlat],
        [nelon, nelat],
        [swlon, nelat],
        [swlon, swlat],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def feature_to_geojson(tile: TileFeature) -> dict:
    """
    Convert TileFeature metadata into a GeoJSON Feature with:
    - properties: human-readable metadata (tile_id, region, sample_file, bounds)
    - geometry: rectangle polygon derived from bounds
    """
    return {
        "type": "Feature",
        "properties": {
            "tile_id": tile.tile_id,
            "region": tile.region,
            "londir": tile.londir,
            "sample_file": tile.sample_file,
            "swlat": tile.swlat,
            "swlon": tile.swlon,
            "nelat": tile.nelat,
            "nelon": tile.nelon,
        },
        "geometry": polygon_from_bounds(tile.swlat, tile.swlon, tile.nelat, tile.nelon),
    }


def pick_sample_nc(dir_path: Path) -> Optional[Path]:
    """
    Pick one NetCDF file from a leaf directory to infer spatial bounds.

    Determinism:
      Uses lexical sort and picks the first file. Any file in the directory should have the
      same lat/lon grid, so choice does not matter for correctness (given the assumption).
    """
    ncs = sorted(dir_path.glob("*.nc"))
    return ncs[0] if ncs else None


def is_leaf_tile_dir(p: Path, root: Path) -> bool:
    """
    Identify leaf tile directories under the root.

    Definition used here:
    - a directory under --root that contains at least one "*.nc" file

    This is intentionally permissive:
    - does not hard-require the directory depth to be exactly root/region/londir
    - but does require the directory to be inside root (via relative_to check)
    """
    if not p.is_dir():
        return False
    if any(p.glob("*.nc")):
        # expect root/region/londir, but don't hard-require
        try:
            _ = p.relative_to(root)
        except Exception:
            return False
        return True
    return False


def main() -> int:
    """
    CLI entrypoint.

    Steps:
    1) Discover leaf tile directories under --root.
    2) For each leaf dir:
         - pick a sample .nc
         - read bounds from lat/lon coords
         - build a FeatureCollection (single feature) and write <tile_id>.geojson
    3) Write combined all_tiles.geojson containing all tile features.

    Overwrite behavior:
    - Per-tile GeoJSON: written only if it doesn't exist, unless --overwrite is set.
    - Combined GeoJSON: written if missing, or if --overwrite is set.
    """
    ap = argparse.ArgumentParser(description="Create GeoJSON polygons for raw leaf dirs.")
    ap.add_argument("--root", type=Path, default=Path("data/raw/era5_land_hourly_temp"),
                    help="Root directory containing region/lonXXX subdirs (default: data/raw/era5_land_hourly_temp)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/geojson/era5_land_hourly_temp"),
                    help="Output directory (default: data/processed/geojson/era5_land_hourly_temp)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing GeoJSON files")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"ERROR: root not found: {root}")
        return 2


    # Find all directories under root that contain .nc files.
    leaf_dirs = [p for p in root.rglob("*") if is_leaf_tile_dir(p, root)]
    leaf_dirs.sort()

    features: list[dict] = []
    wrote = 0
    skipped = 0

    for d in leaf_dirs:
        rel = d.relative_to(root)
        # expected: region/lonXXX
        region = rel.parts[0] if len(rel.parts) >= 1 else "unknown"
        londir = rel.parts[1] if len(rel.parts) >= 2 else rel.name
        tile_id = f"{region}_{londir}"

        sample = pick_sample_nc(d)
        if not sample:
            print(f"SKIP (no .nc): {d}")
            skipped += 1
            continue

        try:
            swlat, swlon, nelat, nelon = get_bounds(sample)
        except Exception as e:
            print(f"SKIP (cannot read coords): {d} sample={sample.name} err={e}")
            skipped += 1
            continue

        tile = TileFeature(
            tile_id=tile_id,
            region=region,
            londir=londir,
            sample_file=str(sample.relative_to(root)),
            swlat=swlat,
            swlon=swlon,
            nelat=nelat,
            nelon=nelon,
        )

        feat = feature_to_geojson(tile)
        features.append(feat)

        out_path = out_dir / f"{tile_id}.geojson"
        if out_path.exists() and not args.overwrite:
            # Still include in combined output, but don't overwrite per-file.
            continue

        out_path.write_text(json.dumps({"type": "FeatureCollection", "features": [feat]}, indent=2))
        wrote += 1

    # Combined file containing all tile features.
    all_path = out_dir / "all_tiles.geojson"
    if (not all_path.exists()) or args.overwrite:
        all_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))

    print(f"Leaf tile dirs found: {len(leaf_dirs)}")
    print(f"Per-tile GeoJSON written: {wrote} (use --overwrite to rewrite existing)")
    print(f"Combined GeoJSON: {all_path}")
    print(f"Skipped: {skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
