#!/usr/bin/env python3
from __future__ import annotations

"""
Create GeoJSON tile polygons for each leaf subdirectory under data/raw/era5_land_hourly_temp.

Assumptions:
- Each leaf dir (e.g. data/raw/era5_land_hourly_temp/north/lon090) contains many .nc files for
  the same spatial extent, differing only by time.
- We can pick any one .nc file per leaf dir, read its lat/lon coordinates, and
  infer bounds.

Output:
- Writes one GeoJSON per leaf dir to:
    data/processed/geojson/era5_land_hourly_temp/<region>_<londir>.geojson
  Example:
    data/processed/geojson/era5_land_hourly_temp/north_lon090.geojson

- Also writes a combined GeoJSON FeatureCollection to:
    data/processed/geojson/era5_land_hourly_temp/all_tiles.geojson

Usage:
  python make_raw_geojson.py \
    --root data/raw/era5_land_hourly_temp \
    --out-dir data/processed/geojson/era5_land_hourly_temp

Dependencies:
- Prefers netCDF4; falls back to xarray if available.
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
    tile_id: str
    region: str
    londir: str
    sample_file: str
    swlat: float
    swlon: float
    nelat: float
    nelon: float


def _find_coord_var(var_names: set[str], preferred: tuple[str, ...]) -> Optional[str]:
    for k in preferred:
        if k in var_names:
            return k
    # common alternatives
    for k in var_names:
        lk = k.lower()
        if lk in [p.lower() for p in preferred]:
            return k
    return None


def _isfinite(x: float) -> bool:
    return math.isfinite(x)


def bounds_from_netcdf4(path: Path) -> Tuple[float, float, float, float]:
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

        if not all(_isfinite(v) for v in (swlat, nelat, swlon, nelat)):
            raise ValueError("non-finite bounds")

        return swlat, swlon, nelat, nelon
    finally:
        ds.close()


def get_bounds(path: Path) -> Tuple[float, float, float, float]:
    if HAVE_NETCDF4:
        return bounds_from_netcdf4(path)
    if HAVE_XARRAY:
        return bounds_from_xarray(path)
    raise RuntimeError("Need netCDF4 or xarray installed.")


def polygon_from_bounds(swlat: float, swlon: float, nelat: float, nelon: float) -> dict:
    # GeoJSON polygon ring: [ [lon, lat], ... ] closed
    ring = [
        [swlon, swlat],
        [nelon, swlat],
        [nelon, nelat],
        [swlon, nelat],
        [swlon, swlat],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def feature_to_geojson(tile: TileFeature) -> dict:
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
    # Prefer smallest filename sort for determinism; any file is fine per your assumption.
    ncs = sorted(dir_path.glob("*.nc"))
    return ncs[0] if ncs else None


def is_leaf_tile_dir(p: Path, root: Path) -> bool:
    # Leaf dir heuristic: contains .nc files
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

    # Combined file
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
