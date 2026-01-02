#!/usr/bin/env python3
"""
Compute annual extreme minimum temperature per tile/year (new raw layout only).

Expected input layout:

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

Outputs:
  data/processed/annual_extreme_min_tiles/{tile_name}_{year}.nc
  where tile_name is like: north_lon090, south_lon270, tropics_lon180

Cold-season month filtering (by region):
  north   -> DJF (12, 1, 2)
  south   -> JJA (6, 7, 8)
  tropics -> all months (1..12)

This version canonicalizes BOTH longitude and latitude grids per tile to avoid
CDS float endpoint/spacing quirks that can produce tiny coordinate drift
(e.g. 20.200000000000855 vs 20.200000000000003) that breaks join="exact".
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

TROPICS_MIN = -20.0
TROPICS_MAX = 20.0
BAND_EPS = 0.1  # 0.1° grid, to avoid overlap between tropics and non-tropics

HOURLY_DIR = Path("data/raw/era5_land_hourly_temp")
OUT_DIR = Path("data/processed/annual_extreme_min_tiles")

OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_FILENAME_RE = re.compile(
    r"^(?P<region>north|south|tropics)_(?P<lon_tag>lon\d{3})_(?P<year>\d{4})_(?P<month>\d{2})\.nc$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HourlyFileInfo:
    tile_name: str  # e.g. "south_lon090"
    region: str     # "north"|"south"|"tropics"
    year: int
    month: int
    path: Path


def parse_hourly_filename(path: Path) -> Optional[HourlyFileInfo]:
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
    region = region.lower()
    if region == "north":
        return {12, 1, 2}
    if region == "south":
        return {6, 7, 8}
    if region == "tropics":
        return set(range(1, 13))
    raise ValueError(f"unknown region: {region}")


def default_min_valid_months(region: str) -> int:
    return 12 if region.lower() == "tropics" else 3


def _time_coord(ds: xr.Dataset) -> Optional[str]:
    for name in ("valid_time", "time"):
        if name in ds.coords:
            return name
    return None


def _pick_temp_var(ds: xr.Dataset) -> Optional[str]:
    if "t2m" in ds.variables:
        return "t2m"
    if "2m_temperature" in ds.variables:
        return "2m_temperature"
    return None


def is_valid_hourly_file(path: Path) -> bool:
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
    lon = (da % 360 + 360) % 360
    lon = lon.assign_attrs(da.attrs)
    return lon


def _lon_bounds_for_tile(tile_name: str) -> tuple[float, float]:
    # tile_name like "tropics_lon000" / "north_lon090"
    region, lon_tag = tile_name.split("_", 1)
    lo = float(int(lon_tag[3:]))
    width = 180.0 if region == "tropics" else 90.0
    return lo, lo + width


def _lat_bounds_for_region(region: str) -> tuple[float, float]:
    """
    Returns bounds as (lo, hi) for a HALF-OPEN selection on a sorted-ascending latitude axis.
    These bounds match your downloader intent (no overlap with tropics).
      - north:   (20.1 .. 90.0]
      - tropics: [-20.0 .. 20.0]
      - south:   [-90.0 .. -20.1)
    When we sort ascending, we can use:
      - north:   [20.1, 90.0]
      - tropics: [-20.0, 20.0]
      - south:   [-90.0, -20.1]
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
      - round coords to round_dp
      - sort
      - select to bounds (half-open by default: [lo,hi) for lon; lat uses inclusive endpoints)
      - if size mismatch vs expected grid, reindex nearest within tolerance
      - drop duplicates after rounding

    For latitude we generally want inclusive endpoints, so pass half_open=False.
    """
    if name not in ds.coords:
        return ds

    coord = ds[name].astype(np.float64)
    coord = xr.apply_ufunc(lambda x: np.round(x, round_dp), coord)
    coord = coord.assign_attrs(ds[name].attrs)

    ds = ds.assign_coords({name: coord}).sortby(name)

    coordv = ds[name]
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
            # fall through to dupe removal
            idx = ds[name].to_index()
            if idx.has_duplicates:
                ds = ds.isel({name: ~idx.duplicated()})
            return ds

    # If mismatch, reindex onto target grid
    if n != expected_n:
        tol = step * 0.25
        ds = ds.reindex({name: target}, method="nearest", tolerance=tol)

    # Drop any duplicates
    idx = ds[name].to_index()
    if idx.has_duplicates:
        ds = ds.isel({name: ~idx.duplicated()})

    return ds


def _canonicalize_lon_grid(ds: xr.Dataset, tile_name: str, lon_name: str = "longitude") -> xr.Dataset:
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
    Fix tiny float drift in latitude coordinates so join="exact" works across months.

    Key points:
      - We want a deterministic 0.1° grid per region.
      - We keep inclusive endpoints for lat tiles (e.g. north: 20.1..90.0).
      - We do NOT change the physical coverage; we just snap coords to canonical values.
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

    for info in sorted(good, key=lambda i: (i.month, i.path.name)):
        ds = xr.open_dataset(info.path, engine="netcdf4")
        try:
            # Canonicalize coords before reducing
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

            # Month filter (usually redundant since each file is a month, but safe)
            try:
                month_mask = ds[tname].dt.month.isin(sorted(allowed_months))
            except Exception:
                month_mask = None

            ds2 = ds.where(month_mask, drop=True) if month_mask is not None else ds
            if ds2.sizes.get(tname, 0) == 0:
                continue

            arr = (ds2[vname] - 273.15).min(dim=tname, skipna=True).load()

            shapes.add((int(arr["latitude"].size), int(arr["longitude"].size)))

            if running_min is None:
                running_min = arr
            else:
                # After canonicalization, this should be stable
                a, b = xr.align(running_min, arr, join="exact")
                running_min = xr.apply_ufunc(np.fmin, a, b).load()
        finally:
            ds.close()

    if running_min is None:
        log.info("[SKIP] %s %d: no usable data after reduction", tile_name, year)
        return

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
