#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import warnings

import xarray as xr
import numpy as np

ANNUAL_DIR = Path("data/processed/annual_extreme_min_tiles")
OUT_DIR = Path("data/processed/usda_zone_temperature_tiles")

OUT_DIR.mkdir(parents=True, exist_ok=True)

HARTFORD_LAT = 41.77
HARTFORD_LON_360 = 360.0 - 72.68  # 287.32


def discover_tiles_from_annual_dir() -> list[str]:
    """
    Discover tiles from Step 2 outputs of the form:
      data/processed/annual_extreme_min_tiles/<tile>_<YYYY>.nc
    """
    tiles: set[str] = set()
    for p in ANNUAL_DIR.glob("*.nc"):
        stem = p.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        try:
            int(parts[-1])  # year
        except ValueError:
            continue
        tile = "_".join(parts[:-1])
        tiles.add(tile)
    return sorted(tiles)


def _open_annual_file(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)

    if "annual_extreme_min" not in ds:
        raise RuntimeError(f"{path}: missing variable 'annual_extreme_min'")

    da = ds["annual_extreme_min"]

    for dim in ("latitude", "longitude"):
        if dim not in da.dims:
            raise RuntimeError(f"{path}: expected '{dim}' dimension not found")

    return da


def _normalize_lon_0_360(lon: xr.DataArray) -> xr.DataArray:
    # Normalize to [0, 360)
    v = (lon % 360 + 360) % 360
    return xr.apply_ufunc(lambda x: np.round(x, 6), v)


def _tile_lon_bounds(tile_name: str) -> tuple[float, float, float]:
    """
    Return (lo, hi, width) in degrees for longitude bounds in 0..360.

    Tile names are expected like:
      north_lon000, north_lon090, north_lon180, north_lon270  (width 90)
      south_lon000, south_lon090, south_lon180, south_lon270  (width 90)
      tropics_lon000, tropics_lon180                          (width 180)
    """
    region, lon_tag = tile_name.split("_", 1)
    lo = float(int(lon_tag[3:]))
    width = 180.0 if region.lower() == "tropics" else 90.0
    return lo, lo + width, width


def _tile_lat_bounds(tile_name: str) -> tuple[float, float]:
    """
    Approximate bounds used by your pipeline:
      tropics: [-20, 20]
      north:   [20, 90]
      south:   [-90, -20]
    """
    region = tile_name.split("_", 1)[0].lower()
    if region == "tropics":
        return -20.0, 20.0
    if region == "north":
        return 20.0, 90.0
    if region == "south":
        return -90.0, -20.0
    return -90.0, 90.0


def _drop_duplicate_coords(da: xr.DataArray, dim: str) -> xr.DataArray:
    idx = da[dim].to_index()
    if idx.has_duplicates:
        mask = ~idx.duplicated()
        da = da.isel({dim: mask})
    return da


def _canonicalize_grid(da: xr.DataArray, tile_name: str) -> xr.DataArray:
    """
    Make lon/lat grids deterministic across years for the same tile:
      - sort coords
      - normalize lon to 0..360
      - round coords
      - drop duplicate coords
      - enforce bounds
      - drop an inclusive endpoint column if present (e.g., 1801 instead of 1800)
    """
    # Latitude
    da = da.sortby("latitude")
    da = da.assign_coords(latitude=xr.apply_ufunc(lambda x: np.round(x, 6), da["latitude"]))
    da = _drop_duplicate_coords(da, "latitude")

    lat_lo, lat_hi = _tile_lat_bounds(tile_name)
    da = da.sel(latitude=(da["latitude"] >= lat_lo) & (da["latitude"] <= lat_hi))

    # Longitude
    da = da.assign_coords(longitude=_normalize_lon_0_360(da["longitude"]))
    da = da.sortby("longitude")
    da = _drop_duplicate_coords(da, "longitude")

    lo, hi, width = _tile_lon_bounds(tile_name)

    lonv = da["longitude"]
    da = da.sel(longitude=(lonv >= lo) & (lonv < hi))
    if da.sizes.get("longitude", 0) == 0:
        return da

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
    lat_lo, lat_hi = _tile_lat_bounds(tile_name)
    lo, hi, _ = _tile_lon_bounds(tile_name)
    return (lat_lo <= HARTFORD_LAT <= lat_hi) and (lo <= HARTFORD_LON_360 < hi)


def _infer_step(coord: xr.DataArray) -> float:
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
    Snap da onto ref coords (lat + lon) using nearest with tight tolerances,
    then require exact coordinate match.
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
    Compute USDA zone number from temp_f using standard 10°F bins starting at -60°F.
    Clamp to [1, 13] for this project. (0 reserved for missing.)
    """
    z = np.floor((temp_f + 60.0) / 10.0) + 1.0
    z = z.clip(1.0, 13.0)
    return z


def _compute_subzone_from_temp_f_and_zone(temp_f: xr.DataArray, zone_num: xr.DataArray) -> xr.DataArray:
    """
    Subzone: 'a' colder half (0), 'b' warmer half (1).
    Boundary at zone_lower + 5°F; exact boundary goes to 'b'.
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
    Compute USDA Plant Hardiness Zone temperature metric:
    mean annual extreme minimum temperature over a climatology period.

    Input:
      data/processed/annual_extreme_min_tiles/<tile>_<YYYY>.nc
      with annual_extreme_min(latitude, longitude) in degC.

    Output:
      data/processed/usda_zone_temperature_tiles/<tile>_mean_<start>_<end>.nc
      with:
        - usda_zone_temp_c (degC)
        - usda_zone_temp_f (degF)
        - usda_zone_num (int16; 0=missing)
        - usda_zone_subzone (int8; 0=a, 1=b; -1=missing)
        - n_years (count contributing years per grid cell)
        - stdev_annual_extreme_min (degC)
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

    tmin = xr.concat(
        yearly,
        dim=xr.IndexVariable("year", np.array(years_found, dtype=np.int32)),
        join="exact",
    )

    n_years = tmin.notnull().sum(dim="year").astype(np.int16)

    mean_c = tmin.mean(dim="year", skipna=True)
    mean_f = mean_c * 9 / 5 + 32

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice", category=RuntimeWarning)
        stdev_c = tmin.std(dim="year", skipna=True)

    stdev_c = stdev_c.where(n_years > 0)

    # Compute zone + subzone everywhere temp is valid (n_years>0)
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

    out = mean_c.to_dataset(name="usda_zone_temp_c")
    out["usda_zone_temp_f"] = mean_f
    out["usda_zone_num"] = zone_num
    out["usda_zone_subzone"] = subzone
    out["n_years"] = n_years
    out["stdev_annual_extreme_min"] = stdev_c

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
        tiles = discover_tiles_from_annual_dir()
        if not tiles:
            raise SystemExit(f"No annual files found in {ANNUAL_DIR}")

        did_any = False
        for tile in tiles:
            out_path = OUT_DIR / f"{tile}_mean_{args.start_year}_{args.end_year}.nc"
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
