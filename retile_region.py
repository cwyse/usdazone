#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import xarray as xr

YEAR_MONTH_RE = re.compile(r"(?P<year>\d{4})_(?P<month>\d{2})(?:\.nc)?$", re.IGNORECASE)

# Non-tropics: 4 bins of 90 degrees starting at lon 0
BINS_NON_TROPICS: List[Tuple[int, int, str]] = [
    (0, 90, "lon000"),
    (90, 180, "lon090"),
    (180, 270, "lon180"),
    (270, 360, "lon270"),
]

# Tropics: 2 bins of 180 degrees starting at lon 0
BINS_TROPICS: List[Tuple[int, int, str]] = [
    (0, 180, "lon000"),
    (180, 360, "lon180"),
]


def detect_year_month(p: Path) -> Tuple[str, str]:
    m = YEAR_MONTH_RE.search(p.name)
    if not m:
        raise ValueError("missing YYYY_MM at end of filename")
    return m.group("year"), m.group("month")


def find_coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    names = set(ds.coords) | set(ds.variables)
    for c in candidates:
        if c in names:
            return c
    raise KeyError(f"missing coord among {list(candidates)}; have coords={sorted(list(ds.coords))[:20]}")


def normalize_lon_0_360(lon: xr.DataArray) -> xr.DataArray:
    lon2 = (lon % 360 + 360) % 360
    lon2 = lon2.assign_attrs(lon.attrs)
    return lon2


def round_lon(ds: xr.Dataset, lon_name: str, decimals: int = 6) -> xr.Dataset:
    lon = ds[lon_name]
    lonr = xr.apply_ufunc(lambda x: np.round(x, decimals), lon)
    lonr = lonr.assign_attrs(lon.attrs)
    return ds.assign_coords({lon_name: lonr})


def drop_duplicate_1d_coord(ds: xr.Dataset, coord_name: str) -> xr.Dataset:
    coord = ds[coord_name].values
    if coord.ndim != 1:
        return ds
    _, idx = np.unique(coord, return_index=True)  # keep first occurrence
    if len(idx) == len(coord):
        return ds
    idx_sorted = np.sort(idx)
    return ds.isel({coord_name: idx_sorted})


def subset_lat(ds: xr.Dataset, lat_name: str, region: str) -> xr.Dataset:
    lat = ds[lat_name]
    if region == "north":
        return ds.sel({lat_name: lat > 20.0})
    if region == "south":
        return ds.sel({lat_name: lat < -20.0})
    if region == "tropics":
        return ds.sel({lat_name: (lat >= -20.0) & (lat <= 20.0)})
    raise ValueError(f"unknown region: {region}")


def subset_lon_range(ds: xr.Dataset, lon_name: str, lo: int, hi: int) -> xr.Dataset:
    lon = ds[lon_name]
    return ds.sel({lon_name: (lon >= lo) & (lon < hi)})


def open_one(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=True, mask_and_scale=True, chunks=None)


def build_out_name(prefix: str, bin_tag: str, year: str, month: str) -> str:
    return f"{prefix}_{bin_tag}_{year}_{month}.nc"


def choose_bins(region: str) -> List[Tuple[int, int, str]]:
    return BINS_TROPICS if region == "tropics" else BINS_NON_TROPICS


def main() -> int:
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

    src_paths = [root / d for d in args.src_dirs]
    missing = [p for p in src_paths if not p.exists()]
    if missing:
        print("ERROR: missing source directories:")
        for p in missing:
            print(f"  - {p}")
        return 2

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

    for _, _, tag in bins:
        (out_root / tag).mkdir(parents=True, exist_ok=True)

    planned = 0
    written = 0
    errors = 0

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

            for p in paths:
                ds = None
                try:
                    ds = open_one(p)
                    lat_name = find_coord_name(ds, ("latitude", "lat", "Latitude", "Lat"))
                    lon_name = find_coord_name(ds, ("longitude", "lon", "Longitude", "Lon"))

                    # normalize/sort lon
                    ds = ds.assign_coords({lon_name: normalize_lon_0_360(ds[lon_name])})
                    ds = round_lon(ds, lon_name, decimals=6)
                    ds = ds.sortby(lon_name)
                    ds = drop_duplicate_1d_coord(ds, lon_name)

                    # region lat filter (north/south remove tropics overlap; tropics keep -20..20)
                    ds = subset_lat(ds, lat_name, args.region)
                    if ds.sizes.get(lat_name, 0) == 0:
                        continue

                    # slice lon bin
                    ds = subset_lon_range(ds, lon_name, lo, hi)
                    if ds.sizes.get(lon_name, 0) == 0:
                        continue

                    pieces.append(ds)
                    ds = None
                except Exception as e:
                    errors += 1
                    print(f"ERROR: {p} ({year}_{month}) bin {tag}: {e}")
                finally:
                    if ds is not None:
                        try:
                            ds.close()
                        except Exception:
                            pass

            if not pieces:
                continue

            try:
                assert lon_name is not None and lat_name is not None
                combined = xr.concat(pieces, dim=lon_name, join="outer", coords="minimal", compat="override")
                combined = combined.sortby(lon_name)
                combined = drop_duplicate_1d_coord(combined, lon_name)
            finally:
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
