#!/usr/bin/env python3
# merge_usda_zone_tiles.py

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import xarray as xr

TILES_DIR = Path("data/processed/usda_zone_temperature_tiles")


def _dedupe_1d(ds: xr.Dataset, dim: str, name: str) -> xr.Dataset:
    idx = ds[dim].to_index()
    if idx.has_duplicates:
        mask = ~idx.duplicated()
        ds = ds.isel({dim: mask})
        print(f"[WARN] {name}: dropped {int((~mask).sum())} duplicate {dim} values")
    return ds


def _normalize(ds: xr.Dataset, name: str) -> xr.Dataset:
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
    Accept both:
      - new tile outputs: mean_annual_extreme_min_f / mean_annual_extreme_min
      - any legacy names (optional): usda_zone_temp_f / usda_zone_temp_c
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
    USDA zone logic:
      zone_num = floor((F + 60)/10) + 1, clipped to 1..13
      subzone  = 'a' if F <= midpoint, else 'b'

    Stored as numeric fields:
      - usda_zone_num: uint8 (1..13), 0 = missing
      - usda_zone_subzone: uint8 (0='a', 1='b'), 255 = missing
      - usda_zone_code: uint16 = zone_num*2 + subzone; 0 = missing
    """
    valid = xr.apply_ufunc(np.isfinite, temp_f)

    zone_num_f = xr.apply_ufunc(np.floor, (temp_f + 60.0) / 10.0) + 1.0
    zone_num_f = zone_num_f.clip(1.0, 13.0)

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

    lower = -60.0 + (zone_num_f - 1.0) * 10.0
    mid = lower + 5.0

    # IMPORTANT: do NOT use 0 as missing because 0 is the real value for 'a'
    sub_fill = np.uint8(255)
    subzone01 = xr.where(temp_f <= mid, 0, 1).astype(np.uint8)
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

    zone_code = (zone_num.astype(np.uint16) * 2 + subzone01.astype(np.uint16)).where(valid, other=0).astype(np.uint16)
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

        if not {"latitude", "longitude"} <= set(ds.dims):
            raise RuntimeError(f"{f} missing spatial dimensions")

        ds = _normalize(ds, f.name)

        if merged is None:
            merged = ds
            continue

        merged, ds = xr.align(merged, ds, join="outer")
        merged = merged.combine_first(ds)

    assert merged is not None
    merged = _normalize(merged, "merged")

    # Add canonical Fahrenheit variable for downstream scripts
    temp_f = _get_temp_f(merged)
    if "usda_zone_temp_f" not in merged:
        merged["usda_zone_temp_f"] = temp_f

    # Compute and store USDA zone fields
    zone_num, subzone, zone_code = _compute_usda_zone_fields(merged["usda_zone_temp_f"])
    merged[zone_num.name] = zone_num
    merged[subzone.name] = subzone
    merged[zone_code.name] = zone_code

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
            "usda_zone_definition": "zone_num=floor((F+60)/10)+1 clipped to 1..13; subzone a if F<=midpoint else b",
        }
    )

    merged.to_netcdf(out_path)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
