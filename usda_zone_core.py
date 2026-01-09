# usda_zone_core.py
from __future__ import annotations

"""
usda_zone_core.py

Single-source-of-truth core logic for:
  1) Longitude normalization + cyclic wrap handling for point sampling.
  2) Interpolating the ERA5-derived USDA-zone temperature metric (°F) at a lat/lon.
  3) Converting that temperature to USDA-like zone number + subzone + label.

Design goals
------------
- Downstream scripts MUST NOT implement their own lon wraparound or interpolation.
  They should call USDAZoneDataset.point(), which calls interp_temp_f_point() here.
- Robust to datasets whose longitude coordinate is either:
    - [-180, 180], or
    - [0, 360)
  and to the 0/360 seam when interpolating.
- Keep behavior deterministic and centralized so tile/domain changes only require edits
  in one place.

Key conventions
---------------
- All point queries internally use lon normalized to [0, 360).
- If cyclic_lon=True, we ensure a cyclic “wrap” column exists at lon0+360 so xarray
  interpolation can span the seam.

Returned semantics
------------------
- interp_temp_f_point() returns:
    - float (°F) if a finite value can be sampled, else None.
- zone_point_from_temp_f() returns ZonePoint with:
    - temp_f=None and zone_label="—" when missing.

Notes on region/tile latitude/longitude bounds
----------------------------------------------
This module does not enforce “tile bounds” or region slicing.
That belongs in your tiling / preprocessing steps. This module just samples the
merged global climatology dataset and must accept arbitrary lon/lat points.
"""

from dataclasses import dataclass
import numpy as np
import xarray as xr

# -----------------------
# Longitude normalization
# -----------------------

def lon_to_0_360(lon: float) -> float:
    """Normalize any longitude to [0, 360)."""
    return float((lon % 360.0 + 360.0) % 360.0)


def _normalize_lon_coord_to_0_360_sorted(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure da.longitude is normalized to [0,360) and strictly ascending.

    This makes interpolation behavior predictable regardless of whether the
    dataset stored longitudes as [-180,180] or [0,360).
    """
    if "longitude" not in da.coords:
        return da

    lon = da["longitude"].astype(np.float64)
    lon = xr.apply_ufunc(lambda x: (x % 360.0 + 360.0) % 360.0, lon)
    lon = xr.apply_ufunc(lambda x: np.round(x, 6), lon)

    out = da.assign_coords(longitude=lon).sortby("longitude")

    # Drop duplicates (can appear after normalization/rounding near 0 seam)
    idx = out["longitude"].to_index()
    if idx.has_duplicates:
        out = out.isel(longitude=~idx.duplicated())

    return out

def add_cyclic_lon_column(da: xr.DataArray) -> xr.DataArray:
    """
    Add a wrap-around longitude column at (lon0 + 360) so xarray interpolation
    across the dateline can work on a monotonic 0..360-ish axis.

    Preconditions:
      - da has dims ('latitude','longitude') (order doesn't matter)
      - longitude is normalized to [0, 360) and sorted ascending
    """
    if "longitude" not in da.dims:
        raise ValueError("DataArray must have a 'longitude' dimension")

    lon = da["longitude"].values
    if lon.size < 2:
        return da

    # If it's already cyclic (last looks like first+360), don't add again
    if np.isfinite(lon[0]) and np.isfinite(lon[-1]) and np.isclose(lon[-1], lon[0] + 360.0):
        return da

    # Keep longitude dimension (size 1) so assign_coords matches dims
    wrap = da.isel(longitude=slice(0, 1)).copy(deep=False)
    wrap = wrap.assign_coords(longitude=(wrap["longitude"] + 360.0))

    out = xr.concat([da, wrap], dim="longitude")
    # Ensure strict increasing lon after concat
    out = out.sortby("longitude")
    return out



def interp_temp_f_point(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    var: str = "usda_zone_temp_f",
    *,
    fallback_nearest: bool = True,
    cyclic_lon: bool = True,
    assume_lon_normalized_sorted: bool = False,
) -> float | None:

    """
    Bilinear interpolation (xarray interp) on a 2D lat/lon grid.

    Parameters
    ----------
    ds:
      Dataset containing var on (latitude, longitude).

    lat, lon:
      Decimal degrees. lon may be in [-180,180] or [0,360); it is normalized here.

    var:
      Variable to sample, default "usda_zone_temp_f".

    fallback_nearest:
      If linear interpolation yields non-finite (masked/outside), try nearest.

    cyclic_lon:
      If True, add a cyclic longitude column to support interpolation across 0/360 seam.

    assume_lon_normalized_sorted:
      If True, skip longitude coordinate normalization/sorting. Use this when the
      DataArray has already been normalized to [0,360) and sorted (and optionally
      made cyclic upstream). This avoids per-call normalization overhead.

    Returns
    -------
    float | None:
      Sampled °F if finite, else None.
    """
    if var not in ds:
        return None

    lon360 = lon_to_0_360(float(lon))
    da = ds[var]

    # Make longitude convention robust before any seam logic/interp.
    # For hot loops, callers may pre-normalize once and set assume_lon_normalized_sorted=True.
    if not assume_lon_normalized_sorted:
        da = _normalize_lon_coord_to_0_360_sorted(da)

    if cyclic_lon:
        da = add_cyclic_lon_column(da)


    v = da.interp(latitude=lat, longitude=lon360, method="linear")
    xv = v.values
    x = float(xv) if np.isscalar(xv) else float(np.asarray(xv).reshape(()))


    if np.isfinite(x):
        return x

    if fallback_nearest:
        vn = da.sel(latitude=lat, longitude=lon360, method="nearest")
        xn = float(vn.values)
        if np.isfinite(xn):
            return xn


    return None

# -----------------------
# USDA zone math
# -----------------------

def zone_num_from_temp_f(temp_f: xr.DataArray | np.ndarray | float) -> xr.DataArray:
    """
    USDA zone number: floor((F + 60)/10)+1, clipped to [1, 13].
    """
    tf = xr.DataArray(temp_f) if not isinstance(temp_f, xr.DataArray) else temp_f
    z = xr.apply_ufunc(np.floor, (tf + 60.0) / 10.0) + 1.0
    return z.clip(1.0, 13.0)


def subzone_from_temp_f(temp_f: xr.DataArray, zone_num_f: xr.DataArray) -> xr.DataArray:
    """
    Subzone:
      - 0 = 'a' (colder half)
      - 1 = 'b' (warmer half)
    Boundary at lower+5°F goes to 'b' (>=).
    """
    lower = -60.0 + (zone_num_f - 1.0) * 10.0
    return xr.where(temp_f >= (lower + 5.0), 1, 0)


@dataclass(frozen=True)
class ZonePoint:
    """
    Result of querying a USDAZoneDataset at a single latitude/longitude.

    This is a lightweight, immutable value object returned by
    USDAZoneDataset.point(). It intentionally contains *derived*
    values only — no raw grid indices or interpolation metadata —
    so that callers do not depend on dataset internals.

    Fields
    ------
    temp_f : float | None
        Interpolated mean annual extreme minimum temperature in °F.
        None indicates missing data or lookup failure.

    zone_num : int | None
        USDA hardiness zone number (1–13), or None if temp_f is missing.

    subzone : str | None
        Subzone letter:
          - 'a' = colder half
          - 'b' = warmer half
        None if temp_f is missing.

    zone_label : str
        Human-readable zone label:
          - e.g. "5a", "6b"
          - "—" if temp_f is missing

    Design notes
    ------------
    - Missing values are represented explicitly (None / "—") rather than
      sentinel numbers to avoid accidental misuse.
    - This object contains *no spatial logic*. All longitude wrapping,
      interpolation, and grid selection must occur upstream.
    - zone_label is provided as a convenience for reporting and UI use;
      numeric fields should be used for analysis.
    """
    temp_f: float | None
    zone_num: int | None
    subzone: str | None
    zone_label: str


def zone_point_from_temp_f(temp_f: float | None) -> ZonePoint:
    """
    Convert a sampled temperature (°F) into a ZonePoint.
    """
    if temp_f is None or not np.isfinite(temp_f):
        return ZonePoint(temp_f=None, zone_num=None, subzone=None, zone_label="—")

    z = int(np.floor((temp_f + 60.0) / 10.0) + 1.0)
    z = min(13, max(1, z))
    lower = -60.0 + (z - 1) * 10.0
    sub = "b" if temp_f >= (lower + 5.0) else "a"
    return ZonePoint(temp_f=float(temp_f), zone_num=z, subzone=sub, zone_label=f"{z}{sub}")
