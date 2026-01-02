# usda_zone_core.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import xarray as xr


def lon_to_0_360(lon: float) -> float:
    return float((lon % 360.0 + 360.0) % 360.0)


def add_cyclic_lon_column(da: xr.DataArray) -> xr.DataArray:
    """
    Adds a wrapped column at lon=360 equal to lon=0 so linear interpolation
    works across the 0/360 seam.
    Assumes longitude is increasing and includes 0.0. No-op otherwise.
    """
    if "longitude" not in da.coords:
        return da

    lon = da["longitude"].values.astype(np.float64)
    if lon.size == 0:
        return da
    if abs(lon[0] - 0.0) > 1e-6:
        return da
    if lon[-1] >= 360.0 - 1e-6:
        return da

    wrap = da.isel(longitude=0).assign_coords(
        longitude=np.array([360.0], dtype=np.float64)
    )
    return xr.concat([da, wrap], dim="longitude")


def interp_temp_f_point(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    var: str = "usda_zone_temp_f",
    *,
    fallback_nearest: bool = True,
    cyclic_lon: bool = True,
) -> float | None:
    """
    Bilinear interpolation (xarray interp) on a 2D lat/lon grid.
    Uses cyclic longitude support to avoid seam failures at 0/360.
    """
    if var not in ds:
        return None

    lon360 = lon_to_0_360(lon)
    da = ds[var]

    if cyclic_lon:
        da = add_cyclic_lon_column(da)

    v = da.interp(latitude=lat, longitude=lon360, method="linear")
    x = float(v.values)

    if np.isfinite(x):
        return x

    if fallback_nearest:
        vn = ds[var].sel(latitude=lat, longitude=lon360, method="nearest")
        xn = float(vn.values)
        if np.isfinite(xn):
            return xn

    return None


def zone_num_from_temp_f(temp_f: xr.DataArray | np.ndarray | float) -> xr.DataArray:
    """
    USDA zone number: floor((F + 60)/10)+1, clipped to [1, 13].
    """
    tf = xr.DataArray(temp_f) if not isinstance(temp_f, xr.DataArray) else temp_f
    z = xr.apply_ufunc(np.floor, (tf + 60.0) / 10.0) + 1.0
    return z.clip(1.0, 13.0)


def subzone_from_temp_f(temp_f: xr.DataArray, zone_num_f: xr.DataArray) -> xr.DataArray:
    """
    Subzone: a if F < lower+5, b if F >= lower+5 (boundary goes to b).
    Return numeric 0=a, 1=b.
    """
    lower = -60.0 + (zone_num_f - 1.0) * 10.0
    return xr.where(temp_f >= (lower + 5.0), 1, 0)


@dataclass(frozen=True)
class ZonePoint:
    temp_f: float | None
    zone_num: int | None
    subzone: str | None
    zone_label: str


def zone_point_from_temp_f(temp_f: float | None) -> ZonePoint:
    if temp_f is None or not np.isfinite(temp_f):
        return ZonePoint(temp_f=None, zone_num=None, subzone=None, zone_label="—")

    z = int(np.floor((temp_f + 60.0) / 10.0) + 1.0)
    z = min(13, max(1, z))
    lower = -60.0 + (z - 1) * 10.0
    sub = "b" if temp_f >= (lower + 5.0) else "a"
    return ZonePoint(temp_f=float(temp_f), zone_num=z, subzone=sub, zone_label=f"{z}{sub}")
