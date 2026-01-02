#!/usr/bin/env python3
# usda_zone_access.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class USDAZonePoint:
    lat_req: float
    lon_req: float
    lat_grid: float
    lon_grid: float
    temp_f: float | None
    zone_num: int | None
    subzone: int | None  # 0='a', 1='b'
    zone_code: int | None

    @property
    def zone_label(self) -> str:
        if self.zone_num is None or self.zone_num <= 0:
            return "—"
        if self.subzone not in (0, 1):
            return "—"
        return f"{self.zone_num}{'a' if self.subzone == 0 else 'b'}"


class USDAZoneDataset:
    """
    Central accessor for the merged USDA zone NetCDF.

    Solves in one place:
      - lon domain conversion ([-180,180] vs [0,360))
      - 0/360 seam clamp/wrap for 0.1° grids (359.967 -> 359.9)
      - missing decoding for usda_zone_subzone (NaN, _FillValue, missing_value)
      - consistent point selection and return format

    Notes:
      - temps can be interpolated if you want, but zone/subzone are categorical
        so the default is nearest-cell selection for everything.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.ds = xr.open_dataset(self.path)

        if "latitude" not in self.ds.coords or "longitude" not in self.ds.coords:
            raise RuntimeError("Dataset missing latitude/longitude coords")

        self.lat = self.ds["latitude"]
        self.lon = self.ds["longitude"]

        # Determine lon domain
        self.lon_min = float(np.nanmin(self.lon.values))
        self.lon_max = float(np.nanmax(self.lon.values))
        self.lon_step = self._infer_step(self.lon.values)

        # Cache commonly used variables (fall back if needed)
        self.temp_f_name = self._pick_temp_f_name()
        self.zone_num_name = "usda_zone_num" if "usda_zone_num" in self.ds else None
        self.subzone_name = "usda_zone_subzone" if "usda_zone_subzone" in self.ds else None
        self.zone_code_name = "usda_zone_code" if "usda_zone_code" in self.ds else None

    def close(self) -> None:
        self.ds.close()

    def __enter__(self) -> "USDAZoneDataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------- Introspection ----------

    def variables(self) -> list[str]:
        return sorted(list(self.ds.data_vars.keys()))

    def summary(self) -> str:
        vars_s = ", ".join(self.variables())
        return (
            f"path={self.path}\n"
            f"lon_domain=[{self.lon_min}, {self.lon_max}] step≈{self.lon_step}\n"
            f"vars={vars_s}"
        )

    # ---------- Core helpers ----------

    @staticmethod
    def _infer_step(vals: np.ndarray) -> float:
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        vals.sort()
        if vals.size < 2:
            return 0.0
        diffs = np.diff(vals)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        return float(np.median(diffs)) if diffs.size else 0.0

    def _pick_temp_f_name(self) -> str:
        # Prefer canonical Fahrenheit if present
        if "usda_zone_temp_f" in self.ds:
            return "usda_zone_temp_f"
        if "mean_annual_extreme_min_f" in self.ds:
            return "mean_annual_extreme_min_f"
        # Otherwise we can derive from C
        if "usda_zone_temp_c" in self.ds:
            return "usda_zone_temp_c"
        if "mean_annual_extreme_min" in self.ds:
            return "mean_annual_extreme_min"
        raise RuntimeError("No temperature variable found")

    def _temp_f_da(self) -> xr.DataArray:
        name = self.temp_f_name
        da = self.ds[name]
        if name.endswith("_f") or name == "usda_zone_temp_f":
            return da
        # C -> F
        return da * 9 / 5 + 32

    def _normalize_lon_to_dataset(self, lon_in: float) -> float:
        """
        Convert lon_in (either [-180,180] or [0,360) style) into the dataset lon domain.
        Also fix the 0/360 seam for 0..360 grids by clamping near-max values to lon_max
        when within half a grid step, else wrapping to lon_min.
        """
        lon = float(lon_in)

        # If dataset uses 0..360 domain
        if self.lon_min >= 0.0 and self.lon_max > 180.0:
            # Accept either style: if lon looks like [-180,180], convert to [0,360)
            if lon < 0.0:
                lon = lon + 360.0
            else:
                lon = lon % 360.0

            # Seam handling: grid often ends at 359.9, but (lon % 360) can yield 359.967...
            if lon > self.lon_max:
                if self.lon_step > 0 and (lon - self.lon_max) <= (self.lon_step / 2.0):
                    lon = self.lon_max
                else:
                    lon = self.lon_min  # wrap to 0.0 typically
            if lon >= 360.0:
                lon = self.lon_min
            return lon

        # Else dataset likely uses [-180,180]
        lon = ((lon + 180.0) % 360.0) - 180.0
        # Clamp if just outside because of floating behavior
        if lon < self.lon_min and self.lon_step > 0 and (self.lon_min - lon) <= (self.lon_step / 2.0):
            lon = self.lon_min
        if lon > self.lon_max and self.lon_step > 0 and (lon - self.lon_max) <= (self.lon_step / 2.0):
            lon = self.lon_max
        return lon

    @staticmethod
    def _decode_missing_int(value: Any, da: xr.DataArray) -> int | None:
        """
        Turn a scalar from xarray into int, respecting NaN/_FillValue/missing_value.
        """
        v = value
        # Unwrap numpy scalars/0-d arrays
        if isinstance(v, np.ndarray):
            if v.shape == ():
                v = v.item()
            else:
                raise ValueError("Expected scalar")

        # Float NaN
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return None

        # Discover fill / missing values
        fill = None
        if "_FillValue" in da.encoding and da.encoding["_FillValue"] is not None:
            fill = da.encoding["_FillValue"]
        elif "_FillValue" in da.attrs:
            fill = da.attrs["_FillValue"]
        elif "missing_value" in da.attrs:
            fill = da.attrs["missing_value"]

        # Compare fill values safely
        if fill is not None:
            try:
                if v == fill:
                    return None
            except Exception:
                pass

        try:
            return int(v)
        except Exception:
            return None

    # ---------- Public API ----------

    def point(self, lat: float, lon: float, *, method: str = "nearest") -> USDAZonePoint:
        """
        Return a single point with both requested and snapped grid coordinates.
        Default selection is nearest-cell.

        If you ever want interpolated temperatures, you can add a separate method
        for temp only; keep zone/subzone as nearest.
        """
        lon_norm = self._normalize_lon_to_dataset(lon)

        temp_f = self._temp_f_da()
        sel = dict(latitude=float(lat), longitude=float(lon_norm), method=method)

        temp_cell = temp_f.sel(**sel)
        lat_grid = float(temp_cell.latitude.values)
        lon_grid = float(temp_cell.longitude.values)

        temp_val = float(temp_cell.values)
        if not np.isfinite(temp_val):
            temp_out: float | None = None
        else:
            temp_out = temp_val

        zone_num_out: int | None = None
        subzone_out: int | None = None
        zone_code_out: int | None = None

        if self.zone_num_name:
            zone_num_out = self._decode_missing_int(self.ds[self.zone_num_name].sel(**sel).values, self.ds[self.zone_num_name])
        if self.subzone_name:
            subzone_out = self._decode_missing_int(self.ds[self.subzone_name].sel(**sel).values, self.ds[self.subzone_name])
        if self.zone_code_name:
            zone_code_out = self._decode_missing_int(self.ds[self.zone_code_name].sel(**sel).values, self.ds[self.zone_code_name])

        # If zone_num is 0 or missing, treat as missing
        if zone_num_out is not None and zone_num_out <= 0:
            zone_num_out = None
            subzone_out = None
            zone_code_out = None

        # If subzone missing, don’t pretend it's 'a'
        if subzone_out is None:
            zone_code_out = None

        return USDAZonePoint(
            lat_req=float(lat),
            lon_req=float(lon),
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            temp_f=temp_out,
            zone_num=zone_num_out,
            subzone=subzone_out,
            zone_code=zone_code_out,
        )

    def points(self, items: Iterable[tuple[float, float]]) -> list[USDAZonePoint]:
        return [self.point(lat, lon) for lat, lon in items]
