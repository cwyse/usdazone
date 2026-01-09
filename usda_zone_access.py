# usda_zone_access.py
from __future__ import annotations

"""
usda_zone_access.py

Thin access layer around the merged global USDA-zone-temperature climatology NetCDF.

Purpose
-------
Provide a single entry point for *all* point sampling of the ERA5-derived “USDA zone
temperature metric” and the derived USDA-like zone label, so that longitude handling,
wraparound/cyclic behavior, and interpolation details are centralized.

Downstream scripts (GBIF cold-edge, smoke tests, etc.) should call ONLY:

    USDAZoneDataset.point(lat, lon)

and must not implement their own:
  - lon normalization (-180..180 vs 0..360)
  - cyclic longitude wrap logic near 0/360 seams
  - gridpoint selection or interpolation
  - masking / out-of-bounds behavior

Those details belong in usda_zone_core.interp_temp_f_point() (and related helpers).

Inputs
------
- NetCDF dataset path (typically produced by merge_usda_zone_tiles.py), expected to contain
  a Fahrenheit climatology variable named "usda_zone_temp_f" by default.

Public API
----------
- USDAZoneDataset(path): context manager that opens/closes the dataset.
- USDAZoneDataset.point(lat, lon) -> ZonePoint:
    - samples temp_f at the requested lat/lon using interp_temp_f_point()
    - converts to a ZonePoint (temp + label) using zone_point_from_temp_f()

Expected behavior / invariants
------------------------------
- point() should be safe to call with longitudes in either:
    - [-180, +180], or
    - [0, 360)
  as long as interp_temp_f_point() implements the normalization you want.
- If the point is outside the dataset coverage or lands on masked data, temp_f should
  come back as None (or NaN depending on your ZonePoint conventions); zone label should
  be the “missing” sentinel (commonly "—").

Notes
-----
- This file intentionally stays small. The “hard” logic lives in usda_zone_core so there is
  exactly one place to fix wraparound/interpolation bugs.
"""

from pathlib import Path
import xarray as xr

from usda_zone_core import (
    interp_temp_f_point,
    zone_point_from_temp_f,
    ZonePoint,
    _normalize_lon_coord_to_0_360_sorted,
    add_cyclic_lon_column,
)

class USDAZoneDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.ds: xr.Dataset | None = None
        self._da_temp_f: xr.DataArray | None = None
        self._ds_view: xr.Dataset | None = None

    def __enter__(self) -> "USDAZoneDataset":
        self.ds = xr.open_dataset(self.path)

        # Normalize/cyclic once for performance and seam correctness
        self._da_temp_f = _normalize_lon_coord_to_0_360_sorted(self.ds["usda_zone_temp_f"])
        self._da_temp_f = add_cyclic_lon_column(self._da_temp_f)

        # Cache a minimal dataset view so point() doesn't allocate a new xr.Dataset each call
        self._ds_view = xr.Dataset({"usda_zone_temp_f": self._da_temp_f})
        return self


    def __exit__(self, exc_type, exc, tb) -> None:
        if self.ds is not None:
            self.ds.close()
            self.ds = None
        self._da_temp_f = None
        self._ds_view = None


    def point(self, lat: float, lon: float) -> ZonePoint:
        """
        Sample the dataset at (lat, lon) and return a ZonePoint.

        Parameters
        ----------
        lat, lon:
          Decimal degrees. lon may be in [-180, 180] or [0, 360) depending on your
          convention; interp_temp_f_point() is responsible for any normalization/wrap.

        var:
          Dataset variable name holding the Fahrenheit climatology metric.
          Default: "usda_zone_temp_f" (the canonical name added by merge step).

        Returns
        -------
        ZonePoint:
          Output of zone_point_from_temp_f(temp_f). If sampling returns missing,
          ZonePoint should represent missing accordingly (per your core logic).
        """
        assert self._ds_view is not None  # __enter__ must have been called

        # Use the core routine so lon normalization + seam behavior stays centralized.
        temp_f = interp_temp_f_point(
            self._ds_view,
            lat=float(lat),
            lon=float(lon),
            var="usda_zone_temp_f",
            fallback_nearest=True,
            cyclic_lon=False,  # already cyclic
            assume_lon_normalized_sorted=True,
        )


        return zone_point_from_temp_f(temp_f)

