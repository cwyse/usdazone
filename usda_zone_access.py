# usda_zone_access.py
from __future__ import annotations

from pathlib import Path
import xarray as xr

from usda_zone_core import interp_temp_f_point, zone_point_from_temp_f, ZonePoint


class USDAZoneDataset:
    def __init__(self, path: Path):
        self.path = path
        self.ds: xr.Dataset | None = None

    def __enter__(self) -> "USDAZoneDataset":
        self.ds = xr.open_dataset(self.path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.ds is not None:
            self.ds.close()
            self.ds = None

    def point(self, lat: float, lon: float) -> ZonePoint:
        assert self.ds is not None
        temp_f = interp_temp_f_point(self.ds, lat, lon, var="usda_zone_temp_f")
        return zone_point_from_temp_f(temp_f)
