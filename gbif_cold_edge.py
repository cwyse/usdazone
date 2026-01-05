#!/usr/bin/env python3
# gbif_cold_edge.py
"""
Estimate a species "cold-edge" hardiness zone using GBIF occurrences + an ERA5-derived
USDA-like climatology.

This module is the reusable library implementation (not a CLI). It is intended to be
imported by:
  - gbif_cold_edge_era5.py      (CLI -> JSON report)
  - species_zone_from_excel.py  (batch spreadsheet processing)
  - any future tooling

Core idea
---------
GBIF provides occurrence coordinates (and sometimes elevation/uncertainty), but no
temperature. We combine:
  1) GBIF occurrence points (lat/lon, optional elevation + uncertainty)
  2) A gridded ERA5-derived climate dataset (mean annual extreme minimum temp in °F)
  3) A consistent point-sampling API (USDAZoneDataset.point) that converts temp->zone

We then define a "cold-edge" point for the species as either:
  - absolute minimum sampled temperature ("min"), or
  - a low quantile of the sampled temperatures (default: 5th percentile), which is
    often more robust to outliers / bad records.

Dominance pruning (recommended)
-------------------------------
To speed up sampling (and reduce obviously-warm candidates), we apply a conservative
Pareto-frontier prune per hemisphere on (polewardness, elevation):

  - Northern Hemisphere: polewardness = latitude (higher is more poleward)
  - Southern Hemisphere: polewardness = -latitude (more negative latitude is more poleward)

A point B is dropped if there exists a point A that is:
  - at least as poleward as B, and
  - at least as high elevation as B (missing elevation treated as 0 for pruning),
  - and strictly better on at least one axis.

This pruning is conservative: it keeps any point that could plausibly be cold-edge.

Tropics:
  Latitude is less predictive of cold extremes, so pruning is skipped for |lat| < tropics_abs_lat_deg.

Thinning (recommended)
----------------------
GBIF records can be extremely dense. After pruning, optional grid-thinning (e.g. 25 km)
keeps at most one record per coarse cell to reduce redundant sampling.

Outputs
-------
The main entrypoint is ColdEdgeEstimator.estimate(), which returns a ColdEdgeResult containing:
  - counts at each stage (raw -> pruned -> thinned -> sampled)
  - the northernmost occurrence (diagnostic)
  - the coldest sampled climate point
  - the selected cold-edge point (min or quantile)
  - driver points (the N coldest sampled points)

The result_to_report_dict() helper converts ColdEdgeResult into a JSON-serializable dict.

Important: climate sampling is delegated to USDAZoneDataset.point(), so longitude
normalization, 0/360 seam handling, interpolation, and temp->zone conversion remain
centralized in usda_zone_core.py / usda_zone_access.py.

Notes on correctness
--------------------
This estimates the coldest (or cold-tail) *climate grid temperature* overlapping known GBIF
occurrences. It is not guaranteed to find the coldest surviving *micro-population*,
but it is a defensible, reproducible, auditable approximation.

Common reasons for failure / "Unknown":
  - GBIF returns no usable occurrences after filtering
  - The dataset grid has missing values at those locations (e.g., masked ocean/ice)
  - All sampled points return missing temp_f from USDAZoneDataset.point()

"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from usda_zone_access import USDAZoneDataset


GBIF_API = "https://api.gbif.org/v1/occurrence/search"

# Conservative defaults. You can broaden later if needed.
BASIS_OF_RECORD = {"HUMAN_OBSERVATION", "PRESERVED_SPECIMEN"}

# Exclude obvious non-wild points when GBIF provides this field.
EXCLUDED_ESTABLISHMENT = {"INTRODUCED", "CULTIVATED", "MANAGED", "ESCAPED", "NATURALISED"}


@dataclass(frozen=True)
class OccPoint:
    """
    A single GBIF occurrence point after basic filtering.

    Fields:
      lat/lon:
        Decimal degrees (WGS84).
      elevation_m:
        Optional; if missing, dominance pruning treats it as 0.
      uncertainty_m:
        coordinateUncertaintyInMeters; optional and often missing/noisy.
      gbif_key:
        Optional GBIF occurrence key for traceability.

    Notes:
      - GBIF does not provide climate; this is only metadata + coordinates.
    """

    lat: float
    lon: float
    country: Optional[str]
    year: Optional[int]
    basis: str
    establishment: Optional[str]
    uncertainty_m: Optional[float]
    elevation_m: Optional[float]
    gbif_key: Optional[int] = None


@dataclass(frozen=True)
class ClimatePoint:
    """
    A climate lookup result corresponding to a GBIF occurrence.

    temp_f / zone are derived by USDAZoneDataset.point() from the gridded climatology.
    grid_lat/grid_lon are optional fields; many implementations won't set them unless
    USDAZoneDataset exposes them.
    """

    lat: float
    lon: float
    temp_f: float
    zone: str
    grid_lat: Optional[float] = None
    grid_lon: Optional[float] = None


@dataclass(frozen=True)
class ColdEdgeConfig:
    """
    Configuration for cold-edge estimation.

    max_records:
      Upper bound of GBIF points fetched (after paging).
    page_size:
      GBIF page size (GBIF allows up to ~300 reliably).
    max_uncertainty_m:
      Filter out points with coordinateUncertaintyInMeters > this value.
    sleep_between_pages_s:
      Politeness delay between GBIF pages.

    dominance_pruning:
      Enable hemisphere-aware Pareto pruning on (polewardness, elevation).
    tropics_abs_lat_deg:
      For |lat| < this, dominance pruning is skipped.

    grid_km:
      Grid thinning cell size; larger = faster but less dense sampling.

    use_min:
      If True, cold-edge is absolute coldest sampled temp. Otherwise use quantile.
    quantile:
      Low quantile (0..1) for cold-edge selection; default 0.05 = 5th percentile.

    drivers:
      Number of coldest sampled points included as "drivers" in output (audit trail).

    http_timeout_s / http_retries:
      Network robustness settings for GBIF API.
    """

    # GBIF fetch / cleaning
    max_records: int = 5000
    page_size: int = 300
    max_uncertainty_m: int = 10_000
    sleep_between_pages_s: float = 0.2

    # Pruning / thinning
    dominance_pruning: bool = True
    tropics_abs_lat_deg: float = 20.0
    grid_km: float = 25.0
    do_thin: bool = True

    # Cold-edge selection
    use_min: bool = False
    quantile: float = 0.05

    # Reporting
    drivers: int = 25

    # Requests retries
    http_timeout_s: int = 60
    http_retries: int = 5


@dataclass(frozen=True)
class ColdEdgeResult:
    """
    Results of a cold-edge estimation run for a single species.

    Counts:
      n_gbif_points_cleaned:
        Points that survived initial GBIF filters.
      n_points_after_prune:
        After dominance pruning (and tropical exception).
      n_points_after_thinning:
        After grid thinning (if enabled).
      n_points_with_climate:
        Points that produced a non-missing climate lookup.

    Key diagnostics:
      northernmost_occurrence:
        Poleward-most point in NH if present, else max-lat overall (diagnostic only).
      coldest:
        Single coldest sampled climate point.
      cold_edge:
        Selected cold-edge point (min or quantile).
      drivers:
        List of N coldest sampled points for auditability.
    """

    species: str
    dataset_path: str
    config: ColdEdgeConfig

    n_gbif_points_cleaned: int
    n_points_after_prune: int
    n_points_after_thinning: int
    n_points_with_climate: int

    northernmost_occurrence: Optional[OccPoint]

    coldest: Tuple[ClimatePoint, OccPoint]

    edge_method: str  # "min" or "quantile"
    quantile: Optional[float]
    selected_index_in_sorted: int
    cold_edge: Tuple[ClimatePoint, OccPoint]

    drivers: List[Tuple[ClimatePoint, OccPoint]]


def _gbif_get(session: requests.Session, params: Dict[str, Any], *, timeout: int, retries: int) -> Dict[str, Any]:
    """
    Fetch one page from GBIF with basic retry/backoff.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(GBIF_API, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(min(2**attempt, 20))
    assert last_err is not None
    raise last_err


def fetch_gbif_occurrences(scientific_name: str, cfg: ColdEdgeConfig, *, session: Optional[requests.Session] = None) -> List[OccPoint]:
    """
    Fetch GBIF occurrence points for a scientific name and apply quality filters.

    Filters applied:
      - hasCoordinate=true
      - occurrenceStatus=PRESENT
      - basisOfRecord in BASIS_OF_RECORD
      - coordinateUncertaintyInMeters <= cfg.max_uncertainty_m (if provided)
      - establishmentMeans not in EXCLUDED_ESTABLISHMENT (if provided)

    Elevation:
      - GBIF fields vary. Prefer 'elevation', fall back to 'elevationInMeters'.

    Returns:
      A list of OccPoint.
    """
    own = False
    if session is None:
        session = requests.Session()
        own = True

    try:
        points: List[OccPoint] = []
        offset = 0

        while offset < cfg.max_records:
            params: Dict[str, Any] = {
                "scientificName": scientific_name,
                "hasCoordinate": "true",
                "occurrenceStatus": "PRESENT",
                "limit": cfg.page_size,
                "offset": offset,
                "basisOfRecord": list(BASIS_OF_RECORD),
            }

            payload = _gbif_get(session, params, timeout=cfg.http_timeout_s, retries=cfg.http_retries)
            records = payload.get("results", [])
            if not records:
                break

            for rec in records:
                lat = rec.get("decimalLatitude")
                lon = rec.get("decimalLongitude")
                if lat is None or lon is None:
                    continue

                cu = rec.get("coordinateUncertaintyInMeters")
                if cu is not None and cu > cfg.max_uncertainty_m:
                    continue

                bor = rec.get("basisOfRecord")
                if bor not in BASIS_OF_RECORD:
                    continue

                em = rec.get("establishmentMeans")
                if em and str(em).upper() in EXCLUDED_ESTABLISHMENT:
                    continue

                elev = rec.get("elevation")
                if elev is None:
                    elev = rec.get("elevationInMeters")

                elev_f: Optional[float]
                try:
                    elev_f = float(elev) if elev is not None else None
                except Exception:
                    elev_f = None

                key = rec.get("key")
                key_i: Optional[int]
                try:
                    key_i = int(key) if key is not None else None
                except Exception:
                    key_i = None

                points.append(
                    OccPoint(
                        lat=float(lat),
                        lon=float(lon),
                        country=rec.get("country"),
                        year=rec.get("year"),
                        basis=str(bor),
                        establishment=em,
                        uncertainty_m=float(cu) if cu is not None else None,
                        elevation_m=elev_f,
                        gbif_key=key_i,
                    )
                )

                if len(points) >= cfg.max_records:
                    break

            if payload.get("endOfRecords"):
                break

            offset += cfg.page_size
            time.sleep(cfg.sleep_between_pages_s)

        return points
    finally:
        if own:
            session.close()


def thin_points_km(points: Sequence[OccPoint], grid_km: float) -> List[OccPoint]:
    """
    Coarse grid thinning to reduce redundant points.

    Implementation:
      - Convert lat/lon to approximate kilometers using a simple equirectangular scale:
        lon_km ≈ lon * 111.320 * cos(lat)
        lat_km ≈ lat * 110.574
      - Keep at most one point per (floor(lon_km/grid), floor(lat_km/grid)) cell.

    This is a speed optimization; it intentionally reduces point density.
    """
    if not points:
        return []
    grid = float(grid_km)
    seen: set[Tuple[int, int]] = set()
    kept: List[OccPoint] = []
    for p in points:
        lat_rad = math.radians(p.lat)
        x_km = p.lon * 111.320 * math.cos(lat_rad)
        y_km = p.lat * 110.574
        key = (int(math.floor(x_km / grid)), int(math.floor(y_km / grid)))
        if key in seen:
            continue
        seen.add(key)
        kept.append(p)
    return kept


def dominance_prune(points: Sequence[OccPoint], cfg: ColdEdgeConfig) -> List[OccPoint]:
    """
    Conservative Pareto-frontier pruning on (polewardness, elevation) per hemisphere.

    Tropics handling:
      - For |lat| < cfg.tropics_abs_lat_deg, pruning is skipped (latitude less predictive).

    Dominance rule (per hemisphere):
      A point B is dropped if there exists point A such that:
        poleward(A) >= poleward(B) and elev(A) >= elev(B)
        and at least one is strictly greater.

      - Missing elevation is treated as 0 for the pruning test (conservative but fast).
      - This keeps any point that is not clearly dominated on BOTH axes.

    Output:
      pruned points (plus untouched tropical points).
    """
    if not cfg.dominance_pruning or not points:
        return list(points)

    tropics: List[OccPoint] = []
    extratropics: List[OccPoint] = []
    for p in points:
        if abs(p.lat) < cfg.tropics_abs_lat_deg:
            tropics.append(p)
        else:
            extratropics.append(p)

    if not extratropics:
        return list(points)

    nh = [p for p in extratropics if p.lat >= 0.0]
    sh = [p for p in extratropics if p.lat < 0.0]

    def prune_group(ps: List[OccPoint], polewardness_fn) -> List[OccPoint]:
        items = [
            (float(polewardness_fn(p)), float(p.elevation_m if p.elevation_m is not None else 0.0), p)
            for p in ps
        ]
        # Sort poleward desc then elevation desc so we can sweep with max elevation.
        items.sort(key=lambda t: (t[0], t[1]), reverse=True)

        kept: List[OccPoint] = []
        best_elev = -1e30
        for pole, elev, p in items:
            # If a previously-seen point was >= poleward (by sort) and had >= elev (best_elev),
            # then this point is dominated and can be dropped.
            if elev <= best_elev:
                continue
            kept.append(p)
            best_elev = elev
        return kept

    nh_kept = prune_group(nh, polewardness_fn=lambda p: p.lat)
    sh_kept = prune_group(sh, polewardness_fn=lambda p: -p.lat)

    return nh_kept + sh_kept + tropics


def _getattr_any(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    """
    Helper for flexible attribute lookup across variants of ZonePoint objects.
    """
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def sample_climate(zds: USDAZoneDataset, points: Sequence[OccPoint]) -> List[Tuple[ClimatePoint, OccPoint]]:
    """
    Sample the climate dataset for each occurrence point.

    Notes:
      - This delegates all spatial logic to USDAZoneDataset.point() (and by extension
        usda_zone_core.py).
      - Points that return missing temp_f are dropped.
    """
    out: List[Tuple[ClimatePoint, OccPoint]] = []
    for p in points:
        r = zds.point(p.lat, p.lon)
        temp_f = _getattr_any(r, ["temp_f"])
        zone = _getattr_any(r, ["zone_label"], default="—")
        if temp_f is None:
            continue

        cp = ClimatePoint(
            lat=p.lat,
            lon=p.lon,
            temp_f=float(temp_f),
            zone=str(zone),
            grid_lat=_getattr_any(r, ["grid_lat"], default=None),
            grid_lon=_getattr_any(r, ["grid_lon"], default=None),
        )
        out.append((cp, p))
    return out


def pick_coldest(sampled: Sequence[Tuple[ClimatePoint, OccPoint]]) -> Tuple[ClimatePoint, OccPoint]:
    """
    Return the single coldest sampled point by temp_f.
    """
    if not sampled:
        raise ValueError("No sampled points with climate.")
    return min(sampled, key=lambda t: t[0].temp_f)


def pick_cold_edge_quantile(
    sampled: Sequence[Tuple[ClimatePoint, OccPoint]],
    quantile: float,
) -> Tuple[ClimatePoint, OccPoint, int]:
    """
    Pick a cold-edge point at a low quantile of the sampled temp distribution.

    quantile:
      - must be in (0,1), e.g. 0.05 for the 5th percentile
      - selection is computed as floor(q*(n-1)) on temp-sorted samples

    Returns:
      (ClimatePoint, OccPoint, selected_index_in_sorted)
    """
    if not sampled:
        raise ValueError("No sampled points with climate.")
    q = float(quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("quantile must be between 0 and 1 (exclusive)")
    s = sorted(sampled, key=lambda t: t[0].temp_f)
    idx = int(math.floor(q * (len(s) - 1)))
    return s[idx][0], s[idx][1], idx


def pick_northernmost(points: Sequence[OccPoint]) -> Optional[OccPoint]:
    """
    Return the most poleward occurrence (diagnostic):
      - NH: max latitude
      - SH-only species: max latitude (least negative) is not "poleward", but still
        useful as a sanity signal for distribution; the cold-edge is climate-based.
    """
    if not points:
        return None
    nh = [p for p in points if p.lat >= 0.0]
    if nh:
        return max(nh, key=lambda p: p.lat)
    return max(points, key=lambda p: p.lat)


class ColdEdgeEstimator:
    """
    Orchestrates GBIF fetch → prune/thin → climate sample → cold-edge selection.

    Usage:
        cfg = ColdEdgeConfig(...)
        est = ColdEdgeEstimator(dataset_path, cfg)
        res = est.estimate("Abies koreana")
        report = result_to_report_dict(res)

    Performance notes:
      - The dataset is opened once per estimator and reused for multiple species if you
        keep the estimator alive.
      - GBIF API calls dominate time for many species; consider caching GBIF responses
        externally if running huge batches repeatedly.
    """

    def __init__(self, dataset_path: str, cfg: Optional[ColdEdgeConfig] = None):
        self.dataset_path = str(dataset_path)
        self.cfg = cfg if cfg is not None else ColdEdgeConfig()
        self._session = requests.Session()
        self._zds: Optional[USDAZoneDataset] = None

    def __enter__(self) -> "ColdEdgeEstimator":
        self._zds = USDAZoneDataset(self.dataset_path)
        self._zds.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._zds is not None:
            self._zds.__exit__(exc_type, exc, tb)
            self._zds = None
        self._session.close()

    def estimate(self, species: str) -> ColdEdgeResult:
        """
        End-to-end cold-edge estimation for one species.

        Steps:
          1) fetch_gbif_occurrences() with filtering
          2) dominance_prune() (optional; enabled by default)
          3) thin_points_km() (optional; enabled by default)
          4) sample climate via USDAZoneDataset.point()
          5) pick coldest point and either:
               - cfg.use_min=True: cold_edge = coldest
               - else: cold_edge = quantile point (default 5th percentile)
          6) compute driver list = N coldest sampled points

        Raises:
          RuntimeError if no climate samples overlap the grid (or all missing temp_f).
        """
        if self._zds is None:
            raise RuntimeError("ColdEdgeEstimator must be used as a context manager: with ColdEdgeEstimator(...) as est:")

        cfg = self.cfg

        pts = fetch_gbif_occurrences(species, cfg, session=self._session)
        pts_pruned = dominance_prune(pts, cfg)
        if cfg.do_thin:
            pts_thin = thin_points_km(pts_pruned, grid_km=cfg.grid_km)
        else:
            pts_thin = list(pts_pruned)

        northmost = pick_northernmost(pts_thin)

        sampled = sample_climate(self._zds, pts_thin)
        if not sampled:
            raise RuntimeError("No GBIF points overlapped the climate grid (or all returned missing temp_f).")

        coldest_cp, coldest_occ = pick_coldest(sampled)

        if cfg.use_min:
            edge_cp, edge_occ = coldest_cp, coldest_occ
            edge_idx = 0
            method = "min"
            qval: Optional[float] = None
        else:
            edge_cp, edge_occ, edge_idx = pick_cold_edge_quantile(sampled, quantile=cfg.quantile)
            method = "quantile"
            qval = float(cfg.quantile)

        sampled_sorted = sorted(sampled, key=lambda t: t[0].temp_f)
        driver_n = max(1, min(int(cfg.drivers), len(sampled_sorted)))
        driver_list = sampled_sorted[:driver_n]

        return ColdEdgeResult(
            species=species,
            dataset_path=self.dataset_path,
            config=cfg,
            n_gbif_points_cleaned=len(pts),
            n_points_after_prune=len(pts_pruned),
            n_points_after_thinning=len(pts_thin),
            n_points_with_climate=len(sampled),
            northernmost_occurrence=northmost,
            coldest=(coldest_cp, coldest_occ),
            edge_method=method,
            quantile=qval,
            selected_index_in_sorted=int(edge_idx),
            cold_edge=(edge_cp, edge_occ),
            drivers=driver_list,
        )


def result_to_report_dict(res: ColdEdgeResult) -> Dict[str, Any]:
    """
    Convert ColdEdgeResult to a JSON-serializable dict.

    Intended for CLIs that write reports to disk (gbif_cold_edge_era5.py) or batch
    tools that want structured output.
    """
    coldest_cp, coldest_occ = res.coldest
    edge_cp, edge_occ = res.cold_edge

    drivers: List[Dict[str, Any]] = []
    for cp, occ in res.drivers:
        drivers.append({"climate": asdict(cp), "occurrence": asdict(occ)})

    return {
        "ok": True,
        "species": res.species,
        "dataset": res.dataset_path,
        "config": asdict(res.config),
        "n_gbif_points_cleaned": res.n_gbif_points_cleaned,
        "n_points_after_dominance_prune": res.n_points_after_prune,
        "n_points_after_thinning": res.n_points_after_thinning,
        "n_points_with_climate": res.n_points_with_climate,
        "northernmost_occurrence": asdict(res.northernmost_occurrence) if res.northernmost_occurrence else None,
        "coldest_point": {"climate": asdict(coldest_cp), "occurrence": asdict(coldest_occ)},
        "edge_method": res.edge_method,
        "quantile": res.quantile,
        "selected_index_in_sorted": res.selected_index_in_sorted,
        "cold_edge": {"climate": asdict(edge_cp), "occurrence": asdict(edge_occ)},
        "drivers_coldest": drivers,
    }
