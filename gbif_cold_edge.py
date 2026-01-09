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

Performance model
-----------------
For large batches, GBIF HTTP calls dominate runtime.

This module supports a fast two-stage workflow:
  1) Fetch GBIF occurrences (parallelizable across species; see species_zone_from_excel.py).
  2) Sample climate grid (usually done in a single thread with one open dataset).

To enable that, the estimator exposes both:
  - fetch_gbif_occurrences()                -> List[OccPoint]
  - estimate_from_occurrences(points, ...)  -> ColdEdgeResult (no GBIF calls)

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

Key bugfixes (rate-limit + resumability)
----------------------------------------
- Cross-thread throttle is applied consistently to BOTH GBIF endpoints used:
    /v1/occurrence/search and /v1/species/match
- Repeated 429s during paging no longer silently result in an empty occurrence list
  that looks like "NO_GBIF_POINTS". If rate limiting prevents getting enough points,
  fetch_gbif_occurrences raises GbifError("GBIF_RATE_LIMIT", ...) so callers can backoff/retry.
"""

from __future__ import annotations

import math
import time
import threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import requests
from usda_zone_access import USDAZoneDataset

@dataclass
class GbifFetchDiagnostics:
    species: str = ""
    usage_key: Optional[int] = None
    used_query_mode: str = ""  # "taxonKey" or "scientificName"

    pages: int = 0
    gbif_total_reported: Optional[int] = None  # payload.get("count") (GBIF provides this)

    # Filtering counters
    n_raw_records_seen: int = 0
    n_missing_coords: int = 0
    n_uncertainty_filtered: int = 0
    n_basis_filtered: int = 0
    n_establishment_filtered: int = 0
    n_kept: int = 0

    # Failure info (if any)
    fail_reason: Optional[str] = None
    fail_detail: Optional[str] = None

class GbifError(RuntimeError):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def make_gbif_session(*, pool_connections: int = 16, pool_maxsize: int = 16) -> requests.Session:
    """
    Create a Session with bounded connection pools so threaded runs do not
    create unbounded sockets.
    """
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=int(pool_connections),
        pool_maxsize=int(pool_maxsize),
        pool_block=True,
        max_retries=0,  # we do retries ourselves in _gbif_get
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

    # Default headers for all requests from this session
    #s.headers.update({
    #    "Accept": "application/json",
    #    "Connection": "close",  # helps avoid lingering CLOSE-WAIT on some networks
    #)


GBIF_API = "https://api.gbif.org/v1/occurrence/search"
# Conservative cross-thread throttle for GBIF calls.
GBIF_SPECIES_MATCH = "https://api.gbif.org/v1/species/match"
# Ensures we don't hammer GBIF even if species fetching is parallelized.
_GBIF_THROTTLE_LOCK = threading.Lock()
_GBIF_NEXT_ALLOWED_TS = 0.0
_GBIF_MIN_INTERVAL_S = 0.25  # ~4 requests/sec across all threads (tune 0.25..0.5)
def _gbif_throttle() -> None:
    """Best-effort cross-thread global pacing for any GBIF request."""
    global _GBIF_NEXT_ALLOWED_TS
    with _GBIF_THROTTLE_LOCK:
        now = time.time()
        if now < _GBIF_NEXT_ALLOWED_TS:
            time.sleep(_GBIF_NEXT_ALLOWED_TS - now)
        _GBIF_NEXT_ALLOWED_TS = time.time() + float(_GBIF_MIN_INTERVAL_S)

# Plants are frequently OBSERVATION; keep this inclusive.
BASIS_OF_RECORD = {
    "HUMAN_OBSERVATION",
    "OBSERVATION",
    "PRESERVED_SPECIMEN",
    "LIVING_SPECIMEN",
    "MACHINE_OBSERVATION",
}

# Exclude obvious non-wild points when GBIF provides this field.
#EXCLUDED_ESTABLISHMENT = {"INTRODUCED", "CULTIVATED", "MANAGED", "ESCAPED", "NATURALISED"}
EXCLUDED_ESTABLISHMENT = {}

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
    do_thin:
      Enable thinning.

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

    min_points_for_partial_ok: int = 500   # proceed if we have at least this many cleaned points
    max_rate_limit_breaks: int = 2         # how many 429s before we give up paging

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


def _gbif_get(
    session: requests.Session,
    params: Dict[str, Any],
    *,
    timeout: int,
    retries: int,
) -> Dict[str, Any]:
    """
    Fetch one page from the GBIF occurrence API with retry/backoff.

    - requests.Response is NOT a context manager
    - response.close() must be called explicitly to avoid FD leaks
    - handle 429 using Retry-After if present
    """
    # Cross-thread global pacing (best-effort)

    last_reason = None
    last_detail = None

    for attempt in range(1, retries + 1):
        r: Optional[requests.Response] = None
        try:
            _gbif_throttle()
            r = session.get(
                GBIF_API,
                params=params,
                timeout=(10, timeout),
            )

            if r.status_code == 429:
                ra = (r.headers.get("Retry-After") or "").strip()
                try:
                    wait_s = int(ra)
                    why = f"429 Retry-After={wait_s}"
                except Exception:
                    wait_s = min(2 ** attempt, 60)
                    why = f"429 no Retry-After, backoff={wait_s}"
                last_reason, last_detail = "GBIF_RATE_LIMIT", f"{why} params={params!r}"
                time.sleep(max(1, wait_s))
                continue

            if r.status_code in (500, 502, 503, 504):
                wait_s = min(2 ** attempt, 20)
                last_reason, last_detail = "GBIF_SERVER_ERROR", f"status={r.status_code} backoff={wait_s} params={params!r}"
                time.sleep(wait_s)
                continue

            if r.status_code != 200:
                body = (r.text or "")[:200].replace("\n", " ")
                raise GbifError("GBIF_HTTP_ERROR", f"status={r.status_code} body={body!r} params={params!r}")

            try:
                return r.json()
            except Exception as e:
                raise GbifError("GBIF_DECODE_ERROR", f"{type(e).__name__}: {e} params={params!r}")

        except GbifError:
            # Non-retryable (except you could choose to retry decode errors)
            raise
        except requests.exceptions.Timeout as e:
            wait_s = min(2 ** attempt, 20)
            last_reason, last_detail = "GBIF_TIMEOUT", f"{type(e).__name__}: {e} backoff={wait_s} params={params!r}"
            time.sleep(wait_s)
        except requests.exceptions.ConnectionError as e:
            wait_s = min(2 ** attempt, 20)
            last_reason, last_detail = "GBIF_CONNECTION_ERROR", f"{type(e).__name__}: {e} backoff={wait_s} params={params!r}"
            time.sleep(wait_s)
        except Exception as e:
            wait_s = min(2 ** attempt, 20)
            last_reason, last_detail = "GBIF_UNKNOWN_ERROR", f"{type(e).__name__}: {e} backoff={wait_s} params={params!r}"
            time.sleep(wait_s)
        finally:
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass

    raise GbifError(last_reason or "GBIF_FAILED", last_detail or f"exhausted retries params={params!r}")



def _gbif_species_match(
    session: requests.Session,
    name: str,
    *,
    timeout: int,
    retries: int,
) -> Optional[int]:
    """
    Resolve a scientific name to a GBIF backbone usageKey (taxonKey for occurrence/search).
    Conservative: requires confidence >= 70.
    Returns usageKey if match is confident enough, else None.
    """

    for attempt in range(1, retries + 1):
        r: Optional[requests.Response] = None
        try:
            _gbif_throttle()
            r = session.get(
                GBIF_SPECIES_MATCH,
                params={"name": name},
                timeout=(10, timeout),
                headers={"Accept": "application/json"},
            )
            if r.status_code == 429:
                ra = (r.headers.get("Retry-After") or "").strip()
                try:
                    wait_s = int(ra)
                except Exception:
                    wait_s = min(2 ** attempt, 20)
                time.sleep(max(1, wait_s))
                continue

            if r.status_code in (500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 20))
                continue

            r.raise_for_status()
            js = r.json()

            usage_key = js.get("usageKey")
            confidence = js.get("confidence", 0)

            # Keep this conservative; you can relax later if needed.
            if usage_key is not None and int(confidence) >= 70:
                return int(usage_key)

            return None

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(min(2 ** attempt, 20))
        except Exception:
            time.sleep(min(2 ** attempt, 20))
        finally:
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass

    # If match endpoint fails repeatedly, just fall back to name-based querying.
    return None


def fetch_gbif_occurrences(
    scientific_name: str,
    cfg: ColdEdgeConfig,
    *,
    session: Optional[requests.Session] = None,
    diag: Optional[GbifFetchDiagnostics] = None,
) -> List[OccPoint]:
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

    Notes:
      - This function does *network I/O* and is therefore parallelizable (threads).
      - For large batch runs, consider fetching in parallel and then calling
        estimate_from_occurrences() to do all climate sampling with one open dataset.

    If `diag` is provided, it will be populated with:
      - query mode (taxonKey vs scientificName), usage_key
      - paging counts, GBIF-reported total count
      - filter counters (basis/uncertainty/etc)
      - fail_reason/fail_detail on GBIF errors

    Important:
      If repeated 429s prevent collecting enough points (and you don't have cfg.min_points_for_partial_ok),
      this raises GbifError("GBIF_RATE_LIMIT", ...) rather than returning [] (which would be indistinguishable
      from "no occurrences").
    """
    own = False
    if session is None:
        session = make_gbif_session()
        own = True

    if diag is not None:
        diag.species = scientific_name  # harmless if already set

    try:
        points: List[OccPoint] = []
        offset = 0

        # Resolve to GBIF backbone usageKey first (synonyms/backbone handling)
        usage_key = _gbif_species_match(
            session,
            scientific_name,
            timeout=cfg.http_timeout_s,
            retries=cfg.http_retries,
        )

        if diag is not None:
            diag.usage_key = usage_key
            diag.used_query_mode = "taxonKey" if usage_key is not None else "scientificName"

        rate_limit_breaks = 0
        last_rate_limit_err: Optional[GbifError] = None
        while offset < cfg.max_records:
            params: Dict[str, Any] = {
                "hasCoordinate": "true",
                "occurrenceStatus": "PRESENT",
                "limit": cfg.page_size,
                "offset": offset,
                # requests encodes list values as repeated query params, which GBIF accepts:
                # basisOfRecord=...&basisOfRecord=...
                "basisOfRecord": list(BASIS_OF_RECORD),
            }

            if usage_key is not None:
                params["taxonKey"] = usage_key
            else:
                params["scientificName"] = scientific_name

            try:
                payload = _gbif_get(
                    session,
                    params,
                    timeout=cfg.http_timeout_s,
                    retries=cfg.http_retries,
                )
            except GbifError as e:
                if getattr(e, "reason", "") == "GBIF_RATE_LIMIT":
                    rate_limit_breaks += 1
                    last_rate_limit_err = e
                    if diag is not None:
                        diag.fail_reason = e.reason
                        diag.fail_detail = e.detail

                    # If we already have enough points, stop paging and return what we have.
                    if len(points) >= cfg.min_points_for_partial_ok:
                        break
                    if rate_limit_breaks >= cfg.max_rate_limit_breaks:
                        raise GbifError(
                            "GBIF_RATE_LIMIT",
                            f"rate_limited during paging; breaks={rate_limit_breaks} "
                            f"points={len(points)} (<{cfg.min_points_for_partial_ok}) last={e.detail}",
                        )

                    # else: keep trying (the outer loop continues)
                    continue

                # non-rate-limit: still a hard failure
                if diag is not None:
                    diag.fail_reason = getattr(e, "reason", "GBIF_ERROR")
                    diag.fail_detail = getattr(e, "detail", str(e))
                raise

            if diag is not None:
                diag.pages += 1
                if diag.gbif_total_reported is None:
                    c = payload.get("count")
                    if isinstance(c, int):
                        diag.gbif_total_reported = c

            records = payload.get("results", [])
            if not records:
                break

            for rec in records:
                if diag is not None:
                    diag.n_raw_records_seen += 1

                lat = rec.get("decimalLatitude")
                lon = rec.get("decimalLongitude")
                if lat is None or lon is None:
                    if diag is not None:
                        diag.n_missing_coords += 1
                    continue

                cu = rec.get("coordinateUncertaintyInMeters")

                cu_f: Optional[float] = None
                if cu is not None:
                    try:
                        cu_f = float(cu)
                    except Exception:
                        cu_f = None  # treat unparsable uncertainty as "missing"

                if cu_f is not None and cu_f > cfg.max_uncertainty_m:
                    if diag is not None:
                        diag.n_uncertainty_filtered += 1
                    continue

                bor = rec.get("basisOfRecord")
                if bor not in BASIS_OF_RECORD:
                    if diag is not None:
                        diag.n_basis_filtered += 1
                    continue

                em = rec.get("establishmentMeans")
                if em and str(em).upper() in EXCLUDED_ESTABLISHMENT:
                    if diag is not None:
                        diag.n_establishment_filtered += 1
                    continue

                elev = rec.get("elevation")
                if elev is None:
                    elev = rec.get("elevationInMeters")

                try:
                    elev_f = float(elev) if elev is not None else None
                except Exception:
                    elev_f = None

                key = rec.get("key")
                try:
                    key_i = int(key) if key is not None else None
                except Exception:
                    key_i = None

                try:
                    lat_f = float(lat)
                    lon_f = float(lon)
                except Exception:
                    if diag is not None:
                        diag.n_missing_coords += 1
                    continue

                points.append(
                    OccPoint(
                        lat=lat_f,
                        lon=lon_f,
                        country=rec.get("country"),
                        year=rec.get("year"),
                        basis=str(bor),
                        establishment=em,
                        uncertainty_m=cu_f,
                        elevation_m=elev_f,
                        gbif_key=key_i,
                    )
                )

                if diag is not None:
                    diag.n_kept += 1

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



# --------------------------------------------------------------------
# Prune / thin
# --------------------------------------------------------------------

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

@dataclass
class ClimateSampleDiagnostics:
    n_total: int = 0
    n_ok: int = 0
    n_temp_missing: int = 0
    n_temp_nan: int = 0
    n_exception: int = 0
    examples_missing: List[Tuple[float, float, str]] = field(default_factory=list)  # (lat, lon, why)
    examples_limit: int = 20

def sample_climate(
    zds: USDAZoneDataset,
    points: Sequence[OccPoint],
    *,
    diag: Optional[ClimateSampleDiagnostics] = None,
) -> List[Tuple[ClimatePoint, OccPoint]]:
    out: List[Tuple[ClimatePoint, OccPoint]] = []
    if diag is None:
        diag = ClimateSampleDiagnostics()

    for p in points:
        diag.n_total += 1
        try:
            r = zds.point(p.lat, p.lon)  # USDAZoneDataset does lon normalization

            temp_f = _getattr_any(r, ["temp_f"])
            zone = _getattr_any(r, ["zone_label"], default="—")

            if temp_f is None:
                diag.n_temp_missing += 1
                if len(diag.examples_missing) < diag.examples_limit:
                    diag.examples_missing.append((p.lat, p.lon, "temp_f is None"))
                continue

            # If temp_f is NaN, treat explicitly (common with xarray interpolation)
            tf = float(temp_f)
            if math.isnan(tf):
                diag.n_temp_nan += 1
                if len(diag.examples_missing) < diag.examples_limit:
                    diag.examples_missing.append((p.lat, p.lon, "temp_f is NaN"))
                continue

            diag.n_ok += 1
            cp = ClimatePoint(
                lat=p.lat,
                lon=p.lon,  # report raw GBIF lon
                temp_f=tf,
                zone=str(zone),
                grid_lat=_getattr_any(r, ["grid_lat"], default=None),
                grid_lon=_getattr_any(r, ["grid_lon"], default=None),
            )
            out.append((cp, p))

        except Exception as e:
            diag.n_exception += 1
            if len(diag.examples_missing) < diag.examples_limit:
                diag.examples_missing.append((p.lat, p.lon, f"exception: {type(e).__name__}: {e}"))
            continue

    return out



def pick_coldest(sampled: Sequence[Tuple[ClimatePoint, OccPoint]]) -> Tuple[ClimatePoint, OccPoint]:
    """Return the single coldest sampled point by temp_f."""
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


def estimate_from_occurrences(
    *,
    species: str,
    dataset_path: str,
    cfg: ColdEdgeConfig,
    occurrences: Sequence[OccPoint],
    zds: USDAZoneDataset,
) -> ColdEdgeResult:
    """
    Estimate cold-edge for `species` using already-fetched occurrences.

    This performs *no network I/O*.

    Typical use:
      - Fetch GBIF occurrences for many species in parallel threads.
      - Open the climate dataset once.
      - Call this function for each species sequentially.
    """
    min_candidates = 25

    pts_pruned = dominance_prune(occurrences, cfg)
    if len(pts_pruned) < min_candidates and len(occurrences) >= min_candidates:
        pts_pruned = list(occurrences)  # fall back: no prune

    pts_thin = thin_points_km(pts_pruned, cfg.grid_km) if cfg.do_thin else list(pts_pruned)
    if len(pts_thin) < min_candidates and len(pts_pruned) >= min_candidates:
        pts_thin = pts_pruned  # fall back: no thinning

    northmost = pick_northernmost(pts_thin)

    def _try_sample(points: Sequence[OccPoint]) -> tuple[list[tuple[ClimatePoint, OccPoint]], ClimateSampleDiagnostics]:
        d = ClimateSampleDiagnostics()
        s = sample_climate(zds, points, diag=d)
        return s, d

    # stage 1: current candidate set
    sampled, diag = _try_sample(pts_thin)

    # stage 2: unthinned pruned
    if not sampled and cfg.do_thin and len(pts_pruned) > len(pts_thin):
        sampled, diag = _try_sample(pts_pruned)
    # stage 3: completely unpruned/unthinned
    if not sampled and len(occurrences) > len(pts_pruned):
        sampled, diag = _try_sample(list(occurrences))

    if not sampled:
        raise RuntimeError(
            f"No climate samples. species={species!r} "
            f"gbif_cleaned={len(occurrences)} pruned={len(pts_pruned)} thinned={len(pts_thin)} "
            f"sample_total={diag.n_total} ok={diag.n_ok} "
            f"missing={diag.n_temp_missing} nan={diag.n_temp_nan} exc={diag.n_exception} "
            f"examples={diag.examples_missing[:5]!r}"
        )


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
        dataset_path=str(dataset_path),
        config=cfg,
        n_gbif_points_cleaned=len(occurrences),
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


class ColdEdgeEstimator:
    """
    Orchestrates GBIF fetch → prune/thin → climate sample → cold-edge selection.

    Usage:
        cfg = ColdEdgeConfig(...)
        with ColdEdgeEstimator(dataset_path, cfg) as est:
            res = est.estimate("Abies koreana")
            report = result_to_report_dict(res)

    Performance notes:
      - The dataset is opened once per estimator and reused for multiple species if you
        keep the estimator alive.
      - For large batches, you can fetch GBIF occurrences in parallel and then call
        estimate_from_occurrences() to avoid opening the dataset in multiple threads.
    """

    def __init__(self, dataset_path: str, cfg: Optional[ColdEdgeConfig] = None):
        self.dataset_path = str(dataset_path)
        self.cfg = cfg if cfg is not None else ColdEdgeConfig()
        self._session: Optional[requests.Session] = None
        self._zds: Optional[USDAZoneDataset] = None

    def __enter__(self) -> "ColdEdgeEstimator":
        # One session and one open dataset for the lifetime of the context.
        self._session = make_gbif_session(pool_maxsize=10)
        self._zds = USDAZoneDataset(self.dataset_path)
        self._zds.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._zds is not None:
            self._zds.__exit__(exc_type, exc, tb)
            self._zds = None
        if self._session is not None:
            self._session.close()
            self._session = None

    def fetch_occurrences(self, species: str) -> List[OccPoint]:
        """Convenience wrapper using the estimator's session."""
        if self._session is None:
            raise RuntimeError("ColdEdgeEstimator must be used as a context manager.")
        return fetch_gbif_occurrences(species, self.cfg, session=self._session)

    def estimate(self, species: str) -> ColdEdgeResult:
        """
        End-to-end cold-edge estimation for one species.

        Raises:
          RuntimeError if no climate samples overlap the grid (or all missing temp_f).
        """
        if self._session is None or self._zds is None:
            raise RuntimeError("ColdEdgeEstimator must be used as a context manager.")

        occ = fetch_gbif_occurrences(species, self.cfg, session=self._session)
        return estimate_from_occurrences(
            species=species,
            dataset_path=self.dataset_path,
            cfg=self.cfg,
            occurrences=occ,
            zds=self._zds,
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
