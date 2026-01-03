#!/usr/bin/env python3
"""
gbif_species_cold_edge_era5.py

Fetch GBIF occurrences for a plant species, apply basic quality filters, optionally
spatially thin the points, sample an ERA5-Land-derived “USDA-zone temperature metric”
using your centralized USDAZoneDataset API, then report:

1) Northernmost occurrence (Northern Hemisphere if available, else global).
2) Coldest climate-sampled point (absolute minimum of the climatology metric).
3) A “cold-edge” point chosen either as:
   - absolute coldest (use-min), or
   - a cold-edge quantile (default 5th percentile among sampled points).

Finally it writes a JSON report capturing the chosen point(s) plus “driver” points that
explain the decision (the coldest N sampled points).

Core assumptions and invariants
-------------------------------
- This script is intentionally thin: it does not implement any longitude wrap logic,
  gridpoint selection, or interpolation. All climate querying must go through:

      USDAZoneDataset.point(lat, lon)

  so that any fixes to wraparound, cyclic longitude, or interpolation are centralized
  in your dataset API.

- ERA5-Land resolution is coarse vs PRISM; point-level USDA labels will differ from the
  official USDA map in many places due to data and method differences.

Inputs/Outputs
--------------
Inputs:
- GBIF Occurrence Search API: https://api.gbif.org/v1/occurrence/search
- Local NetCDF climatology dataset:
    global_usda_zone_temperature_1991_2020.nc

Outputs:
- A JSON report (default: species_edge_era5.json) containing:
  - counts (raw/filtered/thinned/sampled)
  - northernmost occurrence
  - coldest point
  - selected cold-edge point
  - driver points (coldest N sampled points)

CLI usage
---------
Example:

  ./gbif_species_cold_edge_era5.py \
      --species "Aquilegia sibirica" \
      --dataset data/processed/global_usda_zone_temperature_1991_2020.nc \
      --grid-km 25 \
      --quantile 0.05 \
      --drivers 25 \
      --out aquilegia_sibirica_edge.json

Parameter guide (the important part)
------------------------------------

--species (required)
  Scientific name string passed to GBIF as `scientificName`.
  Example: "Acer rubrum" or "Aquilegia sibirica".

--dataset (required)
  Path to your merged global NetCDF climatology file. The file should be readable by
  USDAZoneDataset and contain the ERA5-derived temperature metric used to compute
  temp_f + zone. This script does not care about variable names; it relies on your API.

--max-records (default: 5000)
  Hard cap on how many GBIF records to examine after paging.
  - Higher values increase geographic coverage and robustness, but increase:
    - API time
    - local processing time
    - climate sampling time (each point calls zds.point()).
  Typical: 2000–20000 depending on how common the species is and your patience.

--page-size (default: 300)
  GBIF API page size (`limit`). GBIF caps are typically in the hundreds.
  This affects request count, not correctness.

--max-uncertainty-m (default: 10000)
  Drop points with coordinateUncertaintyInMeters > this value.
  Why: cold-edge inference is sensitive to bad / vague coordinates. A 50km-uncertain
  record could “land” in an incorrect climate cell.
  Guidance:
  - 1000–5000 m: stricter (better for precise edge estimation, fewer points).
  - 10000 m: compromise (default).
  - 20000 m+: permissive (more points, more noise).

--grid-km (default: 25.0)
  Spatial thinning grid size in km. Keeps at most 1 point per grid cell in a crude
  km-projected space to reduce cluster bias and compute.

  What thinning does:
  - GBIF often has dense clusters near roads, cities, botanic gardens, and popular trails.
  - Without thinning, those clusters dominate the cold-edge selection simply because
    there are many more points there, not because that area represents the true range edge.
  - Thinning keeps at most 1 point per approximate grid cell (in a crude km-projected space),
    which reduces oversampling bias and lowers compute cost.

  Why 25 km by default:
  - ERA5-Land is coarse (on the order of ~9 km grid spacing). A 25 km thinning grid:
    - still allows multiple points across climatic gradients,
    - removes extremely dense duplicates (many points often fall in the same or adjacent cells),
    - reduces the number of zds.point() calls substantially.
  - It’s a “middle” value: not so small that you keep lots of near-duplicates, not so
    large that you erase meaningful variation.

  Practical guidance:
  - 10 km: minimal thinning; good if you have few points already.
  - 25 km: good default for most widespread species.
  - 50 km: more aggressive; good for *very* common species with huge GBIF density.
  - 100 km: only if you’re overwhelmed by millions of points; can distort edges.

--quantile (default: 0.05)
  Cold-edge selection uses a percentile among the sampled points’ temp_f values.
  0.05 means “choose a point near the coldest 5% of sampled climates”.

  Why quantile vs absolute minimum:
  - Absolute minimum is extremely sensitive to outliers:
    - a mis-georeferenced specimen,
    - a cultivated/introduced record that slipped through metadata,
    - a single record on a mountaintop far outside the typical range,
    - coastal/inland microclimates not represented well by ERA5.
  - Quantile tends to be a more stable estimate of “edge climate” by ignoring
    the single coldest point and focusing on a small tail of the distribution.

  How to choose quantile:
  - 0.01 (1%): closer to min, still rejects the very worst outlier.
  - 0.05 (5%): robust default (what you want most of the time).
  - 0.10 (10%): more conservative (less “edge-y”), useful when data are noisy.

  Note: quantile acts on the *sampled* points, not the true continuous range.
  If you have very few points, quantile and min converge (see below).

--use-min (flag, default: false)
  If set, ignore quantile and choose the absolute coldest sampled point.

  When to use --use-min:
  - You have high confidence in GBIF filtering and coordinate quality.
  - The species is naturally occurring and data are clean.
  - You want the “most extreme observed” cold-edge, accepting sensitivity.

  When NOT to use it:
  - Common horticultural species with many cultivated escapes.
  - Species with few metadata fields / messy establishmentMeans.
  - You observe obvious outliers in the cold tail.

--drivers (default: 25)
  Number of “driver” points to include in the JSON report, from coldest upward.

  What drivers are for:
  - Debugging and interpretability: they show which records are shaping the edge decision.
  - If the selected cold-edge point looks suspicious, you can inspect the coldest N
    occurrences and decide if filtering/thinning should be adjusted.

  What number should you use?
  - If you are iterating quickly: 10–25.
  - If you want better forensic visibility: 50–100.
  - If points are few (<200): set drivers to min(25, n_points_with_climate).
  - If points are huge: 25 is plenty for most use cases; raising it grows the JSON.

  Rule of thumb:
  - Keep it at 25 unless you routinely see suspicious outliers; then use 50.
  - If you want the cold-edge to be “explainable” as a cluster rather than a single record,
    use 50 and inspect whether multiple cold points agree geographically.

--out (default: species_edge_era5.json)
  Output JSON report path.

Implementation notes (how the selection works)
----------------------------------------------
1) fetch_gbif_occurrences():
   - Requests GBIF occurrences by scientificName.
   - Requires coordinates and PRESENT status.
   - Filters:
     - basisOfRecord must be HUMAN_OBSERVATION or PRESERVED_SPECIMEN.
     - coordinateUncertaintyInMeters <= max_uncertainty_m (if provided).
     - establishmentMeans is excluded if in:
         INTRODUCED, CULTIVATED, MANAGED, ESCAPED, NATURALISED
       (case-normalized). Note: missing establishmentMeans is allowed.

2) thin_points_km():
   - Crude binning of lon/lat into km-ish coordinates:
       x_km ≈ lon * 111.320 * cos(lat)
       y_km ≈ lat * 110.574
   - Keeps the first point seen in each grid cell of size grid_km.

3) sample_era5_points():
   - Calls zds.point(lat, lon) for each thinned point.
   - Extracts temp_f and zone from the returned object using flexible attribute names.
     This is to tolerate either:
       - a dataclass return type
       - a simple namespace
       - an object with differently-named fields
   - Skips points if temp_f is missing (e.g., no data / masked).

4) pick_coldest() / pick_cold_edge_quantile():
   - Coldest: min(temp_f).
   - Quantile: sort by temp_f ascending; choose floor(q*(N-1)).
     Example: N=100, q=0.05 → idx=floor(0.05*99)=4 (5th coldest).

Interpreting results
--------------------
- “Coldest climate-sampled point” is always printed and stored, even if you select
  cold-edge by quantile. It’s useful for spotting outliers.
- “Cold-edge” is what you should treat as the edge estimate.
- Compare coldest vs quantile-selected:
  - If they differ a lot, you likely have outliers or strong microclimate/altitude effects
    not captured well by ERA5.
  - If they’re similar, the cold tail is consistent, and edge estimates are stable.

"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from usda_zone_access import USDAZoneDataset


GBIF_API = "https://api.gbif.org/v1/occurrence/search"

BASIS_OF_RECORD = {"HUMAN_OBSERVATION", "PRESERVED_SPECIMEN"}
EXCLUDED_ESTABLISHMENT = {"INTRODUCED", "CULTIVATED", "MANAGED", "ESCAPED", "NATURALISED"}


@dataclass(frozen=True)
class OccPoint:
    """
    Minimal GBIF occurrence record, normalized for this pipeline.

    Fields are a subset of GBIF response fields; we keep only what is helpful for:
    - filtering/quality control
    - reporting / audit trail of why a point was selected

    lat/lon:
      Decimal degrees.

    country:
      GBIF-reported country string (may be None).

    year:
      GBIF-reported year (may be None).

    basis:
      GBIF basisOfRecord (string), constrained by BASIS_OF_RECORD.

    establishment:
      GBIF establishmentMeans (may be None); filtered against EXCLUDED_ESTABLISHMENT.

    uncertainty_m:
      coordinateUncertaintyInMeters (may be None). If provided and too large, record is dropped.
    """
    lat: float
    lon: float
    country: Optional[str]
    year: Optional[int]
    basis: str
    establishment: Optional[str]
    uncertainty_m: Optional[float]


@dataclass(frozen=True)
class ClimatePoint:
    """
    A GBIF point annotated with climate results from USDAZoneDataset.point().

    lat/lon:
      Original occurrence coordinates.

    temp_f:
      ERA5-derived 1991–2020 mean of annual extreme minimum temperature, converted to °F.
      (Exact definition depends on your dataset.)

    zone:
      USDA-like label computed from temp_f by your centralized logic.

    grid_lat/grid_lon:
      Optional “actual gridpoint used” if your API returns it (e.g., nearest neighbor).
      This is useful for debugging wrap/boundary issues and interpolation behavior.
    """
    lat: float
    lon: float
    temp_f: float
    zone: str
    # Optional extra fields if your API provides them
    grid_lat: Optional[float] = None
    grid_lon: Optional[float] = None


def _gbif_get(params: Dict[str, Any], timeout: int = 60, retries: int = 5) -> Dict[str, Any]:
    """
    Perform a GBIF occurrence/search GET with simple exponential backoff.

    params:
      Passed directly as query parameters to GBIF.

    timeout:
      Requests timeout in seconds for each attempt.

    retries:
      Number of attempts with backoff up to ~20 seconds.

    Returns:
      Parsed JSON dict for the response payload.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(GBIF_API, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 20))
    assert last_err is not None
    raise last_err


def fetch_gbif_occurrences(
    scientific_name: str,
    max_records: int = 5000,
    page_size: int = 300,
    max_uncertainty_m: int = 10_000,
    sleep: float = 0.2,
) -> List[OccPoint]:
    """
    Fetch GBIF occurrences for a scientific name and apply basic cleaning filters.

    Filtering done here (intentionally lightweight, no extra deps):
    - must have coordinates
    - occurrenceStatus must be PRESENT
    - basisOfRecord must be in BASIS_OF_RECORD
    - coordinateUncertaintyInMeters <= max_uncertainty_m (if present)
    - establishmentMeans not in EXCLUDED_ESTABLISHMENT (if present)

    Parameters
    ----------
    scientific_name:
      Species name, passed to GBIF as `scientificName`.

    max_records:
      Maximum number of occurrence records to collect after filtering. This is a hard cap
      to limit runtime and API usage.

    page_size:
      GBIF page size (`limit` parameter). Controls how many records are requested per API call.

    max_uncertainty_m:
      Records with coordinateUncertaintyInMeters > this value are dropped.
      Default 10 km is a compromise; tighten for more precise edge inference.

    sleep:
      Sleep between GBIF pages to be polite and reduce risk of throttling.

    Returns
    -------
    A list of OccPoint records after filtering.
    """
    points: List[OccPoint] = []
    offset = 0

    while offset < max_records:
        params: Dict[str, Any] = {
            "scientificName": scientific_name,
            "hasCoordinate": "true",
            "occurrenceStatus": "PRESENT",
            "limit": page_size,
            "offset": offset,
        }
        # GBIF accepts repeated params for basisOfRecord; requests will encode lists appropriately.
        params["basisOfRecord"] = list(BASIS_OF_RECORD)

        payload = _gbif_get(params)
        records = payload.get("results", [])
        if not records:
            break

        for rec in records:
            lat = rec.get("decimalLatitude")
            lon = rec.get("decimalLongitude")
            if lat is None or lon is None:
                continue

            cu = rec.get("coordinateUncertaintyInMeters")
            if cu is not None and cu > max_uncertainty_m:
                continue

            bor = rec.get("basisOfRecord")
            if bor not in BASIS_OF_RECORD:
                continue

            em = rec.get("establishmentMeans")
            if em and str(em).upper() in EXCLUDED_ESTABLISHMENT:
                continue

            points.append(
                OccPoint(
                    lat=float(lat),
                    lon=float(lon),
                    country=rec.get("country"),
                    year=rec.get("year"),
                    basis=str(bor),
                    establishment=em,
                    uncertainty_m=float(cu) if cu is not None else None,
                )
            )
            if len(points) >= max_records:
                break

        if payload.get("endOfRecords"):
            break

        offset += page_size
        time.sleep(sleep)

    return points


def thin_points_km(points: List[OccPoint], grid_km: float) -> List[OccPoint]:
    """
    Cheap spatial thinning without extra dependencies.

    Why thinning matters:
    - GBIF points are often heavily clustered (popular parks, cities, roadsides).
      Without thinning, clusters dominate the “cold tail” selection and can bias the
      inferred edge climate toward oversampled areas rather than true range limits.

    How thinning works:
    - Convert lat/lon to a crude km space:
        x_km ≈ lon * 111.320 * cos(lat)
        y_km ≈ lat * 110.574
      Then keep at most one point per grid cell of size `grid_km`.

    Why default grid_km=25 km (see CLI docs above):
    - Coarse ERA5 grid + desire to reduce duplicates/compute.
    - Still preserves broad geographic variation for most species.

    Notes:
    - This is a binning heuristic, not a true equal-area projection.
    - Good enough for deduplication / thinning; not intended for distance analysis.

    Returns:
      Thinned list of OccPoint.
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


def _getattr_any(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    """
    Try multiple attribute names on an object and return the first found.

    This is defensive glue to allow USDAZoneDataset.point() to return:
    - a dataclass with fields (temp_f, zone_label, etc.)
    - an object with different attribute naming
    - without coupling this script tightly to a specific return type.

    If no named attribute is present, returns `default`.
    """
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def sample_era5_points(
    zds: USDAZoneDataset,
    points: List[OccPoint],
) -> List[Tuple[ClimatePoint, OccPoint]]:
    """
    Sample climate for each occurrence point via the centralized USDAZoneDataset API.

    Returns:
      List of (ClimatePoint, OccPoint) pairs.
      Points that yield missing temp_f (masked/no-data) are skipped.

    Note:
    - This function does *not* implement any wraparound or interpolation logic.
      All spatial lookup correctness must be handled inside zds.point().
    """
    out: List[Tuple[ClimatePoint, OccPoint]] = []

    for p in points:
        r = zds.point(p.lat, p.lon)

        temp_f = _getattr_any(r, ["temp_f", "tempF", "temperature_f", "temperature_fahrenheit"])
        zone = _getattr_any(r, ["zone_label", "zone", "usda_zone", "zone_ab"], default="—")
        if temp_f is None:
            continue

        grid_lat = _getattr_any(r, ["grid_lat", "latitude", "lat_used", "picked_lat"], default=None)
        grid_lon = _getattr_any(r, ["grid_lon", "longitude", "lon_used", "picked_lon"], default=None)

        cp = ClimatePoint(
            lat=float(p.lat),
            lon=float(p.lon),
            temp_f=float(temp_f),
            zone=str(zone),
            grid_lat=float(grid_lat) if grid_lat is not None else None,
            grid_lon=float(grid_lon) if grid_lon is not None else None,
        )
        out.append((cp, p))

    return out


def pick_northernmost(points: List[OccPoint]) -> Optional[OccPoint]:
    """
    Northernmost occurrence in the Northern Hemisphere (lat >= 0).
    If no NH points exist, returns the global max-lat point.

    This is a geographic diagnostic; it is not necessarily the “cold edge” because
    coldest climate can occur south of the northernmost latitude (continentality,
    elevation, etc.).

    This is included because many species range-edge heuristics use “northernmost”
    as a proxy for cold-edge in the NH, but it can fail due to:
    - coastal moderation vs inland cold
    - elevation effects (mountains)
    - sampling bias
    - hemispheric / global distributions
    """
    if not points:
        return None
    nh = [p for p in points if p.lat >= 0.0]
    if nh:
        return max(nh, key=lambda p: p.lat)
    return max(points, key=lambda p: p.lat)


def pick_coldest(sampled: List[Tuple[ClimatePoint, OccPoint]]) -> Tuple[ClimatePoint, OccPoint]:
    """
    Coldest climate-sampled point (minimum temp_f).

    This is the most “edge-y” selection, but also the most sensitive to:
    - mis-geocoded occurrences
    - cultivated/introduced records that slipped through metadata
    - single extreme points (elevation/microclimate/outlier)

    For robustness, prefer pick_cold_edge_quantile() unless you explicitly want min().
    """
    if not sampled:
        raise ValueError("No sampled points with climate.")
    return min(sampled, key=lambda t: t[0].temp_f)


def pick_cold_edge_quantile(
    sampled: List[Tuple[ClimatePoint, OccPoint]],
    quantile: float,
) -> Tuple[ClimatePoint, OccPoint, int]:
    """
    Select a cold-edge point using a quantile of the cold tail.

    Algorithm:
    - Sort sampled points by temp_f ascending (coldest first).
    - Choose index floor(q * (N - 1)).

    Example:
    - N=100, q=0.05 -> idx=floor(0.05*99)=4 (5th coldest point)

    Why quantile is often better than absolute minimum:
    - rejects a single cold outlier
    - more stable across different GBIF sampling densities and quality

    quantile:
      Must be in (0, 1). Typical values:
      - 0.01 (very edge-like)
      - 0.05 (default)
      - 0.10 (more conservative)

    Returns:
      (ClimatePoint, OccPoint, idx_in_sorted)
    """
    if not sampled:
        raise ValueError("No sampled points with climate.")
    q = float(quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("quantile must be between 0 and 1 (exclusive)")

    s = sorted(sampled, key=lambda t: t[0].temp_f)  # coldest first
    idx = int(math.floor(q * (len(s) - 1)))
    return s[idx][0], s[idx][1], idx


def _fmt_occ(p: OccPoint) -> str:
    """
    Human-readable one-line format for quick terminal inspection.
    """
    parts = [f"({p.lat:.6f}, {p.lon:.6f})"]
    if p.country:
        parts.append(f"country={p.country}")
    if p.year is not None:
        parts.append(f"year={p.year}")
    parts.append(f"basis={p.basis}")
    if p.uncertainty_m is not None:
        parts.append(f"uncertainty_m={p.uncertainty_m:g}")
    if p.establishment:
        parts.append(f"establishment={p.establishment}")
    return "  ".join(parts)


def main() -> int:
    """
    Main CLI workflow.

    1) Fetch + filter GBIF occurrences.
    2) Spatially thin.
    3) Print northernmost occurrence.
    4) Sample climate via USDAZoneDataset.
    5) Compute coldest + cold-edge selection.
    6) Write JSON report.
    """
    ap = argparse.ArgumentParser(description="GBIF -> ERA5 (USDAZoneDataset) -> cold-edge + northernmost reporting")
    ap.add_argument("--species", required=True, help='Scientific name, e.g. "Aquilegia sibirica"')
    ap.add_argument("--dataset", type=Path, required=True, help="Path to global_usda_zone_temperature_1991_2020.nc")
    ap.add_argument("--max-records", type=int, default=5000)
    ap.add_argument("--page-size", type=int, default=300)
    ap.add_argument("--max-uncertainty-m", type=int, default=10_000)
    ap.add_argument("--grid-km", type=float, default=25.0, help="Spatial thinning grid size (km)")
    ap.add_argument("--quantile", type=float, default=0.05, help="Cold-edge quantile (0.05 = coldest 5%)")
    ap.add_argument("--use-min", action="store_true", help="Use absolute coldest point instead of quantile")
    ap.add_argument("--drivers", type=int, default=25, help="How many coldest driver points to include")
    ap.add_argument("--out", type=Path, default=Path("species_edge_era5.json"))
    args = ap.parse_args()

    pts = fetch_gbif_occurrences(
        scientific_name=args.species,
        max_records=args.max_records,
        page_size=args.page_size,
        max_uncertainty_m=args.max_uncertainty_m,
    )
    print(f"GBIF cleaned records: {len(pts)}")

    pts_thin = thin_points_km(pts, grid_km=args.grid_km)
    print(f"After thinning ({args.grid_km:g} km): {len(pts_thin)}")

    northmost = pick_northernmost(pts_thin)
    if northmost is not None:
        print(f"Northernmost occurrence (NH if present): {_fmt_occ(northmost)}")
    else:
        print("Northernmost occurrence: N/A (no points)")


    # IMPORTANT: All spatial lookup logic lives in USDAZoneDataset.point().
    # This script simply calls it per occurrence point.
    with USDAZoneDataset(args.dataset) as zds:
        sampled = sample_era5_points(zds, pts_thin)

    print(f"Sampled (with climate): {len(sampled)}")
    if not sampled:
        print("FAILED: no points overlapped climate grid (or all returned missing).")
        return 2

    # Coldest (absolute) is always computed for inspection, even if you later select
    # cold-edge by quantile. This helps you detect outliers quickly.
    coldest_cp, coldest_occ = pick_coldest(sampled)
    print(f"Coldest climate-sampled point: ({coldest_cp.lat:.6f}, {coldest_cp.lon:.6f})  "
          f"temp_f={coldest_cp.temp_f:.2f}  zone={coldest_cp.zone}")

    # Cold-edge selection: robust quantile (default) or absolute min (--use-min).
    if args.use_min:
        edge_cp, edge_occ = coldest_cp, coldest_occ
        edge_idx = 0
        method = "min"
        qval: Optional[float] = None
    else:
        edge_cp, edge_occ, edge_idx = pick_cold_edge_quantile(sampled, quantile=args.quantile)
        method = "quantile"
        qval = float(args.quantile)

    # “Drivers”: coldest N points (after thinning and climate sampling).
    # These are included in the report to make the selection auditable.
    sampled_sorted = sorted(sampled, key=lambda t: t[0].temp_f)
    driver_n = max(1, min(args.drivers, len(sampled_sorted)))

    drivers: List[Dict[str, Any]] = []
    for cp, occ in sampled_sorted[:driver_n]:
        drivers.append(
            {
                "climate": asdict(cp),
                "occurrence": {
                    "country": occ.country,
                    "year": occ.year,
                    "basis": occ.basis,
                    "establishment": occ.establishment,
                    "uncertainty_m": occ.uncertainty_m,
                },
            }
        )

    report: Dict[str, Any] = {
        "ok": True,
        "species": args.species,
        "dataset": str(args.dataset),
        "n_gbif_points_cleaned": len(pts),
        "n_points_after_thinning": len(pts_thin),
        "n_points_with_climate": len(sampled),
        "northernmost_occurrence": asdict(northmost) if northmost is not None else None,
        "coldest_point": {
            "climate": asdict(coldest_cp),
            "occurrence": {
                "country": coldest_occ.country,
                "year": coldest_occ.year,
                "basis": coldest_occ.basis,
                "establishment": coldest_occ.establishment,
                "uncertainty_m": coldest_occ.uncertainty_m,
            },
        },
        "edge_method": method,
        "quantile": qval,
        "selected_index_in_sorted": int(edge_idx),
        "cold_edge": {
            "climate": asdict(edge_cp),
            "occurrence": {
                "country": edge_occ.country,
                "year": edge_occ.year,
                "basis": edge_occ.basis,
                "establishment": edge_occ.establishment,
                "uncertainty_m": edge_occ.uncertainty_m,
            },
        },
        "drivers_coldest": drivers,
    }

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Cold-edge selection ({method}{'' if qval is None else f', q={qval}'}): "
          f"({edge_cp.lat:.6f}, {edge_cp.lon:.6f})  temp_f={edge_cp.temp_f:.2f}  zone={edge_cp.zone}")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
