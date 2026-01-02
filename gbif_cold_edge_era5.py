#!/usr/bin/env python3
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
    lat: float
    lon: float
    country: Optional[str]
    year: Optional[int]
    basis: str
    establishment: Optional[str]
    uncertainty_m: Optional[float]


@dataclass(frozen=True)
class ClimatePoint:
    lat: float
    lon: float
    temp_f: float
    zone: str
    # Optional extra fields if your API provides them
    grid_lat: Optional[float] = None
    grid_lon: Optional[float] = None


def _gbif_get(params: Dict[str, Any], timeout: int = 60, retries: int = 5) -> Dict[str, Any]:
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
    Cheap spatial thinning without extra deps.
    Projects lon to km using cos(lat) scaling; good enough for binning.
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
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def sample_era5_points(
    zds: USDAZoneDataset,
    points: List[OccPoint],
) -> List[Tuple[ClimatePoint, OccPoint]]:
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
    """
    if not points:
        return None
    nh = [p for p in points if p.lat >= 0.0]
    if nh:
        return max(nh, key=lambda p: p.lat)
    return max(points, key=lambda p: p.lat)


def pick_coldest(sampled: List[Tuple[ClimatePoint, OccPoint]]) -> Tuple[ClimatePoint, OccPoint]:
    """
    Coldest climate-sampled point (min temp_f).
    """
    if not sampled:
        raise ValueError("No sampled points with climate.")
    return min(sampled, key=lambda t: t[0].temp_f)


def pick_cold_edge_quantile(
    sampled: List[Tuple[ClimatePoint, OccPoint]],
    quantile: float,
) -> Tuple[ClimatePoint, OccPoint, int]:
    if not sampled:
        raise ValueError("No sampled points with climate.")
    q = float(quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("quantile must be between 0 and 1 (exclusive)")

    s = sorted(sampled, key=lambda t: t[0].temp_f)  # coldest first
    idx = int(math.floor(q * (len(s) - 1)))
    return s[idx][0], s[idx][1], idx


def _fmt_occ(p: OccPoint) -> str:
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

    with USDAZoneDataset(args.dataset) as zds:
        sampled = sample_era5_points(zds, pts_thin)

    print(f"Sampled (with climate): {len(sampled)}")
    if not sampled:
        print("FAILED: no points overlapped climate grid (or all returned missing).")
        return 2

    # Coldest (absolute)
    coldest_cp, coldest_occ = pick_coldest(sampled)
    print(f"Coldest climate-sampled point: ({coldest_cp.lat:.6f}, {coldest_cp.lon:.6f})  "
          f"temp_f={coldest_cp.temp_f:.2f}  zone={coldest_cp.zone}")

    # Cold-edge (quantile or min)
    if args.use_min:
        edge_cp, edge_occ = coldest_cp, coldest_occ
        edge_idx = 0
        method = "min"
        qval: Optional[float] = None
    else:
        edge_cp, edge_occ, edge_idx = pick_cold_edge_quantile(sampled, quantile=args.quantile)
        method = "quantile"
        qval = float(args.quantile)

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
