#!/usr/bin/env python3
from __future__ import annotations

"""
Download ERA5-Land hourly 2m temperature into the *new* directory layout:

    <raw_dir>/
        north/
            lon000/ north_lon000_YYYY_MM.nc
            lon090/ north_lon090_YYYY_MM.nc
            lon180/ north_lon180_YYYY_MM.nc
            lon270/ north_lon270_YYYY_MM.nc
        south/
            lon000/ south_lon000_YYYY_MM.nc
            lon090/ south_lon090_YYYY_MM.nc
            lon180/ south_lon180_YYYY_MM.nc
            lon270/ south_lon270_YYYY_MM.nc
        tropics/
            lon000/ tropics_lon000_YYYY_MM.nc
            lon180/ tropics_lon180_YYYY_MM.nc

Month rules (same as your pipeline assumptions):
  - north: DJF  (12, 1, 2)
  - south: JJA  (6, 7, 8)
  - tropics: all months

Lat bands (to avoid overlap with tropics on a 0.1° grid):
  - north:   (20.1 .. 90]
  - tropics: [-20 .. 20]
  - south:   [-90 .. -20.1)

This script intentionally drops legacy filename support.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import logging
import time
import zipfile
from typing import Iterable

import cdsapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MIN_EXPECTED_BYTES = 5_000_000  # conservative
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 30

TROPICS_MIN = -20.0
TROPICS_MAX = 20.0
BAND_EPS = 0.1  # 0.1° grid: use +/-0.1 to avoid overlap between bands

# Non-tropics: 4 bins of 90 degrees, ENDING AT .9 to avoid inclusive overlap
BINS_NON_TROPICS: list[tuple[float, float, str]] = [
    (0.0,   90.0  - BAND_EPS, "lon000"),
    (90.0,  180.0 - BAND_EPS, "lon090"),
    (180.0, 270.0 - BAND_EPS, "lon180"),
    (270.0, 360.0 - BAND_EPS, "lon270"),
]

# Tropics: 2 bins of 180 degrees, ENDING AT .9 to avoid 180.0/360.0 extra column
BINS_TROPICS: list[tuple[float, float, str]] = [
    (0.0,   180.0 - BAND_EPS, "lon000"),
    (180.0, 360.0 - BAND_EPS, "lon180"),
]


def _looks_like_netcdf(path: Path) -> bool:
    """
    Quick binary sanity check:
    NetCDF classic starts with b'CDF'; NetCDF4/HDF5 starts with HDF5 magic.
    """
    try:
        with path.open("rb") as f:
            header = f.read(8)
        return header.startswith(b"CDF") or header.startswith(b"\x89HDF")
    except OSError:
        return False


def _parse_years(years_arg: str) -> list[int]:
    """Parse comma-separated list of years or ranges (e.g. "1991-2020,2022")."""
    years: set[int] = set()
    for part in years_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if start > end:
                start, end = end, start
            years.update(range(start, end + 1))
        else:
            years.add(int(part))
    return sorted(years)


def _months_for_region(region: str) -> list[int]:
    if region == "north":
        return [12, 1, 2]
    if region == "south":
        return [6, 7, 8]
    if region == "tropics":
        return list(range(1, 13))
    raise ValueError(f"Unknown region: {region}")


def _lat_band_for_region(region: str) -> tuple[float, float]:
    """
    Returns (north, south) for CDS "area": [N, W, S, E]
    """
    if region == "north":
        return 90.0, TROPICS_MAX + BAND_EPS  # 90 .. 20.1
    if region == "tropics":
        return TROPICS_MAX, TROPICS_MIN     # 20 .. -20
    if region == "south":
        return TROPICS_MIN - BAND_EPS, -90.0  # -20.1 .. -90
    raise ValueError(f"Unknown region: {region}")


def _bins_for_region(region: str) -> list[tuple[int, int, str]]:
    return BINS_TROPICS if region == "tropics" else BINS_NON_TROPICS


def _output_path(raw_dir: Path, region: str, lon_tag: str, year: int, month: int) -> Path:
    out_dir = raw_dir / region / lon_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{region}_{lon_tag}_{year:04d}_{month:02d}.nc"


def _is_valid_existing_file(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size < MIN_EXPECTED_BYTES:
        return False
    if not _looks_like_netcdf(path):
        return False
    try:
        import xarray as xr

        with xr.open_dataset(path, engine="netcdf4") as ds:
            # ERA5-Land 2m temperature should appear as t2m in NetCDF outputs.
            if "t2m" not in ds.variables and "t2m" not in ds.data_vars:
                return False
            if "latitude" not in ds.coords or "longitude" not in ds.coords:
                return False
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class DownloadJob:
    region: str
    lon_west: float
    lon_east: float
    lon_tag: str
    year: int
    month: int

    def out_path(self, raw_dir: Path) -> Path:
        return _output_path(raw_dir, self.region, self.lon_tag, self.year, self.month)

    def area(self) -> list[float]:
        lat_n, lat_s = _lat_band_for_region(self.region)
        # CDS expects [N, W, S, E]
        return [float(lat_n), float(self.lon_west), float(lat_s), float(self.lon_east)]


def build_plan(years: Iterable[int], regions: Iterable[str]) -> list[DownloadJob]:
    plan: list[DownloadJob] = []
    for region in regions:
        months = _months_for_region(region)
        bins = _bins_for_region(region)
        for year in years:
            for month in months:
                for lo, hi, tag in bins:
                    plan.append(
                        DownloadJob(
                            region=region,
                            lon_west=lo,
                            lon_east=hi,
                            lon_tag=tag,
                            year=year,
                            month=month,
                        )
                    )
    return sorted(plan, key=lambda j: (j.year, j.month, j.region, j.lon_west))


def download_one(job: DownloadJob, raw_dir: Path, overwrite: bool, strict: bool) -> Path:
    out = job.out_path(raw_dir)

    if out.exists() and not overwrite:
        if _is_valid_existing_file(out):
            log.info("Exists and valid, skipping: %s", out)
            return out
        log.warning("Existing file invalid/suspicious, re-downloading: %s", out)
        out.unlink(missing_ok=True)

    tmp = out.with_suffix(out.suffix + ".part")

    request = {
        "variable": "2m_temperature",
        "year": f"{job.year}",
        "month": f"{job.month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(0, 24)],
        "area": job.area(),
        "data_format": "netcdf",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        if tmp.exists():
            tmp.unlink()

        try:
            log.info(
                "Downloading %s %04d-%02d area=%s (attempt %d/%d)",
                f"{job.region}/{job.lon_tag}",
                job.year,
                job.month,
                request["area"],
                attempt,
                MAX_RETRIES,
            )

            c = cdsapi.Client()
            c.retrieve("reanalysis-era5-land", request, str(tmp))

            if not tmp.exists():
                raise RuntimeError("CDS returned no file")

            size = tmp.stat().st_size
            if size < MIN_EXPECTED_BYTES:
                raise RuntimeError(f"Downloaded file too small ({size} bytes)")

            # CDS sometimes returns ZIP even when NetCDF requested
            if zipfile.is_zipfile(tmp):
                log.warning("ZIP response detected from CDS, extracting NetCDF: %s", tmp.name)

                with zipfile.ZipFile(tmp) as z:
                    nc_members = [n for n in z.namelist() if n.endswith(".nc")]
                    if len(nc_members) != 1:
                        raise RuntimeError(f"Unexpected ZIP contents: {z.namelist()}")

                    extracted = tmp.parent / nc_members[0]
                    z.extract(nc_members[0], tmp.parent)

                tmp.unlink()

                if not extracted.exists():
                    raise RuntimeError("ZIP extraction failed")

                if extracted.stat().st_size < MIN_EXPECTED_BYTES:
                    extracted.unlink(missing_ok=True)
                    raise RuntimeError("Extracted NetCDF too small")

                if not _looks_like_netcdf(extracted):
                    extracted.unlink(missing_ok=True)
                    raise RuntimeError("Extracted file is not valid NetCDF")

                extracted.replace(out)
            else:
                if not _looks_like_netcdf(tmp):
                    raise RuntimeError("Downloaded file is not NetCDF (bad header)")
                tmp.replace(out)

            # Deep validation
            if not _is_valid_existing_file(out):
                raise RuntimeError("Downloaded NetCDF failed validation (missing coords/t2m?)")

            log.info("Download complete: %s", out)
            return out

        except Exception as e:
            log.error(
                "ERA5 download failed for %s %04d-%02d: %s",
                f"{job.region}/{job.lon_tag}",
                job.year,
                job.month,
                e,
            )

            if tmp.exists():
                tmp.unlink()

            if attempt >= MAX_RETRIES:
                if strict:
                    raise RuntimeError(
                        f"Failed to download {job.region}/{job.lon_tag} "
                        f"{job.year}-{job.month:02d} after {MAX_RETRIES} attempts"
                    ) from e
                log.warning("Skipping %s %04d-%02d due to repeated failures", job.region, job.year, job.month)
                return out

            log.info("Retrying in %d seconds...", RETRY_DELAY_SECONDS)
            time.sleep(RETRY_DELAY_SECONDS)

    raise AssertionError("Unreachable code reached")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--years",
        default="1991-2020",
        help="Comma-separated years or ranges (e.g. 1991-2020,2022)",
    )
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/era5_land_hourly_temp"),
        help="Output root directory for the new layout",
    )
    ap.add_argument(
        "--regions",
        default="north,south,tropics",
        help="Comma-separated subset of regions: north,south,tropics",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only; do not download",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files even if they look valid",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Raise errors on repeated download failures (default: skip)",
    )
    ap.add_argument(
        "--plan-json",
        type=Path,
        help="Optional path to write the computed plan as JSON",
    )
    ap.add_argument(
        "--plan-preview",
        type=int,
        default=12,
        help="How many plan entries to preview in logs (default 12)",
    )
    args = ap.parse_args()

    years = _parse_years(args.years)
    raw_dir: Path = args.raw_dir

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    for r in regions:
        if r not in {"north", "south", "tropics"}:
            raise SystemExit(f"Unknown region in --regions: {r}")

    plan = build_plan(years, regions)
    log.info("Planned %d downloads", len(plan))

    for job in plan[: args.plan_preview]:
        log.info(
            "Plan: %s/%s %04d-%02d area=%s -> %s",
            job.region,
            job.lon_tag,
            job.year,
            job.month,
            job.area(),
            job.out_path(raw_dir),
        )
    if len(plan) > args.plan_preview:
        log.info("... %d additional plan entries omitted", len(plan) - args.plan_preview)

    if args.plan_json:
        payload = [
            {
                "region": j.region,
                "lon_tag": j.lon_tag,
                "lon_west": j.lon_west,
                "lon_east": j.lon_east,
                "year": j.year,
                "month": j.month,
                "area": j.area(),
                "out": str(j.out_path(raw_dir)),
            }
            for j in plan
        ]
        args.plan_json.write_text(json.dumps(payload, indent=2))
        log.info("Wrote plan to %s", args.plan_json)

    if args.dry_run:
        log.info("Dry-run requested; no downloads executed")
        return 0

    start = time.time()
    wrote = 0
    skipped = 0

    for job in plan:
        out = job.out_path(raw_dir)
        if out.exists() and not args.overwrite and _is_valid_existing_file(out):
            skipped += 1
            continue
        download_one(job, raw_dir=raw_dir, overwrite=args.overwrite, strict=args.strict)
        wrote += 1

    elapsed_h = (time.time() - start) / 3600
    log.info("Done. wrote=%d skipped=%d elapsed=%.2f hours", wrote, skipped, elapsed_h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
