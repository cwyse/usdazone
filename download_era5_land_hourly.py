#!/usr/bin/env python3
from __future__ import annotations

"""
download_era5_land_hourly.py

Download ERA5-Land *hourly* 2m temperature (t2m) NetCDF files into the directory layout:

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

Purpose
-------
- Provide a repeatable, “pipeline-compatible” raw download step for ERA5-Land hourly t2m.
- Enforce a stable file naming convention and directory hierarchy (new layout only).
- Split the globe into latitude bands and longitude bins so downloads are smaller and restartable.
- Validate downloads (basic header checks + xarray open + expected coords/variable).

Region logic (matches your processing assumptions)
--------------------------------------------------
Month rules:
- north:   DJF  (12, 1, 2)  → winter minima in the Northern Hemisphere
- south:   JJA  (6, 7, 8)   → winter minima in the Southern Hemisphere
- tropics: all months       → no winter-only shortcut; full year

Latitude bands (avoid overlap with tropics on a 0.1° grid)
- tropics: [-20.0 ..  20.0]
- north:   ( 20.1 ..  90.0]   i.e. start just above 20.0 to avoid shared row
- south:   [-90.0 .. -20.1)   i.e. end just below -20.0 to avoid shared row

BAND_EPS = 0.1 is used because ERA5-Land uses a 0.1° grid. If two downloads include
a boundary latitude/longitude row, they will overlap and you can get off-by-one columns/rows
or stitching issues later. This script avoids overlap by “nudging” end boundaries.

Longitude bins and why they end at 359.9 / 179.9
------------------------------------------------
ERA5 uses 0..360 longitude. CDS “area” bounding boxes are inclusive-ish, and it’s common to
accidentally request an extra 360.0 or 180.0 column that duplicates 0.0 or causes a mismatch.

To avoid that, longitude bin “east” boundaries end at (bin_end - 0.1), e.g. 90.0 → 89.9.

- Non-tropics: 4 bins of 90° (0–89.9, 90–179.9, 180–269.9, 270–359.9)
- Tropics:     2 bins of 180° (0–179.9, 180–359.9)

CDS behavior notes
------------------
- Although `data_format: netcdf` is requested, CDS sometimes returns a ZIP archive containing
  the .nc file. This script detects ZIP responses, extracts the single .nc member, and writes
  it to the final location.

Validation policy
-----------------
An existing file is treated as valid and can be skipped if:
- size >= MIN_EXPECTED_BYTES
- header looks like NetCDF classic (“CDF…”) or HDF5 (NetCDF4)
- xarray can open it with netcdf4 engine
- it contains `t2m` and `latitude`/`longitude` coordinates

CLI usage examples
------------------
Plan only (no downloads), show first 12 plan entries:
  ./download_era5_land_hourly_t2m_new_layout.py --dry-run

Download only tropics for 2019-2020:
  ./download_era5_land_hourly_t2m_new_layout.py --years 2019-2020 --regions tropics

Overwrite and fail fast on repeated errors:
  ./download_era5_land_hourly_t2m_new_layout.py --overwrite --strict

Write the computed plan to a JSON file:
  ./download_era5_land_hourly_t2m_new_layout.py --plan-json plan.json --dry-run

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

# A conservative “is this plausibly a real ERA5-Land month file?” threshold.
# If CDS fails and returns HTML/JSON/error payload, it will often be far smaller than this.
MIN_EXPECTED_BYTES = 5_000_000  # conservative

# Retry policy for CDS calls. CDS outages/throttling happen; this gives a few attempts
# and then either skips (default) or raises (with --strict).
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 30

# Tropics definition for the pipeline. These constants are used to define latitude bands.
TROPICS_MIN = -20.0
TROPICS_MAX = 20.0

# ERA5-Land uses a 0.1° grid. BAND_EPS prevents overlapping boundary rows/cols across downloads.
BAND_EPS = 0.1  # 0.1° grid: use +/-0.1 to avoid overlap between bands

# Non-tropics: 4 bins of 90 degrees, ENDING AT (end - 0.1) to avoid inclusive overlap.
# Example: [0, 90] can include both 0.0 and 90.0 columns; using 89.9 avoids a duplicate
# boundary column when another file begins at 90.0.
BINS_NON_TROPICS: list[tuple[float, float, str]] = [
    (0.0,   90.0  - BAND_EPS, "lon000"),
    (90.0,  180.0 - BAND_EPS, "lon090"),
    (180.0, 270.0 - BAND_EPS, "lon180"),
    (270.0, 360.0 - BAND_EPS, "lon270"),
]

# Tropics: 2 bins of 180 degrees, ENDING AT (end - 0.1) to avoid an extra 180.0/360.0 column.
BINS_TROPICS: list[tuple[float, float, str]] = [
    (0.0,   180.0 - BAND_EPS, "lon000"),
    (180.0, 360.0 - BAND_EPS, "lon180"),
]


def _looks_like_netcdf(path: Path) -> bool:
    """
    Quick binary sanity check on the file header.

    NetCDF classic format starts with ASCII 'CDF' (e.g., b'CDF\\x01' or b'CDF\\x02').
    NetCDF4 files are HDF5 containers and begin with the HDF5 magic b'\\x89HDF...'.

    This is not a full validation; it just filters out obvious error payloads.
    """
    try:
        with path.open("rb") as f:
            header = f.read(8)
        return header.startswith(b"CDF") or header.startswith(b"\x89HDF")
    except OSError:
        return False


def _parse_years(years_arg: str) -> list[int]:
    """
    Parse a comma-separated list of years and/or ranges.

    Examples:
      "1991-2020"         -> [1991, 1992, ..., 2020]
      "1991-1993,2001"    -> [1991, 1992, 1993, 2001]
      "2020,2018-2019"    -> [2018, 2019, 2020]

    Notes:
    - Ranges are inclusive.
    - If a range is reversed (e.g. "2020-2018"), it is normalized.
    """
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
    """
    Define which months to download for each region.

    This is an optimization aligned with your downstream “annual extreme min” pipeline:
    - In non-tropics, annual minimum temperatures are overwhelmingly determined by winter months.
      Downloading only winter months reduces storage and CDS load substantially.
    - In tropics, seasonal variation is weaker and minima can occur in many months, so we download all.

    Regions:
      north   -> [12, 1, 2]
      south   -> [6, 7, 8]
      tropics -> [1..12]
    """
    if region == "north":
        return [12, 1, 2]
    if region == "south":
        return [6, 7, 8]
    if region == "tropics":
        return list(range(1, 13))
    raise ValueError(f"Unknown region: {region}")


def _lat_band_for_region(region: str) -> tuple[float, float]:
    """
    Return (north, south) latitude values for CDS 'area' bounding boxes.

    CDS uses:
      area: [N, W, S, E]  (north lat, west lon, south lat, east lon)

    The epsilon adjustments avoid overlapping lat rows between the tropics and non-tropics.
    """
    if region == "north":
        return 90.0, TROPICS_MAX + BAND_EPS  # 90 .. 20.1
    if region == "tropics":
        return TROPICS_MAX, TROPICS_MIN     # 20 .. -20
    if region == "south":
        return TROPICS_MIN - BAND_EPS, -90.0  # -20.1 .. -90
    raise ValueError(f"Unknown region: {region}")


def _bins_for_region(region: str) -> list[tuple[int, int, str]]:
    """
    Choose longitude bins based on region.

    Tropics uses 2x180° bins to reduce file count. Non-tropics uses 4x90° bins.
    """
    return BINS_TROPICS if region == "tropics" else BINS_NON_TROPICS


def _output_path(raw_dir: Path, region: str, lon_tag: str, year: int, month: int) -> Path:
    """
    Compute output path and ensure the output directory exists.

    Output naming convention:
      <raw_dir>/<region>/<lon_tag>/<region>_<lon_tag>_YYYY_MM.nc
    """
    out_dir = raw_dir / region / lon_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{region}_{lon_tag}_{year:04d}_{month:02d}.nc"


def _is_valid_existing_file(path: Path) -> bool:
    """
    Determine whether an existing file should be trusted and skipped.

    Validation steps:
    1) file exists and meets minimum size
    2) header looks like NetCDF
    3) xarray can open it (netcdf4 engine)
    4) expected elements are present:
       - variable/data_var `t2m` (ERA5-Land 2m temp)
       - coords `latitude` and `longitude`

    If any check fails, caller may delete and re-download.
    """
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
    """
    One “atomic” download unit: a specific region + lon bin + year + month.

    region:
      'north' | 'south' | 'tropics'

    lon_west / lon_east:
      0..360 ERA5 longitudes, with lon_east ending at (bin_end - 0.1)
      to avoid inclusive overlap.

    lon_tag:
      Directory tag: lon000 / lon090 / lon180 / lon270 (or lon000 / lon180 for tropics)

    year / month:
      Year and month to request from CDS.
    """
    region: str
    lon_west: float
    lon_east: float
    lon_tag: str
    year: int
    month: int

    def out_path(self, raw_dir: Path) -> Path:
        """Final NetCDF output path for this job."""
        return _output_path(raw_dir, self.region, self.lon_tag, self.year, self.month)

    def area(self) -> list[float]:
        """
        Return CDS bounding box as [N, W, S, E].

        N/S come from the region latitude band definition.
        W/E come from the longitude bin for this job.
        """
        lat_n, lat_s = _lat_band_for_region(self.region)
        # CDS expects [N, W, S, E]
        return [float(lat_n), float(self.lon_west), float(lat_s), float(self.lon_east)]


def build_plan(years: Iterable[int], regions: Iterable[str]) -> list[DownloadJob]:
    """
    Build the full list of download jobs implied by the input years and regions.

    For each region:
      - select region months (DJF/JJA/all)
      - select region lon bins (4 bins or 2 bins)
      - cross product: years × months × bins

    Returns:
      A sorted list of DownloadJob objects (sorted by year, month, region, lon_west)
      for deterministic logging and execution.
    """
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
    """
    Execute one CDS download job and write the final NetCDF file.

    Behavior
    --------
    - If output exists and overwrite=False:
        - if it validates, skip it
        - if it looks invalid, delete and re-download
    - Downloads to a temporary ".part" file then atomically replaces the final output.
    - Handles CDS returning ZIP instead of NetCDF by extracting a single .nc member.
    - Performs deep validation after download (xarray open + t2m + coords).
    - Retries up to MAX_RETRIES times. After that:
        - if strict=True: raise an exception (fail fast)
        - else: log and skip (continue with other jobs)

    Returns:
      Path to the final output (or the intended path if skipped under non-strict failure mode).
    """
    out = job.out_path(raw_dir)

    if out.exists() and not overwrite:
        if _is_valid_existing_file(out):
            log.info("Exists and valid, skipping: %s", out)
            return out
        log.warning("Existing file invalid/suspicious, re-downloading: %s", out)
        out.unlink(missing_ok=True)

    tmp = out.with_suffix(out.suffix + ".part")

    # CDS request object for ERA5-Land. day/time are requested as complete grids; CDS will
    # return the available timestamps for that month.
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

            # cdsapi reads credentials from ~/.cdsapirc and submits the job to CDS.
            # A new client per attempt is simple and keeps state isolated.
            c = cdsapi.Client()
            c.retrieve("reanalysis-era5-land", request, str(tmp))

            if not tmp.exists():
                raise RuntimeError("CDS returned no file")

            size = tmp.stat().st_size
            if size < MIN_EXPECTED_BYTES:
                raise RuntimeError(f"Downloaded file too small ({size} bytes)")

           # CDS sometimes returns ZIP even when NetCDF requested; handle that case.
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

            # Deep validation: ensures we downloaded what we think we did.
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
    """
    CLI entrypoint.

    Steps:
    1) Parse years and regions.
    2) Build a deterministic download plan (list of DownloadJob).
    3) Optionally write plan JSON and/or dry-run.
    4) Execute downloads, skipping valid existing files unless --overwrite.
    5) Print totals and elapsed time.

    Exit codes:
    - 0 on success (including dry-run)
    - raises SystemExit for invalid CLI arguments
    - may raise RuntimeError if --strict and a job repeatedly fails
    """
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

    # Optional plan dump for reproducibility / debugging / resuming logic outside this script.
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

    # Main execution: skip files that already validate unless you force overwrite.
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
