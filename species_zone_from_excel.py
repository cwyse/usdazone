#!/usr/bin/env python3
# species_zone_from_excel.py
"""
Batch tool: read an Excel spreadsheet containing species names, compute a GBIF-based
cold-edge USDA-like zone for each species, and write a new spreadsheet.

This script is built to support spreadsheets that contain:
  - a "Species" column with Genus/species plus optional ranks (subsp., var., cultivar, etc.)
  - formulas in the species column (see "Formula evaluation" below)

Inputs
------
- An input .xlsx file
- The name of the species column (header text)
- The ERA5-derived climatology NetCDF file (global_usda_zone_temperature_1991_2020.nc)

Processing (per row)
--------------------
For each row:
  1) Read the cell from the species column.
  2) If the cell is a formula, use the *cached evaluated value* (data_only=True).
  3) Parse out only "Genus species" (two tokens) and ignore the rest.
  4) Run cold-edge estimation (GBIF occurrences + dominance pruning + thinning + ERA5 sampling).
  5) Output 4 columns:
       - parsed_species: "Genus species" used for GBIF search
       - latitude: latitude of selected cold-edge occurrence
       - longitude: longitude of selected cold-edge occurrence
       - zone: zone label at that point (or "Unknown" on error)

Verbose output (--verbose)
-------------------------
When --verbose is enabled, the script prints for each processed row:
  - Excel cell reference (e.g., G2)
  - raw cell contents (evaluated value if formula)
  - parsed species
On success it prints:
  - cold-edge lat/lon used
  - zone label
On error it prints a one-line error and continues to the next row.

Formula evaluation (important)
------------------------------
openpyxl does NOT calculate formulas. It can only read:
  - the formula text (data_only=False), or
  - the last saved *cached result* (data_only=True)

This script uses data_only=True so it can read cached results.
Therefore: if your species column contains formulas, you must open the workbook
in Excel or LibreOffice and save it once so the formula results are cached.

Performance notes
-----------------
Cold-edge estimation can be slow if many species are processed:
  - This script opens the climate dataset once and reuses it for all rows.
  - Dominance pruning + thinning greatly reduce point sampling.
  - Network calls to GBIF remain a dominant cost.

Example
-------
python ./species_zone_from_excel.py \
  --in-xlsx your_species.xlsx \
  --species-col "Species" \
  --dataset data/processed/global_usda_zone_temperature_1991_2020.nc \
  --out-xlsx species_zones.xlsx \
  --verbose

"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict
from typing import Any, Optional, Tuple

from openpyxl import load_workbook
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from gbif_cold_edge import ColdEdgeConfig, ColdEdgeEstimator


SPECIES_PARSE_RE = re.compile(r"^\s*([A-Z][a-zA-Z-]+)\s+([a-z][a-zA-Z-]+)\b")


def parse_species_two_words(value: Any) -> Optional[str]:
    """
    Parse a cell value and return only "Genus species" (two tokens), or None.

    Rules:
      - Input may be None, numeric, or string.
      - We accept a leading "Genus species" and ignore any trailing rank/cultivar text.
      - Genus: starts with uppercase letter
      - species epithet: starts with lowercase letter

    Examples:
      "Abies koreana 'Horstmann's Silberlocke'" -> "Abies koreana"
      "Acer tataricum ssp ginnala"              -> "Acer tataricum"
      0                                         -> None
    """
    if value is None:
        return None

    # Convert to string for regex matching, but reject obvious non-text.
    if isinstance(value, (int, float)):
        # Some sheets have 0 or numeric placeholders; treat as empty.
        if float(value) == 0.0:
            return None
        return None

    s = str(value).strip()
    if not s or s == "0":
        return None

    m = SPECIES_PARSE_RE.match(s)
    if not m:
        return None
    return f"{m.group(1)} {m.group(2)}"


def find_header_column(ws: Worksheet, header_name: str, header_row: int) -> int:
    """
    Find the 1-based column index for the given header name in header_row.
    Matches trimmed, case-sensitive header text.
    """
    target = header_name.strip()
    for col in range(1, ws.max_column + 1):
        v = ws.cell(row=header_row, column=col).value
        if v is None:
            continue
        if str(v).strip() == target:
            return col
    raise ValueError(f"Could not find header '{header_name}' in row {header_row}.")


def excel_cell_ref(ws: Worksheet, row: int, col: int) -> str:
    """
    Return A1-style reference (e.g., 'G2').
    """
    return ws.cell(row=row, column=col).coordinate


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute GBIF cold-edge USDA-like zone per species from an Excel column.")
    p.add_argument("--in-xlsx", required=True, help="Input .xlsx file.")
    p.add_argument("--sheet", default=None, help="Sheet name (default: active sheet).")
    p.add_argument("--species-col", required=True, help='Header text of the species column (e.g., "Species").')
    p.add_argument("--header-row", type=int, default=1, help="Header row number (default: 1).")

    p.add_argument("--dataset", required=True, help="Path to global_usda_zone_temperature_1991_2020.nc (or equivalent).")
    p.add_argument("--out-xlsx", required=True, help="Output .xlsx file.")

    # Cold-edge config
    p.add_argument("--max-records", type=int, default=5000, help="Max GBIF records to fetch per species (default: 5000).")
    p.add_argument("--page-size", type=int, default=300, help="GBIF page size (default: 300).")
    p.add_argument("--max-uncertainty-m", type=int, default=10_000, help="Filter out records with uncertainty > this (default: 10000).")

    p.add_argument("--no-dominance-pruning", action="store_true", help="Disable dominance pruning.")
    p.add_argument("--tropics-abs-lat-deg", type=float, default=20.0, help="Skip pruning for |lat| < this (default: 20).")

    p.add_argument("--no-thin", action="store_true", help="Disable thinning (slower).")
    p.add_argument("--grid-km", type=float, default=25.0, help="Thinning grid size in km (default: 25).")

    p.add_argument("--use-min", action="store_true", help="Use absolute coldest point (more sensitive to outliers).")
    p.add_argument("--quantile", type=float, default=0.05, help="Cold-tail quantile (default: 0.05). Ignored if --use-min.")
    p.add_argument("--drivers", type=int, default=25, help="Number of coldest driver points to compute (default: 25).")

    p.add_argument("--verbose", action="store_true", help="Print progress for each row.")
    return p


def main(argv: list[str]) -> int:
    args = build_arg_parser().parse_args(argv)

    # data_only=True is REQUIRED to read cached results of formulas.
    wb_in: Workbook = load_workbook(args.in_xlsx, data_only=True)
    ws_in: Worksheet
    if args.sheet:
        if args.sheet not in wb_in.sheetnames:
            raise SystemExit(f"Sheet '{args.sheet}' not found. Available: {wb_in.sheetnames}")
        ws_in = wb_in[args.sheet]
    else:
        ws_in = wb_in.active

    species_col_idx = find_header_column(ws_in, args.species_col, args.header_row)

    # Create output workbook.
    from openpyxl import Workbook as XLWorkbook
    wb_out = XLWorkbook()
    ws_out = wb_out.active
    ws_out.title = "species_zones"

    # Output headers (exactly four columns as requested).
    ws_out.cell(row=1, column=1).value = "parsed_species"
    ws_out.cell(row=1, column=2).value = "latitude"
    ws_out.cell(row=1, column=3).value = "longitude"
    ws_out.cell(row=1, column=4).value = "zone"

    cfg = ColdEdgeConfig(
        max_records=args.max_records,
        page_size=args.page_size,
        max_uncertainty_m=args.max_uncertainty_m,
        dominance_pruning=not args.no_dominance_pruning,
        tropics_abs_lat_deg=args.tropics_abs_lat_deg,
        do_thin=not args.no_thin,
        grid_km=args.grid_km,
        use_min=args.use_min,
        quantile=args.quantile,
        drivers=args.drivers,
    )

    out_row = 2
    # Iterate from first data row to ws.max_row.
    start_row = args.header_row + 1
    total_rows = max(0, ws_in.max_row - args.header_row)

    with ColdEdgeEstimator(args.dataset, cfg) as est:
        for i, row in enumerate(range(start_row, ws_in.max_row + 1), start=1):
            cell = ws_in.cell(row=row, column=species_col_idx)
            raw = cell.value
            parsed = parse_species_two_words(raw)

            if args.verbose:
                coord = excel_cell_ref(ws_in, row, species_col_idx)
                print(f"[{i}/{total_rows}] {coord} raw={raw!r} parsed={parsed!r}")

            if not parsed:
                # Skip empty/unparseable rows; still write a row with Unknown if there is a raw value?
                # Requested behavior: create output with species+zone; here we only output for parsed entries.
                # If you want one output row per input row regardless, change this to always write.
                continue

            lat_out: Optional[float] = None
            lon_out: Optional[float] = None
            zone_out: str = "Unknown"

            try:
                res = est.estimate(parsed)
                # Use selected cold-edge occurrence coordinates (not "northernmost").
                edge_cp, edge_occ = res.cold_edge
                lat_out = float(edge_occ.lat)
                lon_out = float(edge_occ.lon)
                zone_out = str(edge_cp.zone)

                if args.verbose:
                    print(f"  -> cold_edge lat={lat_out:.6f} lon={lon_out:.6f} zone={zone_out}")
            except Exception as e:
                if args.verbose:
                    print(f"  !! ERROR: {e}")

            # Write output row (always for each parsed species).
            ws_out.cell(row=out_row, column=1).value = parsed
            ws_out.cell(row=out_row, column=2).value = lat_out
            ws_out.cell(row=out_row, column=3).value = lon_out
            ws_out.cell(row=out_row, column=4).value = zone_out
            out_row += 1

    wb_out.save(args.out_xlsx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
