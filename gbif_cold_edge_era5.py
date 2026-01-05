#!/usr/bin/env python3
# gbif_cold_edge_era5.py
"""
CLI: Estimate a species "cold-edge" hardiness zone using GBIF occurrences + an ERA5-derived
USDA-like climatology.

This script is a thin wrapper over the library implementation in gbif_cold_edge.py.

It:
  - parses command-line arguments
  - runs ColdEdgeEstimator on one species
  - writes a JSON report to disk (or stdout)

Design principles
-----------------
- All spatial lookup and temp->zone logic remains inside USDAZoneDataset.point().
- All GBIF fetch + prune + thin + selection logic is centralized in gbif_cold_edge.py.
- This CLI should not implement its own cold-edge logic.

Example
-------
python ./gbif_cold_edge_era5.py \
  --species "Aquilegia sibirica" \
  --dataset data/processed/global_usda_zone_temperature_1991_2020.nc \
  --grid-km 25 \
  --quantile 0.05 \
  --drivers 25 \
  --out aquilegia_sibirica_edge.json

Exit codes
----------
0: success
2: argument parsing / usage error
3: runtime error (no overlap, GBIF issues, etc.)

"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from gbif_cold_edge import ColdEdgeConfig, ColdEdgeEstimator, result_to_report_dict


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate species cold-edge hardiness zone using GBIF + ERA5-derived climatology.")
    p.add_argument("--species", required=True, help='Scientific name (e.g., "Abies koreana").')
    p.add_argument("--dataset", required=True, help="Path to global_usda_zone_temperature_1991_2020.nc (or equivalent).")

    # Controls
    p.add_argument("--max-records", type=int, default=5000, help="Max GBIF records to fetch (default: 5000).")
    p.add_argument("--page-size", type=int, default=300, help="GBIF page size (default: 300).")
    p.add_argument("--max-uncertainty-m", type=int, default=10_000, help="Filter out records with uncertainty > this (default: 10000).")

    p.add_argument("--no-dominance-pruning", action="store_true", help="Disable dominance pruning (slower).")
    p.add_argument("--tropics-abs-lat-deg", type=float, default=20.0, help="Skip pruning for |lat| < this (default: 20).")

    p.add_argument("--no-thin", action="store_true", help="Disable thinning (much slower).")
    p.add_argument("--grid-km", type=float, default=25.0, help="Thinning grid size in km (default: 25).")

    p.add_argument("--use-min", action="store_true", help="Use absolute coldest point (more sensitive to outliers).")
    p.add_argument("--quantile", type=float, default=0.05, help="Cold-tail quantile (default: 0.05). Ignored if --use-min.")

    p.add_argument("--drivers", type=int, default=25, help="Number of coldest driver points in report (default: 25).")

    # Output
    p.add_argument("--out", default=None, help="Output JSON path. If omitted, prints to stdout.")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    return p


def main(argv: list[str]) -> int:
    args = build_arg_parser().parse_args(argv)

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

    try:
        with ColdEdgeEstimator(args.dataset, cfg) as est:
            res = est.estimate(args.species)
        report = result_to_report_dict(res)
    except Exception as e:
        err = {
            "ok": False,
            "species": args.species,
            "dataset": args.dataset,
            "config": asdict(cfg),
            "error": str(e),
        }
        out_s = json.dumps(err, indent=2 if args.pretty else None, sort_keys=False)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(out_s + "\n")
        else:
            print(out_s)
        return 3

    out_s = json.dumps(report, indent=2 if args.pretty else None, sort_keys=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_s + "\n")
    else:
        print(out_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
