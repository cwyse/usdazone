#!/bin/bash

# Download all data.  It will take days.  Comment out unless required.
#python download_era5_land_hourly.py --years 1991-2020             \
#                                    --raw-dir data/raw/era5_land_hourly_temp

# Used to retile only what was needed and remove overlaps due to errors/mistakes in download (deprecated).
# Repeat for north and tropics regions.
#python retile_region.py \
#  --root data/raw/era5_land_hourly_temp \
#  --out-root data/raw/era5_land_hourly_temp_retiled \
#  --region south \
#  --src-dirs era5_land_hourly_lat-90_lon000 era5_land_hourly_lat-90_lon100 era5_land_hourly_lat-90_lon200 era5_land_hourly_lat-90_lon300  \
#  --apply 

# (Optional): Read downloaded data directories and create geojson files 
# Files can be loaded into https://geojson.io to confirm earth coverage
#python make_raw_geojson.py  \
#  --root data/raw/era5_land_hourly_temp \
#  --out-dir data/processed/geojson

# Compute annual extreme minimum temperature per tile/year
python compute_annual_extreme_min.py

# Compute USDA Plant Hardiness Zone temperature metric:
# mean annual extreme minimum temperature over a climatology period.
python compute_usda_zone_temperature.py --all-tiles --start-year 1991 --end-year 2020 --check-hartford  

# Finally, merge the data over 30 years keeping the minimum temperatures at each location
python merge_usda_zone_tiles.py --start-year 1991 --end-year 2020

# At this point we've calculated the mean annual extreme minimum temperature from 1991-2020
# across the globe on a 9 km grid.  All the temperatures are stored in a single file, global_usda_zone_temperature_1991_2020.nc.

# Test the data against cities in the US and abroad
python validate_locations.py

python gbif_cold_edge_era5.py \
  --species "Areca catechu" \
  --dataset data/processed/global_usda_zone_temperature_1991_2020.nc \
  --use-min

python gbif_cold_edge_era5.py \
  --species "Aquilegia sibirica" \
  --dataset data/processed/global_usda_zone_temperature_1991_2020.nc


#!/bin/bash
#
# runme.sh — Rebuild USDA-zone-style products from ERA5-Land, then query zones
#
# Purpose
# -------
# This is a “memory jogger” script: it documents the end-to-end rebuild steps for
# the derived NetCDF products used by your lookup API (USDAZoneDataset / usda_zone_access.py).
#
# Important notes
# --------------
# - This file is intentionally NOT “hardened” (no `set -e`, no argument parsing) so it
#   stays flexible as a personal runbook. All changes here are comments only.
# - Most steps assume you are running from the repository root and that your Python
#   environment has the required dependencies installed (xarray, numpy, netcdf backend, etc.).
# - The ERA5-Land download step can take a very long time and consume significant disk space.
#
# Data flow (high level)
# ----------------------
#   1) (Optional) Download raw ERA5-Land hourly 2m temperature data (large).
#   2) Convert raw hourly data -> annual extreme minimum tiles (per-year, per-tile).
#   3) Collapse annual tiles -> climatology (e.g., 1991–2020 mean annual extreme minimum).
#   4) Merge tiles -> one global product NetCDF used by the query API.
#
# Output artifacts (typical)
# --------------------------
# - data/processed/annual_extreme_min_tiles/...
# - data/processed/usda_zone_temperature_tiles/...
# - data/processed/global_usda_zone_temperature_1991_2020.nc
#
# ------------------------------------------------------------------------------
# STEP 0 — OPTIONAL: Download ERA5-Land hourly data (huge; can take days)
# ------------------------------------------------------------------------------
# This fetches the raw hourly data used to compute the annual extreme minimum.
# You normally do NOT re-run this unless you are rebuilding from scratch.
#
# Expected prerequisites (typical):
# - Copernicus CDS credentials configured (e.g., ~/.cdsapirc) if your downloader uses CDS.
# - Enough disk space for multi-year hourly data.
#
# The command is commented out on purpose to prevent accidental multi-day downloads.
#
#python download_era5_land_hourly.py --years 1991-2020             \
#                                   --raw-dir data/raw/era5_land_hourly_temp
#
# What it should produce (conceptually):
# - A directory tree under data/raw/era5_land_hourly_temp containing hourly temperature
#   data sufficient to compute annual minimum temperatures for each year.
#
# ------------------------------------------------------------------------------
# (Deprecated) Retiling helper notes
# ------------------------------------------------------------------------------
# These comments are retained as context for past fixes:
# - You previously re-tiled only what was needed and removed overlaps caused by
#   download mistakes.
# - If you ever need to reproduce that process, you would repeat it for north and
#   tropics regions (and presumably south if applicable).
#
#python retile_region.py \
#  --root data/raw/era5_land_hourly_temp \
#  --out-root data/raw/era5_land_hourly_temp_retiled \
#  --region south \
#  --src-dirs era5_land_hourly_lat-90_lon000 era5_land_hourly_lat-90_lon100 era5_land_hourly_lat-90_lon200 era5_land_hourly_lat-90_lon300  \
#  --apply 
#
# ------------------------------------------------------------------------------
# (Optional) GeoJSON footprint generation (documentation / sanity checks)
# ------------------------------------------------------------------------------
# This is useful for debugging coverage (tile boundaries, overlaps, gaps).
# It is not required to build the final NetCDFs used by the zone lookup API.
#
# Read downloaded data directories and create geojson files.  Files can be 
# loaded into https://geojson.io to confirm earth coverage
#
#python make_raw_geojson.py  \
#  --root data/raw/era5_land_hourly_temp \
#  --out-dir data/processed/geojson# ...
#
# ------------------------------------------------------------------------------
# STEP 1 — Annual extreme minimum per year, per tile
# ------------------------------------------------------------------------------
# This reads the raw ERA5-Land hourly data and computes, for each grid cell,
# the minimum hourly temperature in each year (annual extreme minimum).
#
# Output location (expected):
# - data/processed/annual_extreme_min_tiles/<tile>_<year>.nc
#
python compute_annual_extreme_min.py
#
# Notes:
# - “--all-tiles” means process every expected tile region (north/tropics/south)
#   and all longitude chunks within each region (as your pipeline defines them).
# - “--year-range 1991-2020” means compute annual-minimum products for each year.
#
# ------------------------------------------------------------------------------
# STEP 2 — Climatology (mean annual extreme minimum) + USDA-zone temperature tiles
# ------------------------------------------------------------------------------
# This step aggregates the annual products over a period (e.g., 1991–2020) to get
# the climatological mean of the annual extreme minimum temperature at each grid cell.
#
# Output location (expected):
# - data/processed/usda_zone_temperature_tiles/<tile>_mean_1991_2020.nc
#
python compute_usda_zone_temperature.py --all-tiles --start-year 1991 --end-year 2020 --check-hartford 
#
# Notes:
# - This produces the “temperature metric” your USDA zone logic is based on
#   (mean of annual extreme minimum over the climatology period).
# - Any USDA zone *labeling / conversion* should be performed consistently via your
#   shared API (usda_zone_access.py / usda_zone_core.py), not re-implemented ad hoc.
#
# ------------------------------------------------------------------------------
# STEP 3 — Merge tiles into a single global NetCDF
# ------------------------------------------------------------------------------
# This combines all of the per-tile climatology outputs into one global dataset file.
#
# Output location (expected):
# - data/processed/global_usda_zone_temperature_1991_2020.nc
#
python merge_usda_zone_tiles.py --start-year 1991 --end-year 2020
#
# Notes:
# - This is the dataset your lookup class typically opens once and queries repeatedly.
# - The merged file should contain a consistent longitude convention (usually 0..360)
#   and consistent variable naming (e.g., usda_zone_temp_f).
#
# ------------------------------------------------------------------------------
# Usage examples (queries)
# ------------------------------------------------------------------------------
# These are intentionally commented out — they are “how to use it” reminders.
#
# 1) Validate known points / boundary behavior (your quick regression test):
#    python validate_locations.py
#
# 2) Query a single point using your API (example pattern — adjust script name/options
#    to match your current CLI wrapper, if any):
#    python -c 'from usda_zone_access import USDAZoneDataset; \
#               from pathlib import Path; \
#               zds=USDAZoneDataset(Path("data/processed/global_usda_zone_temperature_1991_2020.nc")); \
#               print(zds.point(41.77, -72.68))'
#
# 3) Species “cold edge” workflow (GBIF occurrences -> pick coldest / northernmost point
#    -> query equivalent zone). Keep the logic in a dedicated script that calls the same
#    zone API used everywhere else (so interpolation + zone labeling stay consistent).
#
# - Runs gbif_cold_edge_era5.py to estimate the cold-edge hardiness zone for Areca catechu.
# - Fetches GBIF occurrence records for "Areca catechu" (with coordinates + basic filters),
#   then spatially thins them on a ~25 km grid (default), to reduce oversampling bias and runtime.
# - For each remaining occurrence point, queries your ERA5-derived 1991–2020 climatology dataset
#   (data/processed/global_usda_zone_temperature_1991_2020.nc) via USDAZoneDataset.point(lat, lon)
#   to get the temperature metric (°F) and derived zone label.
# - Selects the cold-edge point using --use-min: the single coldest sampled point (absolute min temp_f),
#   rather than a cold-tail quantile, and reports its lat/lon, temp_f, and zone.
# - Writes the JSON report to the default output file (species_edge_era5.json) unless --out is provided.
python gbif_cold_edge_era5.py \
  --species "Areca catechu" \
  --dataset data/processed/global_usda_zone_temperature_1991_2020.nc \
  --use-min

# - Intended: run gbif_cold_edge_era5.py for "Aquilegia sibirica" using the same global ERA5 climatology dataset.
# - However, the line is malformed because "...1991_2020.ncpython ..." is concatenated with no newline/space.
#   In bash, that means "--dataset" is effectively set to:
#       data/processed/global_usda_zone_temperature_1991_2020.ncpython
#   and then it tries to execute "gbif_cold_edge_zone.py" as additional arguments/words.
# - Result: this will fail (likely "file not found" for the dataset path, and/or "unrecognized arguments").
#
# If you meant TWO separate commands, they should be split like this:
#
#   2a) ERA5-based cold-edge for Aquilegia sibirica (writes species_edge_era5.json by default):
#       python gbif_cold_edge_era5.py \
#         --species "Aquilegia sibirica" \
#         --dataset data/processed/global_usda_zone_temperature_1991_2020.nc
#
#   2b) Legacy CHELSA BIO6 GeoTIFF-based cold-edge (older approach) for Aquilegia sibirica,
#       writing to species_cold_edge.json:
#       python gbif_cold_edge_zone.py --species "Aquilegia sibirica" --out species_cold_edge.json
python gbif_cold_edge_era5.py \
  --species "Aquilegia sibirica" \
  --dataset data/processed/global_usda_zone_temperature_1991_2020.ncpython gbif_cold_edge_zone.py --species "Aquilegia sibirica" --out species_cold_edge.json
#
