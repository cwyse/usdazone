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
