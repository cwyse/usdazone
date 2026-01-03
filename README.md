# ERA5-Land–Derived USDA Hardiness Zone Estimation

This repository builds a **global, ERA5-Land–based approximation of USDA Plant Hardiness Zones** and provides a **single, centralized API** for querying estimated zone temperature and zone labels at arbitrary latitude/longitude points.

It also includes tooling to **estimate the cold-edge hardiness zone of a plant species** using GBIF occurrence data sampled against the same climatology.

This is **not** a reproduction of the official USDA map. It is an internally consistent, global, reproducible estimate derived from reanalysis data.

---

## High-Level Overview

### What this project does

1. **Downloads ERA5-Land hourly 2 m air temperature** data in tiled regional chunks.
2. **Computes annual extreme minimum temperature** per grid cell using cold-season logic:
   - **Northern Hemisphere:** cold-season months only
   - **Southern Hemisphere:** cold-season months only
   - **Tropics:** all months (no meaningful cold season)
3. **Averages annual minima over a climatology period** (e.g., 1991–2020).
4. **Converts that temperature metric into USDA-like zones** (numeric + subzone).
5. **Merges tiles into a global dataset** with robust longitude handling.
6. **Provides a single API (`USDAZoneDataset`)** for all point sampling.
7. **Uses the same API to estimate species cold-edge zones** from GBIF records.

### What this project intentionally does *not* do

- It does **not** attempt to exactly match the official USDA zone map.
- It does **not** use PRISM (US-only) or station-based climatologies.
- It does **not** model microclimates, lapse‑rate elevation corrections, or local cold‑air pooling beyond what ERA5 resolves.

---

## Data Source and Rationale

### ERA5-Land

- **Source:** Copernicus Climate Data Store (CDS)
- **Resolution:** ~0.1° (~9–11 km)
- **Coverage:** Global
- **Variables used:** Hourly 2 m air temperature (`t2m`)

ERA5-Land was chosen because it provides:
- global coverage,
- consistent methodology,
- hourly resolution needed to compute *true annual extreme minima*,
- multi-decade historical coverage suitable for climatologies.

### Why annual extreme minimum?

The USDA hardiness zone definition is based on the **mean annual extreme minimum temperature** over a reference period.

This pipeline mirrors that concept by:
1. Computing the **coldest hourly temperature per grid cell per year**
   - Cold-season months only for extratropical hemispheres
   - All months for the tropics
2. Averaging those annual minima across a climatology window (e.g., 1991–2020)
3. Converting to °F
4. Applying standard USDA 10 °F bands

---

## Important Differences from Official USDA Zones

You should expect **systematic differences** between this dataset and the official USDA map:

| Cause | Effect |
|-----|-------|
| ERA5-Land resolution | Smoother fields; reduced local extremes |
| Reanalysis vs stations | Weaker cold extremes in complex terrain |
| Global methodology | No US-specific bias correction |
| Bilinear interpolation | Slight spatial smoothing |
| Cold-season definition | Hemisphere-consistent but not station-specific |

Typical observed effects:
- Mountain locations often appear **warmer** than USDA (elevation smoothing).
- Continental interiors may appear **slightly colder**.
- Coastal moderation is weaker than PRISM-based products.
- Single-point queries can differ by ±1 subzone.

These differences are expected when using a global reanalysis product.

---

## Processing Pipeline

```
ERA5-Land hourly NetCDF
  ↓
download_era5_land_hourly.py
  ↓
retile_region.py (normally not required unless region sizes vary)
  ↓
compute_annual_extreme_min.py
  ↓
compute_usda_zone_temperature.py
  ↓
merge_usda_zone_tiles.py
  ↓
global_usda_zone_temperature_1991_2020.nc
```

---

## Repository Structure

```
.
├── download_era5_land_hourly.py
│   Download ERA5-Land hourly temperature data from CDS.
├── retile_region.py
│   Normalize and retile raw NetCDFs into consistent regional tiles.
├── compute_annual_extreme_min.py
│ Compute annual extreme minimum temperature per grid cell.
├── compute_usda_zone_temperature.py
│   Average annual minima and derive USDA-like zone temperatures.
├── merge_usda_zone_tiles.py
│   Merge regional tiles into a global climatology.
├── usda_zone_core.py
│   Central longitude handling, interpolation, and zone logic.
├── usda_zone_access.py
│   Public API: USDAZoneDataset.point(lat, lon).
├── validate_locations.py
│   Sanity checks for cities and boundary cases.
├── gbif_cold_edge_era5.py
│ Species cold-edge inference using GBIF occurrences.
├── make_raw_geojson.py
│   Utility to emit tile extents as GeoJSON.
├── data/
│   ├── raw/         (raw downloaded data, not versioned)
│   └── processed/   (generated outputs, not versioned)
└── README.md
```

---

## Design Principles

This repository is built around the following non-negotiable principles:

- **Single source of truth for spatial logic**
  All longitude normalization, wraparound handling, and interpolation live in
  `usda_zone_core.py` and are accessed exclusively via `USDAZoneDataset.point()`.

- **Explicit longitude handling**
  Mixed `[-180,180]` and `[0,360)` inputs are normalized deterministically.
  Cyclic longitude seams are handled explicitly, never implicitly.

- **No silent fallback behavior**
  Missing or invalid data produces explicit missing values, not guessed zones.

- **Reproducible scientific results**
  Inputs, processing steps, and assumptions are fixed and documented.
  Dependencies are pinned.

- **Clear separation of concerns**
  - Data acquisition and preprocessing
  - Climate metric derivation
  - Spatial lookup and interpolation
  - Biological inference (species cold-edge estimation)

---

## Requirements

### Python

- **Python ≥ 3.10**

### Python dependencies (pinned)

See `requirements.txt`:

- numpy==1.26.4
- xarray==2024.1.1
- netCDF4==1.6.5
- cdsapi==0.6.1
- requests==2.31.0

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Accessing Remote Data Sources

This repository relies on **external data services**. Credentials are not bundled and must be configured by the user.

### Copernicus Climate Data Store (ERA5-Land)

ERA5-Land data are accessed using the official `cdsapi` Python client.

To enable downloads you must:
1. Create a CDS account
2. Accept the ERA5-Land license terms
3. Create a credentials file at `~/.cdsapirc`

The CDS client reads credentials automatically from this file.

Official setup instructions:
https://cds.climate.copernicus.eu/api-how-to

**Do not commit `.cdsapirc` to version control.**

Only the download step requires CDS access; all later processing can be run offline once data are available.

---

### GBIF Occurrence Data

Species occurrences are fetched from the public GBIF Occurrence Search API.

- No API key is required for standard usage
- Requests are paginated and rate-limited

Users are responsible for complying with GBIF citation and data-use guidelines:
https://www.gbif.org/developer/occurrence---

## Using the Dataset API

```python
from usda_zone_access import USDAZoneDataset

with USDAZoneDataset("data/processed/global_usda_zone_temperature_1991_2020.nc") as zds:
    p = zds.point(41.77, -72.68)
    print(p.temp_f, p.zone_label)
```

- Longitude may be supplied in `[-180, 180]` or `[0, 360]`.
- Seam handling and interpolation are automatic.
- Missing data returns a neutral placeholder (`zone_label="—"`).

---

## Species Cold-Edge Estimation (GBIF)

Example:

```bash
./gbif_species_cold_edge_era5.py \
  --species "Aquilegia sibirica" \
  --dataset data/processed/global_usda_zone_temperature_1991_2020.nc \
  --grid-km 25 \
  --quantile 0.05 \
  --drivers 25 \
  --out aquilegia_sibirica_edge.json
```

The script reports:
- northernmost occurrence,
- coldest sampled climate point,
- selected cold-edge point (quantile or minimum),
- driver points explaining the decision.

All climate sampling uses the same API as all other workflows.

---

## License / Usage

This repository is intended for research, analysis, and personal use.

- ERA5-Land data remains subject to Copernicus licensing terms.
- GBIF data usage must comply with GBIF citation and attribution guidelines.

---

## Differences from Official USDA Hardiness Zones

This project intentionally does **not** reproduce the official USDA Plant Hardiness Zone map.
Differences are expected, understood, and documented.

### Data and Methodology Comparison

| Aspect | This Project | Official USDA |
|------|-------------|---------------|
| Data source | ERA5-Land reanalysis | PRISM + stations |
| Resolution | ~9–11 km | ~800 m |
| Coverage | Global | United States |
| Elevation correction | No | Yes |
| Microclimates | Smoothed | Partially captured |

### Typical Effects You Will See

- Mountain locations often appear **warmer** than USDA zones due to smoothed elevation.
- Coastal moderation is **weaker** than in PRISM-based products.
- Urban heat islands are **not captured**.
- Single-point queries may differ by **±1 subzone** from USDA labels.

This dataset represents a **climate-based equivalent**, not an authoritative planting zone.

---

Summary

If you want:
- a **global, reproducible USDA-like zone metric**,
- a **clean, stable climate lookup API**,
- a **defensible method for estimating species cold-edge hardiness**,

this repository provides exactly that — with assumptions explicit and behavior centralized.
