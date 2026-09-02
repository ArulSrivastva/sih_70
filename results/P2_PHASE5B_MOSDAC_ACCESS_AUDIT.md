# P2 Phase 5B — MOSDAC Data Access & Technical Specification Report

**Date**: September 1, 2026  
**Source**: ISRO / IMD Meteorological and Oceanographic Satellite Data Archival Centre ([MOSDAC](https://mosdac.gov.in)).

---

## 1. Authentication & Query Mechanics

* **Portal**: MOSDAC Open Search (`https://mosdac.gov.in/open_search`)
* **Authentication**: Token-based user authentication using `config.json` credentials and `mdapi.py` query client.
* **Batch Downloader**: Permitted via official REST endpoint using temporal bounding boxes (`start_time`, `end_time`) and spatial polygons.

---

## 2. Technical Data Product Specifications

* **Primary Sensor Product**: `3DIMG_L1C_SGP` / `3RIMG_L1C_SGP`
* **Channel**: Thermal Infrared 1 (**TIR-1**, $10.8\ \mu\text{m}$)
* **Coordinate Projection**: Mercator / Standard Grid Projection (SGP) over South Asia ($44.5^\circ\text{E}–105.5^\circ\text{E}, 10^\circ\text{S}–45.5^\circ\text{N}$).
* **Spatial Resolution**: $4.0\text{ km}$ per pixel at nadir.
* **Temporal Resolution**: Half-hourly ($30\text{ minutes}$).
* **File Formats**: HDF5 (`.h5`) with full radiometric calibration coefficients and GeoTIFF (`.tif`).
* **Storage Footprint**: Full 6-channel L1C file $\approx 18.5\text{ MB}$; single-channel TIR-1 subset $\approx 3.2\text{ MB}$.

---

## 3. Historical Availability (2014–2024)
INSAT-3D has been continuously operational since early 2014; INSAT-3DR was added in late 2016. Every North Indian Ocean tropical cyclone from 2014 to 2024 (e.g., *Hudhud, Fani, Amphan, Tauktae, Gulab, Biparjoy, Remal*) is fully cataloged and accessible in the MOSDAC archive.
