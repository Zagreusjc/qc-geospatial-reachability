# Data Manifest

All datasets are open, static snapshots. Record the exact retrieval date and
version for every download so the study is reproducible (cross-sectional design).
Fill the "Retrieved" and "Version / notes" columns when `00_download_data.py` and
`01b_barangays.py` are run.

| Dataset | Role | Source | Retrieved | Version / notes |
|---|---|---|---|---|
| HOTOSM Philippines Roads | Road-network skeleton (nodes/edges, `highway`, `oneway`) | https://data.humdata.org/dataset/hotosm_phl_roads | _to fill_ | _to fill_ |
| PSA / NAMRIA COD-AB (ADM3 + ADM4) | QC clip boundary + 142 barangay polygons | https://data.humdata.org/dataset/cod-ab-phl | _to fill_ | _to fill_ |
| JICA 2015 speed lookup table | Road-class operating speeds (km/h) | JICA (2015) report; transcribed into `src/config.py` | n/a | Not a file download |

## Barangay boundary source used

The resolver (`src/01b_barangays.py`) tries sources in order and caches the first
that succeeds to `data/processed/qc_barangays.gpkg`. Record which one was used:

- [ ] 1. HDX COD-AB ADM4 (primary) -- https://data.humdata.org/dataset/cod-ab-phl
- [ ] 2. Curated PSA/NAMRIA GeoJSON -- https://github.com/bendlikeabamboo/barangay-boundaries-repository
- [ ] 3. OpenStreetMap `admin_level=10` (via OSMnx)
- [ ] 4. geoBoundaries PHL ADM4 -- https://www.geoboundaries.org
- [ ] 5. Synthesized Voronoi proxy zones (documented limitation)

**Source used:** _to fill_   **Feature count:** _to fill_ (expected ~142)

## Licensing

- HOTOSM / OpenStreetMap data: Open Database License (ODbL).
- COD-AB boundaries: NAMRIA / PSA via OCHA HDX (see dataset page for terms).
- geoBoundaries: CC BY 4.0.
