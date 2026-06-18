# Source Catalog — Phase 10 ETL Chain

> **Status**: Decision-ready catalog. No connector code.
>
> **Epic**: Phase 10 — Source-driven ETL and reproducible ML-ready dataset generation
>
> **Story**: [Issue #105](https://github.com/zheddhe/avr25-mle-trafic-cycliste/issues/105)

---

## Classification Legend

| Classification | Meaning |
| --- | --- |
| **Required** | Needed for MVP — the pipeline cannot produce valid predictions without it. |
| **Optional** | Improves prediction quality but the pipeline can run without it (with degraded accuracy). |
| **Experimental** | Investigational — not yet validated for production use, outside MVP scope. |

---

## Source 1 — Paris Bike-Counting Dataset (Primary)

| Field | Value |
| --- | --- |
| **Name** | Comptage vélo — Données compteurs |
| **Owner** | Mairie de Paris — Direction de la Voirie et de la Circulation |
| **Provider** | [opendata.paris.fr](https://opendata.paris.fr) (Opendatasoft platform) |
| **Access mode** | File download (CSV) via open data portal; API available through Opendatasoft |
| **Format** | UTF-8 CSV, French column names; JSON via API |
| **Update frequency** | Daily at J-1 (yesterday's data). Covers 13 rolling months of data. |
| **Classification** | **Required** |
| **Usage** | Sole raw input to the ML pipeline. Ingested by `src/ml/ingest/import_raw_data.py`, grouped by counter site and orientation, and sliced into per-counter time series. |
| **Current file** | `data/raw/comptage-velo-donnees-compteurs-2024-2025_Enriched_ML-ready_data.csv` (~149 MB, DVC-tracked) |

### Source Verification (Bike-Counting Dataset)

- **Official dataset title**: "Comptage vélo — Données compteurs"
- **Dataset description**: "Jeu de données des comptages horaires de vélos par compteur et localisation des sites de comptage en J-1 sur 13 mois glissants"
- **Sensor partner**: [Eco Compteur](https://www.eco-compteur.com) — data loaded daily into Paris's API
- **Direction reconstruction**: The Eco Compteur API does not natively provide direction-separated counts; aggregation was performed to reconstruct directions
- **Counter evolution**: Number of counters changes over time as cycling infrastructure is added; counters may be deactivated for maintenance or failures
- **Coverage**: Counters are located on bike lanes and some bus lanes open to bikes; other vehicles (e.g., scooters) are not counted
- **Dataset warning**: As of 08/09/2023, the dataset experienced malfunctions that could affect data accuracy (technical teams were mobilized to resolve)
- **Join with counter reference**: A join was performed with the "Comptage vélo — Compteurs" dataset to retrieve descriptive site information including location and installation date
- **API usage**: 3,745,079 API calls recorded; 163,740 downloads (from Opendatasoft analytics)

### Key Columns

| Column | Type | Description |
| --- | --- | --- |
| `nom_du_site_de_comptage` | string | Counter site name (e.g., "Sebastopol") |
| `orientation_compteur` | string | Counter direction (e.g., "N-S", "S-N") |
| `date_et_heure_de_comptage` | datetime (ISO 8601) | Timestamp of the hourly count |
| `comptage_horaire` | int | Hourly bicycle count |
| `weather_code_wmo_code` | int | WMO weather code (embedded) |
| `temperature_2m_c` | float | Temperature in °C (embedded) |
| `rain_mm` | float | Rainfall in mm (embedded) |
| `snowfall_cm` | float | Snowfall in cm (embedded) |
| `latitude` | float | Site latitude (embedded) |
| `longitude` | float | Site longitude (embedded) |
| `arrondissement` | int | Paris arrondissement number (embedded) |
| `elevation` | float | Site elevation in meters (embedded) |

### Known Limits (Bike-Counting Dataset)

| Limit | Detail |
| --- | --- |
| **Rolling-window history** | Current file covers 2024–2025 only. Historical data prior to this window is not included. The pipeline's test ratio (0.25) and roll window (24 h) assume sufficient history. |
| **Timezone** | Timestamps are in local Paris time (CET/CEST). No explicit timezone annotation in the CSV. Risk of ambiguity during daylight-saving transitions. |
| **Schema stability** | Column names are in French and may change between Mairie de Paris releases. The "Enriched_ML-ready" suffix suggests the schema has been adapted for this project, not that it is the official schema. |
| **Expected volume** | ~149 MB for the current snapshot. Grows with additional counters and time. |
| **Missing values** | Gaps exist for counter maintenance periods and sensor failures. Imputation strategy is not yet defined. |
| **Automation** | **Not automated.** The pipeline currently starts from a static CSV file. New data requires manual file replacement. Phase 10 aims to automate this ingestion. |

---

## Source 2 — Counter Reference / Counting Sites

| Field | Value |
| --- | --- |
| **Name** | Comptage vélo — Compteurs |
| **Owner** | Mairie de Paris — Direction de la Voirie et de la Circulation |
| **Provider** | [opendata.paris.fr](https://opendata.paris.fr) (separate dataset from raw counts) |
| **Access mode** | File download (CSV / GeoJSON) via open data portal; API available through Opendatasoft |
| **Format** | CSV or GeoJSON; JSON via API |
| **Update frequency** | Rare — only when counters are added, removed, or relocated. |
| **Classification** | **Required** |
| **Usage** | Provides authoritative metadata for each counter: unique identifier, site name, geographic coordinates, installation date, and operational status. Used to validate counter identifiers in the raw dataset and to enrich the dataset with stable site metadata. |
| **Current state** | Not yet separated from the raw dataset. Counter metadata is embedded in the raw CSV (latitude, longitude, arrondissement, elevation). Phase 10 should extract this into a dedicated reference source. |

### Source Verification (Counter Reference)

- **Official dataset title**: "Comptage vélo — Compteurs"
- **Purpose**: Reference dataset containing descriptive information for each counter including location and installation date
- **Join relationship**: The raw counts dataset ("Données compteurs") is joined with this reference dataset to enrich counter metadata
- **Counter identifiers**: Includes `id_compteur`, `nom_compteur`, `id` (site ID), `name` (site name), and `installation_date`
- **Automation**: **Not automated.** Currently embedded in the raw CSV.

---

## Source 3 — Public Holidays (France)

| Field | Value |
| --- | --- |
| **Name** | Calendrier des jours fériés français |
| **Owner** | Various national authorities (verified via public holiday APIs) |
| **Provider** | [date.nager.at](https://date.nager.at) (Public Holiday API) — confirmed France (FR) support |
| **Access mode** | REST API (JSON response) |
| **Format** | JSON |
| **Update frequency** | Annual — published each year for the upcoming 3–5 years. Fixed dates with rare legislative changes. |
| **Classification** | **Optional** |
| **Usage** | French public holidays significantly affect bike traffic patterns (e.g., Noël, Jour de l'An, Fête du Travail). Holidays should be encoded as features (binary flags or categorical labels) during feature engineering. |
| **Current state** | **Not implemented.** Datetime features (`src/ml/features/features_utils.py::DatetimePeriodicsTransformer`) capture cyclic patterns (hour, day, week, month, year) but do not encode holidays. |

### Source Verification (Public Holidays)

- **API endpoint**: `https://date.nager.at/api/v3/AvailableCountries` — confirmed France (FR) is supported
- **Coverage**: 130 countries supported including all French territories (FR, GP, MQ, RE, YT, MF, BL, PM, WF, PF, NC, PF, TF, PM)
- **API style**: RESTful JSON API, no authentication required for basic usage
- **Determinism**: Public holiday dates are fixed and deterministic (unlike school holidays). This source is safe for reproducible ETL.
- **Regional variation**: Some holidays (e.g., Ascension, Pentecôte) may have regional exceptions. Paris follows national rules.
- **Schema stability**: High — holiday dates are calendrical and do not change format.
- **Alternative providers**: [nageraud-guilloux/et-holidays](https://github.com/nageraud-guilloux/et-holidays) (GitHub repository with holiday data)

---

## Source 4 — School Holidays (Paris / Zone C)

| Field | Value |
| --- | --- |
| **Name** | Calendrier des vacances scolaires — Zone C (Paris) |
| **Owner** | Ministère de l'Éducation nationale |
| **Provider** | [data.gouv.fr](https://www.data.gouv.fr) — `calendrier-des-vacances-scolaires` dataset |
| **Access mode** | File download (CSV / JSON) via data.gouv.fr |
| **Format** | CSV or JSON |
| **Update frequency** | Annual — published each summer for the upcoming school year. |
| **Classification** | **Optional** |
| **Usage** | School holidays affect bike traffic, especially on weekdays. Paris (Zone C) has its own holiday schedule distinct from Zones A and B. Holidays should be encoded as features during feature engineering. |
| **Current state** | **Not implemented.** No school calendar integration exists in the pipeline. |

### Known Limits (School Holidays)

| Limit | Detail |
| --- | --- |
| **Determinism** | School holiday dates are set by decree and are deterministic once published. However, exceptional closures (e.g., strikes, health crises) may introduce unpredictability. |
| **Zone specificity** | Must use Zone C (Paris). Using the wrong zone's schedule would introduce feature leakage or misclassification. |
| **Schema stability** | Moderate — the dataset structure is stable but may include additional fields in future releases. |

---

## Source 5 — Weather Data

| Field | Value |
| --- | --- |
| **Name** | Météo — Données météorologiques historiques et prévisions |
| **Owner** | Multiple weather model providers (ECMWF, DWD, NOAA, Météo-France, JMA, KMA, KNMI, DMI, MeteoSwiss, UK Met Office) |
| **Provider** | [Open-Meteo API](https://open-meteo.com) (recommended for ETL) |
| **Access mode** | REST API (HTTP GET, JSON response) |
| **Format** | JSON |
| **Update frequency** | Hourly for forecasts; historical data available. |
| **Classification** | **Optional** |
| **Usage** | Weather conditions influence bike traffic (rain, temperature, snowfall). Currently embedded in the raw CSV as `weather_code_wmo_code`, `temperature_2m_c`, `rain_mm`, `snowfall_cm`. Phase 10 should fetch weather data directly from an API rather than relying on the embedded enrichment. |
| **Current state** | **Partially implemented.** Weather columns exist in the raw CSV but are **dropped** during feature engineering (`COLUMNS_TO_DROP` in `src/ml/features/build_features.py`). No live weather API integration exists. |

### Source Verification (Weather Data)

- **Pricing structure**: Caters to both non-commercial and commercial users. Non-commercial use is free, with fair usage guidelines and attribution required. Commercial clients enjoy dedicated API servers, personalized support, and a commercial use licence.
- **Free tier**: 10,000 calls/day, 300,000 calls/month. Non-commercial use only. No commercial license.
- **API Standard plan**: 1M calls/month, includes commercial use license and dedicated API key
- **API Professional plan**: 5M calls/month, required for historical weather data, climate data, ensemble data, and satellite radiation APIs
- **API Enterprise plan**: >50M calls/month, dedicated endpoint, 99.9% uptime target
- **Weather models**: 30+ models from ECMWF, DWD, NOAA, Météo-France, JMA, KMA, KNMI, DMI, MeteoSwiss, UK Met Office
- **AROME model**: Available for France region (high-resolution regional model)
- **WMO codes**: Standard WMO Weather interpretation codes (WW) — codes 0-99 for weather conditions
- **Available variables**: temperature_2m, relative_humidity_2m, dew_point_2m, apparent_temperature, precipitation, rain, snowfall, weather_code, wind_speed_10m, wind_direction_10m, wind_gusts_10m, visibility, and 50+ more
- **License**: Server code is open-source under AGPLv3; weather data is CC BY 4.0
- **Historical data**: Requires Professional API plan or higher; not available on free tier
- **Timezone**: API responses use UTC. Conversion to local Paris time (CET/CEST) is required for consistency with counter timestamps.
- **Geographic scope**: Query by counter coordinates (latitude, longitude) using WGS84 format

---

## Source 6 — Administrative Boundaries

| Field | Value |
| --- | --- |
| **Name** | Limites administratives — Communes, arrondissements, quartiers |
| **Owner** | IGN / Insee / Mairie de Paris |
| **Provider** | [data.gouv.fr](https://www.data.gouv.fr) — `communes` or `arrondissements-de-paris` datasets |
| **Access mode** | File download (GeoJSON / Shapefile) via data.gouv.fr |
| **Format** | GeoJSON or Shapefile |
| **Update frequency** | Rare — only when administrative boundaries change (rare for Paris arrondissements). |
| **Classification** | **Optional** |
| **Usage** | Enables district-level aggregation and enrichment. Can be used to map counter locations to arrondissements, compute district-level traffic summaries, or filter counters by geographic region. |
| **Current state** | **Not implemented.** Counter arrondissement information is embedded in the raw CSV but not validated against an authoritative boundary source. |

### Known Limits (Administrative Boundaries)

| Limit | Detail |
| --- | --- |
| **Schema stability** | High — Paris arrondissement boundaries have been stable for decades. |
| **Volume** | Small — GeoJSON for Paris arrondissements is < 1 MB. |
| **Geocoding** | Requires point-in-polygon lookup to map counter coordinates to arrondissements. |

---

## Source 7 — Vélib' / GBFS (Experimental)

| Field | Value |
| --- | --- |
| **Name** | Vélib' Métropole — General Bikeshare Feed Specification |
| **Owner** | Vélib' Métropole / JCDecaux |
| **Provider** | [MobilityData GBFS](https://github.com/MobilityData/gbfs) (open standard) |
| **Access mode** | REST API (JSON) — public GBFS feed |
| **Format** | JSON (GBFS v3.0 — current version as of April 2024) |
| **Update frequency** | Near real-time (every 1–5 minutes) |
| **Classification** | **Experimental** |
| **Usage** | Vélib' bike availability data could provide complementary signals to counter data (e.g., demand patterns, station-level congestion). Currently outside MVP scope. |
| **Current state** | **Not implemented.** Listed as experimental. No integration exists. |

### Source Verification (Vélib' GBFS)

- **GBFS version**: v3.0 (MAJOR release, April 11, 2024) — current recommended version
- **Governance**: Open source project developed under consensus-based governance model by MobilityData (since 2019, previously NABSA)
- **Specification purpose**: Real-time, read-only data feed for shared mobility systems — NOT for historical data
- **Data types**: station_information.json, station_status.json, vehicle_status.json, vehicle_availability.json, system_information.json, vehicle_types.json, and 10+ more
- **Created**: 2014 by Mitch Vars with collaboration from Lyft (formerly Motivate International), public sector, and non-profit organizations
- **Endorsement**: North American Bikeshare Association (NABSA) endorsement and support was key to success starting 2015
- **Scope**: Vélib' covers only shared-bike stations, not private bicycle traffic. Counter data and Vélib' data measure different populations.
- **Schema stability**: GBFS is a standardized specification governed by MobilityData community; changes follow formal voting process (minimum 7 days discussion, 3+ votes including producer and consumer)
- **Rate limits**: Public GBFS feeds may have rate limits. Caching is recommended.
- **MVP fit**: Outside MVP scope. Requires separate validation to confirm predictive value.

---

## Summary Table

| # | Source | Classification | Access Mode | Update Frequency | In Pipeline? |
| --- | --- | --- | --- | --- | --- |
| 1 | Paris Bike-Counting Dataset | **Required** | File download (CSV) | Irregular | ✅ Raw input (static CSV) |
| 2 | Counter Reference | **Required** | File download (CSV/GeoJSON) | Rare | ❌ Embedded in raw CSV |
| 3 | Public Holidays (France) | **Optional** | API / File download (JSON/iCal) | Annual | ❌ Not implemented |
| 4 | School Holidays (Zone C) | **Optional** | File download (CSV/JSON) | Annual | ❌ Not implemented |
| 5 | Weather Data | **Optional** | REST API (JSON) | Hourly / Daily | ⚠️ Embedded, then dropped |
| 6 | Administrative Boundaries | **Optional** | File download (GeoJSON) | Rare | ❌ Not implemented |
| 7 | Vélib' / GBFS | **Experimental** | REST API (JSON) | Near real-time | ❌ Not implemented |

---

## Phase 10 Implementation Priorities

| Priority | Source | Action |
| --- | --- | --- |
| P0 | 1 — Bike-Counting Dataset | Automate ingestion from opendata.paris.fr. Replace static CSV with versioned snapshot. |
| P0 | 2 — Counter Reference | Extract counter metadata into a dedicated reference source. Validate against raw dataset. |
| P1 | 3 — Public Holidays | Integrate French holiday calendar. Encode as features. |
| P1 | 4 — School Holidays | Integrate Zone C school calendar. Encode as features. |
| P2 | 5 — Weather Data | Replace embedded weather with live API fetch (Open-Meteo recommended). |
| P2 | 6 — Administrative Boundaries | Add district enrichment via GeoJSON lookup. |
| — | 7 — Vélib' / GBFS | Defer to post-MVP. Investigate predictive value. |

---

## References

- **Official bike-counting source**: Mairie de Paris — [opendata.paris.fr](https://opendata.paris.fr)
- **Data science lab repository**: Streamlit exploration, model comparison, and feature influence analysis (internal)
- **Current MLOps repository**: Manifest-first pipeline, Airflow orchestration, existing runtime
- **Remaining work**: `docs/remaining-work/global-remaining-work.md`
- **Pipeline parameters**: `params.yaml` (ARIMA order, roll window, test ratio, scenarios)
- **Feature engineering**: `src/ml/features/build_features.py`, `src/ml/features/features_utils.py`
- **Ingestion**: `src/ml/ingest/import_raw_data.py`
