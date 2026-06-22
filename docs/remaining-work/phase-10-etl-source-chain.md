# Phase 10 — ETL Source Chain

> **Status**: Future-state design target. No source connector or dataset
> reconstruction workflow is implemented yet.
>
> **Epic**: Source-driven ETL and reproducible ML-ready dataset generation.
>
> **Story**: [Issue #105](https://github.com/zheddhe/avr25-mle-trafic-cycliste/issues/105)
>
> **Design reference**:
> [zheddhe/mai25-bds-trafic-cycliste](https://github.com/zheddhe/mai25-bds-trafic-cycliste)
> notebooks and `smartcheck/` source code.
>
> **Source data contracts**:
> [Phase 10.2 contract directory](phase-10-etl-contracts/README.md). These
> versioned documentation contracts define future snapshot validation,
> canonicalization, and ML-ready dataset reconstruction inputs; they do not
> implement acquisition, validation, or dataset building.

---

## Purpose and dataset layers

Phase 10 replaces the current static ML-ready CSV starting point with a
reproducible source-driven ETL chain.

The target separates three dataset layers:

| Layer | Responsibility |
| --- | --- |
| **Raw snapshot** | Immutable evidence of one external-source acquisition. |
| **Canonical history** | Validated, reconciled, correction-aware internal history. |
| **ML-ready current window** | Versioned dataset built offline from canonical history and approved enrichments. |

External APIs are an acquisition boundary only. A raw snapshot is not the
canonical history, and it is not automatically an ML-ready dataset. Downstream
feature generation and model execution must use validated internal datasets.

---

## Classification legend

| Classification | Meaning |
| --- | --- |
| **Required** | Needed for the MVP to produce valid predictions. |
| **Optional** | Improves prediction quality; the pipeline can run without it. |
| **Experimental** | Investigational and outside the MVP until predictive value is validated. |

---

## Source catalog contract

Every Phase 10 source implementation must document and persist the following
contract fields in its source catalog and acquisition metadata:

| Contract field | Required decision |
| --- | --- |
| Authority | Which provider is the source of truth. |
| Classification | Required, optional, or experimental. |
| Purpose and keys | Why the source exists and how it joins canonical data. |
| Cadence and coverage | Expected update timing, retention, and historical range. |
| Schema and timezone | Contracted fields, schema-change policy, and temporal semantics. |
| Snapshot strategy | Raw payload, request context, checksum, acquisition time, and source fingerprint. |
| Availability at prediction time | Whether a value is known, forecast, delayed, or unavailable when serving a prediction. |
| Failure policy | Hard failure, quality warning, fallback, or explicit source exclusion. |

A snapshot manifest should include the endpoint or file identity, query or
parameters, headers or validators when available, acquisition timestamp, source
coverage, row count, source-schema fingerprint, and payload checksum.

---

## Source catalog

| # | Source | Class | Purpose | Snapshot strategy | Availability at prediction time |
| --- | --- | --- | --- | --- | --- |
| 1 | Paris bike-counting dataset | Required | Primary observed traffic history. | Immutable raw CSV or API payload per acquisition. | Observations are delayed; only data available before the prediction cut-off may be used. |
| 2 | Counter reference / counting sites | Required | Stable identity, location, and operational metadata. | Versioned reference snapshot when the source changes. | Stable metadata is available after validation and promotion. |
| 3 | French public holidays | Optional | Calendar feature enrichment. | Versioned yearly API response or curated calendar snapshot. | Known in advance once published. |
| 4 | School holidays, Zone C | Optional | Paris-specific calendar enrichment. | Versioned school-calendar export. | Known in advance once published; use Zone C only. |
| 5 | Weather | Optional | Historical and operational weather features. | Historical and forecast snapshots must remain distinct. | Historical observations are training-only; serving may use only a forecast available at the prediction cut-off. |
| 6 | Administrative boundaries | Optional | Geographic validation and arrondissement enrichment. | Versioned GeoJSON or Shapefile snapshot. | Stable after validation; no live lookup is required. |
| 7 | Vélib' / GBFS | Experimental | Potential complementary mobility signal. | Cached feed snapshots only during an experimental study. | Near real-time but not part of the MVP feature contract. |

---

## Historical bootstrap and canonicalization

Sources 1 and 2 require a one-off bootstrap before daily reconstruction can
operate.

The bootstrap must:

1. acquire the full available rolling history from the official Paris source;
2. store the immutable raw snapshot and its acquisition metadata;
3. normalize timezone semantics to an explicit Europe/Paris policy, including
   daylight-saving transitions;
4. apply approved counter-name corrections, outlier rules, and metadata
   reconciliation deterministically;
5. join the authoritative counter reference and optional validated geographic
   enrichment;
6. produce a validated canonical history and a versioned build manifest;
7. materialize an ML-ready current rolling window only after validation passes.

The bootstrap is not a destructive purge. It is a reproducible historical
backfill and canonicalization operation. A failed bootstrap must not replace a
previously promoted valid dataset.

The expected raw source size is currently about 149 MB for a 13-month rolling
window. The actual source window and schema must be captured in each snapshot
manifest instead of inferred from a local filename.

---

## Daily delta and current-window reconstruction

After bootstrap, daily operation must not append blindly to a static baseline.

The target daily workflow is:

1. acquire an immutable J-1 source snapshot, including a configured correction
   lookback when upstream corrections are possible;
2. validate the source contract and apply deterministic quality corrections;
3. merge data into canonical history using the stable logical key
   `(counter_id, timestamp)`;
4. resolve late corrections according to the documented precedence policy;
5. rebuild the current rolling ML-ready window from canonical history;
6. validate coverage, schema, time boundaries, and output checksum;
7. publish the new dataset version and update its `current` pointer atomically;
8. allow feature generation and model execution to consume only that validated
   promoted dataset version.

A failed acquisition, validation, canonical merge, or current-window build must
leave the previously promoted dataset version unchanged.

The existing `bike_traffic_daily` DAG currently operates from the static CSV
baseline. Its future source-driven behaviour belongs to later Phase 10 stories
and must not be described as implemented by this document.

---

## Source 1 — Paris bike-counting dataset

| Field | Value |
| --- | --- |
| **Name** | Comptage vélo — Données compteurs |
| **Authority** | Mairie de Paris — Direction de la Voirie et de la Circulation |
| **Provider** | [opendata.paris.fr](https://opendata.paris.fr) on Opendatasoft |
| **Access mode** | CSV download or Opendatasoft API |
| **Cadence and coverage** | Daily J-1 publication over a rolling 13-month history |
| **Classification** | **Required** |
| **Current state** | Static enriched ML-ready CSV tracked with DVC; no automated acquisition |

Key fields include `nom_du_site_de_comptage`, `orientation_compteur`,
`date_et_heure_de_comptage`, and `comptage_horaire`. The future canonical key
must use a stable counter identity rather than mutable free-text naming whenever
the counter reference permits it.

Known quality work inherited from the mai25 design includes counter-name
reconciliation, missing-metadata propagation, and an approved cap for
`comptage_horaire` values above 3000. Those transformations must become explicit,
versioned ETL rules rather than notebook-only corrections.

The source timestamps are local Paris time without explicit timezone annotation
in the current CSV. Phase 10 must define DST handling and fail or quarantine
ambiguous records according to the data contract.

---

## Source 2 — Counter reference / counting sites

| Field | Value |
| --- | --- |
| **Name** | Comptage vélo — Compteurs |
| **Source 2 — Counter reference / counting sites

| Field | Value |
| --- | --- |
| **Name** | Comptage vélo — Compteurs |
| **Authority** | Mairie de Paris — Direction de la Voirie et la Circulation |
| **Provider** | [opendata.paris.fr](https://opendata.paris.fr) |
| **Access mode** | CSV, GeoJSON, or Opendatasoft API |
| **Cadence** | Rare changes when counters are added, removed, or relocated |
| **Classification** | **Required** |
| **Current state** | Metadata is embedded in the current CSV instead of managed as a separate source |

This source provides stable site metadata, including counter or site identifiers,
coordinates, installation date, and operational state. It is the authority used
to validate raw-count identities and reconcile renamed or malformed records.

---

## Source 3 — French public holidays

| Field | Value |
| --- | --- |
| **Provider** | [calendrier.api.gouv.fr](https://calendrier.api.gouv.fr) |
| **Access mode** | REST JSON by year and country code |
| **Classification** | **Optional** |
| **Current state** | Not implemented |

Use the `metropole` calendar for Paris. Calendar snapshots are deterministic once
published and should be versioned by year. The expected feature is a binary
`jour_ferie` flag or an equivalent documented representation.

---

## Source 4 — School holidays, Zone C

| Field | Value |
| --- | --- |
| **Provider** | [data.education.gouv.fr](https://data.education.gouv.fr) |
| **Access mode** | REST JSON export |
| **Classification** | **Optional** |
| **Current state** | Not implemented |

Use Paris / Zone C records only. The source is deterministic after publication,
but the output must retain the source version and the Zone C filter used. The
expected feature may be a categorical `vacances_scolaires` value or an equivalent
contracted representation.

---

## Source 5 — Weather data

| Field | Value |
| --- | --- |
| **Provider** | [Open-Meteo](https://open-meteo.com), subject to provider and plan constraints |
| **Access mode** | REST CSV or JSON |
| **Classification** | **Optional** |
| **Current state** | Weather columns are embedded in the raw CSV, then dropped during feature generation |

Weather requires an explicit anti-leakage contract:

- historical observations may be used for offline canonicalization and training;
- operational prediction features may use only a forecast or observation that was
  available at the prediction cut-off;
- historical and forecast payloads must be stored as distinct snapshots;
- API UTC timestamps must be converted under the common Europe/Paris policy.

The mai25 design used `temperature_2m`, `weather_code`, `rain`, and `snowfall`,
with a WMO-code categorization step. Feature selection remains a later validated
modelling decision.

---

## Source 6 — Administrative boundaries

| Field | Value |
| --- | --- |
| **Provider** | [data.gouv.fr](https://www.data.gouv.fr), IGN, Insee, or Mairie de Paris |
| **Access mode** | GeoJSON or Shapefile download |
| **Classification** | **Optional** |
| **Current state** | Arrondissement is embedded in the current CSV but not validated against a reference boundary source |

A versioned geographic snapshot may be used for point-in-polygon arrondissement
enrichment. The relevant CRS and geometric predicate must be part of the ETL
contract.

---

## Source 7 — Vélib' / GBFS

| Field | Value |
| --- | --- |
| **Provider** | Vélib' Métropole public GBFS feed |
| **Access mode** | Near-real-time REST JSON |
| **Classification** | **Experimental** |
| **Current state** | Not implemented and outside MVP scope |

GBFS measures shared-bike availability rather than the same population as Paris
counter traffic. It requires a separate experiment, retention policy, and
predictive-value assessment before it can enter the source contract.

---

## Preprocessing and quality rules inherited from mai25

The mai25 design is a reference, not an implementation contract. Phase 10 must
turn approved transformations into versioned, testable code and data contracts:

1. column filtering and name normalization;
2. explicit UTC and Europe/Paris datetime handling;
3. public-holiday and Zone C school-holiday enrichment;
4. weather enrichment with an availability-at-prediction-time contract;
5. WMO-code categorization when the feature is selected;
6. counter-name reconciliation and orientation extraction;
7. outlier correction for documented sensor anomalies;
8. reference-counter and arrondissement enrichment.

---

## Phase 10 implementation priorities

| Priority | Source or concern | Action |
| --- | --- | --- |
| P0 | Bike-counting dataset | Acquire versioned raw snapshots and replace static-file dependence. |
| P0 | Counter reference | Manage it as a separate authority and validate raw-count identities. |
| P0 | Dataset lifecycle | Build canonical history, reconstruct the rolling window, and publish manifests atomically. |
| P1 | Public holidays | Integrate and version the French mainland calendar. |
| P1 | School holidays | Integrate and version the Paris Zone C calendar. |
| P1 | Data contracts | Define schema, timezone, quality, and source-change policies. |
| P2 | Weather | Add historical and operational weather paths without leakage. |
| P2 | Administrative boundaries | Add validated arrondissement enrichment. |
| — | Vélib' / GBFS | Defer until a dedicated experiment validates predictive value. |

---

## References

- **Official bike-counting and counter sources**:
  [opendata.paris.fr](https://opendata.paris.fr)
- **Public holidays**:
  [calendrier.api.gouv.fr](https://calendrier.api.gouv.fr)
- **School holidays**:
  [data.education.gouv.fr](https://data.education.gouv.fr)
- **Weather**: [Open-Meteo](https://open-meteo.com)
- **mai25 design reference**:
  [zheddhe.mai25-bds-trafic-cycliste](https://github.com/zheddhe/mai25-bds-trafic-cycliste)
  - `notebooks/project/01_etapes1a2_trafic_cycliste_comptage_velo.ipynb`
  - `notebooks/project/03_etapes1a2_trafic_cycliste_enrich_with_meteo_holidays.ipynb`
  - `smartcheck/preprocessing_project_specific.py`
  - `smartcheck/dataframe_project_specific.py`
- **Current MLOps pipeline**:
  `src/ml/ingest/import_raw_data.py`, `src/ml/features/build_features.py`, and
  `src/ml/features/features_utils.py`
- **Global roadmap**: `docs/remaining-work/global-remaining-work.md
