# Phase 10 source data contracts

> **Status**: Documentation-level future-state contracts. They do not implement
> source acquisition, snapshot writing, schema fingerprinting, record validation,
> canonicalization, repository persistence, enrichment, or dataset reconstruction.

This directory contains versioned, implementation-neutral source and persistence
contracts for the future Phase 10 ETL source chain. A later implementation may
consume these documents as design input, but this story adds no YAML parser or
runtime coupling.

## Contract vocabulary

Every source contract uses the following top-level fields:

| Field | Meaning |
| --- | --- |
| `contract_version` | Version of the source contract, independent of a source payload version. |
| `source_id` | Stable English identifier used by future manifests and lineage. |
| `source_name` | Human-readable source name. |
| `classification` | `required`, `optional`, or `experimental`, according to the Phase 10 catalog. |
| `implementation_priority` | Planned delivery priority: `P0`, `P1`, or `P2`. |
| `authority` | Organization and source-of-truth statement. |
| `provider` | Distribution service or publisher. |
| `access_mode` | Supported external representation only; not a connector implementation. |
| `purpose` | Intended ETL and ML use, without selecting model features. |
| `current_state` | Explicitly distinguishes future design from existing static-pipeline behaviour. |
| `cadence` and `coverage` | Expected publication rhythm, geography, and historical scope. |
| `timezone_policy` | Source-time semantics and the canonical timezone policy. |
| `availability_at_prediction_time` | Whether a value is observed, known in advance, forecast, delayed, or unavailable at serving time. |
| `snapshot_requirements` | Evidence a future immutable raw snapshot manifest must retain. |
| `logical_keys` | Stable identity used by future canonicalization, not necessarily an upstream raw key. |
| `schema` | Required and optional contract fields plus precise field definitions. |
| `business_rules` | Semantic invariants required for trustworthy use. |
| `quality_rules` | Non-schema anomalies, warnings, and future handling boundaries. |
| `schema_change_policy` | Required response to upstream structural or semantic change. |
| `failure_policy` | Blocking, record-level, warning, fallback, and exclusion outcomes. |
| `lineage_requirements` | Traceability requirements across source, snapshot, and future canonical records. |
| `out_of_scope` | Deliberate exclusions from this contract and story. |

The supporting repository contract uses `contract_id` and `contract_type` rather
than `source_id`. It defines the persistence and extraction boundary shared by
all included source contracts; it is not an additional external source.

### Field-definition vocabulary

`contract_field_name` is a stable English documentation identifier.
`source_field_name` is an exact upstream name only when Phase 10 evidence
confirms it. A `null` value means that an upstream label has **not** been
confirmed: it is not an approved mapping and must not be guessed by a future
implementation. `canonical_target_name` names the future internal target when
known. A field may originate from a source snapshot or from the authoritative
counter-reference contract during future canonicalization.

Each field definition records its expected type, requiredness, nullability,
accepted format, semantic meaning, validation severity, and known limitations.
These descriptions are implementation-neutral: they are not Python, Pydantic,
Pandera, API, or dataframe declarations.

## Contract-wide policy decisions

### Validation levels

| Level | Meaning | Future outcome |
| --- | --- | --- |
| `schema_level_failure` | A required field is missing, has an incompatible type, or has a changed reviewed meaning. | Block the source snapshot from promotion. |
| `record_level_rejection_or_quarantine` | A record violates a deterministic, record-scoped safety rule while the source structure remains valid. | Reject or quarantine that record with evidence; later coverage policy determines whether the build can continue. |
| `quality_warning` | A non-blocking anomaly or non-critical structural addition is observed. | Preserve evidence for review; do not silently rewrite source evidence. |
| `optional_source_fallback` | An optional source is unavailable or incomplete. | Continue only without that enrichment and record the fallback in later lineage. |
| `explicit_source_exclusion` | A source is intentionally outside the MVP or an approved dataset build. | Do not acquire, join, or use it as an implicit substitute. |

Schema validity, record-level quality anomalies, deterministic correction rules,
and quarantine behaviour remain distinct. This story only documents the first
two categories and the future boundaries for the latter two.

### Schema change policy

- A missing required field is a blocking schema-level failure.
- An incompatible type for a required field is a blocking schema-level failure.
- An unexpected optional or non-critical field is a quality warning and must be
  preserved as future schema-fingerprint evidence.
- A changed field meaning is blocking until reviewed and the source contract is
  versioned accordingly.
- A future implementation must never infer an unconfirmed raw field mapping from
  a similar label or a notebook convention.

### Timezone policy

Every temporal contract must identify both source-time semantics and canonical
time semantics. `Europe/Paris` is the canonical local timezone for Paris
calendar and bike-count semantics. When an API emits UTC, the future pipeline
must retain the original UTC instant and derive the Europe/Paris representation
deterministically.

A local timestamp without a timezone annotation must not be silently assumed to
be unambiguous. For daylight-saving transitions:

- ambiguous local times require an explicit disambiguation value from the
  provider or record-level rejection or quarantine;
- nonexistent local times require record-level rejection or quarantine;
- any alternative policy is blocking until documented, reviewed, and versioned.

### Snapshot lineage

A future immutable raw snapshot manifest must capture at least:

- `source_id` and `contract_version`;
- provider dataset, endpoint, or file identity;
- acquisition timestamp;
- requested coverage and observed coverage;
- request context and parameters when applicable;
- response validators or headers when available;
- row count or equivalent record count;
- schema fingerprint;
- payload checksum;
- raw payload identity and storage reference.

Later canonical records and dataset manifests must reference the applicable
source snapshot identity and contract version. This story does not implement
manifest writing, checksums, or schema fingerprints.

### Canonical repository and date-range extraction

[`canonical_history_repository_contract.yaml`](canonical_history_repository_contract.yaml)
defines the future persistence boundary between validated source snapshots and
ML-ready dataset reconstruction. Its intended implementation target is a
PostgreSQL-compatible relational repository with logical isolation from
orchestration and experiment-tracking metadata stores. It does not introduce a
PostgreSQL container, schema, migration, DAG task, or runtime configuration.

The repository preserves immutable raw-payload references and append-only source
and canonical revisions. It supports idempotent nominal daily synchronization,
explicit historical backfills, targeted gap repair, and correction replay. Its
only valid downstream selector is an explicit, timezone-safe half-open date range
`[window_start_inclusive, window_end_exclusive)` plus a repository revision or
snapshot cut-off. Percentage or row-position slicing of a mutable raw file is
not a valid source-repository selector.

### No direct runtime coupling

These contracts stay independent of FastAPI, Airflow, Docker, CLI internals,
local paths, Python class names, and validation-library APIs. Runtime
acquisition, validation, repository implementation, dataset building, and
serving integration belong to later Phase 10 stories.

## Contract inventory

| Contract | Classification | Priority | Purpose |
| --- | --- | --- | --- |
| [`canonical_history_repository_contract.yaml`](canonical_history_repository_contract.yaml) | Required persistence boundary | P0 | Canonical history, revision lineage, gap assessment, and deterministic date-range extraction. |
| [`bike_counts_source.yaml`](bike_counts_source.yaml) | Required | P0 | Primary observed bike-traffic history. |
| [`counter_reference_source.yaml`](counter_reference_source.yaml) | Required | P0 | Stable counter identity and reference metadata. |
| [`public_holidays_source.yaml`](public_holidays_source.yaml) | Optional | P1 | French mainland calendar enrichment. |
| [`school_holidays_source.yaml`](school_holidays_source.yaml) | Optional | P1 | Paris / Zone C calendar enrichment. |
| [`weather_source.yaml`](weather_source.yaml) | Optional | P2 | Historical and prediction-time-safe weather inputs. |
| [`administrative_boundaries_source.yaml`](administrative_boundaries_source.yaml) | Optional | P2 | Offline geographic validation and arrondissement enrichment. |

Vélib' / GBFS remains experimental and is explicitly excluded from this MVP
source-contract scope.
