# Warehouse Layer Modeling Standards

The agent recognizes common data warehouse layers and applies consistent modeling defaults when creating semantic models. Use these standards to classify tables and scaffold schemas consistently.

## Layer Overview

| Layer | Purpose | Naming | Partitioning | Retention |
|-------|---------|--------|--------------|-----------|
| **ODS (Operational Data Store)** | Land raw/ingested events with minimal transformation for traceability. | `ods_{source}_{entity}` | Ingestion or event date/time columns; avoid re-partitioning historical loads. | Short-term raw history (30–90 days) that mirrors source change cadence. |
| **DIM (Conformed Dimension)** | Curated, de-duplicated entities shared across marts. | `dim_{subject}` | Slowly changing dimension effective dates (`effective_start_date`/`effective_end_date`) or `valid_from`/`valid_to`. | Long-lived (12+ months) with carefully versioned history. |
| **DWD (Detail Warehouse Data)** | Cleansed, modeled fact-level detail ready for joins. | `dwd_{subject}_{grain}` | Event/transaction date; optional ingestion date for late-arriving data. | Medium-term (6–24 months) aligned to reconciliation SLAs. |
| **DWS (Data Warehouse Summary)** | Aggregated rollups serving reuse across teams. | `dws_{subject}_{grain}` | Summary window anchors (e.g., week/month start) and refresh timestamps. | Medium/long-term depending on business need; frequently refreshed and backfilled. |
| **ADS (Application Data Service)** | Presentation layer tuned for specific products or dashboards. | `ads_{audience}_{purpose}` | Consumption-friendly partitions (e.g., report_date) with stable schemas. | Align to product/reporting horizons; prune deprecated slices aggressively. |

## Layer-Specific Modeling Guidance

### ODS
- **Purpose**: Preserve source fidelity while enabling downstream replay and reconciliation.
- **Naming**: `ods_{source}_{entity}` (e.g., `ods_crm_contacts`).
- **Partitioning**: Prefer `ingestion_date` or `event_date`; keep `ingestion_ts` as a high-precision load marker.
- **Retention**: Keep the minimum needed for replay (commonly 30–90 days) and mirror source deletions.
- **Sample Table Spec**:
  - Keys: `ingestion_ts` (primary time), `source_system`, `record_hash` for deduplication.
  - Columns: raw payload column (e.g., `payload` VARIANT/JSON), optional `record_type`, and source-specific fields.

### DIM
- **Purpose**: Provide conformed entity context (customers, products, locations) with stable keys.
- **Naming**: `dim_{subject}` (e.g., `dim_customer`).
- **Partitioning**: Use SCD2 validity columns (`effective_start_date`, `effective_end_date`, `is_current`) or `valid_from`/`valid_to` for change tracking.
- **Retention**: Keep full history unless privacy policies dictate trims.
- **Sample Table Spec**:
  - Identifiers: surrogate key (e.g., `customer_sk`), natural/business key (`customer_id`).
  - Time columns: `effective_start_date`, `effective_end_date`, `updated_at`.
  - Attributes: conformed attributes (names, categories, status flags), deduplicated and standardized.

### DWD
- **Purpose**: Cleaned, atomic facts at the base event/transaction grain.
- **Naming**: `dwd_{subject}_{grain}` (e.g., `dwd_order_line`).
- **Partitioning**: Primary event date (`event_date`/`transaction_date`); include ingestion partitions if late data is common.
- **Retention**: Retain detailed rows for compliance windows (often 12–24 months) before archiving to cold storage.
- **Sample Table Spec**:
  - Identifiers: `fact_id` primary key, foreign keys to dimensions (e.g., `customer_sk`, `product_sk`).
  - Measures: raw amounts/quantities, status codes, operational flags.
  - Time columns: `event_ts` (primary), `ingestion_ts` (tracking), optional `processing_date`.

### DWS
- **Purpose**: Reusable summaries/aggregations that power multiple metrics and data products.
- **Naming**: `dws_{subject}_{grain}` (e.g., `dws_order_daily`).
- **Partitioning**: Rollup window anchors (e.g., `summary_date`, `summary_week`), with refresh timestamps to support backfills.
- **Retention**: Keep as long as downstream use requires; rebuild regularly from DWD to ensure consistency.
- **Sample Table Spec**:
  - Grain: clearly defined (daily/weekly/monthly) with `summary_date` or equivalent.
  - Measures: pre-aggregated sums/counts/averages, plus windowed stats where helpful.
  - Dimensions: keys needed for common slicing (e.g., `customer_sk`, `region_id`, `channel`).

### ADS
- **Purpose**: Presentation-ready serving tables tuned for a specific product, report, or API.
- **Naming**: `ads_{audience}_{purpose}` (e.g., `ads_finance_cashflow_dashboard`).
- **Partitioning**: Consumption-friendly partitions such as `report_date` or `snapshot_date` aligned to publication cadence.
- **Retention**: Keep only what the consumer needs (e.g., rolling 12 months), pruning obsolete snapshots quickly.
- **Sample Table Spec**:
  - Keys: consumer-facing identifiers (e.g., `dashboard_id`, `customer_id`), often denormalized.
  - Measures/Derived fields: pre-calculated KPIs and formatted strings ready for display.
  - Refresh tracking: `snapshot_ts`, `source_cutoff_ts`, and `data_quality_status`.

## How the Agent Uses These Layers
- **Classification**: Tables are tagged against the presets to drive naming, partitioning, and retention defaults during semantic model generation.
- **Scaffolding**: When creating new models, the agent uses the sample specs as defaults for identifiers, time dimensions, and measure patterns appropriate to the layer.
- **Consistency**: Ensures downstream metrics and dashboards align to the same warehouse layering standards without manual restatement.
