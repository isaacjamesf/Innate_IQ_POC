# Data Dictionary — InnateIQ Serving Layer

The serving layer is a multi-tenant **star schema**: three fact tables joined to
conformed dimensions on surrogate keys. Staging is normalised (3NF) to preserve source
fidelity and support auditing; the serving layer is denormalised for query performance.

The diagram is `figures/fig_star_schema.png` (Figure B.1 in the report).

## Conventions

| Convention | Meaning |
|---|---|
| **PK** | Primary key — a system-generated surrogate key, never a business key |
| **FK** | Foreign key — joins to the named dimension on its surrogate key |
| **SCD-2** | Slowly changing dimension, type 2 — history preserved via `valid_from` / `valid_to` / `is_current` |
| **Grain** | What one row represents, stated at the top of each table |
| Naming | `Fact*` for fact tables, `Dim*` for dimensions; snake_case fields |
| Multi-tenancy | Every fact row carries `tenant_id`; row-level security and database policies enforce isolation at every layer |

## Privacy

**No personal data enters the serving layer.** Actors are represented as roles, never
names. `DimUser.user_token` is a salted SHA-256 hash; names and emails are dropped at
ETL. The optional cross-tenant benchmark computes anonymised aggregates across
consenting tenants only, with minimum-group-size suppression so no tenant can be
inferred.

## Lineage

InnateIQ reads Meridian's Power App / Excel idea tracker and Jira. It **never writes
back** to those systems. The star schema is InnateIQ's own serving layer, not a
replacement system of record.

---

## Fact tables

### FactIdeaEvent

**Grain:** one row per gate transition, hold, resume, override or reject.

| Field | Type | Key | Description | Source / rule |
|---|---|---|---|---|
| `event_id` | bigint | PK | Surrogate key | Generated |
| `tenant_id` | int | FK → DimTenant | Owning tenant | Configuration |
| `idea_key` | int | FK → DimIdea | Idea (SCD-2 current key) | ETL |
| `date_key` | int | FK → DimDate | Event date | Timestamp |
| `stage_from_key` | int | FK → DimStage | Canonical gate / sub-state before the event | Mapped from legacy stages |
| `stage_to_key` | int | FK → DimStage | Canonical gate / sub-state after the event | Mapped from legacy stages |
| `actor_role_key` | int | FK → DimUser | Role performing the event — never a name | Pseudonymised |
| `event_type` | varchar | | `transition` / `hold` / `resume` / `override` / `reject` | Rule |
| `timestamp` | datetime (UTC) | | Event time | Source, or inferred and flagged |
| `dwell_days` | decimal | | Days since the previous event for the same idea | Derived |
| `inferred_flag` | bit | | 1 if the timestamp was inferred from Jira | Derived — 6.1% of rows |

### FactBenefit

**Grain:** one row per benefit measurement. Populated from the Benefits Profile at G2
and from closure at G4.

| Field | Type | Key | Description | Source / rule |
|---|---|---|---|---|
| `benefit_id` | bigint | PK | Surrogate key | Generated |
| `tenant_id` | int | FK → DimTenant | Owning tenant | Configuration |
| `idea_key` | int | FK → DimIdea | Idea the benefit belongs to | ETL |
| `benefit_type_key` | int | FK → DimBenefitType | `productivity` / `capacity` / `risk` / `revenue` | Benefits Profile |
| `measurement_date_key` | int | FK → DimDate | Date of the measurement | Profile or closure |
| `validated_by_role_key` | int | FK → DimUser | Role validating — never a name | Pseudonymised |
| `baseline_value` | decimal | | Baseline, in `unit` | Benefits Profile |
| `target_value` | decimal | | Target, in `unit` | Benefits Profile |
| `actual_value` | decimal | | Actual, in `unit` | Closure |
| `unit` | varchar | | `hours`, `€`, `%`, `count` | Benefits Profile |
| `evidence_flag` | bit | | Evidence attached | Closure |
| `validated_flag` | bit | | Controller validated | Closure |

### FactScore

**Grain:** one row per idea per nightly batch score. Components are euro-denominated
per the v2 formula in Section 12.5 of the report.

| Field | Type | Key | Description | Source / rule |
|---|---|---|---|---|
| `score_id` | bigint | PK | Surrogate key | Generated |
| `tenant_id` | int | FK → DimTenant | Owning tenant | Configuration |
| `idea_key` | int | FK → DimIdea | Idea being scored | ETL |
| `date_key` | int | FK → DimDate | Scoring date | Batch |
| `p_realization` | decimal(5,4) | | Model A output | Batch score |
| `expected_close_days` | int | | Model B point estimate | Batch score |
| `ci_low`, `ci_high` | int | | Model B 90% interval bounds | Batch score |
| `wsjf_score` | decimal | | Composite priority score | Formula 12.5 (v2) |
| `wsjf_bv` | decimal | | Business value component | Formula 12.5 (v2) |
| `wsjf_tc` | decimal | | Time criticality component | Formula 12.5 (v2) |
| `wsjf_rc` | decimal | | Risk / compliance component | Formula 12.5 (v2) |
| `wsjf_job_size` | decimal | | Job size denominator | Formula 12.5 (v2) |
| `model_version` | varchar | | Model card version used | Registry |
| `override_flag` | bit | | Human override applied | UI |
| `override_reason` | text | | Mandatory free-text reason for the override | UI |

---

## Dimensions

### DimIdea (SCD-2)

**Grain:** one row per idea per version; `is_current = 1` marks the active row.

| Field | Type | Key | Description |
|---|---|---|---|
| `idea_key` | int | PK | Surrogate key |
| `idea_id` | varchar | | Business key from the source tracker |
| `tenant_id` | int | | Owning tenant |
| `title`, `description` | varchar | | Idea text as entered |
| `category_key` | int | FK → DimCategory | Improvement lever |
| `team_key` | int | FK → DimTeam | Execution team |
| `hub_key` | int | FK → DimHub | Delivery hub |
| `savings_hours_est` | decimal | | Estimated annual savings, hours |
| `savings_log` | decimal | | `log1p(savings_hours_est)` — see derived fields |
| `has_evidence_flag` | bit | | Evidence attached at estimate |
| `jira_linked_flag` | bit | | Linked to a Jira execution item |
| `submitted_date_key` | int | FK → DimDate | Submission date |
| `valid_from`, `valid_to`, `is_current` | | | SCD-2 versioning |

### DimStage

| Field | Type | Key | Description |
|---|---|---|---|
| `stage_key` | int | PK | Surrogate key |
| `gate` | varchar | | Canonical gate, G1–G4 |
| `sub_state` | varchar | | Sub-state within the gate |
| `legacy_stage_map` | varchar | | Mapping from each tenant's legacy tracker stages |

### DimUser (pseudonymised)

| Field | Type | Key | Description |
|---|---|---|---|
| `user_key` | int | PK | Surrogate key |
| `user_token` | char(64) | | Salted SHA-256 hash — no name, email or HR data anywhere in this table |
| `role_level` | varchar | | Role, e.g. SPOC, GPO, Controller |
| `hub_key` | int | FK → DimHub | Hub of the role |

### DimTenant

| Field | Type | Key | Description |
|---|---|---|---|
| `tenant_key` | int | PK | Surrogate key |
| `tenant_name` | varchar | | Tenant display name |
| `hosting_region` | varchar | | Data-residency region |
| `configuration_version` | varchar | | Active tenant configuration |

### DimDate

| Field | Type | Key | Description |
|---|---|---|---|
| `date_key` | int | PK | Surrogate key (YYYYMMDD) |
| calendar attributes | | | Day, week, month, quarter, year |
| fiscal period | | | Tenant fiscal calendar |

### DimCategory

| Field | Type | Key | Description |
|---|---|---|---|
| `category_key` | int | PK | Surrogate key |
| lever | varchar | | `Automation` · `Simplification` · `Elimination` · `Standardisation` |

### DimTeam / DimHub

| Field | Type | Key | Description |
|---|---|---|---|
| `team_key` / `hub_key` | int | PK | Surrogate keys |
| `name` | varchar | | Team / hub name |
| `capacity_weight` | decimal | | Relative execution capacity |
| `hub_region` | varchar | | Hub region |

### DimBenefitType

| Field | Type | Key | Description |
|---|---|---|---|
| `benefit_type_key` | int | PK | Surrogate key |
| type | varchar | | `productivity` · `capacity` · `risk` · `revenue` |
| `default_unit` | varchar | | Default measurement unit for the type |

---

## Derived fields

- **`savings_log` = `log1p(savings_hours_est)`.** Used by the predictive layer because
  the raw distribution has skewness 17.23 and a median-to-mean gap of 2.9×, which makes
  the mean actively misleading.
- **`dwell_days`** is derived from the previous event for the same idea, not from
  submission.
- **`inferred_flag`** exists so no consumer mistakes an inferred timestamp for a
  recorded one. 6.1% of event rows are inferred from Jira because the tracker never
  recorded them.
