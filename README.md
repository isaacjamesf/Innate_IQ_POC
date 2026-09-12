# InnateIQ — Proof of Concept

Reproduction code and synthetic dataset for the MSBA capstone **"InnateIQ: The
Innovation Intelligence Platform for Global Business Services"** (Quantic School of
Business and Technology, Class of October 2026).

Every statistic, table and figure in the report is produced by the three scripts in
this repository. No result in the report is hand-entered.

## Contents

- [Data provenance](#data-provenance)
- [Getting started](#getting-started)
- [Repository structure](#repository-structure)
- [Results](#results)
- [Model governance](#model-governance)
- [Documentation](#documentation)
- [Limitations](#limitations)
- [Citation](#citation)
- [Licence](#licence)

## Data provenance

**All data in this repository is synthetic.** It contains no employer data, no client
data and no personal data of any kind. The generator is seeded and deterministic.

The synthetic dataset represents **Meridian GBS**, a fictionalised composite based on a
real Global Business Services organisation. It stands in for Meridian's existing
Power App / Excel idea tracker and its Jira execution records — the systems of record
that InnateIQ reads from and adds prioritisation, forecasting and benefits logic on top
of. InnateIQ does not replace those systems. Section 10.1 of the report states the
generation method, the basis for every generator parameter, and the limitations in
full.

## Getting started

Requires Python 3.11 or later. Runtime is under two minutes.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python etl/generate_data.py    # seeded generator          -> data/*.csv
python run_poc.py              # ETL, EDA, diagnostics,
                               # Models A/B, WSJF-lite,
                               # finance, venture model    -> results.json
python make_figures.py         # 12 rendered figures       -> figures/*.png
```

`results.json` is the single source of truth for every number quoted in the report.
If the report disagrees with `results.json`, `results.json` is correct.

### Verifying reproduction

Re-run from a clean checkout on 29 August 2026, on the pinned versions in
`requirements.txt`:

| Artefact | Result |
|---|---|
| `data/ideas.csv`, `events.csv`, `benefits.csv` | Byte-identical (MD5) |
| `results.json` — all 668 computed values | Identical |
| All 12 figures | Regenerated from `results.json` and `data/` |

To check the generator yourself:

```bash
md5sum data/*.csv > before.txt
rm data/*.csv && python etl/generate_data.py && md5sum -c before.txt
```

## Repository structure

```
innateiq-poc/
├── etl/
│   └── generate_data.py     Seeded synthetic dataset generator (Meridian GBS)
├── data/
│   ├── ideas.csv            12,400 idea records (Meridian's tracker, as read by InnateIQ)
│   ├── events.csv           52,895 event-log rows
│   └── benefits.csv         Benefit records (sparse, as recorded historically)
├── docs/
│   ├── data_dictionary.md   Serving-layer star schema, table by table
│   └── model_cards.md       Model cards for Models A and B
├── figures/
│   ├── *.png                12 rendered figures
│   └── FIGURE_MAP.md        Filename → report figure number
├── run_poc.py               ETL, EDA, diagnostics, Models A/B, WSJF-lite back-test,
│                            pilot finance, venture model
├── make_figures.py          All 12 rendered figures
├── results.json             Every computed statistic, keyed
└── requirements.txt         Pinned, verified dependency versions
```

## Results

Acceptance thresholds were fixed **before** any model was fitted. Three gates failed
and are reported as findings rather than removed.

| Layer | Model | Result | Gate | Outcome |
|---|---|---|---|---|
| Predictive | Realisation (logistic) | AUC 0.769 | ≥ 0.72 | **Pass** |
| Predictive | Challenger (XGBoost) | ΔAUC −0.007 | ≥ +0.03 | **Fail — rejected** |
| Predictive | Cycle time (Weibull AFT) | MAE 30.5 d | ≤ 30 | **Fail (narrow)** |
| Predictive | Throughput (ARIMA) | MAPE 5.8% | ≤ 25% | **Pass** |
| Prescriptive | WSJF-lite back-test v1 | r = −0.149 | ≥ 0.40 | **Fail as specified** |
| Prescriptive | WSJF-lite back-test v2 | r = 0.514 on value per execution day | respecified | **Pass** |
| Prescriptive | Capacity counterfactual | +77.5% vs FIFO | — | **Pass** |

The v1 failure is the substantive finding of Section 12.5: the gate asked a
value-density score to predict absolute value, so no version of the formula could have
passed. The diagnosis also exposed a real defect — the v1 ordering would have delivered
**17.3% less** value than the first-in-first-served baseline it was meant to replace.

The cycle-time model misses its gate by 0.5 days. It still improves on a naive median
predictor by 5.1% and its 90% interval covers 87.4% of outcomes, so it ships as an
interval, not a point estimate, and its enrichment is a Horizon 2 action.

### Traceability

`figures/FIGURE_MAP.md` maps each figure file to its report figure number. Every
statistic quoted in the report traces to a key in `results.json`:

| Report | Claim | `results.json` key |
|---|---|---|
| §1, §11.1 | 12,400 records; 4,122 open; 1,536,216 open hours | `profile` |
| §1, §11.2 | 884-day residence time (WIP 4,122 ÷ 1,701/yr) | `little` |
| §11.1 | Skewness, kurtosis, Gini, concentration | `profile`, `stage_open_hours` |
| §11.2 | 83.3% aged > 180 d; 70.3% inactive; stage shares | `profile`, `flow` |
| §11.3 | Evidence r = 0.273; size ρ = 0.683; H1 / H2 tests | `corr`, `h1`, `h2` |
| §11.4 | Kaplan–Meier by lever; automation 3.6× | `km` |
| §11.5 | Cohorts; control chart; top bottleneck | `cohort`, `control_chart` |
| §10.4 | Null rate 1.4% raw → 0.0% clean; 6.1% inferred timestamps | `dq` |
| §12.4, App C.1 | AUC 0.769, calibration 1.014, CV 0.759 ± 0.011 | `model_a_logit` |
| §12.4 | Challenger ΔAUC −0.007 | `model_a_xgb` |
| §12.6, App C.1 | Bias audit: seniority 0.047 → 0.004; hub 0.084 → 0.026 | `fairness` |
| §12.4, App C.2 | MAE 30.5 d, RMSE 52.7, coverage 87.4% | `model_b_aft` |
| §12.4, App C.2 | ARIMA(2, 1, 2), MAPE 5.8%, Ljung–Box p = 0.143 | `model_b_arima` |
| §12.4 | Portfolio completion range | `monte_carlo` |
| §12.5 | v1 r = −0.149; v2 r = 0.514; rank stability ρ = 0.971 | `wsjf` |
| §12.5 | €5,664,097 vs €3,190,654 on 64,109 days = +77.5% | `counterfactual` |
| App D | Worked scoring examples, v1 and v2 | `wsjf_examples` |
| §4.1 | Bottom-up market sizing and sensitivity | `market` |
| §14 | €823,181 benefit, €92,500 cost, 790% ROI, NPV €1,017,063 | `finance` |
| §14.5, App H | Three-year SaaS model, headcount plan, LTV:CAC, venture NPV | `venture` |

## Model governance

Full model cards — intended use, training data, features, evaluation, fairness audit,
limitations, retraining cadence, version — are in
[`docs/model_cards.md`](docs/model_cards.md).

The production model is **logistic regression**. The XGBoost challenger was fitted and
**rejected**: ΔAUC −0.007 against a required +0.03. It would have been rejected on
explainability grounds even at a marginal gain. The rejection is reproduced by
`run_poc.py`.

Hub and requester-seniority features are engineered so the bias audit can quantify
disparity, then **excluded from the production model**. The audit run is retained in
`run_poc.py` so the exclusion is auditable rather than merely asserted.

> The seniority disparity in the training data was *injected by the generator* so the
> audit had something to detect. This shows the audit method works. It is **not**
> evidence that such a disparity exists in real pipelines.

## Documentation

| Document | Contents |
|---|---|
| [`docs/model_cards.md`](docs/model_cards.md) | Model cards for the Realisation Classifier (A) and Cycle-Time Forecaster (B) |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Serving-layer star schema: conventions, grain, every field of every table |
| [`figures/FIGURE_MAP.md`](figures/FIGURE_MAP.md) | Figure filename → report figure number |

## Limitations

- Synthetic training data. Coefficients are not claims about any real organisation.
- Labels carry selection bias; temporal confounds are not separable.
- Model B misses its gate by 0.5 days and is the weakest model in the stack.
- Generalisation across organisations is untested and deferred to Horizon 2.

## Citation

> Fillingham, Isaac. 2026. *InnateIQ: The Innovation Intelligence Platform for Global
> Business Services.* MSBA Capstone, Quantic School of Business and Technology.

## Licence

MIT — see [LICENSE](LICENSE). Applies to the code and to the synthetic dataset.
