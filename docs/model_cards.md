# Model Cards

Model documentation for the two predictive models in the InnateIQ proof of concept,
following the standard model card structure (model details, intended use, training and
evaluation data, metrics, ethical considerations, caveats and recommendations).

**Governance note.** Both models are advisory to a human decision. No score automates an
outcome, so no appeal against a score is an appeal against an automated decision under
GDPR Article 22. Every human override requires a logged, written reason.

**Data note.** All training data is synthetic and represents Meridian GBS, a
fictionalised composite organisation. No real employer, client or personal data was used.

---

## Model A — Realisation Classifier

### Model details

| | |
|---|---|
| **Model name** | Realisation Classifier |
| **Version** | `A-1.0-POC` |
| **Model type** | Binary classifier — logistic regression |
| **Developed by** | InnateIQ proof of concept (MSBA capstone) |
| **Output** | `p_realization` — probability that an open idea completes with validated benefits within 365 days of submission |
| **Licence** | MIT (with the rest of this repository) |
| **Reproducible from** | `run_poc.py`; all metrics in `results.json` under `model_a_logit` |

### Intended use

- **Primary use.** Advisory input to the WSJF-lite priority score and to governance
  review. The probability feeds the prioritisation formula directly.
- **Primary users.** CI Leads and execution teams reading the ranked backlog; the
  governance board reviewing scores and overrides.
- **Out of scope.** Automated approval or rejection of ideas; performance evaluation of
  individuals or teams; any use where the score is the sole basis for a decision.

### Training data

6,622 closed items with a known outcome (80% of the 8,278 closed records in the
synthetic Meridian GBS dataset). Open items are excluded from training as censored, but
are scored in production.

### Features

13 columns after encoding: `savings_log`, stage dwell at G1 and G2,
`has_evidence_flag`, `jira_linked_flag`, category, and execution team.
Hub and `requester_role_level` are engineered **for the fairness audit run only** and
are excluded from the production model (see the fairness section below).

### Model selection

Logistic regression is the production model. An XGBoost challenger was fitted and
**rejected**: ΔAUC −0.007 against a required gain of +0.03. It would also have been
rejected on explainability grounds even at a marginal gain. The rejection is reproduced
by `run_poc.py` rather than discarded.

### Evaluation

Holdout set, n = 1,656 (20% of closed items).

| Metric | Value | Gate |
|---|---|---|
| AUC | 0.769 | ≥ 0.72 — **pass** |
| Cross-validated AUC | 0.759 ± 0.011 | — |
| Precision | 0.748 | — |
| Recall | 0.831 | — |
| F1 | 0.787 | — |
| Brier score | 0.185 | — |
| Calibration slope | 1.014 | — |

Calibration matters more than discrimination here, because the predicted probability
enters the priority score directly rather than being thresholded.

**What drives the score** (odds ratios on standardised features):

- Strongest positive: evidence attached 1.90 · Simplification 1.66 ·
  Standardisation 1.62 · Elimination 1.41
- Strongest negative: G1 dwell 0.84 · Unassigned team 0.76 · savings (log) 0.61

### Fairness and ethical considerations

Hub and requester-seniority proxies are included in a separate audit run so disparity
can be *quantified*, then **excluded from production**. Including them changes AUC by
only +0.001 but widens the predicted-probability spread across seniority from 0.004 to
0.047 and across hubs from 0.026 to 0.084. The audit run is retained in `run_poc.py` so
the exclusion is auditable rather than merely asserted.

> The seniority disparity in the training data was **injected by the generator** so the
> audit had something to detect. It demonstrates that the audit method works; it is
> **not** evidence that such a disparity exists in real pipelines.

### Caveats and recommendations

- Training data is synthetic; coefficients are not claims about any real organisation.
- Labels carry selection bias, and temporal confounds are not separable.
- Use the score alongside its decomposition, never as an unexplained number.

### Maintenance

Retrained annually, or earlier on a drift alert.

---

## Model B — Cycle-Time Forecaster

### Model details

| | |
|---|---|
| **Model name** | Cycle-Time Forecaster |
| **Version** | `B-1.0-POC` |
| **Model type** | Two-part: Weibull accelerated failure time (AFT) for item-level duration; ARIMA(2, 1, 2) for monthly throughput |
| **Developed by** | InnateIQ proof of concept (MSBA capstone) |
| **Output** | Expected days from G2 to G4 with a 90% interval, conditional on the item closing; portfolio completion range via Monte Carlo |
| **Licence** | MIT (with the rest of this repository) |
| **Reproducible from** | `run_poc.py`; all metrics in `results.json` under `model_b_aft` and `model_b_arima` |

### Intended use

- **Primary use.** Cycle-time expectations for the ranked backlog and portfolio
  completion forecasting. The product reports the **interval**, not the point estimate
  (see caveats).
- **Primary users.** CI Leads and the governance board.
- **Out of scope.** Deadline commitments to stakeholders; item-level guarantees;
  modelling of capacity shocks.

### Model specification

Two-part by design: Model A estimates *whether* an item closes; Model B estimates
*when*, conditional on closing. A single censored AFT was fitted first and
**rejected** — with only 67.7% of G2 items ever closing, the survival median is not
identified (MAE 183.0 days). The rejection is reproduced in `run_poc.py`.

### Training data

Completed items only: n = 5,926 training, 1,482 holdout. Monthly throughput series for
the ARIMA component.

### Evaluation

| Metric | Value | Gate |
|---|---|---|
| MAE | 30.5 days | ≤ 30 — **not met** (by 0.5 days) |
| RMSE | 52.7 days | — |
| 90% interval coverage | 87.4% | — |
| Concordance | 0.711 | — |
| Improvement over naive median | 5.1% | — |
| ARIMA MAPE | 5.8% | ≤ 25% — **pass** |
| Ljung–Box p | 0.143 | — |

### Caveats and recommendations

- The weakest model in the stack: it misses its accuracy gate by 0.5 days, so the
  product ships its 90% interval rather than its point estimate.
- Capacity shocks are not modelled.
- Some category × team cells have small n.
- Enrichment with Jira sub-task features is a Horizon 2 action.

### Maintenance

Retrained annually, or earlier on a drift alert, alongside Model A.
