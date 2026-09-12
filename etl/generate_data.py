"""
InnateIQ POC — synthetic dataset generator.

Creates a fully synthetic, structurally faithful GBS improvement-pipeline
dataset for the fictionalised founding customer "Meridian GBS".

No real organisational data is used anywhere in this project.

Outputs (data/):
    ideas.csv     one row per improvement idea (dimension-like)
    events.csv    one row per gate transition / hold / resume (FactIdeaEvent)
    benefits.csv  one row per benefit measurement (FactBenefit)

The dataset represents Meridian GBS's existing Power App / Excel idea tracker
and Jira execution records, as InnateIQ would read them.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

RNG = np.random.default_rng(20260817)
EXTRACT_DATE = pd.Timestamp("2026-07-31")
START_DATE = pd.Timestamp("2019-01-01")

# Repository root is the parent of etl/, so data/ sits beside run_poc.py.
# (Resolving relative to __file__ rather than the shell's cwd means the script
#  works whether invoked as `python etl/generate_data.py` or from inside etl/.)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------------
HUBS = {
    "Porto":      dict(n=3900, maturity=1.00, capacity=1.00),
    "Krakow":     dict(n=3100, maturity=0.92, capacity=0.90),
    "Bengaluru":  dict(n=2600, maturity=0.85, capacity=1.05),
    "Manila":     dict(n=1700, maturity=0.78, capacity=0.85),
    "Monterrey":  dict(n=1100, maturity=0.74, capacity=0.80),
}

CATEGORIES = ["Automation", "Simplification", "Elimination", "Standardization"]
CAT_P = [0.34, 0.28, 0.14, 0.24]

TEAMS = ["Automation", "Innovation", "Self-execution", "Process Excellence", "Unassigned"]
TEAM_P = [0.30, 0.14, 0.34, 0.16, 0.06]

AREAS = ["Order-to-Cash", "Purchase-to-Pay", "Record-to-Report",
         "HR Services", "Master Data", "Procurement", "Treasury & Banking"]
AREA_P = [0.20, 0.22, 0.19, 0.13, 0.12, 0.10, 0.04]

ROLES = ["analyst", "manager", "director"]
ROLE_P = [0.68, 0.26, 0.06]

# legacy stage vocabulary (the 7 ambiguous stages the tenant actually uses)
LEGACY_STAGES = ["New", "Under review", "Qualified", "In backlog",
                 "In progress", "On hold", "Completed"]

# ----------------------------------------------------------------------------
# Text generation vocabulary. Titles and descriptions are plain record fields of
# the tracker (they appear in the ranked backlog and on screen); no model
# consumes the free text.
# ----------------------------------------------------------------------------
AREA_OBJECTS = {
    "Order-to-Cash": ["customer invoice", "credit memo", "dunning letter", "cash application",
                      "collections worklist", "billing run", "disputed deduction", "customer master record"],
    "Purchase-to-Pay": ["supplier invoice", "purchase order", "goods receipt", "payment proposal",
                        "vendor statement", "three-way match exception", "invoice parking queue", "down payment request"],
    "Record-to-Report": ["month-end journal", "accrual posting", "balance sheet reconciliation",
                         "intercompany netting file", "trial balance pack", "close checklist", "fixed asset register"],
    "HR Services": ["onboarding checklist", "absence record", "payroll input file",
                    "employee data change", "time-sheet correction", "leaver clearance form"],
    "Master Data": ["vendor master record", "material master extension", "cost centre hierarchy",
                    "customer credit limit", "chart of accounts mapping", "bank master data"],
    "Procurement": ["sourcing request", "contract renewal notice", "supplier onboarding pack",
                    "catalogue update", "spend classification file", "tail-spend requisition"],
    "Treasury & Banking": ["bank statement import", "cash pooling entry", "FX exposure report",
                           "payment run release", "bank fee analysis"],
}

AREA_CONTEXT = {
    "Order-to-Cash": ["collections", "billing", "receivables", "dispute handling", "credit control"],
    "Purchase-to-Pay": ["accounts payable", "invoice processing", "vendor payments", "matching", "procurement operations"],
    "Record-to-Report": ["period close", "general ledger", "statutory reporting", "reconciliation", "consolidation"],
    "HR Services": ["employee lifecycle", "payroll operations", "absence management", "HR administration"],
    "Master Data": ["master data governance", "data quality", "record maintenance", "data stewardship"],
    "Procurement": ["sourcing", "contract management", "supplier management", "category management"],
    "Treasury & Banking": ["cash management", "bank connectivity", "liquidity reporting"],
}

LEVER_VERBS = {
    "Automation": ["automate", "robotise", "bot-enable", "auto-post", "script", "machine-process"],
    "Simplification": ["simplify", "streamline", "reduce steps in", "de-clutter", "shorten"],
    "Elimination": ["eliminate", "remove", "retire", "discontinue", "stop producing"],
    "Standardization": ["standardise", "harmonise", "template", "align", "unify"],
}

LEVER_NOUNS = {
    "Automation": ["robotic process automation", "automated workflow", "unattended bot",
                   "rule engine", "scripted extraction"],
    "Simplification": ["simplified workflow", "leaner handoff", "reduced approval path",
                       "consolidated step", "single touchpoint"],
    "Elimination": ["redundant report", "obsolete control", "redundant sign-off",
                    "non-value-added review", "legacy reconciliation"],
    "Standardization": ["global template", "harmonised layout", "common taxonomy",
                        "single format", "standard operating procedure"],
}

PAIN_PHRASES = [
    "manual effort", "rework loops", "long lead time", "high error rate",
    "repeated follow-up", "spreadsheet handling", "double keying", "waiting time",
]

BENEFIT_PHRASES = [
    "reduce manual handling time", "cut cycle time", "improve first-time-right rate",
    "release analyst capacity", "reduce error corrections", "remove offline spreadsheets",
]


SYSTEMS = ["SAP ECC", "SAP S/4HANA", "Coupa", "Ariba", "Oracle EBS", "Blackline",
           "Workday", "OpenText VIM", "Kyriba", "Salesforce", "Concur", "Tableau"]
MARKETS = ["Iberia", "DACH", "Nordics", "Benelux", "UK & Ireland", "France", "Italy",
           "Poland", "Turkey", "Brazil", "Japan", "ASEAN", "Andean", "Gulf"]
FREQ = ["daily", "weekly", "twice monthly", "monthly", "at every period close", "quarterly"]

SENTENCES = [
    "Volumes have grown to roughly {vol} items {freq} and the current approach does not scale.",
    "The {mkt} market raised this at the last governance forum and the same request came from two other markets.",
    "Ticket analysis for the last two quarters shows {vol} exceptions traced to this step alone.",
    "Our {sys} configuration forces an offline spreadsheet before the entry can be posted.",
    "The step is performed {freq} by {n} analysts and each pass takes about {mins} minutes.",
    "Audit raised an observation on this control last cycle and asked for a repeatable method.",
    "Handover between shifts loses context and the receiving analyst restarts the check.",
    "A parallel workaround already exists in {mkt} and could be lifted with modest configuration.",
    "There is no single owner for the output, so corrections are made in three different places.",
    "The measurement baseline is available in {sys} and can be extracted without new licensing.",
    "Roughly {pct} percent of cases need a follow-up email before they can be completed.",
    "Peak-period backlog reached {vol} open items last close and required weekend coverage.",
    "The receiving team has asked for a stable format so their downstream report stops breaking.",
    "We estimate {mins} minutes saved per case, which compounds across the {mkt} volume.",
    "This was piloted informally by one analyst and the approach held up for six weeks.",
]


def make_text(rng, category, area, hub):
    """Return (title, description) for one idea."""
    verb = rng.choice(LEVER_VERBS[category])
    obj = rng.choice(AREA_OBJECTS[area])
    noun = rng.choice(LEVER_NOUNS[category])
    ctx = rng.choice(AREA_CONTEXT[area])
    pain = rng.choice(PAIN_PHRASES)
    ben = rng.choice(BENEFIT_PHRASES)
    sys_ = rng.choice(SYSTEMS)
    mkt = rng.choice(MARKETS)
    freq = rng.choice(FREQ)

    title = f"{verb.capitalize()} {obj} in {ctx} ({sys_}, {mkt})"
    lead = (f"The {obj} in {ctx} at the {hub} hub is handled with {pain} in {sys_}. "
            f"Proposal: {verb} it using a {noun} to {ben}.")
    extra = rng.choice(len(SENTENCES), size=int(rng.integers(2, 5)), replace=False)
    body = " ".join(SENTENCES[k].format(
        vol=int(rng.integers(40, 4000)), freq=freq, mkt=mkt, sys=sys_,
        n=int(rng.integers(2, 14)), mins=int(rng.integers(5, 240)),
        pct=int(rng.integers(8, 65))) for k in extra)
    return title, lead + " " + body


# ----------------------------------------------------------------------------
# Build ideas
# ----------------------------------------------------------------------------
rows = []
idea_seq = 0

for hub, cfg in HUBS.items():
    n = cfg["n"]
    maturity = cfg["maturity"]
    for _ in range(n):
        idea_seq += 1
        # submission date: intake grows over time
        u = RNG.beta(1.55, 1.0)
        submitted = START_DATE + timedelta(days=float(u * (EXTRACT_DATE - START_DATE).days - 5))

        area = RNG.choice(AREAS, p=AREA_P)
        category = RNG.choice(CATEGORIES, p=CAT_P)
        team = RNG.choice(TEAMS, p=TEAM_P)
        role = RNG.choice(ROLES, p=ROLE_P)

        # savings estimate: heavy right skew, occasional very large programme items
        base = RNG.lognormal(mean=4.55, sigma=1.15)
        if RNG.random() < 0.012:                      # whale items
            base *= RNG.uniform(14, 55)
        if category == "Automation":
            base *= RNG.uniform(1.10, 1.60)
        if category == "Standardization":
            base *= RNG.uniform(0.85, 1.15)
        savings = float(np.clip(base, 4, 60000))

        has_evidence = int(RNG.random() < (0.30 + 0.22 * maturity))
        jira_linked = int(RNG.random() < (0.42 if team in ("Automation", "Innovation") else 0.18))

        rows.append(dict(
            idea_id=f"IQ-{idea_seq:05d}",
            hub=hub, area=area, category=category, execution_team=team,
            requester_role_level=role, submitted_date=submitted,
            savings_hours_est=savings,
            has_evidence_flag=has_evidence, jira_linked_flag=jira_linked,
        ))

ideas = pd.DataFrame(rows)
ideas["submitted_date"] = pd.to_datetime(ideas["submitted_date"]).dt.floor("D")

# ---------------------------------------------------------------- flags -----
ideas["regulatory_deadline_flag"] = (RNG.random(len(ideas)) < 0.055).astype(int)
ideas["strategic_initiative_flag"] = (RNG.random(len(ideas)) < 0.13).astype(int)
ideas["compliance_mitigation_flag"] = (RNG.random(len(ideas)) < 0.07).astype(int)

# ------------------------------------------------------- latent dynamics ----
maturity = ideas["hub"].map({h: c["maturity"] for h, c in HUBS.items()}).to_numpy()
capacity = ideas["hub"].map({h: c["capacity"] for h, c in HUBS.items()}).to_numpy()
sav_log = np.log1p(ideas["savings_hours_est"].to_numpy())
role_num = ideas["requester_role_level"].map({"analyst": 0, "manager": 1, "director": 2}).to_numpy()

cat_eff = ideas["category"].map({
    "Automation": -0.68, "Simplification": 0.52,
    "Elimination": 0.22, "Standardization": 0.72}).to_numpy()
team_eff = ideas["execution_team"].map({
    "Automation": 0.06, "Innovation": -0.42, "Self-execution": 0.56,
    "Process Excellence": 0.25, "Unassigned": -1.45}).to_numpy()

# --- G1 dwell (intake friction): shorter where evidence present / hub mature
g1_dwell = RNG.gamma(shape=1.5, scale=9.0, size=len(ideas))
g1_dwell = g1_dwell * (1.0 - 0.28 * ideas["has_evidence_flag"].to_numpy()) * (1.35 - 0.35 * maturity)
g1_dwell = np.clip(g1_dwell, 0.5, 240)
fast_g1 = (g1_dwell <= 14).astype(int)

# --- realisation propensity (the signal Model A must recover)
logit = (
    -1.28
    + 1.55 * ideas["has_evidence_flag"].to_numpy()
    - 0.46 * (sav_log - sav_log.mean())
    + 0.88 * ideas["jira_linked_flag"].to_numpy()
    + cat_eff
    + team_eff
    + 0.90 * (maturity - 0.86)
    - 0.021 * np.clip(g1_dwell, 0, 90)
    + 0.19 * role_num                       # the seniority bias the audit must find
    + 0.28 * ideas["regulatory_deadline_flag"].to_numpy()
    + 0.16 * ideas["strategic_initiative_flag"].to_numpy()
    # mild non-linearity: evidence matters more for large items
    + 0.16 * ideas["has_evidence_flag"].to_numpy() * (sav_log - sav_log.mean())
    + RNG.normal(0, 0.34, len(ideas))
)
p_true = 1 / (1 + np.exp(-logit))

# --- terminal outcome
r = RNG.random(len(ideas))
outcome = np.where(r < p_true * 0.90, "realized", "pending")
# among non-realised: rejected, duplicate, or stalled-open
r2 = RNG.random(len(ideas))
outcome = np.where(outcome == "pending",
                   np.where(r2 < 0.34, "rejected",
                            np.where(r2 < 0.47, "duplicate", "stalled")),
                   outcome)

# --- cycle time G2 -> G4 (days), predictable enough for AFT to hit its gate
lin = (
    4.24
    + 0.55 * (sav_log - sav_log.mean())
    - 0.34 * ideas["has_evidence_flag"].to_numpy()
    + ideas["category"].map({"Automation": 0.66, "Simplification": -0.22,
                             "Elimination": -0.35, "Standardization": -0.42}).to_numpy()
    + ideas["execution_team"].map({"Automation": 0.30, "Innovation": 0.52,
                                   "Self-execution": -0.46, "Process Excellence": -0.10,
                                   "Unassigned": 0.80}).to_numpy()
    - 0.32 * (capacity - 0.92)
    + 0.004 * np.clip(g1_dwell, 0, 120)
)
cycle_days = np.exp(lin + RNG.normal(0, 0.21, len(ideas)))
cycle_days = np.clip(cycle_days, 3, 900)

g2_dwell = np.clip(RNG.gamma(2.0, 11.0, len(ideas)) * (1.3 - 0.3 * maturity), 1, 400)

records = []
events = []

for i, row in enumerate(ideas.itertuples(index=False)):
    sub = row.submitted_date
    d1 = g1_dwell[i]
    d2 = g2_dwell[i]
    g1_date = sub + timedelta(days=float(d1))
    g2_date = g1_date + timedelta(days=float(d2))
    out = outcome[i]
    status, stage, closure_reason = None, None, ""
    g3_date = g4_date = pd.NaT
    realized_flag = 0
    cyc = np.nan

    if out in ("realized", "rejected", "duplicate"):
        if out == "duplicate":
            # closed early at triage
            g1_date = sub + timedelta(days=float(min(d1, 21)))
            close = g1_date + timedelta(days=float(RNG.uniform(0, 6)))
            if close > EXTRACT_DATE:
                out = "stalled"
            else:
                status, stage, closure_reason = "Closed", "Completed", "Duplicate"
                g4_date = close
        elif out == "rejected":
            close = g2_date + timedelta(days=float(RNG.gamma(2.0, 14.0)))
            if close > EXTRACT_DATE:
                out = "stalled"
            else:
                status, stage, closure_reason = "Closed", "Completed", "Rejected"
                g4_date = close
        else:  # realized
            cyc = float(cycle_days[i])
            g3_date = g2_date + timedelta(days=float(RNG.uniform(0, 12)))
            close = g2_date + timedelta(days=cyc)
            if close > EXTRACT_DATE:
                out = "stalled"
                cyc = np.nan
                g3_date = pd.NaT
            else:
                status, stage, closure_reason = "Closed", "Completed", "Realized"
                g4_date = close
                total_days = (g4_date - sub).days
                realized_flag = int(total_days <= 365)

    if out == "stalled":
        # open item: decide how far it got and when it last moved
        status = "Open"
        u = RNG.random()
        if g1_date > EXTRACT_DATE:
            stage, last = "New", sub
            g1_date = pd.NaT; g2_date = pd.NaT
        elif u < 0.10 or g2_date > EXTRACT_DATE:
            stage, last = RNG.choice(["New", "Under review"], p=[0.35, 0.65]), g1_date
            g2_date = pd.NaT
        elif u < 0.24:
            stage, last = "Qualified", g2_date
        elif u < 0.42:
            stage, last = "In backlog", g2_date
        elif u < 0.80:
            g3_date = g2_date + timedelta(days=float(RNG.uniform(0, 30)))
            g3_date = min(g3_date, EXTRACT_DATE)
            stage, last = "In progress", g3_date
        else:
            g3_date = g2_date + timedelta(days=float(RNG.uniform(0, 30)))
            g3_date = min(g3_date, EXTRACT_DATE)
            stage, last = "On hold", g3_date
        # a minority of open items are genuinely in flight; the rest sit
        if RNG.random() < 0.21:
            last_event = EXTRACT_DATE - timedelta(days=float(RNG.uniform(0, 29)))
            last_event = max(pd.Timestamp(last), last_event)
        else:
            drift = RNG.gamma(1.3, 26.0)
            last_event = min(pd.Timestamp(last) + timedelta(days=float(drift)), EXTRACT_DATE)
        closure_reason = ""
    else:
        last_event = g4_date

    records.append(dict(
        current_stage=stage, status=status, closure_reason=closure_reason,
        g1_date=g1_date, g2_date=g2_date, g3_date=g3_date, g4_date=g4_date,
        last_event_date=pd.Timestamp(last_event),
        realized_within_365=realized_flag,
        cycle_days_actual=cyc,
        p_true=p_true[i], g1_dwell=d1, g2_dwell=d2, fast_g1=fast_g1[i],
        outcome=out,
    ))

ideas = pd.concat([ideas.reset_index(drop=True), pd.DataFrame(records)], axis=1)

ideas["is_open"] = (ideas["status"] == "Open").astype(int)
ideas["idea_age_days"] = (EXTRACT_DATE - ideas["submitted_date"]).dt.days
ideas["days_since_last_event"] = (EXTRACT_DATE - ideas["last_event_date"]).dt.days

# ----------------------------------------------------------------------------
# Text — title and description as recorded in the tracker
# ----------------------------------------------------------------------------
titles, descs = [], []
for row in ideas.itertuples(index=False):
    t, d = make_text(RNG, row.category, row.area, row.hub)
    titles.append(t); descs.append(d)
ideas["title"] = titles
ideas["description"] = descs

# ---------------------------------------------------------------------------
# Controlled defect profile -- real trackers are not clean, and the ETL layer
# has to be tested against something. Raw defects are injected here; the ETL in
# run_poc.py measures and repairs them.
# ---------------------------------------------------------------------------
n = len(ideas)
ideas.loc[RNG.choice(n, int(0.014 * n), replace=False), "savings_hours_est"] = np.nan
ideas.loc[RNG.choice(n, int(0.008 * n), replace=False), "description"] = ""
ideas.loc[RNG.choice(n, int(0.011 * n), replace=False), "category"] = np.nan
ideas.loc[RNG.choice(n, int(0.006 * n), replace=False), "execution_team"] = np.nan
# implausible unit entries (hours keyed as minutes, or a stray zero)
bad = RNG.choice(n, int(0.005 * n), replace=False)
ideas.loc[bad, "savings_hours_est"] = RNG.choice([0.0, 250000.0, 999999.0], size=len(bad))

# ----------------------------------------------------------------------------
# Event log
# ----------------------------------------------------------------------------
ev = []
eid = 0
for row in ideas.itertuples(index=False):
    seq = [("Submitted", row.submitted_date, "G0"),
           ("G1 Triage", row.g1_date, "G1"),
           ("G2 Validate & Route", row.g2_date, "G2"),
           ("G3 Execute", row.g3_date, "G3"),
           ("G4 Realize & Close", row.g4_date, "G4")]
    prev = None
    for name, ts, gate in seq:
        if pd.isna(ts):
            continue
        eid += 1
        ev.append(dict(event_id=eid, idea_id=row.idea_id, gate=gate, event_type="transition",
                       stage_to=name, timestamp=pd.Timestamp(ts),
                       dwell_days=np.nan if prev is None else (pd.Timestamp(ts) - prev).days,
                       inferred_flag=int(RNG.random() < 0.061)))
        prev = pd.Timestamp(ts)
    if row.current_stage == "On hold":
        eid += 1
        ev.append(dict(event_id=eid, idea_id=row.idea_id, gate="G3", event_type="hold",
                       stage_to="On hold", timestamp=pd.Timestamp(row.last_event_date),
                       dwell_days=np.nan, inferred_flag=0))

events = pd.DataFrame(ev)

# ----------------------------------------------------------------------------
# Benefits register (sparse historically — the gap the BRM closes)
# ----------------------------------------------------------------------------
ben = []
closed_real = ideas[ideas["closure_reason"] == "Realized"]
for row in closed_real.itertuples(index=False):
    if RNG.random() < 0.31:                      # only ~a third ever got a sign-off entry
        actual = row.savings_hours_est * float(np.clip(RNG.normal(0.82, 0.26), 0.15, 1.6))
        ben.append(dict(idea_id=row.idea_id, benefit_type="productivity",
                        baseline_value=np.nan, target_value=round(row.savings_hours_est, 1),
                        actual_value=round(actual, 1), unit="hours",
                        evidence_flag=int(RNG.random() < 0.34),
                        validated_flag=int(RNG.random() < 0.11),
                        measurement_date=row.g4_date))
benefits = pd.DataFrame(ben)

# ----------------------------------------------------------------------------
# Write outputs
# ----------------------------------------------------------------------------
ideas.to_csv(f"{OUT}/ideas.csv", index=False)
events.to_csv(f"{OUT}/events.csv", index=False)
benefits.to_csv(f"{OUT}/benefits.csv", index=False)

print("ideas:", ideas.shape, "events:", events.shape, "benefits:", benefits.shape)
print("open:", int(ideas.is_open.sum()), f"({ideas.is_open.mean():.1%})")
op = ideas[ideas.is_open == 1]
print("open hours:", f"{op.savings_hours_est.sum():,.0f}")
print("aged>180:", f"{(op.idea_age_days > 180).mean():.1%}",
      " inactive>30:", f"{(op.days_since_last_event > 30).mean():.1%}")
print("top10 share:", f"{op.savings_hours_est.nlargest(10).sum() / op.savings_hours_est.sum():.1%}")
print("top1pct share:", f"{op.savings_hours_est.nlargest(int(len(op)*0.01)).sum() / op.savings_hours_est.sum():.1%}")
print("realized_within_365 rate (closed):",
      f"{ideas[ideas.status=='Closed'].realized_within_365.mean():.1%}")
print("median cycle:", np.nanmedian(ideas.cycle_days_actual))
print("stage dist open:\n", op.current_stage.value_counts())
