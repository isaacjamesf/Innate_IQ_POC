"""InnateIQ — figure generation. Every figure is drawn from computed POC output
(results.json and the synthetic Meridian GBS dataset in data/).

Design system follows the InnateIQ web application: green for value, red for
attention, navy for governed / model series, muted greys for chrome.
"""
import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
from lifelines import KaplanMeierFitter

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
EXTRACT = pd.Timestamp("2026-07-31")

# ------------------------------------------------------------------ tokens --
GREEN, GREEN_DK, GREEN_TINT = "#0E8A5F", "#0B6B4A", "#E4F1EB"
RED, RED_TINT = "#C13A2B", "#FBECE9"
AMBER, AMBER_TINT = "#C07A16", "#FBF3E4"
NAVY, NAVY_LT = "#2E4369", "#8FA3C4"
INK, MUTED, FAINT = "#16202B", "#68727D", "#9AA3AC"
RULE, GRID = "#DFE3E8", "#EEF1F4"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": RULE, "axes.linewidth": 0.9, "axes.labelcolor": MUTED,
    "axes.labelsize": 9,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "legend.frameon": False, "legend.fontsize": 8.5,
})
# Category labels as stored in the tracker use US spelling; the report uses UK spelling.
LABEL = {"Automation": "Automation", "Simplification": "Simplification",
         "Elimination": "Elimination", "Standardization": "Standardisation"}


# ------------------------------------------------------------- helpers ------
def header(fig, title, subtitle=None):
    """Bold title + muted subtitle above the axes, dashboard-card style."""
    h = fig.get_size_inches()[1]
    y = 1 + 0.10 / h
    if subtitle:
        fig.text(0.02, y, subtitle, fontsize=9.2, color=MUTED, va="bottom")
        y += 0.25 / h
    fig.text(0.02, y, title, fontsize=13, fontweight="bold", color=INK, va="bottom")


def footnote(fig, note):
    h = fig.get_size_inches()[1]
    fig.text(0.02, -0.16 / h, note, fontsize=7.6, color=MUTED, va="top",
             linespacing=1.55)


def finish(fig, name, note=None):
    if note:
        footnote(fig, note)
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print("wrote", name)


def style(ax, grid_y=True, bottom=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(length=0)
    if grid_y:
        ax.grid(axis="y", color=GRID, lw=0.9)
        ax.set_axisbelow(True)


R = json.load(open(f"{HERE}/results.json"))
ideas = pd.read_csv(f"{HERE}/data/ideas.csv", parse_dates=[
    "submitted_date", "g1_date", "g2_date", "g3_date", "g4_date", "last_event_date"])
# same repair rules as the ETL in run_poc.py
ideas["savings_hours_est"] = ideas.savings_hours_est.mask(
    (ideas.savings_hours_est <= 0) | (ideas.savings_hours_est > 200000), np.nan)
ideas["category"] = ideas.category.fillna("Uncategorised")
ideas["savings_hours_est"] = ideas.savings_hours_est.fillna(
    ideas.groupby("category").savings_hours_est.transform("median")).fillna(ideas.savings_hours_est.median())
op = ideas[ideas.is_open == 1]

# ============================================================== PARETO ======
v = np.sort(op.savings_hours_est.values)[::-1]
cum = np.cumsum(v) / v.sum() * 100
n_show = 60
fig, ax = plt.subplots(figsize=(8.6, 3.9))
ax.bar(range(n_show), v[:n_show], color=GREEN, width=0.82, linewidth=0)
ax.set_ylabel("Estimated annual savings (hours)")
ax.set_xlabel(f"Open items, ranked by estimated savings (top {n_show} of {len(v):,})")
style(ax)
ax2 = ax.twinx()
ax2.plot(range(n_show + 1), cum[:n_show + 1], color=NAVY, lw=2)
ax2.set_ylim(0, 100)
ax2.set_ylabel("Cumulative share of open hours (%)", color=NAVY)
ax2.tick_params(axis="y", colors=NAVY, length=0)
for s in ("top", "right", "left", "bottom"):
    ax2.spines[s].set_visible(False)
k = R["profile"]["top1pct_n"]
share = R["profile"]["top1pct_share"]
ax2.axhline(share, color=RED, ls="--", lw=1.2)
ax2.annotate(f"Top 1% of items ({k}) hold {share}% of open value",
             xy=(n_show * 0.23, share + 5), ha="left",
             color=RED, fontsize=8.6, fontweight="bold")
ax.set_xlim(-1, n_show)
header(fig, "Concentration of open value",
       "Pareto of the improvement backlog — the largest open items and the cumulative share they hold")
finish(fig, "fig_pareto",
       f"Synthetic Meridian GBS extract, {EXTRACT.date()}.  n = {len(v):,} open items, "
       f"{R['profile']['open_hours']:,} estimated hours.  Gini = {R['profile']['gini']}.")

# =========================================================== HISTOGRAM ======
fig, ax = plt.subplots(figsize=(8.6, 3.6))
ax.hist(np.log10(op.savings_hours_est.clip(lower=1)), bins=46, color=GREEN,
        alpha=.92, linewidth=0)
style(ax)
marks = [(op.savings_hours_est.median(), "Median", NAVY),
         (op.savings_hours_est.mean(), "Mean", RED),
         (op.savings_hours_est.quantile(.9), "P90", AMBER)]
for q, lbl, c in marks:
    ax.axvline(np.log10(q), color=c, lw=1.5, ls="--", label=f"{lbl}   {q:,.0f} h")
leg = ax.legend(loc="upper right", fontsize=9, handlelength=1.6,
                borderaxespad=0.2, labelspacing=0.7)
for txt, (_, _, c) in zip(leg.get_texts(), marks):
    txt.set_color(c); txt.set_fontweight("bold")
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(["1", "10", "100", "1,000", "10,000", "100,000"])
ax.set_xlabel("Estimated annual savings per item (hours, log scale)")
ax.set_ylabel("Open items")
header(fig, "Estimated savings are strongly right-skewed",
       "Distribution of estimated annual savings across open items")
finish(fig, "fig_hist",
       f"Skewness {R['profile']['skewness']}, excess kurtosis {R['profile']['kurtosis']}. "
       f"The gap between median and mean is why the platform models log1p(savings)\n"
       f"and reports median, P90 and concentration rather than the mean.")

# ============================================================== AGEING ======
bands = [(0, 90, "0–90 d"), (90, 180, "90–180 d"), (180, 365, "180–365 d"),
         (365, 730, "1–2 yr"), (730, 10 ** 6, "2 yr +")]
stages = ["New", "Under review", "Qualified", "In backlog", "In progress", "On hold"]
M = np.zeros((len(stages), len(bands)))
for i, st in enumerate(stages):
    for j, (lo, hi, _) in enumerate(bands):
        M[i, j] = ((op.current_stage == st) & (op.idea_age_days >= lo) & (op.idea_age_days < hi)).sum()

cmap = LinearSegmentedColormap.from_list(
    "iq", ["#E7F1EB", "#EFEEE4", "#F6E0D4", "#EFBCA8", "#DD8168", "#B3402C"])
fig, ax = plt.subplots(figsize=(8.6, 3.8))
ax.set_xlim(0, len(bands)); ax.set_ylim(0, len(stages)); ax.axis("off")
vmax = M.max()
for i in range(len(stages)):
    for j in range(len(bands)):
        val = M[i, j]
        y = len(stages) - 1 - i
        c = cmap((val / vmax) ** 0.72)
        ax.add_patch(FancyBboxPatch((j + 0.045, y + 0.075), 0.91, 0.85,
                                    boxstyle="round,pad=0,rounding_size=0.09",
                                    facecolor=c, edgecolor="none"))
        if val:
            dark = (val / vmax) ** 0.72 > .78
            ax.text(j + 0.5, y + 0.5, f"{int(val):,}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white" if dark else INK)
for j, (_, _, lbl) in enumerate(bands):
    ax.text(j + 0.5, len(stages) + 0.14, lbl, ha="center", va="bottom",
            fontsize=8.4, color=MUTED, fontweight="bold")
for i, st in enumerate(stages):
    ax.text(-0.12, len(stages) - 1 - i + 0.5, st, ha="right", va="center",
            fontsize=8.8, color=INK)
ax.text(-0.12, len(stages) + 0.14, "STAGE", ha="right", va="bottom",
        fontsize=7.4, color=FAINT, fontweight="bold")
header(fig, "Ageing by stage",
       "Open items by current tracker stage and age since submission — darker cells hold more items")
finish(fig, "fig_aging",
       f"{R['profile']['aged_180_n']:,} of {R['profile']['n_open']:,} open items "
       f"({R['profile']['aged_180_pct']}%) are older than 180 days; "
       f"{R['profile']['inactive_30_pct']}% have had no event in 30 days.\n"
       f"Accumulation is concentrated in execution-side stages, not at intake.")

# ================================================================= CFD ======
months = pd.date_range("2019-01-31", "2026-07-31", freq="ME")
arr, g1, g2, g3, done = [], [], [], [], []
for m in months:
    arr.append((ideas.submitted_date <= m).sum()); g1.append((ideas.g1_date <= m).sum())
    g2.append((ideas.g2_date <= m).sum()); g3.append((ideas.g3_date <= m).sum())
    done.append((ideas.g4_date <= m).sum())
fig, ax = plt.subplots(figsize=(8.6, 4.0))
for series, lbl, c in [(arr, "Submitted", "#E8ECF1"), (g1, "Past G1 Triage", "#C7D2DE"),
                       (g2, "Past G2 Validate & Route", NAVY_LT),
                       (g3, "In G3 Execute", "#E2B36C"), (done, "Closed (G4)", GREEN)]:
    ax.fill_between(months, series, color=c, label=lbl, linewidth=0)
ax.set_ylabel("Cumulative items")
style(ax)
ax.legend(loc="upper left", ncol=2)
ax.set_xlim(months[0], months[-1] + pd.Timedelta(days=25))
gap = arr[-1] - done[-1]
ax.annotate("", xy=(months[-1], arr[-1]), xytext=(months[-1], done[-1]),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
ax.annotate(f"Work in progress\n{gap:,} items", xy=(months[-4], (arr[-1] + done[-1]) / 2),
            color=RED, fontsize=8.8, fontweight="bold", ha="right", va="center",
            linespacing=1.4)
header(fig, "Cumulative flow, 2019–2026",
       "Ideas arrive faster than they close — the band between intake and closure keeps widening")
finish(fig, "fig_cfd",
       f"By Little's Law, WIP = {R['little']['wip']:,} against a trailing-12-month throughput of "
       f"{R['little']['throughput_per_year']:,} closures implies an average residence time of "
       f"about {R['little']['implied_cycle_days']:,.0f} days.")

# ========================================================= POSITIONING ======
fig, ax = plt.subplots(figsize=(8.2, 5.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.add_patch(Rectangle((5, 5), 5, 5, color=GREEN, alpha=.06, zorder=0))
ax.axvline(5, color=RULE, lw=1); ax.axhline(5, color=RULE, lw=1)
#            name                                x    y    dx   dy   ha
players = [("IdeaScale",                        2.0, 2.4,   0, -16, "center"),
           ("Brightidea",                       2.5, 3.5, -10,  10, "right"),
           ("HYPE Innovation",                  3.6, 2.9,  12,  -3, "left"),
           ("Qmarkets",                         2.9, 4.4,   0,  12, "center"),
           ("innosabi",                         4.0, 3.9,  12,  -3, "left"),
           ("Planview IdeaPlace",               2.4, 6.4,   0,  12, "center"),
           ("KaiNexus",                         6.2, 4.3,   0,  12, "center"),
           ("Home-grown tracker\n(Excel / Power Apps)", 7.4, 1.5, 0, 14, "center"),
           ("InnateIQ",                         8.4, 8.6,   0, -18, "center")]
for name, x, y, dx, dy, ha in players:
    big = name == "InnateIQ"
    ax.scatter([x], [y], s=460 if big else 140, color=GREEN if big else FAINT,
               edgecolor="white", linewidth=1.6, zorder=3)
    ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="bottom" if dy > 0 else "top",
                fontsize=9.6 if big else 8.4,
                fontweight="bold" if big else "normal",
                color=GREEN_DK if big else INK, linespacing=1.35)
ax.set_xlabel("GBS / process specificity  →  gates, SPOC–GPO routing, hub replication, Controller sign-off")
ax.set_ylabel("Analytics depth  →  descriptive to prescriptive")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color(RULE)
ax.text(9.8, 9.65, "INNOVATION\nINTELLIGENCE", ha="right", va="top", fontsize=7.8,
        color=GREEN_DK, fontweight="bold", linespacing=1.5)
ax.text(0.2, 0.35, "GENERIC IDEATION", fontsize=7.8, color=FAINT, fontweight="bold")
header(fig, "Competitive positioning",
       "Analytics depth × GBS process specificity")
finish(fig, "fig_positioning",
       "Placement is derived from the public-feature comparison in Appendix F. "
       "Axes are ordinal, not measured; the claim is relative position, not distance.")

# ======================================================== ARCHITECTURE ======
fig, ax = plt.subplots(figsize=(11.4, 6.6))
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=8.2, bold=False, tc=INK, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.30,rounding_size=1.0",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", linespacing=1.55)


def arrow(x1, y1, x2, y2, c=NAVY, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 color=c, linewidth=lw, shrinkA=0, shrinkB=0))


def elbow(pts, c=RULE, lw=1.0):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=c, lw=lw, solid_capstyle="round", zorder=1)


# column headers
for xx, big, small in [(14, "MERIDIAN GBS", "Systems of record — unchanged"),
                       (50, "INNATEIQ INTELLIGENCE LAYER", "Reads and adds — never replaces"),
                       (86, "OUTPUTS BY PERSONA", "Web application, direct access")]:
    ax.text(xx, 68.2, big, ha="center", va="center", fontsize=8.2,
            fontweight="bold", color=INK)
    ax.text(xx, 66.0, small, ha="center", va="center", fontsize=7.4, color=MUTED)
ax.plot([1, 99], [64.4, 64.4], color=RULE, lw=1.0)

# left: sources
srcs = ["Idea tracker — Power App\n(Dataverse)", "Excel idea registers",
        "Jira (execution items)", "Tenant configuration"]
src_cy = []
for i, sname in enumerate(srcs):
    yy = 54.5 - i * 10.6
    box(2, yy, 24, 7.2, sname, "white", RULE, fs=7.9)
    src_cy.append(yy + 3.6)
ax.text(14, 7.0, "Meridian keeps entering, editing and\nclosing ideas here. InnateIQ reads;\nit does not write back to these records.",
        ha="center", fontsize=7.2, color=MUTED, style="italic", linespacing=1.6)

# middle stack
mid = [
    ("ETL — Python, read-only extract\npseudonymise · map legacy stages\nderive event log",
     "#E9EEF5", NAVY, True),
    ("Staging (3NF)   →   Star schema (serving layer)", "#E9EEF5", NAVY, False),
    ("Analytics engine — Python\nLayer 1  descriptive KPIs\nLayer 2  diagnostics (lifelines · scipy · statsmodels)\n"
     "Layer 3  Model A realisation · Model B cycle time\nLayer 4  WSJF-lite score + capacity pacing",
     GREEN_TINT, GREEN, True),
    ("Governance — benefits register · override & audit log\nmodel cards · DPIA record",
     AMBER_TINT, AMBER, False),
    ("Presentation — web application, used directly\nrole-based views · score decomposition · exports",
     "white", RULE, False),
]
heights = [9.4, 5.4, 15.2, 6.4, 6.4]
gap = 2.5
y_top = 62.0
mid_cy, y = [], y_top
for (text, fc, ec, bold), h in zip(mid, heights):
    y -= h
    box(31.5, y, 37, h, text, fc, ec, fs=7.6, bold=bold)
    mid_cy.append(y + h / 2)
    y -= gap
for i in range(4):
    top_y = y_top - sum(heights[:i + 1]) - i * gap
    arrow(50, top_y, 50, top_y - gap, c=NAVY)

# source collector -> ETL
etl_cy = mid_cy[0]
for cy in src_cy:
    elbow([(26.4, cy), (29.3, cy)])
elbow([(29.3, min(src_cy)), (29.3, etl_cy)])
arrow(29.3, etl_cy, 31.1, etl_cy, c=NAVY)

# presentation -> outputs
outs = [("Portfolio view", "VP Transformation"), ("Ranked backlog", "Execution teams"),
        ("Realisation scores &\ncycle-time forecasts", "CI Lead"),
        ("Benefits register", "Controller"), ("Data quality &\nmodel health", "Platform admin")]
pres_cy = mid_cy[4]
out_cy = []
for i, (o, who) in enumerate(outs):
    yy = 53.6 - i * 10.6
    box(74, yy, 24, 7.6, f"{o}\n{who}", "white", RULE, fs=7.5)
    ax.add_patch(Rectangle((74.06, yy + 0.12), 0.9, 7.35, facecolor=GREEN, lw=0))
    out_cy.append(yy + 3.8)
elbow([(68.8, pres_cy), (71.2, pres_cy)], c=RULE)
elbow([(71.2, pres_cy), (71.2, max(out_cy))], c=RULE)
elbow([(71.2, min(out_cy)), (71.2, max(out_cy))], c=RULE)
for cy in out_cy:
    arrow(71.2, cy, 73.4, cy, c=FAINT, lw=1.0)

ax.text(50, 0.6, "Row-level security by tenant_id and role is enforced at every layer",
        ha="center", fontsize=7.9, color=MUTED, style="italic")
header(fig, "Platform architecture",
       "InnateIQ is an intelligence layer on Meridian GBS's existing records — it reads from the systems of record and never writes back")
finish(fig, "fig_architecture")

# ============================================================== GATES =======
fig, ax = plt.subplots(figsize=(9.6, 3.4))
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
gates = [("G1", "Triage", "SPOC review · DoR checklist\nroute: operational / functional", NAVY),
         ("G2", "Validate & Route", "Estimate validation\nBenefits Profile\npriority score computed", NAVY),
         ("G3", "Execute", "Pull by rank within WIP limit\nJira sync · dwell monitoring", AMBER),
         ("G4", "Realise & Close", "Actual vs target\nController validation\nbenefits register", GREEN)]
terms = ["Closed – duplicate (SPOC decision)", "Closed – rejected",
         "On hold + follow-up date", "Closed – realised"]
for i, (g, name, detail, c) in enumerate(gates):
    x = 1 + i * 25.2
    box(x, 8, 23.0, 17.5, "", "white", RULE, lw=1.1)
    ax.add_patch(Rectangle((x + 0.25, 24.15), 22.5, 1.15, facecolor=c, lw=0))
    ax.text(x + 11.5, 21.0, g, ha="center", fontsize=13.5, fontweight="bold", color=c)
    ax.text(x + 11.5, 17.7, name, ha="center", fontsize=9.8, fontweight="bold", color=INK)
    ax.text(x + 11.5, 12.6, detail, ha="center", fontsize=7.9, color=MUTED, linespacing=1.7)
    if i < 3:
        arrow(x + 23.5, 16.5, x + 25.9, 16.5, c=MUTED, lw=1.4)
ax.plot([1, 99.7], [5.0, 5.0], color=RULE, lw=1.0)
ax.text(1, 3.1, "EXITS", fontsize=7.0, color=FAINT, fontweight="bold", va="center")
for i, lbl in enumerate(terms):
    ax.text(12.5 + i * 25.2, 1.4, lbl, ha="center", fontsize=7.5, color=MUTED)
header(fig, "The four-gate process",
       "Stage gates from intake to validated closure, with the terminal states each gate can produce")
finish(fig, "fig_gates")

# ======================================================== ROC + CALIB =======
cal = R["model_a_logit"]["calibration"]
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
fig.subplots_adjust(wspace=0.32)
ax = axes[0]
auc = R["model_a_logit"]["auc"]
fpr = np.linspace(0, 1, 200)
a = norm.ppf(auc) * np.sqrt(2)
ax.plot(fpr, norm.cdf(a + norm.ppf(fpr)), color=NAVY, lw=2.1,
        label=f"Logistic (production)  AUC {auc}")
a2 = norm.ppf(R["model_a_xgb"]["auc"]) * np.sqrt(2)
ax.plot(fpr, norm.cdf(a2 + norm.ppf(fpr)), color=AMBER, lw=1.5, ls="--",
        label=f"XGBoost (rejected)  AUC {R['model_a_xgb']['auc']}")
ax.plot([0, 1], [0, 1], color=RULE, lw=1, ls=":")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("Discrimination (ROC)", loc="left", fontsize=10, fontweight="bold")
ax.legend(loc="lower right"); style(ax)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax = axes[1]
pred = [c[0] for c in cal]; obs = [c[1] for c in cal]
ax.plot([0, 1], [0, 1], color=RULE, ls=":", lw=1)
ax.plot(pred, obs, "o-", color=GREEN, lw=1.9, ms=5.5)
ax.set_xlabel("Predicted p(realisation), decile mean")
ax.set_ylabel("Observed realisation rate")
ax.set_title("Calibration", loc="left", fontsize=10, fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); style(ax)
ax.annotate(f"Slope {R['model_a_logit']['calibration_slope']}\nBrier {R['model_a_logit']['brier']}",
            xy=(.05, .86), fontsize=8.6, color=GREEN_DK, fontweight="bold", linespacing=1.5)
header(fig, "Realisation model — discrimination and calibration",
       f"Holdout, n = {R['model_a_logit']['n_test']:,}")
finish(fig, "fig_model_a",
       "Calibration matters more than discrimination here because p(realisation) enters the "
       "priority score directly.\nROC curves are binormal fits to the reported holdout AUC.")

# ======================================================== KAPLAN-MEIER ======
km = ideas[ideas.g2_date.notna()].copy()
km["dur"] = np.where(km.g4_date.notna(), (km.g4_date - km.g2_date).dt.days,
                     (EXTRACT - km.g2_date).dt.days)
km["obs"] = km.g4_date.notna().astype(int); km = km[km.dur >= 0]
fig, ax = plt.subplots(figsize=(8.6, 4.0))
cols = {"Automation": RED, "Simplification": NAVY, "Elimination": AMBER,
        "Standardization": GREEN}
kmf = KaplanMeierFitter()
for c, col in cols.items():
    s = km[km.category == c]
    kmf.fit(s.dur, s.obs, label=f"{LABEL[c]}  ·  median {R['km']['median_days'][c]:.0f} d")
    kmf.plot_survival_function(ax=ax, color=col, lw=2, ci_show=False)
ax.set_xlim(0, 500); ax.set_ylim(0, 1)
ax.set_xlabel("Days since G2 Validate & Route")
ax.set_ylabel("Share still open (survival)")
ax.axhline(.5, color=RULE, ls=":", lw=1)
style(ax); ax.legend(loc="upper right")
header(fig, "Time to closure by improvement lever",
       "Kaplan–Meier survival from G2 — open items censored at extract date")
finish(fig, "fig_km",
       f"Log-rank, Automation vs Standardisation: χ² = {R['km']['logrank_stat']}, "
       f"p {R['km']['logrank_p_str']}  (n = {R['km']['n']:,}, {R['km']['n_events']:,} closures observed).\n"
       f"Automation items take roughly {R['km']['ratio_automation_vs_standardisation']}× "
       f"as long to close as standardisation items.")

# ================================================================ WSJF ======
cf = R["counterfactual"]; bt = R["wsjf"]["backtest"]
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), gridspec_kw={"width_ratios": [1.15, 1]})
fig.subplots_adjust(wspace=0.30)
ax = axes[0]
labels = ["v1 vs\nrealised value", "v2 vs\nrealised value",
          "v2 vs value\nper exec. day", "v2 (no decay)\nvs value / day"]
vals = [bt["v1"]["pearson_value"], bt["v2"]["pearson_value"],
        bt["v2"]["pearson_rate"], bt["v2_nodecay"]["pearson_rate"]]
colors = [RED, RED, GREEN, GREEN]
ax.bar(range(4), vals, color=colors, width=.62, linewidth=0)
ax.axhline(0.40, color=NAVY, ls="--", lw=1.4)
ax.annotate("Pre-specified gate  r ≥ 0.40", xy=(-0.45, .435), ha="left",
            fontsize=8.2, color=NAVY, fontweight="bold")
ax.axhline(0, color=RULE, lw=1)
for i, v_ in enumerate(vals):
    ax.text(i, v_ + (.035 if v_ >= 0 else -.045), f"{v_:+.3f}", ha="center",
            va="bottom" if v_ >= 0 else "top", fontsize=9, fontweight="bold", color=colors[i])
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.9)
ax.set_ylabel("Pearson r"); ax.set_ylim(-.32, .74); style(ax, bottom=False)
ax.set_title("Back-test against realised outcomes", loc="left", fontsize=10, fontweight="bold")
ax = axes[1]
names = ["FIFO\n(current practice)", "v1\n(log scale)", "v2\n(euro scale)", "Perfect foresight\n(upper bound)"]
vv = [cf["fifo"]["value"] / 1e6, cf["wsjf_v1"]["value"] / 1e6,
      cf["wsjf_v2"]["value"] / 1e6, cf["oracle"]["value"] / 1e6]
cc = [FAINT, RED, GREEN, NAVY_LT]
ax.bar(range(4), vv, color=cc, width=.64, linewidth=0)
for i, v_ in enumerate(vv):
    ax.text(i, v_ + .14, f"€{v_:.2f}M", ha="center", fontsize=8.8, fontweight="bold", color=cc[i])
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=7.9)
ax.set_ylabel("Validated value delivered (€M)"); ax.set_ylim(0, max(vv) * 1.24)
style(ax)
ax.set_title(f"Same {cf['budget_days']:,} execution-days, different queue",
             loc="left", fontsize=10, fontweight="bold")
header(fig, "Validating the prescriptive layer",
       "Correlation gate versus capacity counterfactual")
finish(fig, "fig_wsjf",
       f"The original gate fails for both specifications because the score is a value-density measure, not a value estimate. "
       f"Against value per execution day, v2 reaches r = {bt['v2']['pearson_rate']}\n({bt['v2_nodecay']['pearson_rate']} without the "
       f"policy-driven age-decay term). Under a fixed capacity budget, v2 delivers {cf['wsjf_v2']['uplift_pct']}% more validated value "
       f"than first-in-first-served and captures\n{cf['capture_of_oracle']}% of the gap to perfect foresight; "
       f"v1 destroys {abs(cf['wsjf_v1']['uplift_pct'])}%.")

# ======================================================= CONTROL CHART ======
g3 = ideas[(ideas.g3_date.notna()) & (ideas.g4_date.notna())].copy()
g3["exec_days"] = (g3.g4_date - g3.g3_date).dt.days
team = R["control_chart"]["chart_team"]
s = (g3[g3.execution_team == team].set_index("g3_date").sort_index()
     .exec_days.resample("ME").mean().dropna())
mu, sd = s.mean(), s.std()
fig, ax = plt.subplots(figsize=(8.6, 3.6))
ax.plot(s.index, s.values, "-o", color=NAVY, ms=3.6, lw=1.3)
x0 = s.index[0] - pd.Timedelta(days=20)
x1 = s.index[-1] + pd.Timedelta(days=25)
ax.hlines(mu, x0, x1, color=INK, lw=1.2)
ax.hlines([mu + 3 * sd, max(mu - 3 * sd, 0)], x0, x1, color=RED, ls="--", lw=1.1)
ax.hlines([mu + 2 * sd, max(mu - 2 * sd, 0)], x0, x1, color=AMBER, ls=":", lw=1.0)
x_lab = s.index[-1] + pd.Timedelta(days=55)
for yy, lbl, c in [(mu, f"Centre {mu:.0f} d", INK), (mu + 3 * sd, "UCL (3σ)", RED),
                   (max(mu - 3 * sd, 0), "LCL (3σ)", RED)]:
    ax.text(x_lab, yy, lbl, fontsize=7.6, color=c, va="center", fontweight="bold")
brk = s[(s > mu + 3 * sd) | (s < mu - 3 * sd)]
ax.scatter(brk.index, brk.values, s=95, facecolor="none", edgecolor=RED, lw=2, zorder=5)
if len(brk):
    ax.annotate("Special-cause signal —\nescalated to the governance board",
                xy=(brk.index[0], brk.values[0]),
                xytext=(30, 18), textcoords="offset points",
                fontsize=8.2, color=RED, fontweight="bold", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.9))
ax.set_ylabel("Mean execution dwell (days)")
ax.set_xlim(s.index[0] - pd.Timedelta(days=20), s.index[-1] + pd.Timedelta(days=320))
style(ax)
header(fig, f"Monthly execution dwell — {team} team",
       "Individuals control chart: points outside the 3σ limits mark a real process change")
finish(fig, "fig_control_chart",
       f"Centre line {R['control_chart']['chart_centre']} days, 3σ limits at {R['control_chart']['chart_lcl']} / "
       f"{R['control_chart']['chart_ucl']} days. Points outside the limits separate a real process change from\n"
       f"ordinary variation, so the governance board discusses {R['control_chart']['chart_breaches']} month(s), not twelve.")

# ========================================================= STAR SCHEMA ======
fig, ax = plt.subplots(figsize=(12.4, 9.0))
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
ax.set_xlim(0, 100); ax.set_ylim(0, 97); ax.axis("off")

ROWH = 2.55
HEADH = 3.4


def table(x, y_top, w, title, fields, ec, key_rows=()):
    """Table drawn from its top edge down. Returns (y_bottom, centre_y)."""
    h = HEADH + len(fields) * ROWH + 1.2
    y = y_top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.28,rounding_size=0.8",
                                facecolor="white", edgecolor=ec, linewidth=1.3))
    ax.add_patch(Rectangle((x + 0.06, y + h - HEADH), w - 0.12, HEADH - 0.06,
                           facecolor=ec, lw=0))
    ax.text(x + w / 2, y + h - HEADH / 2, title, ha="center", va="center",
            fontsize=9.2, fontweight="bold", color="white")
    for i, f in enumerate(fields):
        fy = y + h - HEADH - 1.0 - i * ROWH
        bold = i in key_rows
        ax.text(x + 1.4, fy, f, ha="left", va="center", fontsize=7.5,
                color=INK if bold else "#3B4550", fontweight="bold" if bold else "normal")
    return y, y + h / 2


def link(x1, y1, x2, y2, xm, c=FAINT):
    """Orthogonal connector with a vertical run at xm."""
    ax.plot([x1, xm, xm, x2], [y1, y1, y2, y2], color=c, lw=1.05,
            solid_capstyle="round", zorder=0)
    ax.plot([x2], [y2], marker="o", ms=3.2, color=c, zorder=0)


# --- centre: fact tables ------------------------------------------------
FX, FW = 37, 28
_, ev_cy = table(FX, 96, FW, "FactIdeaEvent",
                 ["event_id  PK", "tenant_id  FK", "idea_key  FK", "date_key  FK",
                  "stage_from_key / stage_to_key  FK", "actor_role_key  FK",
                  "event_type", "timestamp (UTC)", "dwell_days", "inferred_flag"],
                 GREEN, key_rows=(0, 1, 2, 3, 4, 5))
ev_bottom = 96 - (HEADH + 10 * ROWH + 1.2)
_, bn_cy = table(FX, ev_bottom - 3.2, FW, "FactBenefit",
                 ["benefit_id  PK", "tenant_id / idea_key  FK", "benefit_type_key  FK",
                  "measurement_date_key  FK", "validated_by_role_key  FK",
                  "baseline_value · target_value", "actual_value · unit",
                  "evidence_flag · validated_flag"],
                 GREEN, key_rows=(0, 1, 2, 3, 4))
bn_bottom = ev_bottom - 3.2 - (HEADH + 8 * ROWH + 1.2)
_, sc_cy = table(FX, bn_bottom - 3.2, FW, "FactScore",
                 ["score_id  PK", "tenant_id / idea_key  FK", "date_key  FK",
                  "p_realization", "expected_close_days · ci_low · ci_high",
                  "wsjf_score + components (EUR)", "model_version",
                  "override_flag · override_reason"],
                 GREEN, key_rows=(0, 1, 2))

# --- left: conformed dimensions (join to every fact) --------------------
LX, LW = 2, 27
_, idea_cy = table(LX, 90, LW, "DimIdea  (SCD-2)",
                   ["idea_key  PK  ·  idea_id", "tenant_id", "title · description",
                    "category_key · team_key · hub_key",
                    "savings_hours_est · savings_log",
                    "has_evidence_flag · jira_linked_flag",
                    "valid_from · valid_to · is_current"],
                   NAVY, key_rows=(0, 1))
_, date_cy = table(LX, 60, LW, "DimDate",
                   ["date_key  PK", "calendar attributes", "fiscal period"],
                   NAVY, key_rows=(0,))
_, ten_cy = table(LX, 42, LW, "DimTenant",
                  ["tenant_key  PK", "tenant_name · hosting_region",
                   "configuration_version"],
                  NAVY, key_rows=(0,))
ax.text(LX + LW / 2, 92.6, "CONFORMED — JOIN TO EVERY FACT", ha="center",
        fontsize=7.2, color=FAINT, fontweight="bold")

# spine from conformed dims to all three facts
SP = 33.0
for cy in (idea_cy, date_cy, ten_cy):
    ax.plot([LX + LW, SP], [cy, cy], color=FAINT, lw=1.05, zorder=0)
ax.plot([SP, SP], [ev_cy, sc_cy], color=FAINT, lw=1.05, zorder=0)
for cy in (ev_cy, bn_cy, sc_cy):
    ax.plot([SP, FX], [cy, cy], color=FAINT, lw=1.05, zorder=0)
    ax.plot([FX], [cy], marker="o", ms=3.2, color=FAINT, zorder=0)

# --- right: fact-specific dimensions ------------------------------------
RX, RW = 71, 27
_, st_cy = table(RX, 96, RW, "DimStage",
                 ["stage_key  PK", "gate (G1–G4) · sub_state",
                  "legacy_stage_map (per tenant)"], NAVY, key_rows=(0,))
_, us_cy = table(RX, 81, RW, "DimUser  (pseudonymised)",
                 ["user_key  PK", "user_token — salted SHA-256",
                  "role_level · hub_key", "no name, email or HR data"],
                 NAVY, key_rows=(0,))
_, bt_cy = table(RX, 55, RW, "DimBenefitType",
                 ["benefit_type_key  PK", "productivity · capacity · risk · revenue",
                  "default_unit"], NAVY, key_rows=(0,))
_, ct_cy = table(RX, 36, RW, "DimCategory",
                 ["category_key  PK", "lever: Automation · Simplification",
                  "Elimination · Standardisation"], NAVY, key_rows=(0,))
_, th_cy = table(RX, 18, RW, "DimTeam / DimHub",
                 ["team_key / hub_key  PK", "name · capacity_weight", "hub_region"],
                 NAVY, key_rows=(0,))

FR = FX + FW  # right edge of facts
link(RX, st_cy, FR, ev_cy + 2.2, 68.6)
link(RX, us_cy, FR, ev_cy - 2.2, 67.6)
link(RX, us_cy - 1.2, FR, bn_cy + 2.2, 67.6)
link(RX, bt_cy, FR, bn_cy - 2.2, 68.6)
link(RX, ct_cy, FR, sc_cy + 2.2, 68.6)
link(RX, th_cy, FR, sc_cy - 2.2, 67.6)

ax.text(50, -1.6, "Fact tables (green) join to dimensions (navy) on surrogate keys.  "
        "Every row carries tenant_id, and row-level security is enforced on it at every layer.",
        ha="center", va="top", fontsize=8.2, color=MUTED, style="italic")
header(fig, "InnateIQ serving layer — star schema",
       "Three fact tables joined to conformed and fact-specific dimensions")
finish(fig, "fig_star_schema")

print("\nAll figures written to", FIG)
