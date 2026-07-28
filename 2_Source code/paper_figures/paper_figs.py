"""Conceptual figures for the journal paper, rendered at ~3 MP (2000x1500).
Output: thesis_materials/figures/fig1..fig11 (data figures are built by aggregate_seeds.py)."""
import os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")   # the manuscript and its figures live together
os.makedirs(OUT, exist_ok=True)

INK = "#141414"
# A deliberately distinct identity for the manuscript: deep navy, slate, umber,
# forest, wine, indigo and brick, paired with near-white fills, square-ish
# corners and heavier strokes. Kept well away from the tinted pastel palette
# used elsewhere in the project so the two figure sets never read as siblings.
C = {"data": "#1d3557", "build": "#5c677d", "unlearn": "#9c6644", "eval": "#386641",
     "oracle": "#7b2d43", "apply": "#3d348b", "risk": "#bc4749", "grey": "#6c757d"}
BG = {"data": "#eaeef4", "build": "#eff1f4", "unlearn": "#f5efe9", "eval": "#ebf1ec",
      "oracle": "#f6ecef", "apply": "#eeedf6", "risk": "#f8eded", "grey": "#f1f2f4"}
plt.rcParams.update({"font.family": "DejaVu Sans"})

# 3 MP target: 10 x 7.5 in at dpi 200 = 2000 x 1500 px
FIGSIZE, DPI = (10, 7.5), 200


def new(title):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis("off")
    ax.text(8, 11.5, title, ha="center", va="center", fontsize=15,
            fontweight="bold", color=INK)
    return fig, ax


def _renderer(ax):
    fig = ax.figure
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.draw()
        return fig.canvas.get_renderer()


def fit_text(ax, cx, cy, w, h, s, fs, bold, color, maxlines=3, floor=5.5):
    """Centre text at (cx, cy) and guarantee it stays inside a w x h box.
    Wrapping is tried first so the font stays large; the size is only reduced
    when no wrapping of the string fits the available height."""
    r = _renderer(ax)
    inv = ax.transData.inverted()
    flat = " ".join(str(s).split())          # collapse existing newlines

    def measure(t):
        bb = t.get_window_extent(renderer=r)
        p0 = inv.transform((bb.x0, bb.y0)); p1 = inv.transform((bb.x1, bb.y1))
        return abs(p1[0] - p0[0]), abs(p1[1] - p0[1])

    size = fs
    while size >= floor:
        for nchar in range(max(len(flat), 10), 7, -2):
            cand = textwrap.fill(flat, width=nchar)
            if cand.count("\n") + 1 > maxlines:
                continue
            t = ax.text(cx, cy, cand, ha="center", va="center", color=color,
                        fontsize=size, fontweight="bold" if bold else "normal",
                        zorder=3, linespacing=1.15)
            tw, th = measure(t)
            if tw <= w * 0.90 and th <= h * 0.88:
                return t
            t.remove()
        size -= 0.5
    print(f"  !! OVERFLOW RISK: could not fit {flat[:44]!r} in {w:.2f}x{h:.2f}")
    return ax.text(cx, cy, textwrap.fill(flat, width=12), ha="center", va="center",
                   color=color, fontsize=floor, fontweight="bold" if bold else "normal",
                   zorder=3, linespacing=1.1)


def box(ax, cx, cy, w, h, title, sub="", key="grey", fs=10.5, bold=True):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.03",
                 linewidth=2.0, edgecolor=C[key], facecolor=BG[key], zorder=2))
    if sub:
        fit_text(ax, cx, cy + 0.17 * h, w, h * 0.52, title, fs, bold, INK)
        fit_text(ax, cx, cy - 0.26 * h, w, h * 0.40, sub, fs - 2, False, "#3c424e")
    else:
        fit_text(ax, cx, cy, w, h, title, fs, bold, INK)


def arrow(ax, p1, p2, color=INK, ls="-", lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                 color=color, lw=lw, linestyle=ls, zorder=1,
                 connectionstyle=f"arc3,rad={rad}"))


def line(ax, p1, p2, color=INK, ls="-", lw=1.7):
    """Connector without an arrowhead, for orthogonal routing."""
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-", mutation_scale=15,
                 color=color, lw=lw, linestyle=ls, zorder=1,
                 connectionstyle="arc3,rad=0"))


def band(ax, y0, y1, x0, x1, label, key):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="square,pad=0", linewidth=0,
                 facecolor=BG[key], alpha=0.4, zorder=0))
    ax.text(x0 + 0.18, y1 - 0.3, label, ha="left", va="center", color=C[key],
            fontsize=9.5, fontweight="bold", zorder=1)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches=None,
                facecolor="white")
    plt.close(fig); print("wrote", name)


# ------------------------------------------------- FIGURE 1  architecture
fig, ax = new("Trustworthy Machine Unlearning Framework: System Architecture")
band(ax, 8.55, 10.85, 0.3, 15.7, "DATA LAYER", "data")
box(ax, 3.3, 9.55, 5.6, 1.25, "Dataset with\ndeletion request", "retain / forget / held-out", "data", fs=10)
box(ax, 9.6, 9.55, 5.4, 1.25, "Question-answer\nrecords", "structured entity fields", "data", fs=10)
arrow(ax, (6.15, 9.55), (6.85, 9.55))
box(ax, 14.2, 9.55, 2.6, 1.25, "Domain\nadapter", "swap per domain", "data", fs=9.5)
arrow(ax, (12.35, 9.55), (12.85, 9.55))

band(ax, 4.95, 8.25, 0.3, 15.7, "MODEL LAYER", "build")
box(ax, 4.1, 6.95, 5.8, 1.3, "Language model\ntrained to memorisation", "answer-masked objective", "build", fs=10)
box(ax, 11.7, 6.95, 6.4, 1.3, "Unlearning module", "gradient ascent / gradient difference / NPO", "unlearn", fs=10)
arrow(ax, (7.05, 6.95), (8.45, 6.95))
box(ax, 4.1, 5.5, 5.8, 1.0, "Retain anchor and rollback guard", key="build", fs=9.5)
box(ax, 11.7, 5.5, 6.4, 1.0, "Reference model retrained without forget set", key="oracle", fs=9.5)

band(ax, 2.15, 4.65, 0.3, 15.7, "ASSESSMENT LAYER", "eval")
# these three sit to the right of the band label so the vertical drops that feed
# them have a clear corridor and never run through the layer caption
AX1, AX2, AX3 = 5.55, 10.1, 14.1
box(ax, AX1, 3.25, 4.3, 1.5, "Multi-prompt\nbehavioural assessment", "five probe families", "eval", fs=10)
box(ax, AX2, 3.25, 4.1, 1.5, "Trustworthiness\nmetric engine", "UES  KRR  BCM  TS", "eval", fs=10)
box(ax, AX3, 3.25, 3.2, 1.5, "Adversarial\nverification", "PLI  and  ARRM", "risk", fs=10)
# Model layer -> assessment layer, routed as a bus in the clear gap between the
# two bands. Every segment is horizontal or vertical and each drop lands square
# on the box it feeds, so nothing crosses and no connector ends in mid-air.
BUS_DOWN = 4.80
line(ax, (4.1, 5.0), (4.1, BUS_DOWN), color=C["grey"], lw=1.3)
line(ax, (11.7, 5.0), (11.7, BUS_DOWN), color=C["grey"], lw=1.3)
line(ax, (4.1, BUS_DOWN), (AX3, BUS_DOWN), color=C["grey"], lw=1.3)
for x in (AX1, AX2, AX3):
    arrow(ax, (x, BUS_DOWN), (x, 4.05), color=C["grey"], lw=1.3)

box(ax, 8.0, 0.95, 9.0, 1.15, "Trust decision and per-record report",
    "release the model, or flag records for another pass", "apply", fs=11)
# the three assessment outputs converge on one bus before entering the decision
BUS_UP = 2.05
for x in (AX1, AX2, AX3):
    line(ax, (x, 2.5), (x, BUS_UP), color=C["grey"], lw=1.3)
line(ax, (AX1, BUS_UP), (AX3, BUS_UP), color=C["grey"], lw=1.3)
arrow(ax, (8.0, BUS_UP), (8.0, 1.55), color=C["grey"], lw=1.3)
save(fig, "fig1_architecture.png")

# ------------------------------------------------- FIGURE 3  prompt pipeline
fig, ax = new("Multi-Prompt Behavioural Assessment Pipeline")
box(ax, 2.6, 9.8, 4.2, 1.2, "Target record", "structured fields", "data", fs=10.5)
probes = [("Direct", 8.6), ("Fill-in-the-blank", 7.3), ("Reordered", 6.0),
          ("Restated", 4.7), ("Cross-field", 3.4)]
# one-to-many fan-out drawn as a spine plus parallel horizontal arrows, so the
# five connectors stay separated instead of collapsing into a tangle
line(ax, (2.6, 9.2), (2.6, 3.4), color=C["grey"], lw=1.4)
for name, y in probes:
    box(ax, 7.6, y, 3.6, 0.95, name, key="build", fs=10.5)
    arrow(ax, (2.6, y), (5.75, y), color=C["grey"], lw=1.3)

box(ax, 11.4, 6.0, 2.8, 5.9, "Model under test", "before and after unlearning",
    "unlearn", fs=10.5)
for _, y in probes:
    arrow(ax, (9.4, y), (9.95, y), color=C["grey"], lw=1.3)

box(ax, 14.6, 6.0, 2.6, 1.7, "Value-based leak detector", "true value present?",
    "eval", fs=9.5)
arrow(ax, (12.8, 6.0), (13.25, 6.0))

box(ax, 8.0, 1.4, 9.2, 1.4, "Per-record leak vector: UES, BCM and per-record verdict",
    "fully forgotten / partially forgotten / not forgotten", "apply", fs=11)
# route the result down the right-hand side so it never crosses the model box
line(ax, (14.6, 5.15), (14.6, 1.4), color=C["grey"], lw=1.4)
arrow(ax, (14.6, 1.4), (12.65, 1.4), color=C["grey"], lw=1.4)
save(fig, "fig3_prompt_pipeline.png")

# ------------------------------------------------- FIGURE 2  workflow
fig, ax = new("Unlearning and Verification Workflow")
steps = [("Deletion request received", 10.3, "data"), ("Partition into forget and retain sets", 8.9, "data"),
         ("Apply unlearning objective", 7.5, "unlearn"), ("Monitor retain drift and roll back", 6.1, "unlearn"),
         ("Run multi-prompt assessment", 4.7, "eval")]
for t, y, k in steps:
    box(ax, 5.4, y, 7.2, 1.0, t, key=k, fs=11)
for i in range(len(steps) - 1):
    arrow(ax, (5.4, steps[i][1] - 0.52), (5.4, steps[i + 1][1] + 0.52))
dia = Polygon([(5.4, 3.9), (8.6, 3.0), (5.4, 2.1), (2.2, 3.0)], closed=True,
              edgecolor=C["apply"], facecolor=BG["apply"], linewidth=2.0, zorder=2)
ax.add_patch(dia)
ax.text(5.4, 3.0, "Trust score and recovery\nwithin thresholds?", ha="center", va="center",
        fontsize=10, fontweight="bold", color=INK, zorder=3)
arrow(ax, (5.4, 4.2), (5.4, 3.92))
box(ax, 5.4, 0.9, 7.2, 1.0, "Release model with per-record certificate", key="eval", fs=11)
arrow(ax, (5.4, 2.1), (5.4, 1.42)); ax.text(5.75, 1.75, "yes", color=C["eval"], fontsize=10, fontweight="bold")
box(ax, 12.6, 3.0, 3.4, 1.4, "Flag records and\nstrengthen unlearning", key="risk", fs=10)
arrow(ax, (8.6, 3.0), (10.9, 3.0), color=C["risk"]); ax.text(9.5, 3.35, "no", color=C["risk"], fontsize=10, fontweight="bold")
arrow(ax, (12.6, 3.72), (9.2, 7.5), color=C["risk"], ls="--", lw=1.5, rad=-0.35)
save(fig, "fig2_workflow.png")

# ------------------------------------------------- FIGURE 4  metric framework
fig, ax = new("Trustworthiness Evaluation Framework")
box(ax, 3.2, 9.5, 4.6, 1.3, "Forget-set evidence", "leak vectors over probes", "unlearn")
box(ax, 12.8, 9.5, 4.6, 1.3, "Retain-set evidence", "control records", "build")
box(ax, 3.2, 7.2, 4.0, 1.2, "UES", "unlearning effectiveness", "eval", fs=11)
box(ax, 8.0, 7.2, 4.0, 1.2, "BCM", "behavioural consistency", "eval", fs=11)
box(ax, 12.8, 7.2, 4.0, 1.2, "KRR / UPF", "knowledge retention, utility", "eval", fs=11)
arrow(ax, (3.2, 8.85), (3.2, 7.8)); arrow(ax, (4.6, 8.9), (7.2, 7.8), rad=0.1)
arrow(ax, (12.8, 8.85), (12.8, 7.8))
box(ax, 8.0, 4.9, 5.6, 1.3, "Trustworthiness Score  TS", "harmonic combination of forgetting and utility", "apply", fs=12)
for x, xin in ((3.2, 6.5), (8.0, 8.0), (12.8, 9.5)):
    arrow(ax, (x, 6.6), (xin, 5.60), color=C["grey"], lw=1.3, rad=0.05)
box(ax, 3.0, 2.4, 4.4, 1.2, "PLI", "privacy leakage index", "risk", fs=11)
box(ax, 13.0, 2.4, 4.4, 1.2, "ARRM", "adversarial recovery resistance", "risk", fs=11)
arrow(ax, (5.6, 4.9), (4.2, 3.0), color=C["risk"], rad=0.1)
arrow(ax, (10.4, 4.9), (11.8, 3.0), color=C["risk"], rad=-0.1)
box(ax, 8.0, 0.9, 6.0, 1.0, "Release decision", key="apply", fs=11)
arrow(ax, (3.0, 1.8), (6.3, 1.43), color=C["grey"], lw=1.3)
arrow(ax, (13.0, 1.8), (9.7, 1.43), color=C["grey"], lw=1.3)
save(fig, "fig4_metric_framework.png")

# ------------------------------------------------- FIGURE 8  threat model
fig, ax = new("Privacy Leakage Detection and Threat Model")
box(ax, 8.0, 10.2, 6.4, 1.2, "Model after unlearning", "claimed to have forgotten the target records", "unlearn", fs=11)
# the threats are ordered so that each one sits directly above the measurement
# it feeds, which keeps every connector short and free of crossings
threats = [("Membership inference", 2.9, "compares loss on\nmembers and non-members"),
           ("Rephrased querying", 6.4, "asks the same fact\nin new wording"),
           ("Cross-field probing", 9.9, "asks for a linked field\ninstead of the deleted one"),
           ("Relearning attack", 13.4, "brief fine-tune on a\nfew removed records")]
for name, x, sub in threats:
    box(ax, x, 7.3, 3.2, 1.7, name, sub, "risk", fs=10)
    arrow(ax, (8.0, 9.55), (x, 8.2), color=C["risk"], lw=1.3, rad=0.06)
box(ax, 3.0, 4.4, 4.8, 1.3, "PLI  =  attack AUC  -  reference AUC",
    "detectable training footprint", "eval", fs=11)
box(ax, 8.0, 4.4, 4.8, 1.3, "UES and BCM over the probe families",
    "behavioural leakage under rewording", "eval", fs=11)
box(ax, 13.0, 4.4, 4.8, 1.3, "ARRM  =  1  -  recovered fraction",
    "resistance to knowledge recovery", "eval", fs=11)
for xs, xe in ((2.9, 3.0), (6.4, 7.0), (9.9, 9.0), (13.4, 13.0)):
    arrow(ax, (xs, 6.45), (xe, 5.1), color=C["grey"], lw=1.3)
box(ax, 8.0, 1.6, 7.6, 1.3, "Privacy verdict against the retrained reference model",
    "leakage acceptable, or record flagged for a further pass", "apply", fs=11)
for xs, xe in ((3.0, 6.0), (8.0, 8.0), (13.0, 10.0)):
    arrow(ax, (xs, 3.75), (xe, 2.25), color=C["grey"], lw=1.3)
save(fig, "fig8_threat_model.png")

# ------------------------------------------------- FIGURE 5  benchmark design
fig, ax = new("Experimental Benchmark Design")
box(ax, 3.6, 9.9, 5.2, 1.25, "Source records", "100 entities, 5 fields each", "data", fs=11)
box(ax, 10.6, 9.9, 5.6, 1.25, "Question-answer generation", "20 pairs per entity, 2000 total",
    "data", fs=11)
arrow(ax, (6.25, 9.9), (7.75, 9.9))

for t, x, sub, k in [("Retain set", 3.0, "80 entities", "build"),
                     ("Forget set", 8.0, "20 entities", "unlearn"),
                     ("Held-out set", 13.0, "20 entities, never trained", "oracle")]:
    box(ax, x, 7.3, 4.2, 1.3, t, sub, k, fs=11)
    arrow(ax, (10.6, 9.25), (x, 7.98), color=C["grey"], lw=1.3, rad=0.06)

# the main model sees retain AND forget; the reference model sees retain only;
# the held-out set is never trained on and feeds the privacy attack directly
box(ax, 2.9, 4.8, 5.0, 1.3, "Reference model", "trained on retain set only", "oracle", fs=11)
box(ax, 9.9, 4.8, 6.6, 1.3, "Main model", "retain + forget, loss 10.9 -> 0.001", "build", fs=11)
arrow(ax, (2.3, 6.65), (2.7, 5.47), color=C["grey"], lw=1.3)
arrow(ax, (3.9, 6.65), (7.6, 5.47), color=C["grey"], lw=1.3, rad=0.05)
arrow(ax, (8.2, 6.65), (9.4, 5.47), color=C["grey"], lw=1.3)

box(ax, 8.0, 2.0, 12.0, 1.5, "Three unlearning objectives, then assessment and adversarial verification",
    "gradient ascent / gradient difference / NPO, scored by UES, BCM, KRR, TS, PLI, ARRM",
    "eval", fs=11)
arrow(ax, (2.9, 4.15), (5.2, 2.78), color=C["grey"], lw=1.3)
arrow(ax, (9.9, 4.15), (8.8, 2.78), color=C["grey"], lw=1.3)
# held-out set -> assessment: route straight down the far right so it stays clear
# of the main model box (right edge x=13.2) and never looks like it starts there
line(ax, (14.6, 6.65), (14.6, 3.05), color=C["grey"], lw=1.3)
arrow(ax, (14.6, 3.05), (13.7, 2.78), color=C["grey"], lw=1.3)
# label sits in the corridor between the main model and the routing line
ax.text(14.05, 4.6, "non-members\nfor the privacy\nattack", ha="center", va="center",
        color=C["oracle"], fontsize=7.5, style="italic")
save(fig, "fig5_benchmark_design.png")

# ------------------------------------------------- FIGURE 11  roadmap
fig, ax = new("Research Roadmap for Verifiable Machine Unlearning")
stages = [("Near term", 2.9, ["Free-text record formats", "Larger pretrained models", "Wider seed replication"], "build"),
          ("Medium term", 8.0, ["Continual deletion streams", "Certified guarantees", "Cross-domain adapters"], "eval"),
          ("Long term", 13.1, ["Regulator-facing audit trails", "Agentic and multimodal systems", "Standardised trust reporting"], "apply")]
HEAD_Y, ITEM_Y0, STEP, IH = 9.5, 7.9, 1.6, 1.05
for name, x, items, k in stages:
    box(ax, x, HEAD_Y, 4.4, 1.1, name, key=k, fs=12)
    for i, it in enumerate(items):
        cy = ITEM_Y0 - i * STEP
        box(ax, x, cy, 4.4, IH, it, key=k, fs=9.5, bold=False)
        top = cy + IH / 2
        bottom_prev = (HEAD_Y - 1.1 / 2) if i == 0 else (ITEM_Y0 - (i - 1) * STEP - IH / 2)
        arrow(ax, (x, bottom_prev), (x, top), color=C["grey"], lw=1.3)
for x0, x1 in ((5.15, 5.72), (10.25, 10.82)):
    arrow(ax, (x0, ITEM_Y0 - STEP), (x1, ITEM_Y0 - STEP), color=C["grey"], lw=1.7)
box(ax, 8.0, 1.6, 11.0, 1.3, "Goal: deletion that can be demonstrated, not merely asserted",
    "verifiable, auditable and reproducible unlearning", "apply", fs=12)
save(fig, "fig11_roadmap.png")

print("\nAll conceptual figures written to", OUT)
