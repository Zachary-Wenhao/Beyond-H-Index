# %% [markdown]
# Figure + table generation for Sections 7–8
# Saves outputs to figures/ and tables/

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_PATH = Path("data/nlp_researcher_metrics.csv")
SURVEY_BTL_PATH = Path("data/survey_btl_ranks.csv")  # optional
OUT_FIG = Path("figures"); OUT_TAB = Path("tables")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Columns (as provided)
cols = [
    "author_id","name","total_paper_count","total_citation_count","h_index",
    "first_publication_year","last_publication_year","career_span","avg_papers_per_year",
    "first_author_count","last_author_count","single_author_count","mode_venue",
    "unique_venues","venue_diversity","venue_types","conference_journal_ratio",
    "citations_per_paper","institution"
]
df = df[[c for c in cols if c in df.columns]].copy()

# Optional HIC
hic_col = "highly_influential_citations" if "highly_influential_citations" in df.columns else None

# ---- helpers ----
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def winsorize(s, lo=0.01, hi=0.99):
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)

def minmax01(s):
    s = winsorize(s.astype(float))
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)

def balance_score(r):
    r = to_num(r).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    r = r.where(r > 0, 1e-9)
    return 1.0 / (1.0 + np.abs(np.log(r)))

# ---- clean numeric columns used in figures ----
num_cols = [
    "h_index","total_paper_count","total_citation_count","career_span",
    "avg_papers_per_year","first_author_count","last_author_count",
    "unique_venues","venue_diversity","conference_journal_ratio","citations_per_paper"
]
if hic_col: num_cols.append(hic_col)

for c in num_cols:
    if c in df.columns:
        col = to_num(df[c]).replace([np.inf, -np.inf], np.nan)
        med = float(col.median()) if np.isfinite(col.median()) else 0.0
        df[c] = col.fillna(med).clip(lower=0)

# Normalized features for plots
norm = pd.DataFrame(index=df.index)
norm["h_index"] = minmax01(np.log1p(df["h_index"]))
norm["citations_per_paper"] = minmax01(np.log1p(df["citations_per_paper"]))
norm["avg_papers_per_year"] = minmax01(np.log1p(df["avg_papers_per_year"]))
norm["first_last_lead"] = minmax01( (df["first_author_count"] + df["last_author_count"]) /
                                    np.maximum(df["avg_papers_per_year"]*np.maximum(df["career_span"],1), 1) )
norm["unique_venues"] = minmax01(df["unique_venues"])
norm["venue_diversity"] = minmax01(df["venue_diversity"])
norm["conf_jour_balance"] = minmax01(balance_score(df["conference_journal_ratio"]))
norm["career_span_norm"] = minmax01(1.0/(1.0 + np.log1p(np.maximum(df["career_span"] - df["career_span"].median(), 0))))

if hic_col:
    norm["hic_norm"] = minmax01(np.log1p(df[hic_col]))

# %%  Figure 1: feature distributions (histograms)
plt.figure(figsize=(12, 8))
plot_cols = list(norm.columns)
rows = int(np.ceil(len(plot_cols)/3))
for i, c in enumerate(plot_cols, 1):
    ax = plt.subplot(rows, 3, i)
    ax.hist(norm[c].values, bins=30)
    ax.set_title(c, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUT_FIG / "feature_distributions.png", dpi=200)
plt.close()

# %%  Figure 2: correlation heatmap (absolute Spearman)
corr = norm.corr(method="spearman").abs()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, vmin=0, vmax=1)
ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=8)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.ax.set_ylabel('|Spearman|', rotation=270, labelpad=12)
ax.set_title("Correlation heatmap (absolute Spearman)")
plt.tight_layout()
plt.savefig(OUT_FIG / "corr_heatmap.png", dpi=200)
plt.close()

# %%  Composite score (same interpretable design)
# weights sum to 1
W = {
    "h_index": 0.30,
    "citations_per_paper": 0.22,
    "avg_papers_per_year": 0.12,
    "first_last_lead": 0.10,
    "unique_venues": 0.08,
    "venue_diversity": 0.07,
    "conf_jour_balance": 0.06,
    "career_span_norm": 0.05,
}
if "hic_norm" in norm.columns:
    # borrow a bit from other weights to include HIC without changing spirit
    for k in ("h_index","citations_per_paper","avg_papers_per_year"):
        W[k] -= 0.02
    W["hic_norm"] = 0.06

# normalize weights
sw = sum(W.values())
for k in W: W[k] = W[k] / sw

base = sum(W[k]*norm[k] for k in W.keys())

# tiny deterministic tiebreak
eps = 1e-9
tb = (eps*norm["h_index"].values
      + (eps/10)*norm["citations_per_paper"].values
      + (eps/100)*norm["avg_papers_per_year"].values)
df["composite_score"] = (base.values + tb).astype(float)

# %%  Table 1: dataset summary (LaTeX)
summary_cols = ["h_index","total_paper_count","total_citation_count","career_span",
                "avg_papers_per_year","first_author_count","last_author_count",
                "unique_venues","venue_diversity","conference_journal_ratio","citations_per_paper"]
if hic_col: summary_cols.append(hic_col)

desc = df[summary_cols].describe().T[["count","mean","std","min","25%","50%","75%","max"]]
desc = desc.round(2)
latex = desc.to_latex(index=True, escape=False, caption=False)
with open(OUT_TAB / "dataset_summary.tex", "w") as f:
    f.write(latex)

# %%  Figure 3: pair sampling illustration
# near-neighbors on h_index, differing on citations_per_paper and venue_diversity
hi = df.sort_values("h_index")
n = len(hi)
# Take pairs that are consecutive in h_index ranking
a = hi.iloc[:n-1].reset_index(drop=True)   # first of each pair
b = hi.iloc[1:n].reset_index(drop=True)    # second of each pair
dx = (b["citations_per_paper"] - a["citations_per_paper"]).abs()
dy = (b["venue_diversity"] - a["venue_diversity"]).abs()

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(a["h_index"], dx, s=10, alpha=0.6, label="Δ CPP vs. H-index proximity")
plt.xlabel("H-index"); plt.ylabel("Δ Citations per paper")
plt.title("Pair sampling illustration")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG / "pair_sampling_scatter.png", dpi=200)
plt.close()

# %%  Figure 4: rank alignment (composite vs. BTL if available, else H-index)
rank_df = df[["author_id","h_index","composite_score"]].copy()
rank_df["rank_h"] = rank_df["h_index"].rank(ascending=False, method="min")
rank_df["rank_c"] = rank_df["composite_score"].rank(ascending=False, method="min")

use_btl = False
if SURVEY_BTL_PATH.exists():
    btl = pd.read_csv(SURVEY_BTL_PATH)  # expects columns: author_id, btl_rank (1=best)
    rank_df = rank_df.merge(btl[["author_id","btl_rank"]], on="author_id", how="left")
    if rank_df["btl_rank"].notna().sum() > 0:
        use_btl = True

plt.figure(figsize=(7,6))
if use_btl:
    x = rank_df["btl_rank"]; y = rank_df["rank_c"]
    plt.scatter(x, y, s=10)
    plt.xlabel("Survey BTL rank (1=best)"); plt.ylabel("Composite rank (1=best)")
    plt.title("Rank alignment: Composite vs. Survey BTL")
else:
    x = rank_df["rank_h"]; y = rank_df["rank_c"]
    plt.scatter(x, y, s=10)
    plt.xlabel("H-index rank (1=best)"); plt.ylabel("Composite rank (1=best)")
    plt.title("Rank alignment: Composite vs. H-index (baseline)")
plt.tight_layout()
plt.savefig(OUT_FIG / "rank_alignment_scatter.png", dpi=200)
plt.close()

# %%  Figure 5: tie-rate reduction bar (composite vs. h-index)
def tie_rate(ranks):
    # proportion of duplicated ranks
    return 1 - (ranks.nunique() / len(ranks))

tr_h = tie_rate(rank_df["rank_h"])
tr_c = tie_rate(rank_df["rank_c"])
plt.figure(figsize=(5,4))
plt.bar(["H-index","Composite"], [tr_h, tr_c])
plt.ylabel("Tie rate (lower is better)")
plt.title("Tie-rate reduction")
for i, v in enumerate([tr_h, tr_c]):
    plt.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
plt.ylim(0, max(tr_h, tr_c) + 0.05)
plt.tight_layout()
plt.savefig(OUT_FIG / "tie_rate_bar.png", dpi=200)
plt.close()

print("Saved figures to figures/ and summary table to tables/")
