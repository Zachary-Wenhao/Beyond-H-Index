import numpy as np
import pandas as pd

# ---------- helpers (focused set) ----------
def _to_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _winsorize(s, p_low=0.01, p_high=0.99):
    lo, hi = s.quantile(p_low), s.quantile(p_high)
    return s.clip(lo, hi)

def _minmax01(s):
    s = _winsorize(s.astype(float))
    rng = s.max() - s.min()
    if not np.isfinite(rng) or rng == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / rng

def _balance_around_one(r):
    """peak at 1.0; smooth & symmetric: larger is better when closer to 1"""
    r = _to_num(r).fillna(1.0)
    r = r.where(r > 0, 1e-9)
    score = 1.0 / (1.0 + np.abs(np.log(r)))
    return _minmax01(pd.Series(score, index=r.index))

def _impute_with_median(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            col = _to_num(out[c])
            med = float(col.median()) if np.isfinite(col.median()) else 0.0
            out[c] = col.fillna(med)
        else:
            out[c] = 0.0
    return out

# ---------- main ----------
def add_composite_scores(
    df: pd.DataFrame,
    weights: dict | None = None,
    return_components: bool = True,
):
    """
    Compute composite score using ONLY:
    h_index, career_span, avg_papers_per_year, first_author_count, last_author_count,
    unique_venues, venue_diversity, conference_journal_ratio, citations_per_paper
    """
    need = [
        "h_index", "career_span", "avg_papers_per_year",
        "first_author_count", "last_author_count",
        "unique_venues", "venue_diversity",
        "conference_journal_ratio", "citations_per_paper"
    ]
    df2 = _impute_with_median(df, need)

    # Pull clean series
    hidx  = _to_num(df2["h_index"])
    span  = _to_num(df2["career_span"]).clip(lower=0)
    avgpy = _to_num(df2["avg_papers_per_year"]).clip(lower=0)
    fa    = _to_num(df2["first_author_count"]).clip(lower=0)
    la    = _to_num(df2["last_author_count"]).clip(lower=0)
    uniqv = _to_num(df2["unique_venues"]).clip(lower=0)
    vdiv  = _to_num(df2["venue_diversity"]).clip(lower=0)  
    cjr   = _to_num(df2["conference_journal_ratio"]).clip(lower=1e-9)
    cpp   = _to_num(df2["citations_per_paper"]).clip(lower=0)

    # Derived components (continuous to reduce ties)
    # 1) Impact proxies (log-compressed to spread the mid-range)
    C_hidx = _minmax01(np.log1p(hidx))
    C_cpp  = _minmax01(np.log1p(cpp))

    # 2) Productivity (log-compressed)
    C_avgpy = _minmax01(np.log1p(avgpy))

    # 3) Leadership share — normalize by estimated papers ~= avg/yr * span (guarded)
    est_papers = (avgpy * np.maximum(span, 1)).replace(0, 1)
    lead_share = (fa + la) / est_papers
    C_lead = _minmax01(lead_share)

    # 4) Venue breadth & diversity
    C_uniqv = _minmax01(uniqv)
    C_vdiv  = _minmax01(vdiv)

    # 5) Conf/Journal balance (peak at 1)
    C_cjbal = _balance_around_one(cjr)

    # 6) Career span normalization (reward comparable outcomes in shorter careers)
    #    Smooth diminishing-return penalty for very long spans
    span_penalty = 1.0 / (1.0 + np.log1p(np.maximum(span - span.median(), 0)))
    C_span_norm = _minmax01(pd.Series(span_penalty, index=span.index))

    # Weights. Tuned to reduce ties by mixing several continuous signals.
    default_weights = {
        "C_hidx":   0.30,
        "C_cpp":    0.22,
        "C_avgpy":  0.12,
        "C_lead":   0.10,
        "C_uniqv":  0.08,
        "C_vdiv":   0.07,
        "C_cjbal":  0.06,
        "C_span":   0.05,
    }
    W = (weights or default_weights).copy()
    sw = sum(W.values())
    for k in W: W[k] = W[k] / sw

    # Weighted base score
    base = (
        W["C_hidx"]  * C_hidx +
        W["C_cpp"]   * C_cpp +
        W["C_avgpy"] * C_avgpy +
        W["C_lead"]  * C_lead +
        W["C_uniqv"] * C_uniqv +
        W["C_vdiv"]  * C_vdiv +
        W["C_cjbal"] * C_cjbal +
        W["C_span"]  * C_span_norm
    )

    # Micro tiebreaker (deterministic, tiny; preserves ordering but breaks equalities)
    # Use a cascade of components with decreasing epsilons; optional author_id seed.
    eps = 1e-9
    tb = (
        eps * (C_hidx.values) +
        eps/10 * (C_cpp.values) +
        eps/100 * (C_avgpy.values) +
        eps/1_000 * (C_lead.values) +
        eps/10_000 * (C_uniqv.values) +
        eps/100_000 * (C_vdiv.values) +
        eps/1_000_000 * (C_cjbal.values) +
        eps/10_000_000 * (C_span_norm.values)
    )

    # Optional: include stable per-row jitter from author_id (very tiny) to break rare exact ties
    if "author_id" in df2.columns:
        # stable hash in [0,1)
        h = pd.util.hash_pandas_object(df2["author_id"], index=False).astype("uint64") % (10**6)
        tb = tb + (h.values / 1e6) * (eps / 100_000_000)

    composite = base.values + tb

    # Pack outputs
    out = df.copy()
    out["composite_score"] = composite.astype(float)

    if return_components:
        out = pd.concat([
            out,
            pd.DataFrame({
                "C_hidx": C_hidx,
                "C_cpp": C_cpp,
                "C_avgpy": C_avgpy,
                "C_lead": C_lead,
                "C_uniqv": C_uniqv,
                "C_vdiv": C_vdiv,
                "C_cjbal": C_cjbal,
                "C_span": C_span_norm,
            }, index=df.index)
        ], axis=1)

    return out

def calculate_composite_score(row):
    df1 = pd.DataFrame([row])
    scored = add_composite_scores(df1, return_components=False)
    return float(scored.loc[df1.index[0], "composite_score"])
