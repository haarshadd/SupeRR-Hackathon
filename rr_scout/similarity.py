"""Z-scoring and cosine similarity between player style vectors."""
import numpy as np
import pandas as pd
from . import config as C


def zscore(df, feats, ref_min=C.REF_MIN_BALLS):
    ref = df[df.n >= ref_min][feats].astype(float)
    return (df[feats].astype(float) - ref.mean()) / ref.std()


def cosine(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def candidate_pool(prof, meta, kind, group, exclude_ids):
    """kind: 'bat'|'bowl'; group: 'established'|'local' (see config.POOL_RULES)."""
    r = C.POOL_RULES[group]
    p = prof.join(meta, how="inner")
    ipl = p["bat_ipl" if kind == "bat" else "bowl_ipl"]
    t20 = p["bat_t20" if kind == "bat" else "bowl_t20"]
    base = (p.last_match >= C.LAST_ACTIVE) & ~p.segment.isin(C.EXCLUDE_SEGMENTS) & ~p.index.isin(exclude_ids)
    est = (p.n >= C.POOL_RULES["established"][kind]) & (ipl >= C.POOL_RULES["established"]["ipl_" + kind])
    if group == "established":
        keep = base & est
    else:
        keep = base & ~est & p.is_local & (p.n >= r[kind]) & (t20 >= r["ipl_" + kind])
    return p[keep]


def value_pct(prof, kind):
    """Percentile of raa100 versus established players (n >= QUALITY_REF_BALLS)."""
    ref = prof[prof.n >= C.QUALITY_REF_BALLS].raa100.values
    return prof.raa100.apply(lambda v: (ref < v).mean())


def rank_similar(core_id, prof, Z, pool, vpct):
    s = cosine(Z.loc[[core_id]].values, Z.loc[pool.index].values)[0]
    out = pool.assign(sim=s, value_pct=vpct.reindex(pool.index))
    # similarity x (some value weighting) x confidence in the sample
    out["fit_score"] = out.sim * (0.6 + 0.4 * out.value_pct) * (0.5 + 0.5 * out.conf)
    return out.sort_values("fit_score", ascending=False)
