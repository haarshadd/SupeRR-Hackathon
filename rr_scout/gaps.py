"""What the core is missing, and who fills each gap."""
import pandas as pd
from . import config as C


def core_coverage(b, core_ids):
    """Ball-weighted look at what the core batting/bowling group covers."""
    bt = b[b.bat_balls & b.striker_id.isin(core_ids)]
    bw = b[b.bowler_id.isin(core_ids)]
    rows = {}
    for ph in ["PP", "MID", "DEATH"]:
        x = bt[bt.phase == ph]
        y = bw[bw.phase == ph]
        rows[f"bat_sr_{ph}"] = x.runs.mean() * 100
        rows[f"bowl_overs_{ph}"] = y.legal.sum() / 6
        rows[f"bowl_eco_{ph}"] = y.rt.sum() / max(y.legal.sum(), 1) * 6
    for p in ["Pace", "Spin"]:
        rows[f"bat_sr_vs_{p}"] = bt[bt.pace_or_spin == p].runs.mean() * 100
    lg = b[b.legal]
    rows["bowl_spin_balls_core"] = int((bw.legal & (bw.pace_or_spin == "Spin")).sum())
    rows["league_spin_share"] = (lg.pace_or_spin == "Spin").mean()
    rows["n_core_bowlers(>=200 balls)"] = int((bw.groupby("bowler_id").legal.sum() >= 200).sum())
    return pd.Series(rows)


def core_percentiles(core_ids, bat, bowl):
    """Where each core player sits vs established league players (higher = better)."""
    ref_b, ref_w = bat[bat.n >= C.QUALITY_REF_BALLS], bowl[bowl.n >= C.QUALITY_REF_BALLS]
    out = []
    for i in core_ids:
        if i in bat.index:
            r = bat.loc[i]
            out.append(dict(player_id=i, role="bat", **{f: (ref_b[f] < r[f]).mean() for f in ["sr_PP", "sr_MID", "sr_DEATH", "sr_Pace", "sr_Spin", "bpd"]}))
        if i in bowl.index and bowl.loc[i, "n"] >= 100:
            r = bowl.loc[i]
            out.append(dict(player_id=i, role="bowl", **{f: (ref_w[f] > r[f]).mean() for f in ["eco_PP", "eco_MID", "eco_DEATH"]},
                            wkt100=(ref_w.wkt100 < r.wkt100).mean()))
    return pd.DataFrame(out)


def _pct(s, ref, inverse=False):
    return s.apply(lambda v: (ref > v).mean() if inverse else (ref < v).mean())


def score_gap_modules(pool_bat, pool_bowl, ref_bat, ref_bowl):
    """Return {module: scored DataFrame}. Each module = one gap in the RR core."""
    rb, rw = ref_bat[ref_bat.n >= C.QUALITY_REF_BALLS], ref_bowl[ref_bowl.n >= C.QUALITY_REF_BALLS]
    B, W = pool_bat.copy(), pool_bowl.copy()
    for c in ["sr_MID", "sr_Spin", "bpd", "raa100"]:
        B["p_" + c] = _pct(B[c], rb[c])
    for c in ["eco_PP", "eco_MID", "eco_DEATH"]:
        W["p_" + c] = _pct(W[c], rw[c], True)
    for c in ["wkt100", "raa100", "dotp"]:
        W["p_" + c] = _pct(W[c], rw[c])
    W["pp_balls"], W["mid_balls"], W["death_balls"] = W.share_PP * W.n, W.share_MID * W.n, W.share_DEATH * W.n
    W["is_spin"] = W.pace < 0.3
    conf = lambda d: (0.5 + 0.5 * d.conf)
    out = {}
    s = W[W.is_spin & (W.mid_balls >= 24)].copy()
    s["gap_score"] = (s.p_eco_MID + s.p_wkt100 + s.p_raa100 + s.p_dotp) / 4 * conf(s)
    out["spin_mid"] = s
    s = W[~W.is_spin & (W.death_balls >= 24)].copy()
    s["gap_score"] = (s.p_eco_DEATH * 2 + s.p_wkt100 + s.p_raa100) / 4 * conf(s)
    out["death_pace"] = s
    s = W[~W.is_spin & (W.pp_balls >= 24)].copy()
    s["gap_score"] = (s.p_eco_PP * 2 + s.p_wkt100 + s.p_raa100) / 4 * conf(s)
    out["pp_pace"] = s
    s = B[B.pos.between(2.5, 5.5)].copy()
    s["gap_score"] = (s.p_sr_MID + s.p_sr_Spin + s.p_bpd + s.p_raa100) / 4 * conf(s)
    out["mid_bat_vs_spin"] = s
    return {k: v.sort_values("gap_score", ascending=False) for k, v in out.items()}
