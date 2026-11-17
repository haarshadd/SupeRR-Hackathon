"""Pressure = how much the next ball can move the match (leverage), measured from win probability.

leverage(state) = sum over next-ball outcomes o of  P(o | phase) * |WP(state after o) - WP(state now)|

Each ball also gets exp_wpa (the win-probability change an AVERAGE delivery would produce from that state)
and wpae = wpa - exp_wpa ("win probability added over expectation"), which is what ratings credit.

Opposition adjustment: when the delivery table carries the opposing-strength buckets (see opposition.py),
two more expectations are computed with outcome probabilities conditional on (group, bucket, phase):
  wpae_bat   expectation vs the strength of the BOWLING side  -> credit for batters
  wpae_bowl  expectation vs the strength of the BATTING side  -> credit for bowlers
Leverage and the pressure flag always use the neutral (phase-only) probabilities, so which balls count as
"high pressure" does not depend on the adjustment.

* Chases (innings 2): high pressure = top (1 - HIGH_PRESSURE_QUANTILE) of leverage.
* First innings: the first-innings win-probability model is weak (a score says little about who wins
  until late), so leverage there is unreliable. First-innings pressure is a simple collapse flag instead:
  >= FIRST_INN_COLLAPSE_WKTS wickets down inside the first FIRST_INN_COLLAPSE_BALLS legal balls.
"""
import numpy as np
import pandas as pd
from . import config as C


def _codes(L):
    oc = np.where(L.is_wicket, "W", L.runs.clip(upper=6).astype(int).astype(str))
    return np.where(np.isin(oc, ["5"]), "4", oc)        # 5s are rare; fold into 4


def outcome_probs(s):
    """Next-ball outcome distribution by phase, from legal deliveries."""
    L = s[s.legal]
    P = pd.crosstab(L.phase.values, _codes(L), normalize="index").reindex(columns=C.OUTCOMES).fillna(0.0)
    return P.div(P.sum(axis=1), axis=0)


def outcome_table(s, bucket_col):
    """Outcome distribution by (group, opposing-strength bucket, phase)."""
    L = s[s.legal]
    P = pd.crosstab([L.pool.values, L[bucket_col].values, L.phase.astype(str).values], _codes(L), normalize="index")
    P = P.reindex(columns=C.OUTCOMES).fillna(0.0)
    return P.div(P.sum(axis=1), axis=0)


def _lookup(P, d, bucket_col, neutral):
    idx = pd.MultiIndex.from_arrays([d.pool.values, d[bucket_col].values, d.phase.astype(str).values])
    out = np.array(P.reindex(idx).values, dtype=float)
    miss = np.isnan(out).any(axis=1)
    out[miss] = neutral[miss]
    return out


def compute_pressure(s, wp):
    """Return one row per delivery with win probabilities, leverage and the pressure flag."""
    s = s[s.legal_before < C.BALLS_PER_INNINGS].copy()          # drop deliveries after ball 120 (data oddities)
    P = outcome_probs(s)
    adj = "bucket_faced_by_bat" in s.columns
    if adj:
        P_bat, P_bowl = outcome_table(s, "bucket_faced_by_bat"), outcome_table(s, "bucket_faced_by_bowl")
    out = []
    for inn in (1, 2):
        d = s[s.innings == inn]
        args = lambda r, w, l: (inn, r, w, l, d.target.values, d.t20i.values, d.match_id.values)
        before = wp.predict(*args(d.runs_before.values, d.wkts_before.values, d.legal_before.values))
        after = wp.predict(*args(d.runs_before.values + d.team_runs.values,
                                 d.wkts_before.values + d.is_wicket.values.astype(int),
                                 d.legal_before.values + d.legal.values.astype(int)))
        lev = np.zeros(len(d))
        exp = np.zeros(len(d))
        exp_b = np.zeros(len(d))
        exp_w = np.zeros(len(d))
        pp = P.loc[d.phase.astype(str).values].values
        if adj:
            pp_b = _lookup(P_bat, d, "bucket_faced_by_bat", pp)
            pp_w = _lookup(P_bowl, d, "bucket_faced_by_bowl", pp)
        for k, o in enumerate(C.OUTCOMES):
            r, w = (0, 1) if o == "W" else (int(o), 0)
            hyp = wp.predict(*args(d.runs_before.values + r, d.wkts_before.values + w, d.legal_before.values + 1))
            lev += pp[:, k] * np.abs(hyp - before)
            exp += pp[:, k] * (hyp - before)
            if adj:
                exp_b += pp_b[:, k] * (hyp - before)
                exp_w += pp_w[:, k] * (hyp - before)
        if not adj:
            exp_b, exp_w = exp, exp
        wpa = after - before
        out.append(d.assign(wp_before=before, wp_after=after, wpa=wpa, leverage=lev, exp_wpa=exp, wpae=wpa - exp,
                            wpae_bat=wpa - exp_b, wpae_bowl=wpa - exp_w))
    r = pd.concat(out)
    thr = r.loc[(r.innings == 2) & r.legal, "leverage"].quantile(C.HIGH_PRESSURE_QUANTILE)
    chase = (r.innings == 2) & (r.leverage >= thr)
    collapse = (r.innings == 1) & (r.wkts_before >= C.FIRST_INN_COLLAPSE_WKTS) & (r.legal_before < C.FIRST_INN_COLLAPSE_BALLS)
    r["pressure_tier"] = np.select([chase, collapse], ["chase_leverage", "first_inn_collapse"], "ordinary")
    r["high_pressure"] = r.pressure_tier != "ordinary"
    r.attrs["leverage_threshold"] = float(thr)
    return r.sort_values(["match_id", "innings", "innings_ball_num"])


KEEP = ["match_id", "innings", "innings_ball_num", "overid", "phase", "batting_team_id", "bowling_team_id", "striker_id",
        "bowler_id", "runs", "team_runs", "is_wicket", "legal", "bowler_wkt", "dis_bat", "runs_before", "wkts_before",
        "legal_before", "target", "t20i", "comp_name", "wide", "nb", "rt", "wp_before", "wp_after", "wpa",
        "exp_wpa", "wpae", "wpae_bat", "wpae_bowl", "leverage", "pressure_tier", "high_pressure"]