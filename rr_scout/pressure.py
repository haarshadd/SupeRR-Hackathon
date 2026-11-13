"""Pressure = how much the next ball can move the match (leverage), measured from win probability.

leverage(state) = sum over next-ball outcomes o of  P(o | phase) * |WP(state after o) - WP(state now)|

* Chases (innings 2): high pressure = top (1 - HIGH_PRESSURE_QUANTILE) of leverage.
* First innings: the first-innings win-probability model is weak (a score says little about who wins
  until late), so leverage there is unreliable. First-innings pressure is a simple collapse flag instead:
  >= FIRST_INN_COLLAPSE_WKTS wickets down inside the first FIRST_INN_COLLAPSE_BALLS legal balls.
"""
import numpy as np
import pandas as pd
from . import config as C


def outcome_probs(s):
    """Next-ball outcome distribution by phase, from legal deliveries."""
    L = s[s.legal]
    oc = np.where(L.is_wicket, "W", L.runs.clip(upper=6).astype(int).astype(str))
    oc = np.where(np.isin(oc, ["5"]), "4", oc)        # 5s are rare; fold into 4
    P = pd.crosstab(L.phase.values, oc, normalize="index").reindex(columns=C.OUTCOMES).fillna(0.0)
    return P.div(P.sum(axis=1), axis=0)


def compute_pressure(s, wp):
    """Return one row per delivery with win probabilities, leverage and the pressure flag."""
    s = s[s.legal_before < C.BALLS_PER_INNINGS].copy()          # drop deliveries after ball 120 (data oddities)
    P = outcome_probs(s)
    out = []
    for inn in (1, 2):
        d = s[s.innings == inn]
        args = lambda r, w, l: (inn, r, w, l, d.target.values, d.t20i.values, d.match_id.values)
        before = wp.predict(*args(d.runs_before.values, d.wkts_before.values, d.legal_before.values))
        after = wp.predict(*args(d.runs_before.values + d.team_runs.values,
                                 d.wkts_before.values + d.is_wicket.values.astype(int),
                                 d.legal_before.values + d.legal.values.astype(int)))
        lev = np.zeros(len(d))
        pp = P.loc[d.phase.astype(str).values].values
        for k, o in enumerate(C.OUTCOMES):
            r, w = (0, 1) if o == "W" else (int(o), 0)
            hyp = wp.predict(*args(d.runs_before.values + r, d.wkts_before.values + w, d.legal_before.values + 1))
            lev += pp[:, k] * np.abs(hyp - before)
        out.append(d.assign(wp_before=before, wp_after=after, wpa=after - before, leverage=lev))
    r = pd.concat(out)
    thr = r.loc[(r.innings == 2) & r.legal, "leverage"].quantile(C.HIGH_PRESSURE_QUANTILE)
    chase = (r.innings == 2) & (r.leverage >= thr)
    collapse = (r.innings == 1) & (r.wkts_before >= C.FIRST_INN_COLLAPSE_WKTS) & (r.legal_before < C.FIRST_INN_COLLAPSE_BALLS)
    r["pressure_tier"] = np.select([chase, collapse], ["chase_leverage", "first_inn_collapse"], "ordinary")
    r["high_pressure"] = r.pressure_tier != "ordinary"
    r.attrs["leverage_threshold"] = float(thr)
    return r.sort_values(["match_id", "innings", "innings_ball_num"])


KEEP = ["match_id", "innings", "innings_ball_num", "overid", "phase", "batting_team_id", "striker_id", "bowler_id",
        "runs", "team_runs", "is_wicket", "legal", "bowler_wkt", "dis_bat", "runs_before", "wkts_before",
        "legal_before", "target", "t20i", "comp_name", "wp_before", "wp_after", "wpa", "leverage",
        "pressure_tier", "high_pressure"]