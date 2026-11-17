"""Opposition strength, measured on the balls themselves.

For each group of matches that can be compared (T20I, IPL, other T20 such as A-team events) a ridge
regression of run value per legal ball (team runs minus WICKET_RUNS per wicket) on batting-team and
bowling-team effects (plus phase) gives every side:
  bat_effect   how much better/worse than average its batters score
  bowl_effect  how many more/fewer run-value points per ball its attack allows (negative = strong attack)
Each delivery is then tagged with the strength bucket (tertiles within its group) of the OPPOSING side:
  bucket_faced_by_bat   strength of the bowling attack the batter is facing
  bucket_faced_by_bowl  strength of the batting side the bowler is facing
pressure.py uses outcome probabilities conditional on that bucket as the expectation, so a bowler who
beats a weak batting side, or a batter who feasts on a weak attack, earns less credit.
Own-team strength is deliberately ignored, so a star is not penalised for lifting his own side.
"""
import numpy as np
import pandas as pd
from . import config as C


def _pool(s):
    return np.where(s.t20i == 1, "T20I", np.where(s.is_ipl, "IPL", "EMERGING"))


def fit_team_effects(s, raw):
    L = s[s.legal].copy()
    L["pool"] = _pool(L)
    L["y"] = L.team_runs - C.WICKET_RUNS * L.is_wicket.astype(int)
    nm = pd.concat([raw["matches"][["home_team_id", "home_team"]].set_axis(["team_id", "team"], axis=1),
                    raw["matches"][["away_team_id", "away_team"]].set_axis(["team_id", "team"], axis=1)]
                   ).drop_duplicates("team_id").set_index("team_id").team
    out = []
    for pool, g in L.groupby("pool"):
        teams = sorted(set(g.batting_team_id) | set(g.bowling_team_id))
        ix = {t: i for i, t in enumerate(teams)}
        k = len(teams)
        X = np.zeros((len(g), 2 * k + 2))
        X[np.arange(len(g)), g.batting_team_id.map(ix).values] = 1
        X[np.arange(len(g)), k + g.bowling_team_id.map(ix).values] = 1
        X[:, 2 * k] = (g.phase == "MID").values
        X[:, 2 * k + 1] = (g.phase == "DEATH").values
        y = g.y.values
        pen = np.eye(2 * k + 2) * C.TEAM_EFFECT_L2
        pen[2 * k:, 2 * k:] = 0                                  # do not shrink the phase terms
        w = np.linalg.solve(X.T @ X + pen, X.T @ (y - y.mean()))
        balls = g.groupby("batting_team_id").size()
        out.append(pd.DataFrame(dict(pool=pool, team_id=teams, team=[nm[t] for t in teams],
                                     bat_effect=w[:k] - w[:k].mean(), bowl_effect=w[k:2 * k] - w[k:2 * k].mean(),
                                     balls=[int(balls.get(t, 0)) for t in teams])))
    return pd.concat(out).sort_values(["pool", "bowl_effect"]).reset_index(drop=True)


def attach_opposition(s, raw):
    """Add pool, team effects and the opposing-strength buckets to the delivery table."""
    R = fit_team_effects(s, raw)
    s = s.copy()
    s["pool"] = _pool(s)
    key = R.set_index(["pool", "team_id"])
    s["eff_bowl_team"] = key.bowl_effect.reindex(pd.MultiIndex.from_arrays([s.pool, s.bowling_team_id])).values
    s["eff_bat_team"] = key.bat_effect.reindex(pd.MultiIndex.from_arrays([s.pool, s.batting_team_id])).values
    s[["eff_bowl_team", "eff_bat_team"]] = s[["eff_bowl_team", "eff_bat_team"]].fillna(0.0)
    s["bucket_faced_by_bat"] = "mid"
    s["bucket_faced_by_bowl"] = "mid"
    for pool, idx in s.groupby("pool").groups.items():
        a = s.loc[idx, "eff_bowl_team"]                       # low = strong attack
        q1, q2 = a.quantile([1 / 3, 2 / 3])
        s.loc[idx, "bucket_faced_by_bat"] = np.where(a <= q1, "strong", np.where(a <= q2, "mid", "weak"))
        b = s.loc[idx, "eff_bat_team"]                        # high = strong batting side
        q1, q2 = b.quantile([1 / 3, 2 / 3])
        s.loc[idx, "bucket_faced_by_bowl"] = np.where(b <= q1, "weak", np.where(b <= q2, "mid", "strong"))
    return s, R