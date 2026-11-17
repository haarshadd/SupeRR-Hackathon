"""Player ratings under pressure.

Credit per delivery is wpae (win probability added over what an average delivery would produce from
that state).  Batter credit = wpae on balls he faced (not wides); bowler credit = -wpae on balls bowled.

Per player we compute, over HIGH-PRESSURE balls:
  abs_pts100  shrunk mean credit x 100 balls  ("win-probability points added per 100 pressure balls").
              Stakes-weighted: a ball with more at stake can move the score more.
and versus the player's own ORDINARY balls:
  delta       pressure index - ordinary index, where index = sum(credit)/sum(leverage)
              (share of the available swing captured; scale-free so pressure and ordinary compare).
blend = BLEND_ABSOLUTE*z(abs_pts100) + BLEND_DELTA*z(delta), z-scored across eligible players.
Players are ranked by the lower bootstrap quantile of blend (resampling MATCHES), which penalises
thin and erratic samples at once.
"""
import numpy as np
import pandas as pd
from . import config as C


def credit_rows(r, role, adjust=None):
    adjust = C.OPPOSITION_ADJUST if adjust is None else adjust
    if role == "bat":
        c = r[~r.wide].assign(player_id=r.striker_id, credit=r.wpae_bat if adjust else r.wpae)
    else:
        c = r.assign(player_id=r.bowler_id, credit=-(r.wpae_bowl if adjust else r.wpae))
    hp = c.high_pressure.values
    return c.assign(cp=np.where(hp, c.credit, 0.0), lp=np.where(hp, c.leverage, 0.0), n_p=hp.astype(int),
                    co=np.where(~hp, c.credit, 0.0), lo=np.where(~hp, c.leverage, 0.0), n_o=(~hp).astype(int))


def _describe(c, role):
    """Plain cricket numbers in pressure vs ordinary balls, for the report."""
    hp = c.high_pressure
    out = {}
    for tag, m in [("p", hp), ("o", ~hp)]:
        d = c[m]
        if role == "bat":
            g = d.groupby("player_id")
            out[f"sr_{tag}"] = g.runs.mean() * 100
            out[f"dis_per100_{tag}"] = g.dis_bat.mean() * 100
        else:
            g = d.groupby("player_id")
            legal = g.legal.sum()
            out[f"eco_{tag}"] = g.rt.sum() / legal * 6
            out[f"wkts_per100_{tag}"] = g.bowler_wkt.sum() / legal * 100
    return pd.DataFrame(out)


def rate_players(r, role, names, adjust=None):
    c = credit_rows(r, role, adjust)
    pm = c.groupby(["player_id", "match_id"])[["cp", "lp", "n_p", "co", "lo", "n_o"]].sum().reset_index()
    k_bal = C.PRESSURE_SHRINK_K
    lbar_p = c.loc[c.high_pressure, "leverage"].mean()
    lbar_o = c.loc[~c.high_pressure, "leverage"].mean()
    min_balls = C.MIN_PRESSURE_BALLS_BAT if role == "bat" else C.MIN_PRESSURE_BALLS_BOWL

    def metrics(cp, lp, n_p, co, lo):
        abs_ = cp / (n_p + k_bal) * 100
        delta = cp / (lp + k_bal * lbar_p) - co / (lo + k_bal * lbar_o)
        return abs_, delta

    tot = pm.groupby("player_id")[["cp", "lp", "n_p", "co", "lo", "n_o"]].sum()
    pm_p = pm[pm.n_p > 0]
    tot["matches_p"] = pm_p.groupby("player_id").size()
    tot["consistency"] = pm_p.assign(pos=pm_p.cp > 0).groupby("player_id").pos.mean()
    tot["t20i_share"] = c[c.high_pressure].groupby("player_id").t20i.mean()
    tot = tot[(tot.n_p >= min_balls) & (tot.matches_p >= C.MIN_PRESSURE_MATCHES)].copy()
    tot["abs_pts100"], tot["delta"] = metrics(tot.cp, tot.lp, tot.n_p, tot.co, tot.lo)
    mu_a, sd_a, mu_d, sd_d = tot.abs_pts100.mean(), tot.abs_pts100.std(), tot.delta.mean(), tot.delta.std()
    blend = lambda a, d: C.BLEND_ABSOLUTE * (a - mu_a) / sd_a + C.BLEND_DELTA * (d - mu_d) / sd_d
    tot["blend"] = blend(tot.abs_pts100, tot.delta)

    # match-level bootstrap of the blended score
    rng = np.random.RandomState(C.RANDOM_SEED)
    groups = {pid: g for pid, g in pm.groupby("player_id") if pid in tot.index}
    lcb, spread = {}, {}
    for pid, g in groups.items():
        A = g[["cp", "lp", "n_p", "co", "lo"]].values
        idx = rng.randint(0, len(A), size=(C.N_BOOTSTRAP, len(A)))
        S = A[idx].sum(axis=1)
        a, d = metrics(S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4])
        b = blend(a, d)
        lcb[pid], spread[pid] = np.quantile(b, C.LCB_QUANTILE), b.std()
    tot["blend_lcb"], tot["blend_sd"] = pd.Series(lcb), pd.Series(spread)
    tot = tot.join(_describe(c, role))
    tot.insert(0, "name", tot.index.map(names))
    tot["rank_point"] = tot.blend.rank(ascending=False, method="min").astype(int)
    tot["rank"] = tot.blend_lcb.rank(ascending=False, method="min").astype(int)
    keep = ["name", "rank", "rank_point", "blend", "blend_lcb", "blend_sd", "abs_pts100", "delta", "n_p", "matches_p",
            "consistency", "t20i_share"] + [x for x in tot.columns if x.startswith(("sr_", "eco_", "dis_per100", "wkts_per100"))]
    return tot[keep].sort_values("rank")