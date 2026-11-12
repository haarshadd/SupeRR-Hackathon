"""Style profiles per player: shrunk phase rates, matchup splits, value-over-average.

Small samples are shrunk toward the league mean, which keeps tiny-sample flukes from
looking like stars -- but it also pulls them toward 'average', so every profile carries
n and a confidence weight and downstream ranking uses both.
"""
import numpy as np
import pandas as pd
from . import config as C


def _shr(x, n, mu, k=C.SHRINK_K):
    return (x * n + mu * k) / (n + k)


def batting_profiles(b, min_balls):
    bt = b[b.bat_balls].copy()
    bt["bnd"] = bt.is_four | bt.is_six
    bt["dotb"] = bt.runs.eq(0)
    lg = bt.groupby("phase", observed=True).runs.mean()
    bt["exp"] = bt.phase.map(lg).astype(float)
    mu = dict(bnd=bt.bnd.mean(), six=bt.is_six.mean(), dot=bt.dotb.mean(),
              bpd=len(bt) / bt.dis_bat.sum(),
              Pace=bt[bt.pace_or_spin == "Pace"].runs.mean(), Spin=bt[bt.pace_or_spin == "Spin"].runs.mean())
    rows = {}
    for pid, g in bt.groupby("striker_id"):
        n = len(g)
        if n < min_balls:
            continue
        d = dict(n=n, pos=g.batting_order.mean())
        for ph in ["PP", "MID", "DEATH"]:
            gp = g[g.phase == ph]
            d[f"sr_{ph}"] = _shr(gp.runs.mean() if len(gp) else lg[ph], len(gp), lg[ph]) * 100
            d[f"share_{ph}"] = len(gp) / n
        d["bnd"] = _shr(g.bnd.mean(), n, mu["bnd"])
        d["six"] = _shr(g.is_six.mean(), n, mu["six"])
        d["dot"] = _shr(g.dotb.mean(), n, mu["dot"])
        for p in ["Pace", "Spin"]:
            gp = g[g.pace_or_spin == p]
            d[f"sr_{p}"] = _shr(gp.runs.mean() if len(gp) else mu[p], len(gp), mu[p]) * 100
        outs = g.dis_bat.sum()
        d["bpd"] = _shr(n / outs, n, mu["bpd"]) if outs else float(n)
        raw = ((g.runs - g.exp).sum() - C.WICKET_RUNS * outs) / n * 100   # runs above avg per 100 balls
        d["raa100"] = raw * n / (n + C.CONF_K)
        rows[pid] = d
    out = pd.DataFrame(rows).T
    out["conf"] = out.n / (out.n + C.CONF_K)
    return out


def bowling_profiles(b, min_balls):
    bw = b.copy()
    bw["dotb"] = bw.rt.eq(0) & bw.legal
    bw["bnd"] = bw.is_four | bw.is_six
    lg = bw.groupby("phase", observed=True).rt.sum() / bw.groupby("phase", observed=True).legal.sum()
    tot = bw.legal.sum()
    mu = dict(dot=bw.dotb.sum() / tot, bnd=bw.bnd.sum() / tot, wk=bw.bowler_wkt.sum() / tot)
    rows = {}
    for pid, g in bw.groupby("bowler_id"):
        nl = int(g.legal.sum())
        if nl < min_balls:
            continue
        d = dict(n=nl, pace=(g.pace_or_spin == "Pace").mean(),
                 btype=g.bowling_type.mode().iloc[0] if g.bowling_type.notna().any() else "")
        for ph in ["PP", "MID", "DEATH"]:
            gp = g[g.phase == ph]
            k = int(gp.legal.sum())
            d[f"eco_{ph}"] = _shr(gp.rt.sum() / k * 6 if k else lg[ph] * 6, k, lg[ph] * 6)
            d[f"share_{ph}"] = k / nl
        d["dotp"] = _shr(g.dotb.sum() / nl, nl, mu["dot"])
        d["bndp"] = _shr(g.bnd.sum() / nl, nl, mu["bnd"])
        d["wkt100"] = _shr(g.bowler_wkt.sum() / nl, nl, mu["wk"], 60) * 100
        exp = g.phase.map(lg).astype(float)
        raw = (((exp * g.legal) - g.rt).sum() + C.WICKET_RUNS * g.bowler_wkt.sum()) / nl * 100
        d["raa100"] = raw * nl / (nl + C.CONF_K)
        rows[pid] = d
    out = pd.DataFrame(rows).T
    for c in out.columns:
        if c != "btype":
            out[c] = pd.to_numeric(out[c])
    out["conf"] = out.n / (out.n + C.CONF_K)
    return out
