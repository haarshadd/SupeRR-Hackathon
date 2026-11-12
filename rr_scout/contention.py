"""Proxy for rival demand. NOT a price model -- the data pack has no auction prices."""
import pandas as pd
from . import config as C
from .similarity import cosine


def rival_similarity(player_id, Z, prof, cores, names, my_team=C.MY_TEAM):
    """Rival teams that already own a core player stylistically close to this candidate."""
    hits = []
    for team, grp in cores[cores.team != my_team].groupby("team"):
        ids = [i for i in grp.player_id if i in Z.index and prof.n.get(i, 0) >= C.REF_MIN_BALLS]
        if not ids or player_id not in Z.index:
            continue
        s = cosine(Z.loc[[player_id]].values, Z.loc[ids].values)[0]
        if s.max() >= C.SIM_RIVAL_THRESHOLD:
            hits.append(f"{team}:{names[ids[int(s.argmax())]]}({s.max():.2f})")
    return hits


def rival_niche_gaps(cores, bowl, my_team=C.MY_TEAM):
    """Rivals whose core lacks a spinner / elite death bowler / elite powerplay pacer."""
    ref = bowl[bowl.n >= C.QUALITY_REF_BALLS]
    q_d, q_p = ref.eco_DEATH.quantile(0.4), ref.eco_PP.quantile(0.4)
    res = {"spin": [], "death": [], "pp": []}
    for team, grp in cores[cores.team != my_team].groupby("team"):
        B = bowl.loc[[i for i in grp.player_id if i in bowl.index and bowl.n[i] >= 200]]
        if not (B.pace < 0.3).any():
            res["spin"].append(team)
        if not ((B.share_DEATH >= 0.2) & (B.eco_DEATH <= q_d)).any():
            res["death"].append(team)
        if not ((B.pace > 0.7) & (B.share_PP >= 0.3) & (B.eco_PP <= q_p)).any():
            res["pp"].append(team)
    return res
