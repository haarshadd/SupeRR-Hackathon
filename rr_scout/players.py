"""Player metadata: who is 'local', who is excluded, who is still active, who is in a 2026 squad."""
import pandas as pd
from . import config as C


def build_meta(b, raw, names):
    mp, comps = raw["mapping"], raw["comps"]
    ipl_comp_ids = comps[comps.comp_name.str.startswith("IPL")].comp_id
    franchises = set(mp[mp.comp_id.isin(ipl_comp_ids)].team_name)
    nat = mp[~mp.team_name.isin(franchises)].groupby("player_id").team_name.agg(set)

    def segment(teams):
        if not isinstance(teams, set):
            return "domestic_uncapped"           # never in a national/A squad in this window
        if {"Pakistan", "Pakistan A"} & teams:
            return "excluded_pakistan"
        senior = {t for t in teams if not t.endswith(" A")}
        if "India" in senior:
            return "india_capped"
        if senior:
            return "overseas_capped"
        return "india_a_uncapped" if "India A" in teams else "other_a_uncapped"

    meta = pd.DataFrame(index=names.index)
    meta["name"] = names
    meta["segment"] = [segment(nat.get(i)) for i in meta.index]
    last = pd.concat([b.groupby("striker_id").match_date.max(), b.groupby("bowler_id").match_date.max()], axis=1).max(axis=1)
    meta["last_match"] = last.reindex(meta.index)
    ipl26 = comps.loc[comps.comp_name == "IPL 2026", "comp_id"].iloc[0]
    meta["squad_2026"] = mp[mp.comp_id == ipl26].drop_duplicates("player_id").set_index("player_id").team_name.reindex(meta.index)
    meta["bat_t20"] = b[b.is_t20_league & b.bat_balls].groupby("striker_id").size().reindex(meta.index).fillna(0)
    meta["bowl_t20"] = b[b.is_t20_league & b.legal].groupby("bowler_id").size().reindex(meta.index).fillna(0)
    meta["bat_ipl"] = b[b.is_ipl & b.bat_balls].groupby("striker_id").size().reindex(meta.index).fillna(0)
    meta["bowl_ipl"] = b[b.is_ipl & b.legal].groupby("bowler_id").size().reindex(meta.index).fillna(0)
    meta["is_local"] = meta.segment.isin(C.LOCAL_SEGMENTS)
    return meta
