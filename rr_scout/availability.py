"""Who can realistically be bought at the auction.

A player is UNAVAILABLE if he is (a) in cores.csv (the predicted retentions / confirmed trades) or
(b) has appeared in a Pakistan / Bangladesh squad in the data pack (config.EXCLUDE_NATIONAL_TEAMS), since
those players are assumed not to enter the auction.  Ratings are computed for everyone (so the benchmark
is the full population); availability is applied only when building the auction top 10.

cores.csv can be comma- or tab-separated and can be edited right up to the deadline: just rerun.
This module is self-contained (it does not depend on anything in data.py beyond the data it is given).
"""
import difflib
import pandas as pd
from . import config as C


def read_cores(path):
    """Read cores.csv whether it is comma- or tab-separated; needs columns team, player."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"team", "player"} <= set(df.columns):
        raise ValueError(f"{path} must have columns team and player, found {list(df.columns)}")
    df["team"], df["player"] = df.team.astype(str).str.strip(), df.player.astype(str).str.strip()
    return df[(df.team != "") & (df.player != "")].drop_duplicates(["team", "player"]).reset_index(drop=True)


def unavailable_players(raw, names):
    """Return a Series: player_id -> reason (only unavailable players are present)."""
    reasons, review = {}, []
    by_name = names.reset_index().groupby("name").id.apply(list)
    ret = read_cores(C.CORES_CSV)
    for _, r in ret.iterrows():
        hit = r.player if r.player in by_name.index else None
        if hit is None:
            close = difflib.get_close_matches(r.player, list(by_name.index), n=1, cutoff=0.85)
            hit = close[0] if close else None
            review.append((r.team, r.player, hit or "NOT FOUND"))
        if hit is None:
            continue
        for pid in by_name[hit]:
            reasons[pid] = f"retained ({r.team})"
    mp = raw["mapping"]
    nat = mp[mp.team_name.isin(C.EXCLUDE_NATIONAL_TEAMS)].groupby("player_id").team_name.first()
    for pid, team in nat.items():
        reasons.setdefault(pid, f"excluded nationality ({team})")
    if review:
        print("CHECK cores.csv names that were not an exact match (given -> data name):")
        for t, g, h in review:
            print(f"   {t}: {g} -> {h}")
    return pd.Series(reasons, name="reason")