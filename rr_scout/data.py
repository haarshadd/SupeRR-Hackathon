"""Load the data pack, clean it, and add the derived columns every later step uses."""
import difflib
import pandas as pd
from . import config as C

DISMISSALS_NOT_BOWLER = ["RunOut", "Retire Out", "Obstracting the field", "Obstructing the field"]


def load_raw(data_dir=C.DATA_DIR):
    r = lambda f: pd.read_csv(data_dir / f)
    return dict(balls=r("ball_by_ball.csv"), matches=r("matches.csv"), comps=r("competitions.csv"),
                mapping=r("competition_team_player_mapping.csv"))


def prepare_balls(raw):
    """Keep innings 1-2 (drop super overs) and derive columns.

    Data quirks handled here (found during inspection):
      * extras1_runs is a CODE, not runs -> never summed.
      * wides/byes/leg-byes carry runs=0, so team totals are approximate.
      * wides and 'Overstep' (no-balls) are not legal deliveries.
    """
    b, m, c = raw["balls"], raw["matches"], raw["comps"]
    m = m.merge(c, on="comp_id")
    b = b[b.innings <= 2].merge(m[["match_id", "comp_name", "match_format", "match_date"]], on="match_id")
    b["wide"] = b.extras_type.eq("wide")
    b["nb"] = b.extras_type.eq("Overstep")
    b["legal"] = ~(b.wide | b.nb)
    b["bat_balls"] = ~b.wide                      # balls the batter faced
    b["phase"] = pd.cut(b.overid, [0, 6, 15, 20], labels=["PP", "MID", "DEATH"])
    b["rt"] = b.runs + (b.wide | b.nb | b.extras_type.eq("other")).astype(int)   # runs charged to bowler
    b["bowler_wkt"] = b.is_wicket & ~b.dismissal_type.isin(DISMISSALS_NOT_BOWLER)
    b["dis_bat"] = b.dismissed_player_id.eq(b.striker_id) & b.is_wicket
    b["is_ipl"] = b.comp_name.str.startswith("IPL")
    b["is_t20_league"] = b.match_format.eq("T20")   # IPL + emerging events
    return b


def player_names(b):
    n = pd.concat([b[["striker_id", "striker_name"]].set_axis(["id", "name"], axis=1),
                   b[["bowler_id", "bowler_name"]].set_axis(["id", "name"], axis=1)])
    return n.dropna().drop_duplicates("id").set_index("id").name


def resolve_cores(cores_csv, names):
    """Map core player names to ids (fuzzy, so spelling slips still resolve). Fails loudly."""
    cores = pd.read_csv(cores_csv)
    pool = list(names.unique())
    by_name = names.reset_index().groupby("name").id.first()
    out = []
    for _, r in cores.iterrows():
        hit = difflib.get_close_matches(r.player, pool, n=1, cutoff=0.6)
        if not hit:
            raise ValueError(f"No match in data for core player {r.player!r}")
        out.append(dict(team=r.team, given=r.player, matched=hit[0], player_id=int(by_name[hit[0]])))
    return pd.DataFrame(out)
