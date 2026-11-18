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


def read_cores(path):
    """Read cores.csv whether it is comma- or tab-separated; needs columns team, player."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"team", "player"} <= set(df.columns):
        raise ValueError(f"{path} must have columns team and player, found {list(df.columns)}")
    df["team"], df["player"] = df.team.astype(str).str.strip(), df.player.astype(str).str.strip()
    return df[(df.team != "") & (df.player != "")].drop_duplicates(["team", "player"]).reset_index(drop=True)


def resolve_cores(cores_csv, names):
    """Map core player names to ids (fuzzy, so spelling slips still resolve). Fails loudly."""
    cores = read_cores(cores_csv)
    pool = list(names.unique())
    by_name = names.reset_index().groupby("name").id.first()
    out = []
    for _, r in cores.iterrows():
        hit = difflib.get_close_matches(r.player, pool, n=1, cutoff=0.6)
        if not hit:
            raise ValueError(f"No match in data for core player {r.player!r}")
        out.append(dict(team=r.team, given=r.player, matched=hit[0], player_id=int(by_name[hit[0]])))
    return pd.DataFrame(out)



def add_match_state(b, raw):
    """State BEFORE each delivery, needed by the win-probability model.

    Adds: team_runs, runs_before, wkts_before, legal_before, balls_left, target (chases only),
    bat_wins, t20i.  Drops matches that cannot be modelled cleanly: no winner (no result /
    abandoned) or rain-affected (revised overs / target).

    Team runs are approximate: byes and leg-byes are stored with runs=0, so each is credited
    config.UNRECORDED_EXTRA_RUNS run. About 85% of chase outcomes then reconcile with the
    recorded winner; most mismatches are within a few runs.
    """
    
    m = raw["matches"].set_index("match_id").copy()

    # The recorded winner must be the winner of the regular match,
    # not a Super Over winner.
    result = m["result"].fillna("").astype(str)

    is_super_over = result.str.contains(
        r"\bsuper[\s-]*over\b",
        case=False,
        regex=True
    )

    is_tied = result.str.contains(
        r"\btied\b|\bmatch\s*tie\b",
        case=False,
        regex=True
    )

    # Some DLS matches in the dataset have no revised-overs fields,
    # so we must also inspect the result description.
    is_dls = result.str.contains(
        r"\bDLS\b|\bD/L\b|Duckworth",
        case=False,
        regex=True
    )

    is_no_result = result.str.contains(
        r"\bno result\b|abandon",
        case=False,
        regex=True
    )

    # Exclude any match with revised playing conditions.
    has_revised_rules = m[
        [
            "first_inn_revised_overs",
            "second_inn_revised_overs",
            "revised_target"
        ]
    ].notna().any(axis=1)

    usable_mask = (
        m["win_team_id"].notna()
        & ~has_revised_rules
        & ~is_super_over
        & ~is_tied
        & ~is_dls
        & ~is_no_result
    )

    usable = m.loc[usable_mask]

    s = b[
        b.match_id.isin(usable.index)
    ].sort_values(
        ["match_id", "innings", "innings_ball_num"]
    ).copy()


    # Base extras:
    # - wides, no-balls and other extras receive a one-run base
    # - byes and leg-byes receive the configured one-run approximation
    base_extra = (
        (s.wide | s.nb | s.extras_type.eq("other")).astype(int)
        + s.extras_type.isin(["byes", "legbyes"]).astype(int)
        * C.UNRECORDED_EXTRA_RUNS
    )

    # extras_2_runs is the numeric additional-extras field.
    recorded_extra = (
        pd.to_numeric(s["extras_2_runs"], errors="coerce")
        .fillna(0)
    )

    s["team_runs"] = (
        pd.to_numeric(s["runs"], errors="coerce").fillna(0)
        + base_extra
        + recorded_extra
    )

    g = s.groupby(["match_id", "innings"], sort=False)
    s["runs_before"] = g.team_runs.cumsum() - s.team_runs
    s["wkts_before"] = g.is_wicket.cumsum() - s.is_wicket.astype(int)
    s["legal_before"] = g.legal.cumsum() - s.legal.astype(int)
    s["balls_left"] = C.BALLS_PER_INNINGS - s.legal_before
    first_total = s[s.innings == 1].groupby("match_id").team_runs.sum()
    s["target"] = (s.match_id.map(first_total) + 1).where(s.innings == 2)
    s["bat_wins"] = (s.batting_team_id == s.match_id.map(usable.win_team_id)).astype(int)
    s["t20i"] = s.match_format.eq("T20I").astype(int)
    return s