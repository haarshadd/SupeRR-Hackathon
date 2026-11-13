"""Central configuration. Change thresholds here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CORES_CSV = ROOT / "cores.csv"

MY_TEAM = "RR"

# profile construction
SHRINK_K = 40            # pseudo-balls pulling small samples toward the league mean
CONF_K = 150             # confidence = n / (n + CONF_K)
WICKET_RUNS = 6.0        # runs a wicket is worth in the value metric
REF_MIN_BALLS = 150      # reference population for z-scoring
QUALITY_REF_BALLS = 300  # reference population for value percentiles
LAST_ACTIVE = "2025-01-01"

# candidate pools
# established = enough IPL-only balls to trust (any nationality)
# local       = no international record in the pack AND below the established volume
#               (the "hidden gem" pool: IPL/emerging-event balls, lower bar, confidence-penalised)
POOL_RULES = {
    "established": dict(bat=200, bowl=200, ipl_bat=150, ipl_bowl=120),
    "local":       dict(bat=30,  bowl=30,  ipl_bat=30,  ipl_bowl=24),
}
EXCLUDE_SEGMENTS = {"excluded_pakistan"}   # not IPL-eligible
LOCAL_SEGMENTS = {"india_a_uncapped", "domestic_uncapped"}   # "domestic_uncapped" = no intl squad record in this pack

# similarity
SIM_RIVAL_THRESHOLD = 0.80
BAT_FEATURES = ["sr_PP", "sr_MID", "sr_DEATH", "sr_Pace", "sr_Spin", "bnd", "six",
                "dot", "pos", "share_PP", "share_MID", "share_DEATH"]
BOWL_FEATURES = ["eco_PP", "eco_MID", "eco_DEATH", "share_PP", "share_MID",
                 "share_DEATH", "dotp", "bndp", "wkt100", "pace"]

# shortlist: (module, how many picks)
GAP_PLAN = [("spin_mid", 2), ("death_pace", 1), ("pp_pace", 1), ("mid_bat_vs_spin", 1)]
N_REPLACEMENTS = 5



# ---------------------------------------------------------------------------
# Round 1: pressure model
# ---------------------------------------------------------------------------
PRESSURE_OUT_DIR = OUT_DIR / "pressure"
RANDOM_SEED = 42

# match-state reconstruction
UNRECORDED_EXTRA_RUNS = 1    # byes / leg-byes are stored with runs=0; assume 1 run each (see README)
BALLS_PER_INNINGS = 120

# win-probability model
WP_TEST_FRACTION = 0.20      # share of matches held out to check calibration
WP_PARAMS = dict(max_depth=4, learning_rate=0.08, max_iter=200, min_samples_leaf=200)

# pressure definition
HIGH_PRESSURE_QUANTILE = 0.75    # top 25% of leverage = "high pressure"
OUTCOMES = ["W", "0", "1", "2", "3", "4", "6"]   # next-ball outcomes used for leverage

# ratings
MIN_PRESSURE_BALLS_BAT = 100
MIN_PRESSURE_BALLS_BOWL = 120
MIN_PRESSURE_MATCHES = 10
PRESSURE_SHRINK_K = 100          # pseudo-balls pulling pressure estimates toward the league mean
BLEND_ABSOLUTE = 0.75            # weight on absolute pressure performance
BLEND_DELTA = 0.25               # weight on (pressure - own ordinary level)
N_BOOTSTRAP = 500
