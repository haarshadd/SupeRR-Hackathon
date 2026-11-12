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
