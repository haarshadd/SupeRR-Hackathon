"""Round 1 pipeline.  python run_pressure.py
Step 2: win-probability model + pressure labelling. Later steps extend this file."""
import json
import pandas as pd
from rr_scout import config as C
from rr_scout.data import load_raw, prepare_balls, add_match_state
from rr_scout.win_prob import WinProb
from rr_scout.pressure import compute_pressure, KEEP


def main():
    C.PRESSURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    s = add_match_state(prepare_balls(raw), raw)
    print(f"usable matches={s.match_id.nunique()}  deliveries={len(s):,}")

    wp = WinProb().fit(s)
    print("\nWin-probability model (out-of-fold, by match):\n", wp.metrics.round(3).to_string(index=False))
    wp.metrics.to_csv(C.PRESSURE_OUT_DIR / "wp_model_metrics.csv", index=False)

    r = compute_pressure(s, wp)
    r[KEEP].to_csv(C.PRESSURE_OUT_DIR / "ball_pressure.csv.gz", index=False)
    thr = r.attrs["leverage_threshold"]
    L = r[r.legal]
    print(f"\nchase leverage threshold (top {1 - C.HIGH_PRESSURE_QUANTILE:.0%}) = {thr:.4f}")

    print("\nShare of legal balls flagged as pressure, by innings and phase:")
    print(L.groupby(["innings", "phase"], observed=True).high_pressure.mean().unstack().round(3).to_string())
    print("\nTier counts (legal balls):\n", L.pressure_tier.value_counts().to_string())

    ch = L[L.innings == 2].copy()
    ch["dot"] = (ch.runs == 0) & ~ch.is_wicket
    ch["bnd"] = ch.runs >= 4
    print("\nChases: outcomes in high-pressure vs ordinary balls, by phase")
    print(ch.groupby(["phase", "high_pressure"], observed=True).agg(
        balls=("runs", "size"), runs_per_ball=("runs", "mean"), wicket_rate=("is_wicket", "mean"),
        dot_rate=("dot", "mean"), boundary_rate=("bnd", "mean"),
        mean_abs_wp_gap=("wp_before", lambda x: (x - 0.5).abs().mean())).round(3).to_string())
    top = ch[ch.leverage >= ch.leverage.quantile(0.99)]
    print(f"\nTop 1% leverage chase balls: median win prob {top.wp_before.median():.2f}, median over {top.overid.median():.0f}")
    print("Matches / balls per tier:", json.dumps(L.groupby("pressure_tier").match_id.nunique().to_dict()))


if __name__ == "__main__":
    main()