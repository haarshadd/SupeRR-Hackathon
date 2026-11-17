"""Round 1 pipeline.  python run_pressure.py
Steps 2-4a: win-probability model, opposition strength, pressure labelling, player ratings, auction top 10.
Later steps extend this file."""
import json
import pandas as pd
from rr_scout import config as C
from rr_scout.data import load_raw, prepare_balls, add_match_state, player_names
from rr_scout.win_prob import WinProb
from rr_scout.pressure import compute_pressure, KEEP
from rr_scout.ratings import rate_players
from rr_scout.availability import unavailable_players
from rr_scout.opposition import attach_opposition


def main():
    C.PRESSURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    b = prepare_balls(raw)
    names = player_names(b)
    s = add_match_state(b, raw)
    print(f"usable matches={s.match_id.nunique()}  deliveries={len(s):,}")
    s, team_fx = attach_opposition(s, raw)
    team_fx.to_csv(C.PRESSURE_OUT_DIR / "team_effects.csv", index=False)
    chk = s[s.legal].assign(value=lambda d: d.team_runs - C.WICKET_RUNS * d.is_wicket.astype(int))
    print("\nRun value per ball by strength of the attack the batter faces (justifies the adjustment):")
    print(chk.groupby(["pool", "bucket_faced_by_bat"]).value.agg(balls="size", value_per_ball="mean").round(3).to_string())

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

    # ---- ratings for everyone, then the auction-available top 10
    unavailable = unavailable_players(raw, names)
    for role, label in [("bat", "BATTERS"), ("bowl", "BOWLERS")]:
        t0 = rate_players(r, role, names, adjust=False)          # no opposition adjustment (for comparison)
        t0["unavailable_reason"] = t0.index.map(unavailable).fillna("")
        t0.to_csv(C.PRESSURE_OUT_DIR / f"ratings_{role}_unadjusted.csv")
        t = rate_players(r, role, names, adjust=True)
        t["unavailable_reason"] = t.index.map(unavailable).fillna("")
        t["available"] = t.unavailable_reason == ""
        t["rank_auction"] = t.blend_lcb.where(t.available).rank(ascending=False, method="min")
        t.to_csv(C.PRESSURE_OUT_DIR / f"ratings_{role}.csv")
        t.sort_values("rank").head(10).to_csv(C.PRESSURE_OUT_DIR / f"top10_{role}_all_players.csv")
        auction = t[t.available].sort_values("rank_auction")
        auction.head(10).to_csv(C.PRESSURE_OUT_DIR / f"top10_{role}_auction.csv")
        cols = ["name", "rank_auction", "rank", "blend", "blend_lcb", "abs_pts100", "delta", "n_p", "matches_p",
                "consistency", "t20i_share"]
        cols += [c for c in t.columns if c.startswith(("sr_p", "sr_o", "eco_p", "eco_o"))]
        print(f"\n=== TOP 10 {label} AVAILABLE AT AUCTION (rated {len(t)}, available {int(t.available.sum())}) ===")
        print(auction.head(10)[cols].round(2).to_string())
        gone = t.sort_values("rank").head(10)
        gone = gone[~gone.available]
        print("Removed from the unfiltered top 10: " + "; ".join(f"{n} ({x})" for n, x in zip(gone.name, gone.unavailable_reason)))
        before = set(t0[t0.unavailable_reason == ""].sort_values("blend_lcb", ascending=False).head(10).name)
        print(f"Opposition adjustment: {len(before & set(auction.head(10).name))}/10 of the auction top 10 unchanged. "
              f"Dropped out: {sorted(before - set(auction.head(10).name))}; came in: {sorted(set(auction.head(10).name) - before)}")


if __name__ == "__main__":
    main()