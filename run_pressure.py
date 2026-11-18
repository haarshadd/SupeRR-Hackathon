"""Round 1 — Final pressure-ranking pipeline."""

import json
import pandas as pd

from rr_scout import config as C
from rr_scout.data import (
    load_raw, prepare_balls, add_match_state, player_names
)
from rr_scout.win_prob import WinProb
from rr_scout.pressure import compute_pressure, KEEP
from rr_scout.ratings import rate_players
from rr_scout.opposition import attach_opposition


def main():
    C.PRESSURE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load and prepare match data
    # --------------------------------------------------
    raw = load_raw()
    b = prepare_balls(raw)
    names = player_names(b)
    s = add_match_state(b, raw)

    print(f"Usable matches: {s.match_id.nunique()}")
    print(f"Deliveries: {len(s):,}")

    # --------------------------------------------------
    # 2. Opposition-strength adjustment
    # --------------------------------------------------
    s, team_fx = attach_opposition(s, raw)

    team_fx.to_csv(
        C.PRESSURE_OUT_DIR / "team_effects.csv",
        index=False
    )

    chk = s[s.legal].assign(
        value=lambda d: (
            d.team_runs - C.WICKET_RUNS * d.is_wicket.astype(int)
        )
    )

    print("\nRun value per ball by opposition strength:")
    print(
        chk.groupby(
            ["pool", "bucket_faced_by_bat"]
        ).value.agg(
            balls="size",
            value_per_ball="mean"
        ).round(3).to_string()
    )

    # --------------------------------------------------
    # 3. Fit win-probability model
    # --------------------------------------------------
    wp = WinProb().fit(s)

    print("\nWin-probability model:")
    print(wp.metrics.round(3).to_string(index=False))

    wp.metrics.to_csv(
        C.PRESSURE_OUT_DIR / "wp_model_metrics.csv",
        index=False
    )

    # --------------------------------------------------
    # 4. Calculate delivery-level pressure
    # --------------------------------------------------
    r = compute_pressure(s, wp)

    missing = [col for col in KEEP if col not in r.columns]
    if missing:
        raise RuntimeError(
            f"Pressure output is missing required columns: {missing}"
        )

    pressure_path = C.PRESSURE_OUT_DIR / "ball_pressure.csv.gz"

    r[KEEP].to_csv(pressure_path, index=False)

    print("\nPressure methodology:")
    print(
        r.attrs.get(
            "first_innings_pressure_rule",
            "First innings: configured collapse rule"
        )
    )

    print(
        f"Chase leverage threshold: "
        f"{r.attrs['leverage_threshold']:.5f}"
    )

    legal = r[r.legal.astype(bool)].copy()

    print("\nPressure share by innings and phase:")
    print(
        legal.groupby(
            ["innings", "phase"], observed=True
        ).high_pressure.mean().unstack().round(3).to_string()
    )

    print("\nPressure-tier counts:")
    print(
        legal.pressure_tier.value_counts().to_string()
    )

    # Chase diagnostics
    ch = legal[legal.innings == 2].copy()

    ch["dot"] = (
        (ch.team_runs == 0) & ~ch.is_wicket.astype(bool)
    )
    ch["boundary"] = ch.team_runs >= 4

    print("\nChase outcomes by phase and pressure:")
    print(
        ch.groupby(
            ["phase", "high_pressure"], observed=True
        ).agg(
            balls=("team_runs", "size"),
            runs_per_ball=("team_runs", "mean"),
            wicket_rate=("is_wicket", "mean"),
            dot_rate=("dot", "mean"),
            boundary_rate=("boundary", "mean"),
        ).round(3).to_string()
    )

    print(
        "\nMatches represented by pressure tier:",
        json.dumps(
            legal.groupby("pressure_tier").match_id.nunique().to_dict()
        )
    )

    # --------------------------------------------------
    # 5. Calculate final player ratings and Top 10s
    # --------------------------------------------------
    for role, label in [
        ("bat", "BATTERS"),
        ("bowl", "BOWLERS")
    ]:
        print("\n" + "=" * 65)
        print(f"FINAL ROUND 1 — TOP 10 {label}")
        print("=" * 65)

        # Unadjusted ratings retained for comparison.
        unadjusted = rate_players(
            r, role, names, adjust=False
        )

        unadjusted.to_csv(
            C.PRESSURE_OUT_DIR /
            f"ratings_{role}_unadjusted.csv"
        )

        # Configured opposition adjustment.
        ratings = rate_players(
            r, role, names, adjust=C.OPPOSITION_ADJUST
        )

        ratings = ratings.sort_values(
            "rank", kind="stable"
        )

        if len(ratings) < 10:
            raise RuntimeError(
                f"Only {len(ratings)} eligible {label.lower()} found. "
                "Cannot produce a Top 10."
            )

        # Verify eligibility requirements.
        min_balls = (
            C.MIN_PRESSURE_BALLS_BAT
            if role == "bat"
            else C.MIN_PRESSURE_BALLS_BOWL
        )

        if (
            (ratings.n_p < min_balls).any()
            or (ratings.matches_p < C.MIN_PRESSURE_MATCHES).any()
        ):
            raise RuntimeError(
                f"{label}: eligibility requirements were not applied correctly."
            )

        ratings.to_csv(
            C.PRESSURE_OUT_DIR / f"ratings_{role}.csv"
        )

        top10 = ratings.head(10).copy()

        # Save the all-player Round 1 ranking.
        top_path = (
            C.PRESSURE_OUT_DIR /
            f"top10_{role}_all_players.csv"
        )

        top10.to_csv(top_path)

        # Read back and verify the actual saved file.
        saved = pd.read_csv(top_path)

        if len(saved) != 10:
            raise RuntimeError(
                f"{top_path.name}: expected 10 players, got {len(saved)}"
            )

        if saved["name"].isna().any():
            raise RuntimeError(
                f"{top_path.name}: missing player names."
            )

        if saved["name"].duplicated().any():
            raise RuntimeError(
                f"{top_path.name}: duplicate player names detected."
            )

        display_cols = [
            "rank", "name", "blend_lcb", "blend",
            "abs_pts100", "delta", "n_p",
            "matches_p", "consistency",
            "sr_p", "eco_p"
        ]

        display_cols = [
            col for col in display_cols
            if col in top10.columns
        ]

        print(
            top10[display_cols].round(3).to_string(index=False)
        )

        print(f"\nSaved and verified: {top_path}")

    print("\n" + "=" * 65)
    print("ROUND 1 PRESSURE RANKING PIPELINE COMPLETED")
    print("=" * 65)

    print("\nFinal files:")
    print(
        C.PRESSURE_OUT_DIR / "top10_bat_all_players.csv"
    )
    print(
        C.PRESSURE_OUT_DIR / "top10_bowl_all_players.csv"
    )


if __name__ == "__main__":
    main()